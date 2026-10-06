"""Final outcome-free acceptance gate before formal V18.5 UJI execution.

This runner is intentionally separate from the formal experiment driver.
It compiles the registered implementation, runs every V18.5 contract test,
executes CPU/CUDA training-only smoke, runs the multi-dataset revision audit,
checks stale formal artifacts, and records a method-contract hash.

It never opens UJI validation or any external provided test split and never
creates a formal model checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import py_compile
import re
import subprocess
import sys
from typing import Sequence

import torch

from fhm_dapp_tim.fhm_dapp.v18_5_contract import (
    METHOD_CONTRACT_PATHS,
    method_contract,
)
from fhm_dapp_tim.fhm_dapp.v18_5_environment import environment_fingerprint
from fhm_dapp_tim.fhm_dapp.v18_5_realdata_smoke import run_smoke
from fhm_dapp_tim.fhm_dapp.v18_5_revision_audit import (
    _formal_artifact_absence,
    run_audit,
)
from fhm_dapp_tim.fhm_dapp.uji_v18_5_development import (
    MODEL_SEED,
    PREDICTIVE_DRAWS,
    SCORING_SEED,
    _set_seed,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "fhm_dapp_tim"
EVIDENCE_ROOT = PACKAGE_ROOT / "evidence" / "v18_5"
DEFAULT_JSON = EVIDENCE_ROOT / "PRE_RUN_ACCEPTANCE_V18_5.json"
DEFAULT_MD = EVIDENCE_ROOT / "PRE_RUN_ACCEPTANCE_V18_5.md"

EXTERNAL_SLUGS = (
    "uts",
    "tampere",
    "tuji-pooled",
    "tuji-lodo-A12",
    "tuji-lodo-POCO",
    "tuji-lodo-S20",
    "tuji-lodo-S7",
    "tuji-lodo-tabS7",
)

AUXILIARY_VERIFICATION_PATHS = (
    "fhm_dapp_tim/fhm_dapp/v18_5_realdata_smoke.py",
    "fhm_dapp_tim/fhm_dapp/v18_5_revision_audit.py",
    "fhm_dapp_tim/fhm_dapp/v18_5_artifact_audit.py",
    "fhm_dapp_tim/fhm_dapp/v18_5_external_artifact_audit.py",
    "fhm_dapp_tim/fhm_dapp/v18_5_pre_run_acceptance.py",
    ".github/workflows/v18_5_verification.yml",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_state() -> dict[str, object]:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    if len(head) != 40:
        raise RuntimeError("invalid Git HEAD")
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    dirty = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--"],
        cwd=REPO_ROOT,
        check=False,
    ).returncode
    if dirty == 1:
        raise RuntimeError("pre-run acceptance requires clean tracked source")
    if dirty != 0:
        raise RuntimeError("cannot inspect tracked source state")
    return {
        "source_commit_sha": head,
        "branch": branch,
        "tracked_source_dirty": False,
    }


def _hash_manifest(paths: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in paths:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        result[relative] = _sha256_file(path)
    return result


def _method_contract() -> dict[str, object]:
    return method_contract(repo_root=REPO_ROOT)


def _verification_contract() -> dict[str, object]:
    files = _hash_manifest(AUXILIARY_VERIFICATION_PATHS)
    return {
        "files": files,
        "verification_sha256": _canonical_sha(files),
    }


def _compile_targets() -> list[str]:
    """Compile every Python file that participates in method or verification."""

    relative_paths = {
        path
        for path in (*METHOD_CONTRACT_PATHS, *AUXILIARY_VERIFICATION_PATHS)
        if path.endswith(".py")
    }
    if not relative_paths:
        raise RuntimeError("no V18.5 method/verification modules found")
    compiled = []
    for relative in sorted(relative_paths):
        path = REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        py_compile.compile(str(path), doraise=True)
        compiled.append(relative)
    return compiled


def _test_targets() -> list[str]:
    test_root = REPO_ROOT / "tests"
    paths = sorted(test_root.glob("test_v18_5*.py"))
    uji = test_root / "test_uji_v18_5_development.py"
    if uji.is_file() and uji not in paths:
        paths.append(uji)
    if not paths:
        raise RuntimeError("no V18.5 contract tests found")
    return [str(path.relative_to(REPO_ROOT)) for path in paths]


def _parse_pytest_output(output: str) -> dict[str, object]:
    if re.search(r"\b(failed|error|errors)\b", output, flags=re.IGNORECASE):
        raise RuntimeError("pytest output contains failure/error")
    for forbidden in ("skipped", "xfailed", "xpassed"):
        if re.search(rf"\b{forbidden}\b", output, flags=re.IGNORECASE):
            raise RuntimeError(f"pytest output contains {forbidden}")
    matches = re.findall(r"(\d+) passed(?:,| in)", output)
    if not matches:
        raise RuntimeError("could not parse pytest passed count")
    passed = int(matches[-1])
    if passed <= 0:
        raise RuntimeError("pytest did not execute any tests")
    return {"passed": passed, "no_skip_xfail": True}


def _run_tests(targets: Sequence[str]) -> dict[str, object]:
    command = [sys.executable, "-m", "pytest", "-q", *targets]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "V18.5 contract tests failed:\n" + completed.stdout[-12000:]
        )
    summary = _parse_pytest_output(completed.stdout)
    summary.update({
        "targets": list(targets),
        "target_hashes": _hash_manifest(targets),
    })
    return summary


def _external_artifact_absence() -> dict[str, bool]:
    evidence = EVIDENCE_ROOT / "external"
    models = PACKAGE_ROOT / "models" / "v18_5" / "external"
    suffixes = (
        "_training_manifest.json",
        "_training_freeze.json",
        "_test_consumption_lock.json",
        "_test_result.json",
    )
    paths: list[Path] = []
    for slug in EXTERNAL_SLUGS:
        for suffix in suffixes:
            paths.append(evidence / f"{slug}{suffix}")
        paths.append(models / f"{slug}_selection_refit.pt")
    checked: list[Path] = []
    for path in paths:
        checked.append(path)
        checked.append(path.with_name(path.name + ".tmp"))
    result = {
        str(path.relative_to(REPO_ROOT)): not path.exists()
        for path in checked
    }
    present = sorted(key for key, absent in result.items() if not absent)
    if present:
        raise RuntimeError(
            "stale formal V18.5 external artifacts exist: "
            + ", ".join(present)
        )
    return result


def _environment_fingerprint() -> dict[str, object]:
    return environment_fingerprint(
        model_seed=MODEL_SEED,
        scoring_seed=SCORING_SEED,
        predictive_draws=PREDICTIVE_DRAWS,
    )

def _smoke(*, require_cuda: bool) -> dict[str, object]:
    cpu = run_smoke("cpu")
    if cpu.get("status") != "success":
        raise RuntimeError("V18.5 CPU smoke did not succeed")
    cuda_available = bool(torch.cuda.is_available())
    if require_cuda and not cuda_available:
        raise RuntimeError("CUDA is required by formal V18.5 acceptance")
    cuda: dict[str, object]
    if cuda_available:
        cuda = run_smoke("cuda")
        if cuda.get("status") != "success":
            raise RuntimeError("V18.5 CUDA smoke did not succeed")
    else:
        cuda = {"status": "not_available"}
    return {
        "cpu": cpu,
        "cuda": cuda,
        "cuda_required": bool(require_cuda),
        "cuda_available": cuda_available,
    }


def run_acceptance(*, require_cuda: bool = False) -> dict[str, object]:
    _set_seed(MODEL_SEED)
    start_git = _git_state()
    method = _method_contract()
    verification = _verification_contract()
    uji_absence_before = _formal_artifact_absence()
    external_absence_before = _external_artifact_absence()

    compiled = _compile_targets()
    tests = _run_tests(_test_targets())
    smoke = _smoke(require_cuda=require_cuda)
    revision = run_audit(include_smoke=False)
    if revision.get("status") != "PASS":
        raise RuntimeError("V18.5 revision audit did not PASS")
    environment = _environment_fingerprint()
    if environment["torch"]["deterministic_algorithms_enabled"] is not True:
        raise RuntimeError("deterministic algorithms are not enabled")
    if environment["determinism"]["cublas_workspace_config"] != ":4096:8":
        raise RuntimeError("unexpected CUBLAS workspace configuration")

    # No verification step is allowed to create or alter formal outputs.
    uji_absence_after = _formal_artifact_absence()
    external_absence_after = _external_artifact_absence()
    end_git = _git_state()
    if end_git["source_commit_sha"] != start_git["source_commit_sha"]:
        raise RuntimeError("Git HEAD changed during pre-run acceptance")
    if _method_contract() != method:
        raise RuntimeError("method contract changed during pre-run acceptance")
    if _verification_contract() != verification:
        raise RuntimeError(
            "verification contract changed during pre-run acceptance"
        )

    payload = {
        "acceptance_version": "v18_5_pre_run_acceptance_v1_20261006",
        "status": "PASS",
        "scientific_status": "OUTCOME_FREE_PRE_RUN_ACCEPTED",
        "git": start_git,
        "method_contract": method,
        "verification_contract": verification,
        "environment": environment,
        "environment_sha256": _canonical_sha(environment),
        "compile": {
            "modules": compiled,
            "count": int(len(compiled)),
        },
        "tests": tests,
        "smoke": smoke,
        "revision_audit": revision,
        "formal_uji_artifacts_absent_before": uji_absence_before,
        "formal_uji_artifacts_absent_after": uji_absence_after,
        "formal_external_artifacts_absent_before": external_absence_before,
        "formal_external_artifacts_absent_after": external_absence_after,
        "official_uji_validation_accessed": False,
        "external_test_performance_accessed": False,
        "formal_v185_training_executed": False,
        "formal_v185_test_scoring_executed": False,
    }
    payload["acceptance_sha256"] = _canonical_sha(payload)
    return payload


def _markdown(payload: dict[str, object]) -> str:
    smoke = payload["smoke"]
    revision = payload["revision_audit"]
    lines = [
        "# V18.5 Pre-run Acceptance",
        "",
        f"Status: **{payload['status']}**",
        "",
        f"- Source commit: `{payload['git']['source_commit_sha']}`",
        f"- Method contract SHA256: `{payload['method_contract']['contract_sha256']}`",
        f"- Environment SHA256: `{payload['environment_sha256']}`",
        f"- PyTorch: {payload['environment']['packages']['torch']}",
        f"- CUDA build: {payload['environment']['torch']['cuda_build']}",
        f"- CUDA devices: {payload['environment']['torch']['cuda_device_count']}",
        f"- Compiled modules: {payload['compile']['count']}",
        f"- Contract tests: {payload['tests']['passed']} passed; no skip/xfail",
        f"- CPU smoke: {smoke['cpu']['status']}",
        f"- CUDA required: {smoke['cuda_required']}",
        f"- CUDA smoke: {smoke['cuda']['status']}",
        f"- Multi-dataset training-only audit: {revision['status']}",
        "",
        "All registered UJI and external formal artifacts were absent before and",
        "after acceptance. No UJI validation or external test performance was",
        "accessed. No formal V18.5 model training or test scoring was executed.",
        "",
    ]
    return "\n".join(lines)


def _write_new(path: Path, content: str) -> None:
    """Atomically write one acceptance artifact and never overwrite it."""

    temporary = path.with_name(path.name + ".tmp")
    if path.exists():
        raise FileExistsError(path)
    if temporary.exists():
        raise RuntimeError(f"stale partial acceptance artifact: {temporary}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(path)
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--json", default=str(DEFAULT_JSON))
    parser.add_argument("--markdown", default=str(DEFAULT_MD))
    args = parser.parse_args()

    json_path = Path(args.json)
    markdown_path = Path(args.markdown)
    outputs = (json_path, markdown_path)
    if any(path.exists() for path in outputs):
        raise FileExistsError("V18.5 acceptance outputs are write-once")
    stale = [
        path.with_name(path.name + ".tmp")
        for path in outputs
        if path.with_name(path.name + ".tmp").exists()
    ]
    if stale:
        raise RuntimeError(
            "stale partial V18.5 acceptance output(s): "
            + ", ".join(map(str, stale))
        )

    payload = run_acceptance(require_cuda=bool(args.require_cuda))
    _write_new(
        json_path,
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    _write_new(markdown_path, _markdown(payload))
    print(json.dumps({
        "status": payload["status"],
        "acceptance_sha256": payload["acceptance_sha256"],
        "method_contract_sha256": payload["method_contract"]["contract_sha256"],
        "tests_passed": payload["tests"]["passed"],
        "cuda_required": payload["smoke"]["cuda_required"],
        "cuda_status": payload["smoke"]["cuda"]["status"],
        "json": str(json_path),
        "json_sha256": _sha256_file(json_path),
        "markdown": str(markdown_path),
        "markdown_sha256": _sha256_file(markdown_path),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
