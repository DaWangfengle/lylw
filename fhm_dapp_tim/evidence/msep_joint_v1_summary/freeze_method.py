"""Freeze the audited method and execution plan before any formal result."""
from pathlib import Path
import datetime
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    audit = json.loads((HERE/"DATA_AUDIT.json").read_text(encoding="utf-8"))
    smoke = json.loads((HERE/"SMOKE_AUDIT.json").read_text(encoding="utf-8"))
    numerical = json.loads((HERE/"NUMERICAL_AUDIT.json").read_text(encoding="utf-8"))
    assert smoke["status"] == numerical["status"] == "PASS"
    sources = json.loads((HERE/"SOURCE_BASELINE.json").read_text(encoding="utf-8"))["files"]
    for item in sources:
        assert sha(ROOT/item["path"]) == item["sha256"], item["path"]
    inputs = {}
    for dataset in audit["datasets"].values():
        inputs.update(dataset["input_sha256"])
    for path, expected in inputs.items():
        assert sha(ROOT/path) == expected
    commands = []
    for seed in (17301, 17302, 17303):
        for domain in ("tuji", "uts", "uji"):
            out = f"fhm_dapp_tim/evidence/msep_joint_v1_{domain}_seed{seed}"
            if (ROOT/out).exists():
                raise FileExistsError(out)
            args = ["-m", "fhm_dapp_tim.fhm_dapp.msep_joint_ablation_v1", "--dataset", domain,
                    "--seed", str(seed), "--split-seed", "20261017", "--base-epochs", "80",
                    "--joint-epochs", "10", "--batch", "64", "--es-draws", "32", "--output", out]
            if seed == 17301:
                args.append("--save-checkpoints")
            commands.append({"label": f"full_{domain}_{seed}", "arguments": args})
    frozen = {"frozen_at": datetime.datetime.now().astimezone().isoformat(),
              "initial_branch": "msep-joint-location-ablation-v1",
              "initial_commit": "92181f4ce6e11449b0972fffd5a2d290939dff57",
              "source_files": sources, "input_sha256": inputs,
              "data_audit_sha256": sha(HERE/"DATA_AUDIT.json"),
              "original_instructions_sha256": sha(HERE/"original/MSEP_JOINT_V1_CODEX_RUN.txt"),
              "smoke_audit_sha256": sha(HERE/"SMOKE_AUDIT.json"),
              "numerical_audit_sha256": sha(HERE/"NUMERICAL_AUDIT.json"),
              "source_changes_before_freeze": [], "formal_commands": commands,
              "test_policy": "No TEST predictions/scores/training/calibration or decisions. Partition metadata only.",
              "scope": "three seeds on identical exposed TRAIN-internal sites; not external/independent blind validation",
              "allowed_runtime_changes": "batch reduction only if documented OOM; none currently planned"}
    path = HERE/"METHOD_FREEZE.json"
    with path.open("x", encoding="utf-8") as f:
        json.dump(frozen, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"status": "FROZEN", "source_files": len(sources), "input_files": len(inputs),
                      "commands": len(commands), "method_freeze_sha256": sha(path)}))


if __name__ == "__main__":
    main()
