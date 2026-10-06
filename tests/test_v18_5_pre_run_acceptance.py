import json
from pathlib import Path
from unittest.mock import patch

import pytest

from fhm_dapp_tim.fhm_dapp import v18_5_pre_run_acceptance as acceptance


def test_parse_pytest_output_accepts_clean_pass_summary():
    result = acceptance._parse_pytest_output(
        "................................................................ [100%]\n"
        "73 passed in 6.10s\n"
    )
    assert result == {"passed": 73, "no_skip_xfail": True}


@pytest.mark.parametrize(
    "summary",
    [
        "72 passed, 1 skipped in 5.0s",
        "72 passed, 1 xfailed in 5.0s",
        "72 passed, 1 xpassed in 5.0s",
        "1 failed, 72 passed in 5.0s",
        "ERROR collecting tests/test_demo.py",
    ],
)
def test_parse_pytest_output_rejects_nonclean_suite(summary):
    with pytest.raises(RuntimeError):
        acceptance._parse_pytest_output(summary)


def test_test_target_discovery_includes_external_and_uji_contracts():
    targets = acceptance._test_targets()
    assert "tests/test_uji_v18_5_development.py" in targets
    assert "tests/test_v18_5_external.py" in targets
    assert "tests/test_v18_5_external_artifact_audit.py" in targets
    assert "tests/test_v18_5_pre_run_acceptance.py" in targets


def test_external_artifact_absence_checks_all_eight_protocols(tmp_path: Path):
    package = tmp_path / "fhm_dapp_tim"
    evidence = package / "evidence" / "v18_5"
    models = package / "models" / "v18_5"
    with patch.object(acceptance, "PACKAGE_ROOT", package), patch.object(
        acceptance, "EVIDENCE_ROOT", evidence
    ), patch.object(acceptance, "REPO_ROOT", tmp_path):
        result = acceptance._external_artifact_absence()
        assert len(result) == 8 * 5 * 2
        target = (
            evidence
            / "external"
            / "tuji-lodo-A12_test_consumption_lock.json"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}", encoding="utf-8")
        with pytest.raises(RuntimeError, match="stale formal"):
            acceptance._external_artifact_absence()


def test_acceptance_contract_hash_includes_runner_itself():
    assert (
        "fhm_dapp_tim/fhm_dapp/v18_5_pre_run_acceptance.py"
        in acceptance.AUXILIARY_VERIFICATION_PATHS
    )


def test_external_artifact_absence_rejects_stale_partial(tmp_path: Path):
    package = tmp_path / "fhm_dapp_tim"
    evidence = package / "evidence" / "v18_5"
    models = package / "models" / "v18_5"
    with patch.object(acceptance, "PACKAGE_ROOT", package), patch.object(
        acceptance, "EVIDENCE_ROOT", evidence
    ), patch.object(acceptance, "REPO_ROOT", tmp_path):
        partial = (
            models
            / "external"
            / "uts_selection_refit.pt.tmp"
        )
        partial.parent.mkdir(parents=True, exist_ok=True)
        partial.write_bytes(b"partial")
        with pytest.raises(RuntimeError, match="stale formal"):
            acceptance._external_artifact_absence()


def test_method_contract_hash_includes_contract_definition_itself():
    assert (
        "fhm_dapp_tim/fhm_dapp/v18_5_contract.py"
        in acceptance.METHOD_CONTRACT_PATHS
    )
    contract = acceptance._method_contract()
    assert set(contract["files"]) == set(acceptance.METHOD_CONTRACT_PATHS)
    assert len(contract["contract_sha256"]) == 64


def test_method_contract_hash_includes_all_direct_shared_dependencies():
    required = {
        "fhm_dapp_tim/fhm_dapp/model.py",
        "fhm_dapp_tim/fhm_dapp/dapp.py",
        "fhm_dapp_tim/fhm_dapp/coherent_marked_v9.py",
        "fhm_dapp_tim/fhm_dapp/v18_single.py",
        "fhm_dapp_tim/fhm_dapp/v18_4_datasets.py",
        "fhm_dapp_tim/fhm_dapp/v18_4_protocol.py",
    }
    assert required.issubset(set(acceptance.METHOD_CONTRACT_PATHS))


def test_acceptance_compile_targets_cover_method_contract_python_files():
    compiled = set(acceptance._compile_targets())
    required_python = {
        path for path in acceptance.METHOD_CONTRACT_PATHS
        if path.endswith(".py")
    }
    assert required_python.issubset(compiled)


def test_environment_fingerprint_is_json_serializable_and_records_determinism():
    acceptance._set_seed(acceptance.MODEL_SEED)
    environment = acceptance._environment_fingerprint()
    json.dumps(environment, sort_keys=True, allow_nan=False)
    assert environment["torch"]["deterministic_algorithms_enabled"] is True
    assert environment["determinism"]["cublas_workspace_config"] == ":4096:8"
    assert environment["determinism"]["model_seed"] == acceptance.MODEL_SEED
    assert environment["determinism"]["scoring_seed"] == acceptance.SCORING_SEED
    assert (
        environment["determinism"]["predictive_draws"]
        == acceptance.PREDICTIVE_DRAWS
    )
    assert set(environment["packages"]) == {
        "numpy", "pandas", "scipy", "torch"
    }
    if environment["torch"]["cuda_available"]:
        assert environment["torch"]["cuda_device_count"] >= 1
        assert len(environment["torch"]["cuda_devices"]) == (
            environment["torch"]["cuda_device_count"]
        )
