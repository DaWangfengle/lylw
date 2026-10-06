import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from fhm_dapp_tim.fhm_dapp import uji_v18_5_development as driver
from fhm_dapp_tim.fhm_dapp.uji_v18_5_development import _select_global_lambda
from fhm_dapp_tim.fhm_dapp.v18_5_model import V185_CONFIG


def _report(lam, es, nll, mae):
    return {
        "lambda_spatial": lam,
        "selected_epoch": 12,
        "selected_base_metrics": {
            "energy_score_m": es,
            "nll": nll,
            "posterior_mean_mae_m": mae,
        },
    }


def test_global_lambda_selector_uses_es_shortlist_then_nll_mean_and_smaller_lambda():
    reports = [
        _report(0.0, 5.00, 6.3, 7.0),
        _report(0.1, 5.03, 6.1, 7.1),
        _report(0.25, 5.04, 6.1, 6.9),
        _report(0.5, 5.20, 5.0, 5.0),
    ]
    selected = _select_global_lambda(reports)
    assert selected["lambda_spatial"] == 0.25


def test_joint_surface_is_exactly_192_cells_and_contains_identity():
    assert (
        len(V185_CONFIG["temperature_grid"])
        * len(V185_CONFIG["epsilon_kl_grid"])
        * len(V185_CONFIG["scale_grid"])
        == 192
    )
    assert 0.0 in V185_CONFIG["epsilon_kl_grid"]


def test_v185_driver_has_no_gm_or_geometry_correction_selection_path():
    root = Path(__file__).resolve().parents[1]
    driver = (
        root / "fhm_dapp_tim" / "fhm_dapp" / "uji_v18_5_development.py"
    ).read_text(encoding="utf-8")
    model = (
        root / "fhm_dapp_tim" / "fhm_dapp" / "v18_5_model.py"
    ).read_text(encoding="utf-8")
    assert "weighted_component_geometric_median" not in driver
    assert "_masked_geometry_energy" not in model
    assert "geometry_reliability" not in model
    assert "posterior_mean" in driver


def test_development_does_not_select_epsilon():
    root = Path(__file__).resolve().parents[1]
    driver = (
        root / "fhm_dapp_tim" / "fhm_dapp" / "uji_v18_5_development.py"
    ).read_text(encoding="utf-8")
    select_block = driver.split("def _select_global_lambda", 1)[1].split(
        "def develop", 1
    )[0]
    assert "epsilon" not in select_block


def test_atomic_json_write_is_valid_and_write_once(tmp_path: Path):
    path = tmp_path / "artifact.json"
    driver._write_json_new(path, {"value": 3})
    assert json.loads(path.read_text(encoding="utf-8")) == {"value": 3}
    assert not driver._atomic_temp_path(path).exists()
    with pytest.raises(FileExistsError):
        driver._write_json_new(path, {"value": 4})


def test_atomic_json_write_rejects_stale_partial(tmp_path: Path):
    path = tmp_path / "artifact.json"
    partial = driver._atomic_temp_path(path)
    partial.write_text("partial", encoding="utf-8")
    with pytest.raises(RuntimeError, match="stale partial"):
        driver._write_json_new(path, {"value": 3})
    assert partial.exists()
    assert not path.exists()


def test_atomic_checkpoint_write_is_loadable_and_write_once(tmp_path: Path):
    path = tmp_path / "checkpoint.pt"
    payload = {"tensor": torch.tensor([1.0, 2.0])}
    driver._save_torch_new(path, payload)
    restored = torch.load(path, map_location="cpu", weights_only=False)
    assert torch.equal(restored["tensor"], payload["tensor"])
    assert not driver._atomic_temp_path(path).exists()
    with pytest.raises(FileExistsError):
        driver._save_torch_new(path, payload)


def test_atomic_checkpoint_failure_cleans_partial(tmp_path: Path):
    path = tmp_path / "checkpoint.pt"
    partial = driver._atomic_temp_path(path)

    def fail_after_partial(payload, handle):
        handle.write(b"partial")
        raise RuntimeError("synthetic save failure")

    with patch.object(driver.torch, "save", side_effect=fail_after_partial):
        with pytest.raises(RuntimeError, match="synthetic save failure"):
            driver._save_torch_new(path, {"value": 1})
    assert not path.exists()
    assert not partial.exists()



def _fake_acceptance_payload(source_sha: str):
    contract = driver.method_contract(repo_root=driver.REPO_ROOT)
    payload = {
        "acceptance_version": "test",
        "status": "PASS",
        "scientific_status": "OUTCOME_FREE_PRE_RUN_ACCEPTED",
        "git": {
            "source_commit_sha": source_sha,
            "branch": "v18-5-joint-radio-calibration",
            "tracked_source_dirty": False,
        },
        "method_contract": contract,
        "verification_contract": {
            "files": {},
            "verification_sha256": "test",
        },
        "environment": {
            "python": {
                "version": "test",
                "implementation": "CPython",
                "executable": "python",
            },
            "platform": {
                "system": "test",
                "release": "test",
                "machine": "test",
                "platform": "test",
            },
            "packages": {
                "numpy": "test",
                "pandas": "test",
                "scipy": "test",
                "torch": "test",
            },
            "torch": {
                "default_dtype": "torch.float32",
                "cuda_build": "test",
                "cudnn_version": 1,
                "cuda_available": True,
                "cuda_device_count": 1,
                "cuda_devices": [{
                    "index": 0,
                    "name": "test-gpu",
                    "compute_capability": [8, 0],
                    "total_memory_bytes": 1,
                }],
                "deterministic_algorithms_enabled": True,
                "cudnn_available": True,
                "cudnn_deterministic": True,
                "cudnn_benchmark": False,
                "scaled_dot_product_attention": {
                    "flash_enabled": False,
                    "memory_efficient_enabled": False,
                    "math_enabled": True,
                },
            },
            "determinism": {
                "cublas_workspace_config": ":4096:8",
                "model_seed": driver.MODEL_SEED,
                "scoring_seed": driver.SCORING_SEED,
                "predictive_draws": driver.PREDICTIVE_DRAWS,
            },
        },
        "compile": {"modules": [], "count": 0},
        "tests": {"passed": 1, "no_skip_xfail": True},
        "smoke": {
            "cpu": {"status": "success"},
            "cuda": {"status": "success"},
            "cuda_required": True,
            "cuda_available": True,
        },
        "revision_audit": {"status": "PASS"},
        "formal_uji_artifacts_absent_before": {},
        "formal_uji_artifacts_absent_after": {},
        "formal_external_artifacts_absent_before": {},
        "formal_external_artifacts_absent_after": {},
        "official_uji_validation_accessed": False,
        "external_test_performance_accessed": False,
        "formal_v185_training_executed": False,
        "formal_v185_test_scoring_executed": False,
    }
    payload["environment_sha256"] = driver._canonical_sha(
        payload["environment"]
    )
    payload["acceptance_sha256"] = driver._canonical_sha(payload)
    return payload


def _write_acceptance(path: Path, payload):
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_formal_driver_accepts_matching_cuda_pre_run_acceptance(tmp_path: Path):
    source = "a" * 40
    path = tmp_path / "acceptance.json"
    payload = _fake_acceptance_payload(source)
    _write_acceptance(path, payload)
    with patch.object(
        driver,
        "_git_source_state",
        return_value={"source_commit_sha": source, "tracked_source_dirty": False},
    ):
        loaded = driver.load_pre_run_acceptance(path)
    assert loaded["acceptance_sha256"] == payload["acceptance_sha256"]


def test_formal_driver_rejects_acceptance_from_different_source(tmp_path: Path):
    path = tmp_path / "acceptance.json"
    payload = _fake_acceptance_payload("a" * 40)
    _write_acceptance(path, payload)
    with patch.object(
        driver,
        "_git_source_state",
        return_value={"source_commit_sha": "b" * 40, "tracked_source_dirty": False},
    ):
        with pytest.raises(RuntimeError, match="different source"):
            driver.load_pre_run_acceptance(path)


def test_formal_driver_rejects_changed_method_contract(tmp_path: Path):
    source = "a" * 40
    path = tmp_path / "acceptance.json"
    payload = _fake_acceptance_payload(source)
    payload["method_contract"] = dict(payload["method_contract"])
    payload["method_contract"]["contract_sha256"] = "0" * 64
    payload["acceptance_sha256"] = driver._canonical_sha(
        payload, omit="acceptance_sha256"
    )
    _write_acceptance(path, payload)
    with patch.object(
        driver,
        "_git_source_state",
        return_value={"source_commit_sha": source, "tracked_source_dirty": False},
    ):
        with pytest.raises(RuntimeError, match="method contract"):
            driver.load_pre_run_acceptance(path)


def test_formal_driver_rejects_acceptance_without_cuda_pass(tmp_path: Path):
    source = "a" * 40
    path = tmp_path / "acceptance.json"
    payload = _fake_acceptance_payload(source)
    payload["smoke"]["cuda"]["status"] = "not_available"
    payload["smoke"]["cuda_available"] = False
    payload["acceptance_sha256"] = driver._canonical_sha(
        payload, omit="acceptance_sha256"
    )
    _write_acceptance(path, payload)
    with patch.object(
        driver,
        "_git_source_state",
        return_value={"source_commit_sha": source, "tracked_source_dirty": False},
    ):
        with pytest.raises(ValueError, match="CUDA"):
            driver.load_pre_run_acceptance(path)


def test_formal_driver_rejects_tampered_acceptance_environment(tmp_path: Path):
    source = "a" * 40
    path = tmp_path / "acceptance.json"
    payload = _fake_acceptance_payload(source)
    payload["environment"]["determinism"]["model_seed"] += 1
    payload["acceptance_sha256"] = driver._canonical_sha(
        payload, omit="acceptance_sha256"
    )
    _write_acceptance(path, payload)
    with patch.object(
        driver,
        "_git_source_state",
        return_value={"source_commit_sha": source, "tracked_source_dirty": False},
    ):
        with pytest.raises(ValueError, match="environment hash"):
            driver.load_pre_run_acceptance(path)


def test_formal_driver_rejects_wrong_accepted_seed_even_with_valid_hash(
    tmp_path: Path,
):
    source = "a" * 40
    path = tmp_path / "acceptance.json"
    payload = _fake_acceptance_payload(source)
    payload["environment"]["determinism"]["model_seed"] += 1
    payload["environment_sha256"] = driver._canonical_sha(
        payload["environment"]
    )
    payload["acceptance_sha256"] = driver._canonical_sha(
        payload, omit="acceptance_sha256"
    )
    _write_acceptance(path, payload)
    with patch.object(
        driver,
        "_git_source_state",
        return_value={"source_commit_sha": source, "tracked_source_dirty": False},
    ):
        with pytest.raises(ValueError, match="model seed"):
            driver.load_pre_run_acceptance(path)
