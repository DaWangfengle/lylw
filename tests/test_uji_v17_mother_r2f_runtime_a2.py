from __future__ import annotations

import json
from pathlib import Path

import torch

from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r2f as registered_r2f
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r2f_runtime_a2 as runtime
from fhm_dapp_tim.fhm_dapp.experiment_v7 import _canonical_json_sha256, _sha256_file


def _audit_fixture() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prior = torch.tensor(
        [
            [[0.60, 0.30, 0.10], [0.55, 0.35, 0.10]],
            [[0.50, 0.30, 0.20], [0.50, 0.25, 0.25]],
        ],
        dtype=torch.float32,
    )
    indices = torch.tensor(
        [
            [[0, 2, 4], [0, 1, 3]],
            [[1, 2, 3], [0, 2, 4]],
        ],
        dtype=torch.long,
    )
    radial = torch.tensor(
        [[1.0, 1.5], [2.0, 1.0]], dtype=torch.float32
    )
    return prior, indices, radial


def test_operator_equivalence_audit_compares_global_anchor_measures() -> None:
    result = runtime._operator_equivalence_audit(
        *_audit_fixture(),
        anchor_count=5,
        top_k=2,
        chunk_size=1,
        atol=runtime.LAW_EQUIVALENCE_ATOL,
    )
    assert result["passed"] is True
    assert result["rows"] == 2
    assert result["execution_device"] == "cpu"
    assert result["maximum_anchor_probability_difference"] <= 1e-6


def test_operator_audit_is_repeatable() -> None:
    first = runtime._operator_equivalence_audit(
        *_audit_fixture(), anchor_count=5, top_k=2
    )
    second = runtime._operator_equivalence_audit(
        *_audit_fixture(), anchor_count=5, top_k=2
    )
    assert first == second


def test_a2_amendment_is_self_hashed_and_records_both_failed_predecessors() -> None:
    payload = json.loads(runtime.DEFAULT_AMENDMENT.read_text(encoding="utf-8"))
    assert payload["amendment_sha256"] == _canonical_json_sha256(
        payload, omit="amendment_sha256"
    )
    assert payload["scientific_change"] is False
    assert payload["selection_authority_for_v17_r2f"] is False
    assert payload["may_modify_v17_r2f_model"] is False
    assert payload["created_after_failed_r2f_a1_evaluation_attempt"] is True
    assert payload["incident"]["development_result_existed_after_a1_failure"] is False
    assert payload["incident"]["outer_or_official_data_accessed"] is False


def test_a2_binds_original_and_a1_artifacts() -> None:
    payload = runtime._load_amendment()
    for record in payload["bindings"].values():
        assert Path(record["path"]).is_file()
        assert _sha256_file(record["path"]) == record["file_sha256"]
    assert payload["bindings"]["a1_amendment"]["path"].endswith(
        "amendment_v17_r2f_a1_deterministic_consensus.json"
    )


def test_a2_preserves_registered_science_and_forbids_energy_identity_gate() -> None:
    payload = runtime._load_amendment()
    unchanged = payload["unchanged_scientific_contract"]
    assert payload["operator_equivalence_atol"] == runtime.LAW_EQUIVALENCE_ATOL
    assert unchanged["member_grid"] == list(registered_r2f.MEMBER_GRID)
    assert unchanged["support_l_grid"] == list(registered_r2f.SUPPORT_L_GRID)
    assert unchanged["support_k"] == registered_r2f.SUPPORT_K
    assert unchanged["alpha"] == registered_r2f.CONSENSUS_ALPHA
    assert unchanged["bootstrap_repetitions"] == registered_r2f.BOOTSTRAP_REPETITIONS
    assert unchanged["bootstrap_seed"] == registered_r2f.BOOTSTRAP_SEED
    assert unchanged["historical_energy_identity_is_a_gate"] is False
    assert unchanged["outer_test_access"] is False
    assert unchanged["official_validation_access"] is False
