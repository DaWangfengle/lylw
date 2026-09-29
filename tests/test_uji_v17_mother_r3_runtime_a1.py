import json

import torch

from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r3 as registered
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r3_runtime_a1 as runtime
from fhm_dapp_tim.fhm_dapp.experiment_v7 import _canonical_json_sha256


def test_flatten_changes_only_the_unused_fourth_return():
    prior = torch.randn(2, 3, 4)
    indices = torch.arange(24).reshape(2, 3, 4)
    radial = torch.randn(2, 3)
    retained = torch.randn(2, 3)

    def original(*_args, **_kwargs):
        return prior, indices, radial, retained

    observed = runtime._flatten_diagnostic_retained_mass(original, None)
    assert observed[0] is prior
    assert observed[1] is indices
    assert observed[2] is radial
    assert torch.equal(observed[3], retained.reshape(-1))


def test_adapter_augments_write_and_restores_registered_functions(tmp_path):
    original_top_l = registered._member_top_l
    original_write = registered._write_json_new
    destination = tmp_path / "result.json"
    captured = {}

    def fake_top_l(*_args, **_kwargs):
        return (
            torch.ones(1, 2, 1),
            torch.zeros(1, 2, 1, dtype=torch.long),
            torch.ones(1, 2),
            torch.ones(1, 2),
        )

    def fake_write(path, payload):
        captured.update(payload)
        path.write_text(json.dumps(payload), encoding="utf-8")

    registered._member_top_l = fake_top_l
    registered._write_json_new = fake_write
    amendment = {"amendment_sha256": "abc"}
    try:
        with runtime._diagnostic_flatten_adapter(amendment):
            outputs = registered._member_top_l(None)
            assert outputs[3].shape == (2,)
            registered._write_json_new(destination, {"value": 7, "result_sha256": "old"})
        assert registered._member_top_l is fake_top_l
        assert registered._write_json_new is fake_write
    finally:
        registered._member_top_l = original_top_l
        registered._write_json_new = original_write

    assert captured["runtime_protocol"] == runtime.RUNTIME_PROTOCOL
    assert captured["outer_test_access_number"] == 2
    assert captured["outer_test_execution_attempts"] == 2
    assert captured["successful_outer_result_writes"] == 1
    assert captured["execution_amendment"]["scientific_change"] is False
    assert captured["execution_amendment"]["predictive_law_changed"] is False
    assert captured["result_sha256"] == _canonical_json_sha256(
        captured, omit="result_sha256"
    )


def test_registered_execution_source_is_not_modified_by_runtime_adapter():
    assert runtime.registered_r3 is registered
    assert runtime.RUNTIME_PROTOCOL.endswith("v1_20260822")

