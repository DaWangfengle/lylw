import inspect
from pathlib import Path

from fhm_dapp_tim.fhm_dapp import historical_reproduction as repro
from fhm_dapp_tim.fhm_dapp import uji_v9_experiment as v9
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_r3_protocol as v17_r3_protocol


def test_stage_uses_only_historical_v9_v17_modules():
    source = Path(repro.__file__).read_text(encoding="utf-8")
    assert "import uji_v18" not in source
    assert "import uji_v19" not in source
    assert "from fhm_dapp_tim.fhm_dapp.uji_v18" not in source
    assert "from fhm_dapp_tim.fhm_dapp.uji_v19" not in source


def test_official_validation_is_not_a_predictive_input():
    source = Path(repro.__file__).read_text(encoding="utf-8")
    assert "validationData.csv" not in source
    assert repro.TRAIN_DATA.name == "trainingData.csv"


def test_v9_seed_budget_and_initial_member_count_are_frozen():
    assert tuple(v9.MEMBER_SEEDS) == tuple(range(9311, 9323))
    assert v9.INITIAL_MEMBER_COUNT == 8
    assert len(v9.MEMBER_SEEDS) == 12


def test_v17_r3_selected_and_historical_laws_are_exact():
    assert tuple(v17_r3_protocol.MEMBER_SEEDS) == tuple(range(17301, 17326))
    assert v17_r3_protocol.CONFIGURATIONS[v17_r3_protocol.SELECTED_NAME] == {
        "E": 16,
        "L": 192,
        "K": 192,
        "alpha": 1.0,
    }
    assert v17_r3_protocol.CONFIGURATIONS[v17_r3_protocol.HISTORICAL_NAME] == {
        "E": 5,
        "L": 24,
        "K": 24,
        "alpha": 0.25,
    }


def test_v17_replay_uses_all_twenty_five_registered_seeds():
    assert repro.V17_MEMBER_SEEDS == tuple(range(17301, 17326))
    assert repro.V17_PANEL_MEMBERS == 25
    assert repro._v17_member_path(1).name == "member_01_seed17301.pt"
    assert repro._v17_member_path(25).name == "member_25_seed17325.pt"


def test_v9_workspace_patches_paths_only_and_restores_them():
    fields = (
        "DEFAULT_EVIDENCE",
        "DEFAULT_DESIGN",
        "DEFAULT_REGISTRY",
        "DEFAULT_MODELS",
        "DEFAULT_FINAL_MODELS",
        "DEFAULT_FREEZE",
        "DEFAULT_RESULT",
    )
    before = {name: getattr(v9, name) for name in fields}
    with repro._v9_workspace():
        assert v9.DEFAULT_EVIDENCE == repro.V9_EVIDENCE
        assert v9.DEFAULT_MODELS == repro.V9_SELECTION_MODELS
        assert v9.DEFAULT_FINAL_MODELS == repro.V9_FINAL_MODELS
        assert v9.DEFAULT_FREEZE == repro.V9_FREEZE
    assert {name: getattr(v9, name) for name in fields} == before


def test_accuracy_gate_is_sub_six_and_blocks_successor_design_until_both_pass():
    assert repro.SUB_SIX_GATE_M == 6.0
    assert repro._gate(5.99)
    assert not repro._gate(6.0)
    assert not repro._gate(7.0)
    source = inspect.getsource(repro.summarize)
    assert "next_stage_authorized" in source
    assert "successor_model_implemented_in_this_stage" in source


def test_v9_reproduction_registry_binds_isolated_design_without_changing_science(
    tmp_path, monkeypatch
):
    output = tmp_path / "registry_v9_reproduction.json"
    monkeypatch.setattr(repro, "V9_REGISTRY", output)
    with repro._v9_workspace():
        registry = repro._ensure_v9_registry()
    loaded = v9.load_registry(output)
    assert loaded["registry_sha256"] == registry["registry_sha256"]
    assert loaded["design"] == str(repro.V9_DESIGN.resolve())
    assert loaded["design_sha256"] == repro._sha256_file(repro.V9_DESIGN)
    assert loaded["constants"]["member_seeds"] == list(range(9311, 9323))
    assert loaded["constants"]["initial_member_count"] == 8
    assert loaded["joint_grid"]["E"] == list(v9.E_GRID)
    assert loaded["joint_grid"]["K"] == list(v9.K_GRID)
    assert loaded["joint_grid"]["lambda"] == list(v9.LAMBDA_GRID)
    assert loaded["candidates"] == list(v9.CANDIDATES)
    assert loaded["official_validation_file_in_scope"] is False
