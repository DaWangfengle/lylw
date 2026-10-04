import numpy as np
import torch

from fhm_dapp_tim.fhm_dapp.v18_4_datasets import CanonicalFingerprintData
from fhm_dapp_tim.fhm_dapp.v18_5_datasets import (
    build_v185_anchor_memory,
    model_input,
)


def _canonical(rss_dbm):
    rss_dbm = np.asarray(rss_dbm, dtype=np.float32)
    rows, aps = rss_dbm.shape
    return CanonicalFingerprintData(
        dataset="synthetic",
        split="train",
        rss_dbm=rss_dbm,
        coords_m=np.stack(
            [np.arange(rows, dtype=np.float32), np.zeros(rows, dtype=np.float32)],
            axis=1,
        ),
        building=np.zeros(rows, dtype=np.int64),
        floor=np.zeros(rows, dtype=np.int64),
        device=np.zeros(rows, dtype=np.int64),
        user=np.zeros(rows, dtype=np.int64),
        timestamp_s=np.arange(rows, dtype=np.float64),
        ap_ids=tuple(f"AP{i}" for i in range(aps)),
        metadata={},
    )


def test_observed_minus_110_and_missing_can_share_numeric_value_not_semantics():
    data = _canonical([[-110.0, np.nan], [-90.0, -110.0]])
    inputs = model_input(data, missing_fill=-1.0)
    assert float(inputs.rss[0, 0]) == -1.0
    assert float(inputs.rss[0, 1]) == -1.0
    assert bool(inputs.observed_mask[0, 0]) is True
    assert bool(inputs.observed_mask[0, 1]) is False
    assert bool(inputs.observed_mask[1, 1]) is True


def test_missing_fill_cannot_change_observation_mask():
    data = _canonical([[-100.0, np.nan], [np.nan, -80.0]])
    a = model_input(data, missing_fill=-1.0)
    b = model_input(data, missing_fill=0.73)
    assert torch.equal(a.observed_mask, b.observed_mask)
    assert torch.equal(
        a.rss[a.observed_mask],
        b.rss[b.observed_mask],
    )


def test_count_aware_scale_uses_local_pooled_and_global_fallbacks():
    rss = torch.tensor(
        [
            [-0.8, -0.5, 0.2],
            [-0.4,  0.0, 0.2],
            [ 0.0,  0.0, 0.2],
            [ 0.3,  0.5, 0.2],
            [ 0.6,  0.0, 0.2],
        ],
        dtype=torch.float32,
    )
    observed = torch.tensor(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [1, 1, 1],
        ],
        dtype=torch.bool,
    )
    coords = np.array(
        [[0, 0], [0, 0], [0, 0], [1, 0], [2, 0]],
        dtype=np.float32,
    )
    building = np.zeros(5, dtype=np.int64)
    floor = np.zeros(5, dtype=np.int64)
    memory, sample_anchor, diagnostics = build_v185_anchor_memory(
        rss,
        observed,
        coords,
        building,
        floor,
        np.arange(5, dtype=np.int64),
    )
    # Anchor 0/AP0 has three observations with nonzero local MAD.
    assert float(memory["rss_scale"][0, 0]) > 0.05
    # Anchor 0/AP1 has one observation; pooled AP1 dispersion is informative.
    assert float(memory["rss_scale"][0, 1]) == float(
        memory["ap_pooled_scale"][1]
    )
    # AP2 is constant everywhere, forcing the global robust fallback.
    assert torch.all(
        memory["rss_scale"][:, 2]
        == memory["global_rss_scale"]
    )
    assert diagnostics["local_scale_fraction"] > 0.0
    assert diagnostics["ap_pooled_fallback_fraction"] > 0.0
    assert diagnostics["global_fallback_fraction"] > 0.0
    assert diagnostics["singleton_anchors"] == 2
    assert set(sample_anchor.tolist()) == {0, 1, 2}


def test_memory_scale_is_finite_and_never_below_registered_floor():
    rss = torch.tensor([[0.1], [0.1]], dtype=torch.float32)
    observed = torch.ones_like(rss, dtype=torch.bool)
    coords = np.array([[0, 0], [1, 0]], dtype=np.float32)
    memory, _, diagnostics = build_v185_anchor_memory(
        rss,
        observed,
        coords,
        np.zeros(2, dtype=np.int64),
        np.zeros(2, dtype=np.int64),
        np.arange(2),
    )
    assert torch.isfinite(memory["rss_scale"]).all()
    assert float(memory["rss_scale"].min()) >= 0.05
    assert diagnostics["global_rss_scale"] >= 0.05
