import numpy as np
import torch

from fhm_dapp_tim.fhm_dapp.v18_5_datasets import build_v185_anchor_memory
from fhm_dapp_tim.fhm_dapp.v18_5_model import (
    V185_CONFIG,
    build_v185_model,
    prototype_retrieval_score_v185,
    training_spatial_scale_m,
    v185_distribution,
    v185_training_loss,
)


def _state():
    coords_anchor = np.array(
        [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]],
        dtype=np.float32,
    )
    coords = np.repeat(coords_anchor, 2, axis=0)
    rss = torch.tensor(
        [
            [-1.0, -0.8, -0.4],
            [-0.9, -0.7, -0.5],
            [-0.2, -0.8, -0.5],
            [-0.1, -0.7, -0.4],
            [-0.8, -0.1, -0.5],
            [-0.7,  0.0, -0.4],
            [-0.1,  0.0, -0.2],
            [ 0.0,  0.1, -0.1],
        ],
        dtype=torch.float32,
    )
    observed = torch.ones_like(rss, dtype=torch.bool)
    # Make the first numeric -1 a genuinely observed RSS.
    assert float(rss[0, 0]) == -1.0 and bool(observed[0, 0])
    building = np.zeros(8, dtype=np.int64)
    floor = np.zeros(8, dtype=np.int64)
    memory, sample_anchor, _diag = build_v185_anchor_memory(
        rss,
        observed,
        coords,
        building,
        floor,
        np.arange(8),
    )
    spatial = training_spatial_scale_m(memory)
    model, _geometry, neighbours, weights = build_v185_model(
        memory=memory,
        n_aps=3,
        n_buildings=1,
        n_devices=0,
        spatial_scale_m=spatial,
        device=torch.device("cpu"),
    )
    return model, memory, sample_anchor, neighbours, weights, rss, observed, coords


def test_config_has_no_geometry_posterior_correction_grid():
    forbidden = [
        key for key in V185_CONFIG
        if "geometry" in key and key not in {"geometry_seed_k"}
    ]
    assert forbidden == []
    assert tuple(V185_CONFIG["lambda_spatial_grid"]) == (0.0, 0.1, 0.25, 0.5)
    assert 0.0 in V185_CONFIG["epsilon_kl_grid"]


def test_encoder_keeps_observed_numeric_minus_one_visible():
    model, *_rest = _state()
    rss = torch.tensor([[-1.0, 0.0, 0.0]])
    mask = torch.tensor([[True, False, False]])
    _tokens, padding = model.encoder._tokens_masked(rss, mask)
    assert int((~padding).sum()) == 1


def test_mask_aware_prototype_score_ignores_missing_fill():
    model, *_rest = _state()
    first = torch.tensor([[-1.0, 0.7, -0.2]])
    second = torch.tensor([[-1.0, -0.9, -0.2]])
    mask = torch.tensor([[True, False, True]])
    a = prototype_retrieval_score_v185(model, first, mask)
    b = prototype_retrieval_score_v185(model, second, mask)
    assert torch.allclose(a, b, atol=1e-7, rtol=0)


def test_lambda_zero_training_path_is_finite_and_backpropagates():
    (
        model,
        _memory,
        sample_anchor,
        neighbours,
        weights,
        rss,
        observed,
        coords,
    ) = _state()
    rows = np.array([0, 2, 4, 6])
    target_anchor = torch.from_numpy(sample_anchor[rows]).long()
    target = torch.from_numpy(coords[rows]).float()
    device_ids = torch.full((len(rows),), -1, dtype=torch.long)
    result = v185_training_loss(
        model,
        rss[rows],
        observed[rows],
        target,
        target_anchor,
        device_ids,
        neighbours,
        weights,
        lambda_spatial=0.0,
    )
    assert torch.isfinite(result["loss"])
    assert float(result["spatial_contribution"]) == 0.0
    result["loss"].backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(value).all() for value in gradients)


def test_epsilon_zero_distribution_is_identity_path_without_radio_reference():
    model, memory, *_rest, rss, observed, _coords = _state()
    model.eval()
    result = v185_distribution(
        model,
        rss[:2],
        observed[:2],
        radio_memory={
            "detection_prob": model.anchor_detection_prob,
            "rss_center": model.anchor_rss_center,
            "rss_scale": model.anchor_rss_scale,
        },
        radio_reference=None,
        temperature=1.0,
        epsilon_kl=0.0,
        scale_multiplier=1.0,
    )
    assert result["correction"] is None
    assert torch.equal(
        result["probability"],
        result["temperature_probability"],
    )
    assert result["posterior_mean"].shape == (2, 2)
    assert torch.isfinite(result["posterior_mean"]).all()
