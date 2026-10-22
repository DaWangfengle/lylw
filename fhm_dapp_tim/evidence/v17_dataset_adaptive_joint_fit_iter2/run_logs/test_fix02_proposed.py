"""Review-only synthetic regression proposal; no real data or model access."""
import math
import numpy as np
import pytest
import torch
from fhm_dapp_tim.fhm_dapp import msep_v17_dataset_adaptive_joint_fit_v1 as joint


@pytest.mark.parametrize("device,counts", [
    ("cpu", [3] * 10),
    pytest.param("cuda", [7] * 30 + [6] * 39,
                 marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
])
@pytest.mark.parametrize("one_wrong", [False, True])
def test_accuracy_uses_integer_site_counts_without_relaxing_the_guard(device, counts, one_wrong):
    n_rows = sum(counts)
    coords = torch.tensor([[0., 0.], [1., 0.], [10., 0.]], device=device)
    edges = torch.tensor([[0, 1]], device=device)
    graph, _, _ = joint.build_cutoff_graph(
        coords.cpu().numpy(), np.array([0, 0, 1]), np.zeros(3, dtype=np.int64), edges, 2., 1.,
    )
    buildings = torch.zeros(n_rows, dtype=torch.long, device=device)
    if one_wrong:
        buildings[-1] = 1
    ctx = {"memory": {"coords": coords, "building": torch.tensor([0, 0, 1], device=device),
                      "floor": torch.zeros(3, dtype=torch.long, device=device)},
           "edges": edges, "classes": torch.tensor([0, 0, 1], device=device),
           "class_pairs": torch.tensor([[0, 0], [1, 0]], device=device)}
    keys = [f"site_{i}" for i, count in enumerate(counts) for _ in range(count)]
    pack = {"scores": torch.tensor([[3., -1., -5.]], device=device).repeat(n_rows, 1),
            "z": torch.zeros((n_rows, 3, 2), device=device),
            "virtual_scores": torch.zeros((n_rows, 1), device=device),
            "virtual_z": torch.zeros((n_rows, 1, 2), device=device),
            "xy": coords[None].repeat(n_rows, 1, 1),
            "sigma": torch.ones((n_rows, 3, 2), device=device),
            "virtual_xy": torch.tensor([[[.5, 0.]]], device=device).repeat(n_rows, 1, 1),
            "virtual_sigma": torch.ones((n_rows, 1, 2), device=device),
            "target": torch.zeros((n_rows, 2), device=device),
            "building": buildings, "floor": torch.zeros_like(buildings),
            "weight": torch.tensor([1. / (len(counts) * count) for count in counts for _ in range(count)], device=device),
            "keys": keys}
    candidates = [{"candidate_id": f"beta_{beta}", "temperature": 1., "theta_v": .75,
                   "theta_s": .75, "alpha_prior": .1, "alpha_obs": .25, "beta": beta}
                  for beta in joint.BETAS]
    rows = joint.evaluate_calibration_grid(pack, ctx, graph, edges, {"tau_virt_m": 1.}, candidates)
    exact = ((len(counts) - 1) + (counts[-1] - 1) / counts[-1]) / len(counts) if one_wrong else 1.
    assert len(rows) == 9
    for row in rows:
        assert math.isfinite(row["bf_accuracy"])
        assert row["bf_accuracy"] == pytest.approx(exact, abs=1e-15, rel=0.)
        if not one_wrong:
            base = dict(row, bf_accuracy=1.)
            checked = joint._add_objectives([dict(row)], base)[0]
            assert checked["bf_accuracy_non_degradation"] is True
