"""Synthetic, label-safe checks of objectives and independent refit execution."""
from pathlib import Path
import copy
import json
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from fhm_dapp_tim.fhm_dapp import msep_joint_ablation_v1 as study
from fhm_dapp_tim.fhm_dapp import msep_joint_science_v1 as science


def main():
    torch.set_num_threads(4)
    centers = torch.tensor([[[0., 1.], [3., -2.]]], dtype=torch.float64)
    scales = torch.tensor([[[.7, 1.2], [.8, 2.]]], dtype=torch.float64)
    y = torch.tensor([[.2, -.4]], dtype=torch.float64)
    independent_lp = torch.distributions.StudentT(4., loc=centers, scale=scales).log_prob(y[:, None]).sum(-1)
    assert torch.allclose(science.student_t_logpdf(centers, scales, y), independent_lp, atol=1e-12, rtol=1e-12)
    theta = torch.tensor([.2, .2], requires_grad=True)
    scores = torch.tensor([[.2, -.3]])
    z = torch.tensor([[[1., -.7], [-.5, 1.]]])
    q = torch.softmax(scores/.6+(z*theta).sum(-1), dim=-1)
    loss = science.full_joint_loss(q, torch.tensor([[0, 1]]), centers.float(), scales.float(), y.float(),
                                   torch.tensor([0]), torch.tensor([0, 1]), point_weight=.3)
    loss.backward()
    assert torch.isfinite(theta.grad).all() and bool((theta.grad.abs() > 1e-6).all())

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(.1))

    first_states, orders = [], []
    current = {"seen": False}

    def candidates(model, x, obs, k):
        if not current["seen"]:
            first_states.append(copy.deepcopy(model.state_dict()))
            current["seen"] = True
        orders[-1].extend(x[:, 0].tolist())
        b = len(x)
        s = (model.bias * torch.tensor([1., 2., -1.])).expand(b, -1)
        ids = torch.arange(3).expand(b, -1)
        c = torch.tensor([[[0., 0.], [2., 0.], [10., 0.]]]).expand(b, -1, -1)
        ev = torch.tensor([[[1., -.3], [-.5, .7], [.8, -.4]]]).expand(b, -1, -1)
        return s, ids, c, torch.ones_like(c), ev, None

    base = Tiny()
    models, coefficients = [], []
    base_state = copy.deepcopy(base.state_dict())
    with patch.object(study, "candidate_evidence", candidates):
        for kind, ev in (("xy", False), ("joint", False), ("joint", True), ("joint", False)):
            current["seen"] = False
            orders.append([])
            model, coefs, history = study.fit_refit(
                base, kind=kind, evidence=ev, rss=torch.tensor([[0.], [1.], [2.], [3.]]),
                xy=torch.tensor([[1., 0.], [1.2, 0.], [.8, 0.], [1.1, 0.]]),
                sample_anchor=np.array([0, 1, 0, 1]), obs={}, roles={"fit": np.arange(4)},
                anchor_class=torch.tensor([0, 0, 1]), epochs=1, batch=2, seed=17301,
                device=torch.device("cpu"))
            assert history[0]["batches"] == 2
            models.append(model)
            coefficients.append(coefs)
    assert all(torch.equal(base_state[k], base.state_dict()[k]) for k in base_state)
    assert all(torch.equal(v["bias"], base_state["bias"]) for v in first_states)
    assert all(order == orders[0] for order in orders)
    assert len({model.bias.data_ptr() for model in models} | {base.bias.data_ptr()}) == 5
    assert torch.equal(models[1].bias, models[3].bias)
    assert coefficients[0] == coefficients[1] == coefficients[3] == [0., 0.]
    assert any(abs(v-.2) > 1e-5 for v in coefficients[2])
    assert all(0 <= v <= 1.5 for v in coefficients[2])
    result = {"status": "PASS", "student_t_matches_torch_distribution": True,
              "evidence_theta_gradients": theta.grad.tolist(),
              "refits_independent_from_same_initial_state": True,
              "identical_refit_query_order_and_budget": True,
              "base_parameters_unchanged": True, "repeated_no_evidence_refit_identical": True,
              "synthetic_evidence_final_theta": coefficients[2],
              "scope": "synthetic correctness only; no real CAL/DEV/TEST scores"}
    with Path(__file__).with_name("NUMERICAL_AUDIT.json").open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
