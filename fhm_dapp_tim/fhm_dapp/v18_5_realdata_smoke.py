"""Outcome-free real-data smoke for V18.5.

Uses only UJI trainingData roles.  No optimizer step is taken.
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
import torch

from fhm_dapp_tim.fhm_dapp.v18_4_datasets import dataset_summary
from fhm_dapp_tim.fhm_dapp.uji_v18_5_development import (
    _build_training_state,
)
from fhm_dapp_tim.fhm_dapp.v18_5_core import temperature_base_posterior
from fhm_dapp_tim.fhm_dapp.v18_5_datasets import load_uji
from fhm_dapp_tim.fhm_dapp.v18_5_model import (
    V185_CONFIG,
    base_posterior_v185,
    candidate_support_v185,
    prototype_retrieval_score_v185,
    v185_distribution,
    v185_training_loss,
)
from fhm_dapp_tim.fhm_dapp.v18_5_protocol import (
    ROLE_SEEDS,
    construct_roles,
    row_indices_for_roles,
)
from fhm_dapp_tim.fhm_dapp.v18_5_radio import (
    fit_temperature_radio_reference,
    profile_radio_nuisance,
)


def _small_reference(state, rows, device, *, temperature: float):
    model = state["model"]
    chunks = []
    batch_size = 16
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start:start + batch_size]
        rss = state["rss"][batch_rows].to(device)
        mask = state["observed_mask"][batch_rows].to(device)
        with torch.no_grad():
            logits, embedding = model.retrieval_logits_v185(rss, mask)
            prototype = prototype_retrieval_score_v185(model, rss, mask)
            support = candidate_support_v185(model, logits, prototype)
            base = base_posterior_v185(
                model, rss, mask, embedding, logits, support
            )
            q_t = temperature_base_posterior(
                base["probability"],
                base["candidate_mask"],
                temperature=temperature,
            )
        raw = profile_radio_nuisance(
            rss,
            mask,
            base["candidate_indices"],
            {
                "detection_prob": model.anchor_detection_prob,
                "rss_center": model.anchor_rss_center,
                "rss_scale": model.anchor_rss_scale,
            },
            q_t,
            ap_reliability=model.v185_ap_reliability,
            steps=int(V185_CONFIG["radio_profile_steps"]),
            learning_rate=float(V185_CONFIG["radio_profile_learning_rate"]),
            offset_max=float(V185_CONFIG["radio_profile_offset_max"]),
            log_scale_max=float(V185_CONFIG["radio_profile_log_scale_max"]),
            offset_weight=float(V185_CONFIG["radio_profile_offset_weight"]),
            log_scale_weight=float(V185_CONFIG["radio_profile_log_scale_weight"]),
        )
        chunks.append(raw["profiled_objective"].cpu())
    return fit_temperature_radio_reference({
        float(temperature): torch.cat(chunks)
    })


def run_smoke(device_name: str = "cpu") -> dict[str, object]:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    data, _ = load_uji("train")
    roles = construct_roles(data, seed=ROLE_SEEDS["UJIIndoorLoc"])
    rows = row_indices_for_roles(data, roles)
    state = _build_training_state(
        data=data,
        row_indices=rows["fit"],
        device=device,
    )
    model = state["model"]

    width = min(8, len(rows["fit"]))
    train_rows = rows["fit"][:width]
    anchor = torch.from_numpy(
        state["sample_anchor"][train_rows]
    ).long().to(device)
    target = torch.from_numpy(
        state["centered"][train_rows].astype(np.float32)
    ).to(device)
    device_ids = torch.from_numpy(data.device[train_rows]).long().to(device)
    rss = state["rss"][train_rows].to(device)
    observed = state["observed_mask"][train_rows].to(device)

    losses = {}
    gradient_norms = {}
    for lam in (0.0, 0.25):
        model.zero_grad(set_to_none=True)
        result = v185_training_loss(
            model,
            rss,
            observed,
            target,
            anchor,
            device_ids,
            state["neighbours"],
            state["neighbour_weights"],
            lambda_spatial=lam,
        )
        if not bool(torch.isfinite(result["loss"])):
            raise RuntimeError("nonfinite smoke loss")
        result["loss"].backward()
        squared = 0.0
        for parameter in model.parameters():
            if parameter.grad is not None:
                if not bool(torch.isfinite(parameter.grad).all()):
                    raise RuntimeError("nonfinite smoke gradient")
                squared += float(parameter.grad.detach().square().sum())
        losses[str(lam)] = {
            key: float(value.detach())
            for key, value in result.items()
        }
        gradient_norms[str(lam)] = math.sqrt(squared)

    model.zero_grad(set_to_none=True)
    reference_rows = rows["fit"][: min(32, len(rows["fit"]))]
    reference = _small_reference(
        state,
        reference_rows,
        device,
        temperature=1.5,
    )
    eval_rows = rows["development"][: min(4, len(rows["development"]))]
    rss_eval = state["rss"][eval_rows].to(device)
    mask_eval = state["observed_mask"][eval_rows].to(device)

    before = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    identity = v185_distribution(
        model,
        rss_eval,
        mask_eval,
        radio_memory={
            "detection_prob": model.anchor_detection_prob,
            "rss_center": model.anchor_rss_center,
            "rss_scale": model.anchor_rss_scale,
        },
        radio_reference=None,
        temperature=1.5,
        epsilon_kl=0.0,
        scale_multiplier=1.0,
    )
    corrected = v185_distribution(
        model,
        rss_eval,
        mask_eval,
        radio_memory={
            "detection_prob": model.anchor_detection_prob,
            "rss_center": model.anchor_rss_center,
            "rss_scale": model.anchor_rss_scale,
        },
        radio_reference=reference,
        temperature=1.5,
        epsilon_kl=0.025,
        scale_multiplier=1.0,
    )
    after = model.state_dict()
    if any(
        not torch.equal(before[key], after[key].detach().cpu())
        for key in before
    ):
        raise RuntimeError("radio smoke mutated model state")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("radio smoke created network parameter gradients")
    if not torch.equal(
        identity["probability"],
        identity["temperature_probability"],
    ):
        raise RuntimeError("epsilon=0 failed identity")

    correction = corrected["correction"]
    assert correction is not None
    if not bool(torch.isfinite(corrected["posterior_mean"]).all()):
        raise RuntimeError("nonfinite posterior mean")
    if bool(
        (
            correction["realized_kl"]
            > correction["epsilon_kl_effective"]
            + correction["kl_budget_tolerance"]
        ).any()
    ):
        raise RuntimeError("smoke KL budget violation")

    summary = dataset_summary(data)
    result = {
        "status": "success",
        "device": str(device),
        "rows": int(len(data.coords_m)),
        "fit_rows": int(len(rows["fit"])),
        "development_rows": int(len(rows["development"])),
        "anchors": int(len(state["memory"]["coords"])),
        "aps": int(state["rss"].shape[1]),
        "observed_values_at_or_below_clip_min": int(
            summary["observed_values_at_or_below_clip_min"]
        ),
        "memory_diagnostics": state["memory_diagnostics"],
        "losses": losses,
        "gradient_norms": gradient_norms,
        "epsilon_zero_identity": True,
        "mean_compatibility": float(
            correction["radio_compatibility"].double().mean()
        ),
        "mean_realized_kl": float(
            correction["realized_kl"].double().mean()
        ),
        "mean_effective_budget": float(
            correction["epsilon_kl_effective"].double().mean()
        ),
        "finite_beta_fraction": float(
            (~correction["dual_beta_infinite"]).double().mean()
        ),
        "posterior_mean_finite": True,
        "optimizer_steps": 0,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(
        run_smoke(args.device),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ))


if __name__ == "__main__":
    main()
