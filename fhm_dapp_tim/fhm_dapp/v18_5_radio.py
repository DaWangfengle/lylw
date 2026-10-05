"""V18.5 explicit-mask radio consistency and compatibility.

The radio model is generative evidence only.  It never votes as a peer
posterior.  A scan-level affine nuisance is profiled under q_T, and the final
profiled objective is compared with a temperature-matched training reference.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch

from fhm_dapp_tim.fhm_dapp.dapp import (
    anchor_ap_reliability,
    robust_hurdle_anchor_energy,
)
from fhm_dapp_tim.fhm_dapp.v18_5_core import (
    empirical_survival_reliability,
    fit_empirical_reference,
)


@dataclass(frozen=True)
class TemperatureRadioReference:
    """One label-free compatibility reference per registered temperature."""

    references: dict[float, torch.Tensor]

    def __post_init__(self) -> None:
        if not self.references:
            raise ValueError("temperature reference cannot be empty")
        for temperature, values in self.references.items():
            if not math.isfinite(float(temperature)) or float(temperature) <= 0:
                raise ValueError("reference temperature must be positive")
            if values.ndim != 1 or len(values) < 2:
                raise ValueError("each reference needs >=2 scalar mismatches")
            if not bool(torch.isfinite(values).all()):
                raise ValueError("reference mismatch values must be finite")
            if bool((values[1:] < values[:-1]).any()):
                raise ValueError("reference values must be sorted")

    def for_temperature(
        self,
        temperature: float,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key = float(temperature)
        if key not in self.references:
            raise KeyError(
                f"no radio compatibility reference registered for T={key}"
            )
        return self.references[key].to(device=device, dtype=dtype)



def training_seen_ap_reliability(
    memory: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Estimate AP reliability only on APs observed in the training role.

    V18.5 AP dictionaries are inherited from the dataset training schema, but
    a particular FIT/selection-refit role can contain columns that are never
    observed.  Such columns must contribute exactly zero radio evidence and
    must not distort the normalization of genuinely seen AP reliabilities.
    """

    required = {
        "detection_prob",
        "rss_center",
        "rss_scale",
        "seen_ap_mask",
    }
    missing = required - set(memory)
    if missing:
        raise KeyError(f"radio memory misses fields: {sorted(missing)}")
    seen = memory["seen_ap_mask"].bool()
    if seen.ndim != 1:
        raise ValueError("seen_ap_mask must be one-dimensional")
    if memory["detection_prob"].shape[1] != len(seen):
        raise ValueError("seen_ap_mask width mismatch")
    output = torch.zeros(
        len(seen),
        dtype=memory["rss_center"].dtype,
        device=memory["rss_center"].device,
    )
    if bool(seen.any()):
        output[seen] = anchor_ap_reliability(
            memory["detection_prob"][:, seen],
            memory["rss_center"][:, seen],
            memory["rss_scale"][:, seen],
        ).to(output)
    return output


def support_radio_fields(
    anchor_indices: torch.Tensor,
    memory: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    required = {"detection_prob", "rss_center", "rss_scale"}
    missing = required - set(memory)
    if missing:
        raise KeyError(f"radio memory misses fields: {sorted(missing)}")
    return (
        memory["detection_prob"][anchor_indices],
        memory["rss_center"][anchor_indices],
        memory["rss_scale"][anchor_indices],
    )


def marked_radio_energy(
    query_rss: torch.Tensor,
    observed_mask: torch.Tensor,
    anchor_indices: torch.Tensor,
    memory: Mapping[str, torch.Tensor],
    *,
    ap_reliability: torch.Tensor | None = None,
    degrees_freedom: float = 4.0,
    style_offset: torch.Tensor | float | None = None,
    style_scale: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Channel-balanced marked energy with explicit AP observation mask."""

    if query_rss.ndim != 2 or observed_mask.shape != query_rss.shape:
        raise ValueError("query_rss/observed_mask must share [B,A] shape")
    if observed_mask.dtype != torch.bool:
        raise ValueError("observed_mask must be bool")
    if anchor_indices.ndim != 2 or len(anchor_indices) != len(query_rss):
        raise ValueError("anchor_indices must be [B,K]")
    probability, center, scale = support_radio_fields(anchor_indices, memory)
    if ap_reliability is None:
        ap_reliability = torch.ones(
            query_rss.shape[1],
            dtype=query_rss.dtype,
            device=query_rss.device,
        )
    if ap_reliability.shape != (query_rss.shape[1],):
        raise ValueError("ap_reliability must contain one value per AP")
    return robust_hurdle_anchor_energy(
        probability,
        center,
        scale,
        query_rss,
        observed_mask.to(query_rss.dtype),
        ap_reliability,
        style_offset=style_offset,
        style_scale=style_scale,
        visible_factor=1.0,
        invisible_factor=1.0,
        distribution="student_t",
        degrees_freedom=float(degrees_freedom),
        score="balanced_log",
    )


def profile_radio_nuisance(
    query_rss: torch.Tensor,
    observed_mask: torch.Tensor,
    anchor_indices: torch.Tensor,
    memory: Mapping[str, torch.Tensor],
    anchor_prior: torch.Tensor,
    *,
    ap_reliability: torch.Tensor | None = None,
    degrees_freedom: float = 4.0,
    steps: int = 6,
    learning_rate: float = 0.20,
    offset_max: float = 0.20,
    log_scale_max: float = 0.25,
    offset_weight: float = 0.10,
    log_scale_weight: float = 0.10,
) -> dict[str, torch.Tensor]:
    """Profile only two temporary scan-style tensors under q_T.

    The returned mismatch is the final profiled objective itself:

      E_qT[E_radio] + w_o offset^2 + w_g log(gain)^2.

    No target coordinate or localization outcome is accepted by this API.
    """

    if anchor_prior.shape != anchor_indices.shape:
        raise ValueError("anchor_prior and anchor_indices must align")
    if int(steps) <= 0:
        raise ValueError("profile steps must be positive")
    prior = anchor_prior.detach()
    prior = prior / prior.sum(dim=1, keepdim=True).clamp_min(1e-30)
    query = query_rss.detach()
    observed = observed_mask.detach()
    indices = anchor_indices.detach()
    fields = {
        key: value.detach()
        for key, value in memory.items()
        if key in {"detection_prob", "rss_center", "rss_scale"}
    }
    reliability = (
        None if ap_reliability is None else ap_reliability.detach()
    )

    raw_offset = torch.zeros(
        len(query), dtype=query.dtype, device=query.device, requires_grad=True
    )
    raw_log_gain = torch.zeros_like(raw_offset, requires_grad=True)
    optimizer = torch.optim.Adam(
        [raw_offset, raw_log_gain],
        lr=float(learning_rate),
    )

    def nuisance() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        offset = float(offset_max) * torch.tanh(raw_offset)
        log_gain = float(log_scale_max) * torch.tanh(raw_log_gain)
        return offset, log_gain, torch.exp(log_gain)

    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        offset, log_gain, gain = nuisance()
        energy = marked_radio_energy(
            query,
            observed,
            indices,
            fields,
            ap_reliability=reliability,
            degrees_freedom=degrees_freedom,
            style_offset=offset,
            style_scale=gain,
        )
        expected = torch.sum(prior * energy, dim=1)
        objective = (
            expected
            + float(offset_weight) * offset.square()
            + float(log_scale_weight) * log_gain.square()
        )
        objective.mean().backward()
        optimizer.step()

    with torch.no_grad():
        offset, log_gain, gain = nuisance()
        energy = marked_radio_energy(
            query,
            observed,
            indices,
            fields,
            ap_reliability=reliability,
            degrees_freedom=degrees_freedom,
            style_offset=offset,
            style_scale=gain,
        )
        expected = torch.sum(prior * energy, dim=1)
        objective = (
            expected
            + float(offset_weight) * offset.square()
            + float(log_scale_weight) * log_gain.square()
        )
    return {
        "energy": energy.detach(),
        "style_offset": offset.detach(),
        "style_scale": gain.detach(),
        "profiled_objective": objective.detach(),
    }


def fit_temperature_radio_reference(
    mismatch_by_temperature: Mapping[float, torch.Tensor],
) -> TemperatureRadioReference:
    """Freeze one empirical training compatibility reference for each T."""

    references = {
        float(temperature): fit_empirical_reference(values).detach().cpu()
        for temperature, values in mismatch_by_temperature.items()
    }
    return TemperatureRadioReference(references=references)



def temperature_radio_reference_from_payload(
    payload: Mapping[str, torch.Tensor],
) -> TemperatureRadioReference:
    """Deserialize a checkpoint payload without relaxing exact-T matching."""

    references: dict[float, torch.Tensor] = {}
    for key, value in payload.items():
        temperature = float(key)
        if temperature in references:
            raise ValueError("duplicate temperature reference")
        references[temperature] = torch.as_tensor(value).detach().cpu()
    return TemperatureRadioReference(references=references)

def radio_compatibility(
    profiled_objective: torch.Tensor,
    reference: TemperatureRadioReference,
    *,
    temperature: float,
) -> torch.Tensor:
    """Return a label-free training-compatibility score in [0,1]."""

    sorted_reference = reference.for_temperature(
        temperature,
        device=profiled_objective.device,
        dtype=profiled_objective.dtype,
    )
    return empirical_survival_reliability(
        profiled_objective,
        sorted_reference,
    )


__all__ = [
    "TemperatureRadioReference",
    "fit_temperature_radio_reference",
    "marked_radio_energy",
    "profile_radio_nuisance",
    "radio_compatibility",
    "support_radio_fields",
    "training_seen_ap_reliability",
    "temperature_radio_reference_from_payload",
]
