# 从 rf_anchor_physics.py 内联，自包含 fhm_dapp_tim 实验
"""Identifiable RF likelihood for datasets with known AP metadata."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def load_sod_ap_metadata(
    xlsx_path: str | Path,
    sheet_name: str,
    wap_cols: list[str],
    coord_origin: np.ndarray,
) -> dict[str, torch.Tensor]:
    """Align fixed SOD AP coordinates/frequencies with RSS feature columns."""
    frame = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    by_attribute: dict[str, tuple[float, float, float]] = {}
    for row in frame.to_dict(orient="records"):
        for attribute_col, frequency_col in (
            ("Attribute_2.4", "Frequency_2.4"),
            ("Attribute_5", "Frequency_5"),
        ):
            attribute = row.get(attribute_col)
            frequency = row.get(frequency_col)
            if pd.isna(attribute) or pd.isna(frequency):
                continue
            by_attribute[str(attribute)] = (
                float(row["ECoord"]) - float(coord_origin[0]),
                float(row["NCoord"]) - float(coord_origin[1]),
                float(frequency),
            )

    coords = torch.full((len(wap_cols), 2), float("nan"))
    frequencies = torch.full((len(wap_cols),), float("nan"))
    for index, column in enumerate(wap_cols):
        metadata = by_attribute.get(str(column))
        if metadata is None:
            continue
        coords[index] = torch.tensor(metadata[:2], dtype=torch.float32)
        frequencies[index] = metadata[2]

    known = torch.isfinite(frequencies) & torch.isfinite(coords).all(dim=1)
    if not known.any():
        raise ValueError(
            f"no AP metadata from sheet {sheet_name!r} matched RSS columns")
    return {
        "ap_coords": coords,
        "ap_frequencies": frequencies,
        "known_mask": known,
    }


class FixedAPCensoredLikelihood(nn.Module):
    """Constrained log-distance likelihood with fixed AP coordinates.

    Detected RSS values use a Gaussian likelihood. Non-detections use the
    probability that RSS falls below the receiver threshold, rather than an
    MSE target at the missing-value sentinel.
    """

    def __init__(
        self,
        ap_coords: torch.Tensor,
        ap_frequencies: torch.Tensor,
        threshold_dbm: float = -104.0,
        invisible_weight: float = 0.20,
    ) -> None:
        super().__init__()
        known = (
            torch.isfinite(ap_frequencies)
            & torch.isfinite(ap_coords).all(dim=1)
        )
        self.register_buffer(
            "ap_coords", torch.nan_to_num(ap_coords.float()), persistent=True)
        self.register_buffer(
            "ap_frequencies",
            torch.nan_to_num(ap_frequencies.float()),
            persistent=True,
        )
        self.register_buffer("known_mask", known, persistent=True)
        band = (self.ap_frequencies >= 4900.0).long()
        self.register_buffer("band_index", band, persistent=True)

        self.p0_raw = nn.Parameter(torch.zeros(2))
        self.path_loss_raw = nn.Parameter(torch.zeros(2))
        self.sigma_raw = nn.Parameter(torch.zeros(2))
        self.ap_bias_raw = nn.Parameter(torch.zeros(len(ap_coords)))
        self.threshold_dbm = float(threshold_dbm)
        self.invisible_weight = float(invisible_weight)

    @property
    def p0_dbm(self) -> torch.Tensor:
        return -45.0 + 10.0 * torch.tanh(self.p0_raw)

    @property
    def path_loss_exponent(self) -> torch.Tensor:
        return 2.5 + 1.5 * torch.tanh(self.path_loss_raw)

    @property
    def sigma_db(self) -> torch.Tensor:
        return 6.0 + 3.0 * torch.tanh(self.sigma_raw)

    @property
    def ap_bias_db(self) -> torch.Tensor:
        return 4.0 * torch.tanh(self.ap_bias_raw)

    def predicted_rss(self, coords: torch.Tensor) -> torch.Tensor:
        """Predict RSS in dBm for coordinates shaped [..., 2]."""
        delta = coords.unsqueeze(-2) - self.ap_coords
        distance = delta.square().sum(dim=-1).sqrt().clamp(min=1.0)
        band = self.band_index
        return (
            self.p0_dbm[band]
            - 10.0 * self.path_loss_exponent[band] * torch.log10(distance)
            + self.ap_bias_db
        )

    @staticmethod
    def normalized_to_dbm(rss: torch.Tensor) -> torch.Tensor:
        return (rss + 1.0) * 40.0 - 110.0

    def log_likelihood(
        self, rss: torch.Tensor, coords: torch.Tensor
    ) -> torch.Tensor:
        """Return an average physical log likelihood per candidate.

        ``rss`` is [B, A]. ``coords`` may be [B, 2] or [B, K, 2].
        The returned shape is [B] or [B, K].
        """
        candidate_axis = coords.ndim == 3
        if not candidate_axis:
            coords = coords.unsqueeze(1)

        mean = self.predicted_rss(coords)
        observed = self.normalized_to_dbm(rss).unsqueeze(1)
        detected = (rss > -0.999).unsqueeze(1)
        known = self.known_mask.view(1, 1, -1)
        detected = detected & known
        censored = (~detected) & known

        sigma = self.sigma_db[self.band_index].view(1, 1, -1)
        z = (observed - mean) / sigma
        detected_lp = (
            -0.5 * z.square()
            - torch.log(sigma)
            - 0.5 * np.log(2.0 * np.pi)
        )
        censor_z = (self.threshold_dbm - mean) / sigma
        censored_lp = torch.special.log_ndtr(censor_z)

        detected_count = detected.sum(dim=-1).clamp(min=1)
        censored_count = censored.sum(dim=-1).clamp(min=1)
        score = (
            (detected_lp * detected).sum(dim=-1) / detected_count
            + self.invisible_weight
            * (censored_lp * censored).sum(dim=-1) / censored_count
        )
        return score.squeeze(1) if not candidate_axis else score

    def negative_log_likelihood(
        self, rss: torch.Tensor, coords: torch.Tensor
    ) -> torch.Tensor:
        regularizer = 0.01 * self.ap_bias_db.square().mean()
        return -self.log_likelihood(rss, coords).mean() + regularizer

    def candidate_scores(
        self, rss: torch.Tensor, coords: torch.Tensor
    ) -> torch.Tensor:
        score = self.log_likelihood(rss, coords)
        centered = score - score.mean(dim=1, keepdim=True)
        scale = score.std(dim=1, keepdim=True, correction=0).clamp(min=0.25)
        return centered / scale

    def summary(self) -> dict[str, list[float] | int]:
        return {
            "known_aps": int(self.known_mask.sum()),
            "p0_dbm": self.p0_dbm.detach().cpu().tolist(),
            "path_loss_exponent": (
                self.path_loss_exponent.detach().cpu().tolist()
            ),
            "sigma_db": self.sigma_db.detach().cpu().tolist(),
        }


