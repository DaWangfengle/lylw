"""Can the mean-preserving support operator be selected on CAL-fit alone?"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
for path in (str(REPO), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

import run_hypotheses as rh  # noqa: E402
import run_validation as rv  # noqa: E402
from fhm_dapp_tim.fhm_dapp import msep_v17_spatial_operators_v1 as op  # noqa: E402
from fhm_dapp_tim.fhm_dapp import msep_v17_spatial_structure_v1 as s3  # noqa: E402

torch.set_grad_enabled(False)
OUT = REPO / "outputs/v17_gmrf_bandwidth_v1"


def main():
    hyp = [r for r in json.loads((OUT / "hypotheses/HYPOTHESES.json").read_text(encoding="utf-8"))
           if "error" not in r and r["arm"] == "D"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for record in hyp:
        dataset, seed = record["dataset"], record["seed"]
        ctx = s3.reconstruct(dataset, seed, rv.DEFAULT_ROOT, device)
        model = s3.enable_training_arm(ctx["initial"], "D", ctx["graph"])
        ckpt = rv.DEFAULT_CKPT / "trained" / f"{dataset}_seed{seed}" / "D" / "best_dev_epoch_21_80.pt"
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["state_dict"], strict=True)
        model.eval()
        spacing = rv.neighbor_spacing(
            ctx["memory"]["coords"], ctx["memory"]["building"], ctx["memory"]["floor"]).to(device)
        fit = s3.make_pack(ctx, model, ctx["cal_fit"])
        ev = s3.make_pack(ctx, model, ctx["cal_eval"], virtual_gamma=0.0)
        params = record["h1_calibration"]["w0.30"]["params"]
        rho = record["h1_calibration"]["w0.30"]["rho_star"]
        q_fit = torch.softmax(s3.complete_logits(fit, params), dim=1)
        q_eval = torch.softmax(s3.complete_logits(ev, params), dim=1)

        def nll(pack, q, centers=None, scales=None, ids=None):
            centers = pack["xy"] if centers is None else centers
            scales = pack["sigma"] if scales is None else scales
            ids = pack["ids"] if ids is None else ids
            return float(op.joint_bf_xy_nll(
                q, ids, centers, scales, pack["target"], pack["building"], pack["floor"],
                ctx["memory"]["building"], ctx["memory"]["floor"]).mean())

        methods = {}
        methods["base"] = dict(
            fit=nll(fit, q_fit), eval=nll(ev, q_eval),
            fit_mae=rv.light_metrics(fit, ctx, q_fit)["mae"],
            eval_mae=rv.light_metrics(ev, ctx, q_eval)["mae"])
        for name, K, rq in (("2Q4", 4, 0.4), ("2Q8", 8, 0.4)):
            qf, cf, sf, idf = rh.quadrature_components(fit, q_fit, spacing, rq, K=K)
            qe, ce, se, ide = rh.quadrature_components(ev, q_eval, spacing, rq, K=K)
            methods[name] = dict(fit=nll(fit, qf, cf, sf, idf), eval=nll(ev, qe, ce, se, ide),
                                 fit_mae=rv.light_metrics(fit, ctx, qf, cf, sf, idf)["mae"],
                                 eval_mae=rv.light_metrics(ev, ctx, qe, ce, se, ide)["mae"])
        methods["2K"] = dict(
            fit=nll(fit, q_fit, scales=rv.inflate_scales(fit["sigma"], spacing, rho)),
            eval=nll(ev, q_eval, scales=rv.inflate_scales(ev["sigma"], spacing, rho)),
            fit_mae=rv.light_metrics(fit, ctx, q_fit,
                                     scales=rv.inflate_scales(fit["sigma"], spacing, rho))["mae"],
            eval_mae=rv.light_metrics(ev, ctx, q_eval,
                                      scales=rv.inflate_scales(ev["sigma"], spacing, rho))["mae"])

        fit_choice = min(("2K", "2Q4", "2Q8"), key=lambda m: methods[m]["fit"])
        eval_choice = min(("2K", "2Q4", "2Q8"), key=lambda m: methods[m]["eval"])
        rows.append({"dataset": dataset, "seed": seed, "methods": methods,
                     "fit_choice": fit_choice, "eval_choice": eval_choice})
        print(f"{dataset} {seed}: fit->{fit_choice} eval->{eval_choice} "
              f"nll fit {methods[fit_choice]['fit']:.3f} eval {methods[fit_choice]['eval']:.3f} "
              f"(oracle eval {methods[eval_choice]['eval']:.3f})", flush=True)

    (OUT / "selection_check.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    for dataset in ("tuji", "uts", "uji"):
        subset = [r for r in rows if r["dataset"] == dataset]
        picks = [r["fit_choice"] for r in subset]
        print(dataset, "fit picks", picks, "eval-best", [r["eval_choice"] for r in subset])


if __name__ == "__main__":
    main()
