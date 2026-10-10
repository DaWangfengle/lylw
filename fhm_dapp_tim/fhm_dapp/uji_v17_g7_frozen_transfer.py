"""G7: frozen TUJI-derived posterior temperature transfer stress test.

Reuses ONLY existing G4 UTS and G5 Tampere five-member checkpoints. No
training. No new heldout/blind test claim: all these internal splits were
previously observed. No candidate tuning on either target domain.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from fhm_dapp_tim.fhm_dapp import uji_v17_g4_observable_support as g4
from fhm_dapp_tim.fhm_dapp import uji_v17_g5_bayes_action as g5
from fhm_dapp_tim.fhm_dapp import uji_v17_g6_calibration as g6
from fhm_dapp_tim.fhm_dapp.experiment_v7 import (
    _canonical_json_sha256, _sha256_file, _write_json_new,
)

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT/"evidence/v17_g7_frozen_transfer"
DESIGN = EVIDENCE/"DESIGN_V17_G7.md"
MANIFEST = EVIDENCE/"manifest_v17_g7.json"
LOCK = EVIDENCE/"transfer_v17_g7.lock"
RESULT = EVIDENCE/"transfer_v17_g7.json"
REPORT = EVIDENCE/"RUN_REPORT_V17_G7.md"
PROTOCOL = "v17_g7_frozen_tuji_policy_uts_tampere_transfer_20261010"
POLICY = {"policy":"T0.6_S0.5","T":0.6,"scale":0.5}
BASE = "T1_S1"
JOINT = "T0.6_S0.5"
TEMP_ONLY = "T0.6_S1"
SCALE_ONLY = "T1_S0.5"
DOMAINS = ("uts","tampere")
UTS_ROLES = ("calibration","confirmation")
TAMPERE_ROLES = ("confirmation",)
MIN_DOMAIN_GAIN_M = 0.15
MAX_DOMAIN_ES_INCREASE_M = 0.0
RANDOM_STREAM_SEED = g6.SCORE_SEED
EXPECTED_CANDIDATES = (BASE, JOINT, TEMP_ONLY, SCALE_ONLY)


class G7Error(RuntimeError):
    pass


def _hash_record(record:dict, key:str="result_sha256") -> dict:
    value=dict(record)
    value[key]=_canonical_json_sha256(value)
    return value


def _save(path:Path,record:dict,key:str="result_sha256") -> dict:
    if path.exists():
        raise FileExistsError(f"G7 immutable artifact already exists: {path}")
    value=_hash_record(record,key)
    path.parent.mkdir(parents=True,exist_ok=True)
    _write_json_new(path,value)
    return value


def _read(path:Path,key:str="result_sha256") -> dict:
    obj=json.loads(path.read_text(encoding="utf-8"))
    if obj.get(key)!=_canonical_json_sha256(obj,omit=key):
        raise G7Error(f"G7 integrity failure: {path}")
    return obj


def _source_hashes() -> dict:
    paths={
        "g7_design":DESIGN,"g7_code":Path(__file__),
        "g7_tests":ROOT.parent/"tests/test_uji_v17_g7_frozen_transfer.py",
        "g6_calibration_and_scorer":Path(g6.__file__),
        "g4_uts_historical_protocol":Path(g4.__file__),
        "g5_tampere_historical_protocol":Path(g5.__file__),
    }
    return {k:_sha256_file(p) for k,p in paths.items()}


def _hash_local_checkpoints(paths:Sequence[Path]) -> list[str]:
    found=[]
    for path in paths:
        if not path.is_file():
            raise G7Error(
                f"Missing original local frozen checkpoint: {path}. "
                "Restore the exact G4/G5 .pt file from YOUR original run; "
                "G7 MUST NOT retrain or substitute historical weights."
            )
        found.append(_sha256_file(path))
    return found


def _data_and_old_protocol(domain:str) -> dict:
    """Reads canonical TRAIN files only; no prior outcome JSON is parsed."""
    if domain=="uts":
        if not g4.MANIFEST.is_file():
            raise G7Error("G4 historical manifest absent; cannot verify UTS panel")
        old=g4._read(g4.MANIFEST,"manifest_sha256")
        if g4._source_hashes()!=old["source_sha256"]:
            raise G7Error("G4 training/data/scoring source hashes changed")
        data,groups,rows=g4._data_roles()
        frozen_hashes=_hash_local_checkpoints([g4.member_path(i) for i in range(1,6)])
        if int(old["panel"])!=5 or int(old["L"])!=192 or int(old["K"])!=192:
            raise G7Error("G4 original E5/L192/K192 protocol differs")
        if old["coordinate_keys"]!={k:[list(x) for x in sorted(v)] for k,v in groups.items()}:
            raise G7Error("historical G4 physical coordinate split mismatch")
        if (g4.check_members()["members"]!=[{"member":i,"sha256":sha}
             for i,sha in enumerate(frozen_hashes,1)]):
            raise G7Error("G4 five-member checkpoint identity failed")
        roles=UTS_ROLES
        original_provenance={
            "historical_manifest_sha256":_sha256_file(g4.MANIFEST),
            "historical_manifest_self_sha256":old["manifest_sha256"],
            "historical_protocol":g4.PROTOCOL,
            "checkpoints":frozen_hashes,
            "dataset_train_source":"UTSIndoorLoc/UTS_training.csv",
            "old_confirm_labels_previously_observed":True
        }
    elif domain=="tampere":
        if not g5.MANIFEST.is_file() or not g5.FREEZE.is_file():
            raise G7Error("G5 manifest/freeze unavailable; cannot verify Tampere panel")
        old=g5._read(g5.MANIFEST,"manifest_sha256")
        frozen=g5._read(g5.FREEZE)
        if g5._source_hashes()!=old["source_sha256"]:
            raise G7Error("G5 training/data/scoring source hashes changed")
        data,groups,rows=g5._training()
        frozen_hashes=_hash_local_checkpoints([g5.member_path(i) for i in range(1,6)])
        if (old["model_count"]!=5 or int(old["L"])!=192 or int(old["K"])!=192
            or int(old["alpha"])!=1):
            raise G7Error("G5 E5/L192/K192/alpha1 protocol differs")
        if old["role_coordinate_keys"]!={k:[list(x) for x in sorted(v)] for k,v in groups.items()}:
            raise G7Error("historical G5 physical coordinate split mismatch")
        if frozen["manifest_sha256"]!=old["manifest_sha256"]:
            raise G7Error("G5 freeze does not match G5 manifest")
        if frozen["members"]!=[{"index":i,"sha256":sha}
             for i,sha in enumerate(frozen_hashes,1)]:
            raise G7Error("G5 frozen checkpoint hashes changed")
        if not g5.check_members()["all_complete"]:
            raise G7Error("G5 5 member panel incomplete")
        roles=TAMPERE_ROLES
        original_provenance={
            "historical_manifest_sha256":_sha256_file(g5.MANIFEST),
            "historical_freeze_sha256":_sha256_file(g5.FREEZE),
            "historical_manifest_self_sha256":old["manifest_sha256"],
            "historical_protocol":g5.PROTOCOL,
            "checkpoints":frozen_hashes,
            "dataset_train_source":"Tampere/Training_rss.csv + Training_coordinates.csv",
            "old_confirm_labels_previously_observed":True
        }
    else:
        raise G7Error("only UTS and Tampere internal training roles permitted")
    if groups["fit"] & set().union(*(groups[r] for r in roles)):
        raise G7Error(f"{domain} FIT site leakage")
    for role in roles:
        if set(data["keys"][int(i)] for i in rows[role])!=groups[role]:
            raise G7Error(f"{domain}/{role} site membership changed")
    original_provenance.update({
        "role_coordinate_count":{role:len(groups[role]) for role in roles},
        "role_rows":{role:len(rows[role]) for role in roles},
        "role_coordinate_sha256":{
            role:hashlib.sha256(json.dumps(
                [list(x) for x in sorted(groups[role])],
                ensure_ascii=False,separators=(",",":")).encode("utf-8")).hexdigest()
            for role in roles
        },
        "raw_training_rows":len(data["keys"]),
        "anchor_count":len(groups["fit"]),
        "ap_width":int(data["rss"].shape[1]),
    })
    return {"data":data,"groups":groups,"rows":rows,
            "roles":roles,"original":original_provenance}


def preflight() -> dict:
    checked={name:_data_and_old_protocol(name)["original"] for name in DOMAINS}
    # G6 fixed selection must be matched to its original CAL evidence,
    # not re-selected by looking at old UTS/Tampere labels.
    if not g6.CALIBRATION.is_file() or not g6.MANIFEST.is_file():
        raise G7Error("G6 calibration+manifest required for frozen original source")
    cal=g6._read(g6.CALIBRATION)
    g6manifest=g6._read(g6.MANIFEST,"manifest_sha256")
    if (cal["selected_policy"]["policy"]!=JOINT or
        cal["selected_policy"]["T"]!=POLICY["T"] or
        cal["selected_policy"]["scale"]!=POLICY["scale"] or
        cal["manifest_sha256"]!=g6manifest["manifest_sha256"]):
        raise G7Error("G6 CAL policy provenance is not the preregistered policy")
    return {
        "protocol":PROTOCOL,
        "status":"G7_REUSE_TWO_HISTORICAL_FROZEN_E5_PANELS",
        "source_sha256":_source_hashes(),
        "g6_frozen_calibration_file_sha256":_sha256_file(g6.CALIBRATION),
        "g6_frozen_manifest_file_sha256":_sha256_file(g6.MANIFEST),
        "fixed_policy":dict(POLICY),
        "frozen_scorer_seed":RANDOM_STREAM_SEED,
        "variants":list(EXPECTED_CANDIDATES),
        "domains":checked,
        "new_network_training_required":False,
        "previously_observed_uts_tampere_internal_labels":True,
        "blind_external_evaluation":False,
        "provided_test_files_accessed":False,
    }


def prepare() -> dict:
    meta=preflight()
    rec={
        "protocol":PROTOCOL,"source_sha256":meta["source_sha256"],
        "fixed_policy":meta["fixed_policy"],
        "g6_calibration_file_sha256":meta["g6_frozen_calibration_file_sha256"],
        "g6_manifest_file_sha256":meta["g6_frozen_manifest_file_sha256"],
        "models_and_roles":meta["domains"],
        "domain_order":list(DOMAINS),
        "scored_roles":{"uts":list(UTS_ROLES),"tampere":list(TAMPERE_ROLES)},
        "variants":list(EXPECTED_CANDIDATES),
        "g6_original_energy_draws_per_stream":g6.r3p.PREDICTIVE_DRAWS_PER_STREAM,
        "energy_random_seed":RANDOM_STREAM_SEED,
        "diagnostic_transfer_gate":{
            "required_domains":list(DOMAINS),
            "each_domain_mae_improvement_min_m":MIN_DOMAIN_GAIN_M,
            "each_domain_es_worsening_max_m":MAX_DOMAIN_ES_INCREASE_M,
            "not_formal_prospective_statistical_evidence":True
        },
        "historically_seen_labels_not_independent":True,
        "no_retraining_no_temperature_grid_search_no_provided_test":True,
    }
    rec["manifest_sha256"]=_canonical_json_sha256(rec)
    if MANIFEST.exists():
        old=_read(MANIFEST,"manifest_sha256")
        if old!=rec:
            raise G7Error("immutable G7 plan differs from checkpoint/data/scorer")
        return old
    _write_json_new(MANIFEST,rec)
    return rec


def transfer_gate(by_domain:dict[str,dict]) -> dict:
    checks={}
    for domain in DOMAINS:
        one=by_domain[domain]
        b=one["candidate_metrics"][BASE]
        c=one["candidate_metrics"][JOINT]
        dmae=float(c["mae_2d_m"]-b["mae_2d_m"])
        des=float(c["energy_score_2d_m"]-b["energy_score_2d_m"])
        if not np.isfinite([dmae,des]).all():
            raise G7Error(f"{domain} nonfinite transfer result")
        checks[domain]={
            "delta_mae_m":dmae,"delta_es_m":des,
            "mae_improvement_met":dmae<=-MIN_DOMAIN_GAIN_M,
            "es_nonworsening_met":des<=MAX_DOMAIN_ES_INCREASE_M,
        }
    passed=all(d["mae_improvement_met"] and d["es_nonworsening_met"]
               for d in checks.values())
    return {"status":("TWO_DOMAIN_DESCRIPTIVE_TRANSFER_SIGNAL" if passed
                      else "TRANSFER_NOT_ESTABLISHED"),
            "passed_descriptive_gate":passed,"checks":checks,
            "not_independent_blind_evidence":True}


def _load_historical_models(domain:str,manifest:dict,device_name:str):
    expected=manifest["models_and_roles"][domain]["checkpoints"]
    paths=([g4.member_path(i) for i in range(1,6)] if domain=="uts"
           else [g5.member_path(i) for i in range(1,6)])
    if _hash_local_checkpoints(paths)!=expected:
        raise G7Error(f"{domain} checkpoint file changed after G7 plan")
    if domain=="uts":
        old=g4._read(g4.MANIFEST,"manifest_sha256")
        models,hashes,origin=g4._load_models(device_name,old)
        if hashes!=expected:raise G7Error("G4 loaded hashes differ")
    else:
        frozen=g5._read(g5.FREEZE)
        models,origin=g5._load_models(device_name,frozen)
    return models,origin


def run(device_name:str="cuda") -> dict:
    if LOCK.exists() or RESULT.exists():
        raise G7Error("G7 frozen transfer already attempted; do not overwrite")
    manifest=prepare()
    # Write irreversible lock before ANY historical target-role label scoring.
    lock=_save(LOCK,{
        "protocol":PROTOCOL,
        "manifest_file_sha256":_sha256_file(MANIFEST),
        "g6_calibration_sha256":manifest["g6_calibration_file_sha256"],
        "historical_weights_are_original":True,
        "reanalysis_of_seen_domains":True,
    })
    # Once locked, do NOT recompute/retune the old target confirmation.
    output={}
    for domain in DOMAINS:
        ds=_data_and_old_protocol(domain)
        models,origin=_load_historical_models(domain,manifest,device_name)
        role_results={}
        for role in ds["roles"]:
            one=g6._score_role(
                "confirmation",models,ds["data"],ds["rows"][role],origin,
                selected=dict(POLICY)
            )
            if set(one["candidate_metrics"])!=set(EXPECTED_CANDIDATES):
                raise G7Error("unexpected candidate set from frozen G6 scorer")
            role_results[role]=one
        output[domain]={
            "scored_roles":role_results,
            "historical_model_provenance":ds["original"],
        }
        del models
        if torch.cuda.is_available():torch.cuda.empty_cache()
    confirm={d:output[d]["scored_roles"]["confirmation"] for d in DOMAINS}
    gate=transfer_gate(confirm)
    result={
        "protocol":PROTOCOL,
        "status":gate["status"],
        "gate":gate,
        "frozen_policy":dict(POLICY),
        "by_domain":output,
        "manifest_file_sha256":_sha256_file(MANIFEST),
        "lock_file_sha256":_sha256_file(LOCK),
        "lock_result_sha256":lock["result_sha256"],
        "provided_test_files_accessed":False,
        "no_model_weights_retrained":True,
        "previously_observed_target_domain_labels":True,
        "scientific_limit":"Re-analysis of G4/G5 internal partitions. No new blind external confirmation; do not use nominal intervals to claim an independent discovery.",
    }
    saved=_save(RESULT,result)
    REPORT.write_text(make_report(saved),encoding="utf-8",newline="\n")
    return {"status":saved["status"],"result_sha256":saved["result_sha256"],
            "gate":saved["gate"]}


def make_report(result:dict) -> str:
    lines=[
        "# G7 | Frozen G6 calibration transfer stress test",
        "",
        "### Scientific status",
        "",
        "No new training. G4 UTS and G5 Tampere previously seen internal role labels;",
        "**this is NOT a blind external transfer claim**.",
        "TUJI-derived T=0.6,s=0.5 was fixed before all G7 target scoring.",
        "G7 target labels NEVER select a new T or s. E5/L192/K192/alpha1 frozen.",
        "",
        "| Domain / historic role | Action | Macro MAE (m) | Energy Score (m) |",
        "|---|---|---:|---:|",
    ]
    for domain in DOMAINS:
        for role in result["by_domain"][domain]["scored_roles"]:
            one=result["by_domain"][domain]["scored_roles"][role]
            for name in EXPECTED_CANDIDATES:
                m=one["candidate_metrics"][name]
                lines.append(
                    f"| {domain} / {role} | {name} | "
                    f"{m['mae_2d_m']:.6f} | {m['energy_score_2d_m']:.6f} |"
                )
    lines.extend(["", "### Fixed transfer decision", ""])
    for domain in DOMAINS:
        g=result["gate"]["checks"][domain]
        lines.append(
            f"- {domain}: candidate-baseline MAE={g['delta_mae_m']:+.6f}m, "
            f"ES={g['delta_es_m']:+.6f}m; "
            f"MAE threshold={g['mae_improvement_met']}, "
            f"ES threshold={g['es_nonworsening_met']}"
        )
    lines.extend([
        "",
        f"**{result['status']}**",
        "",
        "T-only/s-only are predeclared ablations, never replacements.",
        "Historical labels were exposed in G4/G5, so even a positive",
        "G7 gate is useful directional evidence, NOT independent validation.",
        "No access to UTS_test, Tampere Test or official UJI/TUJI tests.",
        "Do not claim JCR Q1 originality for fixed mixture temperature.",
        f"Immutable result: {result['result_sha256']}",
        "",
    ])
    return "\n".join(lines)


def check() -> dict:
    if not MANIFEST.exists():return {"status":"NO_G7_PLAN"}
    manifest=prepare()
    if not LOCK.exists():return {"status":"FROZEN_READY"}
    lock=_read(LOCK)
    if lock["manifest_file_sha256"]!=_sha256_file(MANIFEST):
        raise G7Error("G7 lock differs from frozen plan")
    if not RESULT.exists():
        return {"status":"G7_LOCK_WITHOUT_RESULT_STOP"}
    result=_read(RESULT)
    if (result["lock_file_sha256"]!=_sha256_file(LOCK) or
        result["manifest_file_sha256"]!=_sha256_file(MANIFEST) or
        result["lock_result_sha256"]!=lock["result_sha256"]):
        raise G7Error("G7 result provenance changed")
    if not REPORT.is_file() or REPORT.read_text(encoding="utf-8")!=make_report(result):
        raise G7Error("G7 rendered report differs from result")
    return {"status":"VERIFIED","scientific_status":result["status"],
            "result_sha256":result["result_sha256"]}


def main(argv:Sequence[str]|None=None) -> int:
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    for action in ("preflight","prepare","run","check"):
        obj=sub.add_parser(action)
        if action=="run":obj.add_argument("--device",default="cuda")
    args=p.parse_args(argv)
    if args.command=="preflight":v=preflight()
    elif args.command=="prepare":v=prepare()
    elif args.command=="run":v=run(args.device)
    else:v=check()
    print(json.dumps(v,ensure_ascii=False,indent=2,default=str,allow_nan=False))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
