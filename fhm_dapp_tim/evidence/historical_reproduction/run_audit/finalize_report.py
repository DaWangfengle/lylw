"""Write the final report only after every authorized experiment stage completes."""
from __future__ import annotations
import hashlib
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from fhm_dapp_tim.fhm_dapp.experiment_v7 import _canonical_json_sha256

ROOT = Path(__file__).resolve().parents[4]
E = ROOT / "fhm_dapp_tim/evidence/historical_reproduction"
A = E / "run_audit"
M = ROOT / "fhm_dapp_tim/models/historical_reproduction"
read = lambda p: json.loads(p.read_text(encoding="utf-8"))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
q = lambda text: chr(96) + str(text) + chr(96)
gate = lambda value: math.isfinite(float(value)) and 0.0 < float(value) < 6.0

def recorded_stage(name):
    rows = [read(p) for p in sorted(A.glob(name + "_attempt*.json"))]
    assert rows, name
    if name == "run-v17":
        assert len(rows) == 1 and rows[0]["exit_code"] == 1
        result = rows[0]
        assert "Expected size 32 but got size 2" in Path(result["log_path"]).read_text(encoding="utf-8")
    else:
        successes = [record for record in rows if record["exit_code"] == 0]
        assert successes, (name, rows)
        assert name in ("compile", "pytest", "preflight") or len(successes) == 1
        result = successes[-1]
    assert sha(Path(result["log_path"])) == result["log_sha256"], name
    return result

def write_once(path, value):
    assert path.resolve().is_relative_to(E.resolve()), path
    if path.exists():
        assert path.read_text(encoding="utf-8") == value, path
    else:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(value)

def main():
    initial = read(A / "INITIAL_EXECUTION_AUDIT.json")
    env_audit = read(A / "ENVIRONMENT_AND_STATIC_PRECHECK.json")
    repair = read(A / "PRECHECK_TRAINING_INPUT_PATH_REPAIR.json")
    diagnostic_repair = read(A / "V17_RETAINED_MASS_IMPLEMENTATION_REPAIR.json")
    diagnostic_plan = read(A / "V17_RETAINED_MASS_FAILURE_AND_REPAIR_PLAN.json")
    post_precheck = read(A / "POST_TRAINING_IMPLEMENTATION_PRECHECK.json")
    effective_repair_hashes = {
        **repair["changed_file_sha256"], **diagnostic_repair["changed_file_sha256"]
    }
    registry = read(E / "registry_historical_reproduction.json")
    v9 = read(E / "v9/REPRODUCTION_V9.json")
    v9_result = read(E / "v9/outer_test_result_v9_uji.json")
    freeze = read(E / "v9/selection_freeze_v9_uji.json")
    v17_registry = read(E / "v17/registry_v17_reproduction.json")
    v17 = read(E / "v17/outer_result_v17_reproduction.json")
    overall = read(E / "REPRODUCTION_SUMMARY.json")
    stages = {name: recorded_stage(name) for name in (
        "compile", "pytest", "preflight", "prepare", "run-v9",
        "prepare-v17", "run-v17", "check-v17-members", "evaluate-v17", "summarize"
    )}
    check_log = Path(stages["check-v17-members"]["log_path"]).read_text(encoding="utf-8")
    check_text = check_log.split("HISTORICAL_STAGE_EXECUTION=", 1)[0]
    member_check, _ = json.JSONDecoder().raw_decode(check_text[check_text.index("{"):])
    assert member_check["complete_members"] == 25 and member_check["all_complete"] is True
    assert len(member_check["members"]) == 25
    assert member_check["members"] == v17["member_epoch_records"]
    assert member_check == diagnostic_plan["complete_members_before_repair"]
    assert diagnostic_repair["non_retained_diagnostic_AST_unchanged"] is True
    assert diagnostic_repair["diagnostic_mean_and_quantile_definitions_unchanged"] is True
    for relative, expected in diagnostic_plan["registry_file_sha256_before_repair"].items():
        assert sha(E / relative) == expected, relative
    source_snapshot_deltas = {
        name: {
            "registered_sha256": record["sha256"],
            "current_sha256": sha(Path(record["path"])),
        }
        for name, record in registry["source_hashes"].items()
        if sha(Path(record["path"])) != record["sha256"]
    }
    assert source_snapshot_deltas == post_precheck["registration_time_source_snapshot_deltas"]
    assert set(source_snapshot_deltas) == {"runner"}
    assert source_snapshot_deltas["runner"]["current_sha256"] == diagnostic_repair["source_after_sha256"]
    model_records = []
    for index, record in enumerate(member_check["members"], 1):
        assert record["member_index"] == index and record["seed"] == 17300 + index
        assert record["exists"] is True and 1 <= record["selected_epoch"] <= 80
        path = M / f"v17/member_{index:02d}_seed{17300 + index}.pt"
        assert sha(path) == record["checkpoint_sha256"], path
        model_records.append({**record, "path": str(path)})
    for payload, field in (
        (registry, "registry_sha256"), (v17_registry, "registry_sha256"),
        (freeze, "freeze_sha256"), (v9_result, "result_sha256"),
        (v17, "result_sha256"), (overall, "summary_sha256")
    ):
        assert payload[field] == _canonical_json_sha256(payload, omit=field), field
    assert v9["result_sha256"] == sha(E / "v9/outer_test_result_v9_uji.json")
    assert v9["selected"] == freeze["selected"] == v9_result["selected"]
    assert freeze["outer_test_accesses_before_freeze"] == 0
    assert v9_result["outer_test_access_number"] == 1
    assert v17_registry["selected"] == v17["selected_configuration"] == {"E":16,"L":192,"K":192,"alpha":1.0}
    assert v17_registry["historical"] == v17["historical_configuration"] == {"E":5,"L":24,"K":24,"alpha":0.25}
    assert v17_registry["member_seeds"] == list(range(17301, 17326))
    assert v17["outer_test_rows"] == 4066 and v17["outer_test_coordinate_groups"] == 187
    assert v17_registry["configuration_reselection"] is False
    for payload in (v9, v17, overall):
        assert payload["official_validation_predictions"] == 0 and payload["external_predictions"] == 0
    assert overall["v9_coordinate_macro_mae_m"] == v9["outer_coordinate_macro_mae_2d_m"]
    assert overall["v17_coordinate_macro_mae_m"] == v17["selected_coordinate_macro_mae_m"]
    assert overall["sub_six_gate_m"] == 6.0
    assert overall["v9_passed"] == gate(v9["outer_coordinate_macro_mae_2d_m"])
    assert overall["v17_passed"] == gate(v17["selected_coordinate_macro_mae_m"])
    assert overall["next_stage_authorized"] == (overall["v9_passed"] and overall["v17_passed"])
    assert overall["successor_model_implemented_in_this_stage"] is False
    assert env_audit["pytest"]["passed"] == 17 and env_audit["pytest"]["failed"] == 0
    assert post_precheck["pytest"]["passed"] == 19 and post_precheck["pytest"]["failed"] == 0
    raw_sha = sha(ROOT / "data/UJIndoorLoc/trainingData.csv")
    assert raw_sha == registry["training_data_sha256"] == v17_registry["training_data_sha256"]
    source_records = {}
    for relative, original in initial["original_source_file_sha256"].items():
        name = relative.replace("\\", "/")
        actual = sha(ROOT / name)
        expected = effective_repair_hashes.get(name, original)
        assert actual == expected, name
        source_records[name] = {"original_sha256":original,"final_sha256":actual,
                                "authorized_implementation_repair":name in effective_repair_hashes}
    git_env = {**os.environ, "GIT_OPTIONAL_LOCKS":"0"}
    def git_read(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, env=git_env).decode("utf-8").strip()
    branch, head = git_read("branch","--show-current"), git_read("rev-parse","HEAD")
    assert branch == initial["branch"] and head == initial["head"]
    changed = git_read("diff","--name-only").splitlines()
    assert set(changed) == set(effective_repair_hashes), changed
    tracked_status = git_read("status","--short","--untracked-files=no")
    env = env_audit["environment"]
    selected, historical = v17["selected_name"], v17["historical_name"]
    cfg = v9["selected"]
    endpoint_rows = []
    for name, label in ((selected, "selected E16"), (historical, "historical E5")):
        values = v17["metrics"][name]
        endpoint_rows.append(
            f"| {label} | {q(name)} | {values['exact_mixture_mean_mae_m']['coordinate_macro_m']:.10f} | "
            f"{values['energy_score_2d_m']['coordinate_macro_m']:.10f} | "
            f"{values['exact_mixture_mean_mae_m']['row_mean_m']:.10f} |"
        )
    epoch_rows = [f"| {r['member_index']:02d} | {r['seed']} | {r['selected_epoch']} |" for r in model_records]
    mass_rows = []
    for name, label in ((selected, "selected E16"), (historical, "historical E5")):
        mass = v17["retained_mass_diagnostic"][name]
        mass_rows.append(f"| {label} | {mass['mean']:.10f} | {mass['q10']:.10f} | {mass['median']:.10f} | {mass['q90']:.10f} |")
    inference = v17["paired_familywise_inference"]
    inference_rows = []
    for endpoint, record in inference["comparisons"].items():
        lo, hi = record["paired_percentile_ci95_m"]
        inference_rows.append(
            f"| {q(endpoint)} | {record['mean_difference_m']:.10f} | {record['standard_error_m']:.10f} | "
            f"[{lo:.10f}, {hi:.10f}] | {record['simultaneous_upper_bound_m']:.10f} |"
        )
    stage_rows = [
        f"| {q(name)} | {r['attempt']} | {r['exit_code']} | {r['runtime_seconds']/60:.2f} | [\u65e5\u5fd7](run_audit/{Path(r['log_path']).name}) |"
        for name, r in stages.items()
    ]
    v9_gate, v17_gate = ("PASS" if overall["v9_passed"] else "FAIL"), ("PASS" if overall["v17_passed"] else "FAIL")
    finished = datetime.now(timezone.utc).isoformat()
    artifact_paths = (
        E / "v9/REPRODUCTION_V9.json", E / "v9/outer_test_result_v9_uji.json",
        E / "v9/selection_freeze_v9_uji.json", E / "v17/outer_result_v17_reproduction.json",
        E / "v17/registry_v17_reproduction.json", E / "REPRODUCTION_SUMMARY.json"
    )
    final_audit = {
        "status":"FINAL_VERIFICATION_PASS","recorded_utc":finished,
        "branch":branch,"head":head,"git_mutations_performed":False,
        "tracked_changed_files":changed,"tracked_status":tracked_status,
        "source_file_hash_verification":source_records,"training_data_sha256":raw_sha,
        "member_check":member_check,"v17_checkpoint_records":model_records,
        "v17_checkpoints_byte_identical_to_before_diagnostic_repair":True,
        "registries_byte_identical_to_before_diagnostic_repair":True,
        "registration_time_source_snapshot_deltas":source_snapshot_deltas,
        "post_training_implementation_repair":diagnostic_repair,
        "final_static_precheck":post_precheck,
        "stage_executions":stages,
        "artifact_file_sha256":{str(p.relative_to(E)).replace("\\","/"):sha(p) for p in artifact_paths},
        "v9_gate":v9_gate,"v17_gate":v17_gate,"next_stage_authorized":overall["next_stage_authorized"],
        "strict_gate_waived":False,"successor_implemented":False,
        "official_validation_predictions":0,"external_predictions":0,
        "v18_v19_model_execution":False,"outer_outcome_tuning":False
    }
    report = f"""# V9 + V17 \u5386\u53f2\u5f3a\u57fa\u7ebf\u590d\u73b0\u8fd0\u884c\u62a5\u544a

\u5b8c\u6210\u65f6\u95f4\uff08UTC\uff09\uff1a{finished}\u3002\u672c\u8f6e\u5b8c\u6574\u6267\u884c\u6307\u5b9a V9/V17 \u6d41\u7a0b\uff0c\u7ed3\u675f\u72b6\u6001\u4e3a **{overall['status']}**\u3002

| \u95e8\u69db\u5bf9\u8c61 | outer coordinate-macro 2D MAE\uff08m\uff09 | \u4e25\u683c <6.0 m |
| --- | ---: | --- |
| V9 frozen system | {v9['outer_coordinate_macro_mae_2d_m']:.10f} | **{v9_gate}** |
| V17 selected E16/L192/K192/alpha=1 | {v17['selected_coordinate_macro_mae_m']:.10f} | **{v17_gate}** |

{q('next_stage_authorized')} = **{str(overall['next_stage_authorized']).lower()}**\u3002\u672a\u653e\u5bbd\u3001\u7ed5\u8fc7\u6216 waiver \u95e8\u69db\uff1b\u672c\u9636\u6bb5\u7ed3\u675f\uff0c\u4e0d\u5b9e\u73b0\u540e\u7ee7\u6a21\u578b\u3002

## 1. \u73af\u5883\u4e0e\u8f93\u5165

- Python\uff1a{q(env['python_executable'])}\uff1b\u7248\u672c {q(env['python'])}\u3002
- PyTorch\uff1a{q(env['torch'])}\uff1bCUDA runtime\uff1a{q(env['torch_cuda'])}\uff1bCUDA available\uff1a{env['cuda_available']}\u3002
- GPU\uff1a{env['gpu']}\uff0c\u663e\u5b58 {env['gpu_total_memory_bytes']} bytes\uff1b\u7cfb\u7edf\uff1a{q(env['os'])}\u3002
- \u5206\u652f\uff1a{q(branch)}\uff1bHEAD\uff1a{q(head)}\uff0c\u4e0e\u542f\u52a8\u65f6\u4e00\u81f4\u3002
- \u552f\u4e00\u539f\u59cb\u6570\u636e\u8f93\u5165\uff1a{q(str(ROOT / 'data/UJIndoorLoc/trainingData.csv'))}\u3002
- trainingData.csv SHA256\uff1a{q(raw_sha)}\u3002
- \u672c\u9636\u6bb5\u65b0\u4ea7\u7269\u4f4d\u4e8e {q('fhm_dapp_tim/evidence/historical_reproduction/')} \u548c {q('fhm_dapp_tim/models/historical_reproduction/')}\u3002
- V17 outer_refit\uff1a15871 rows / 746 coordinate anchors\uff1bouter-test\uff1a4066 rows / 187 coordinate groups\u3002\u8bad\u7ec3\u5747\u6765\u81ea trainingData.csv \u7684\u6ce8\u518c\u5212\u5206\u3002

## 2. \u9884\u68c0\u4e0e\u6267\u884c

\u6b63\u5f0f\u8bad\u7ec3\u524d\u7f16\u8bd1\u901a\u8fc7\uff0c\u6307\u5b9a\u4e09\u4e2a\u6d4b\u8bd5\u6587\u4ef6\u4e3a **17 passed**\u3002\u8bad\u7ec3\u540e\u8bca\u65ad\u4fee\u590d\u7684\u6700\u7ec8\u7ed3\u679c\u4e3a **19 passed\u30010 failed\u30010 skipped\u30010 xfailed**\uff0816 \u4e2a\u539f\u6d4b\u8bd5\u30011 \u4e2a\u771f\u5b9e registry \u96c6\u6210\u56de\u5f52\u6848\u4f8b\u30012 \u4e2a retained-mass \u56de\u5f52\u6848\u4f8b\uff09\u3002\u4fee\u590d\u540e CUDA preflight \u4e3a ready\uff0c\u56fa\u5b9a seed panel\u30016.0 m gate\u3001official/external authorization=false \u5747\u518d\u6b21\u6838\u9a8c\u901a\u8fc7\u3002

| \u9636\u6bb5 | attempt | exit code | \u8017\u65f6\uff08min\uff09 | \u5b8c\u6574\u8bb0\u5f55 |
| --- | ---: | ---: | ---: | --- |
{chr(10).join(stage_rows)}

\u521d\u59cb\u5931\u8d25\u8bb0\u5f55\u4fdd\u7559\uff0c\u672a\u7528\u6700\u7ec8\u901a\u8fc7\u8bb0\u5f55\u8986\u76d6\u3002\u8be6\u89c1 [\u8bad\u7ec3\u524d\u73af\u5883\u4e0e\u9884\u68c0\u5ba1\u8ba1](run_audit/ENVIRONMENT_AND_STATIC_PRECHECK.json)\u3001[\u8bad\u7ec3\u540e\u4fee\u590d\u9884\u68c0](run_audit/POST_TRAINING_IMPLEMENTATION_PRECHECK.json) \u548c\u4e0b\u6587\u4fee\u590d\u8bb0\u5f55\u3002run-v17 \u5b8c\u6210\u5168\u90e8 25 \u540d\u6210\u5458\u5e76\u5904\u7406 4066 \u6761 outer rows \u540e\uff0c\u5728 retained-mass \u8bca\u65ad\u62fc\u63a5\u5931\u8d25\uff08exit code=1\uff09\uff1b\u4fee\u590d\u540e\u5355\u72ec\u6267\u884c evaluate-v17 \u5b8c\u6210\u540c\u4e00\u51bb\u7ed3\u8bc4\u4ef7\uff0c\u672a\u91cd\u8dd1\u4efb\u4f55\u6210\u5458\u8bad\u7ec3\u3002\u5931\u8d25\u65f6\u672a\u751f\u6210\u6b63\u5f0f V17 \u7ed3\u679c\uff0c\u4e5f\u672a\u8bfb\u53d6\u5176\u6570\u503c\u7528\u4e8e\u9009\u62e9\u6216\u8c03\u53c2\u3002

## 3. V9 \u539f\u6d41\u7a0b\u7ed3\u679c

- \u5019\u9009\u4e0e\u7f51\u683c\uff1ageometry_invariant / historical_affine\uff0cseed 9311..9322\uff0cK=[8,16,24,32,48]\uff0clambda=[1/64,1/32,1/16,1/8,1/4,1/2,1,2]\uff0c\u4fdd\u6301\u539f\u6837\u3002
- \u539f\u59cb E=8\u3001E=10 \u5224\u5b9a\u5747\u8981\u6c42\u589e\u52a0 2 \u540d\u6210\u5458\uff0c\u6700\u7ec8\u4e24\u5019\u9009\u5404\u5b8c\u6210 E=12 \u7684\u8bad\u7ec3\u4e0e grid\u3002
- \u51bb\u7ed3\u914d\u7f6e\uff1acandidate={q(cfg['candidate'])}\uff0cE={cfg['E']}\uff0cK={cfg['K']}\uff0clambda={cfg['lambda']}\u3002
- development coordinate-macro MAE\uff1a{cfg['development_coordinate_macro_mae_2d_m']:.10f} m\uff1bEnergy Score\uff1a{cfg['development_coordinate_macro_energy_score_2d_m']:.10f} m\u3002
- \u51bb\u7ed3\u4f9d\u636e\uff1a{q(freeze['selection_basis'])}\uff1bsaturation_claim={str(freeze['saturation_claim']).lower()}\u3002
- E=12 \u4ecd\u672a\u8bc1\u660e\u4fe1\u606f\u9971\u548c\u3002\u4f7f\u7528\u8fd0\u884c\u524d\u5df2\u5b58\u5728\u4e8e\u539f V9 \u6e90\u7801\u7684 V9-A1-terminal-selection \u6761\u6b3e\uff0c\u5728\u9884\u6ce8\u518c\u9884\u7b97\u4e0a\u9650\u9009\u62e9 development proper-risk \u5168\u5c40\u6700\u5c0f\u503c\uff0c\u5e76\u660e\u786e\u4e0d\u5ba3\u79f0\u9971\u548c\u3002\u672c\u8f6e\u672a\u4fee\u6539\u8be5\u6761\u6b3e\u3002
- 12 \u540d final member \u5747\u7528\u81ea\u5df1\u7684\u539f\u59cb early-stop \u6240\u9009 epoch\uff0c\u5728 outer_refit \u4e0a\u4ece\u5934\u91cd\u8bad\u3002
- \u51bb\u7ed3\u524d outer-test accesses=0\uff1b\u6700\u7ec8 outer-test access number=1\u3002
- outer coordinate-macro MAE\uff1a**{v9['outer_coordinate_macro_mae_2d_m']:.10f} m**\uff1bEnergy Score\uff1a{v9_result['metrics']['coordinate_macro_energy_score_2d_m']:.10f} m\u3002
- \u8f85\u52a9 row MAE\uff1a{v9_result['metrics']['row_mae_2d_m']:.10f} m\u3002\u4e25\u683c\u95e8\u69db\u4ee5 coordinate-macro MAE \u5224\u5b9a\uff0cV9 \u4e3a **{v9_gate}**\u3002

\u8bc1\u636e\uff1a[V9 \u7ed3\u679c\u6c47\u603b](v9/REPRODUCTION_V9.json)\u3001[\u6b63\u5f0f outer-test \u7ed3\u679c](v9/outer_test_result_v9_uji.json)\u3001[\u914d\u7f6e\u51bb\u7ed3](v9/selection_freeze_v9_uji.json)\u3002

## 4. V17 25-member \u51bb\u7ed3 R3 \u7ed3\u679c

\u6bcf\u4e2a seed \u72ec\u7acb\u8c03\u7528\u672a\u4fee\u6539\u7684\u539f V17-R0 _fit_schedule\uff0c\u5728 member_fit/early_stop \u4e0a\u6309 coordinate-macro MAE \u9009\u62e9 epoch\uff1b\u968f\u540e\u4ee5\u540c seed \u8c03\u7528 _fit_refit\uff0c\u5728 outer_refit \u4e0a\u4ece\u5934\u91cd\u8bad selected_epoch \u8f6e\u3002\u6700\u5927 80 epoch\u3001patience=10 \u6cbf\u7528\u539f\u5b9a\u4e49\uff0c\u4e0d\u7edf\u4e00\u56fa\u5b9a epoch\uff0c\u4e0d\u4f9d\u636e outer outcome \u9009 epoch\u3002

\u5b98\u65b9 check-v17-members\uff1acomplete_members=25\uff0call_complete=true\u3002\u5168\u90e8\u6210\u5458 identity\u3001seed\u3001registry\u3001phase=r3_outer_refit\u3001anchor_count=746 \u53ca\u8bad\u7ec3\u672a\u6d88\u8d39 outer-test/official labels \u68c0\u67e5\u901a\u8fc7\u3002checkpoint \u6587\u4ef6 SHA256 \u5df2\u4e0e\u6b63\u5f0f\u8bb0\u5f55\u9010\u4e00\u6838\u5bf9\u3002

| member | seed | selected_epoch |
| ---: | ---: | ---: |
{chr(10).join(epoch_rows)}

\u91c7\u7528\u539f R3 balanced cyclic subset estimand\uff1a\u5168\u90e8 25 seed\uff0c\u6bcf\u4e2a\u914d\u7f6e 25 \u4e2a\u5b50\u96c6\uff0cgenerator=7\uff0c\u6210\u5458\u5404\u51fa\u73b0 E \u6b21\u3002\u672a\u9009\u62e9\u6700\u4f18\u5b50\u96c6\uff0c\u672a\u91cd\u9009 E/L/K/alpha\u3002Student-t df=4\uff0c\u6bcf stream 64 predictive draws\uff0c\u9884\u6d4b seed=17931\u3002

| \u914d\u7f6e | \u6b63\u5f0f\u540d\u79f0 | coordinate-macro MAE\uff08m\uff09 | coordinate-macro Energy Score\uff08m\uff09 | \u8f85\u52a9 row MAE\uff08m\uff09 |
| --- | --- | ---: | ---: | ---: |
{chr(10).join(endpoint_rows)}

### \u914d\u5bf9 familywise \u6bd4\u8f83

\u65b9\u5411\uff1a{q(inference['difference_orientation'])}\u3002\u6309 {inference['coordinate_groups']} \u4e2a coordinate groups \u5171\u540c\u91cd\u91c7\u6837\uff0cstudentized max-T\uff0cconfidence={inference['confidence']}\uff0cfamily_size={inference['family_size']}\uff0cbootstrap={inference['bootstrap_repetitions']} \u6b21\uff0cseed={inference['seed']}\uff0ccritical max-T={inference['critical_max_t']:.10f}\u3002

| endpoint | E16\u2212E5 \u5747\u503c\u5dee\uff08m\uff09 | \u6807\u51c6\u8bef\uff08m\uff09 | paired percentile CI95\uff08m\uff09 | familywise simultaneous upper bound\uff08m\uff09 |
| --- | ---: | ---: | --- | ---: |
{chr(10).join(inference_rows)}

\u5dee\u503c\u4e3a\u8d1f\u8868\u793a selected E16 \u7684\u8be5\u6307\u6807\u8f83\u4f4e\uff1bsimultaneous upper bound <0 \u624d\u652f\u6301\u8be5 endpoint \u7684 familywise \u6539\u5584\u3002\u914d\u5bf9\u6bd4\u8f83\u4e0d\u66ff\u4ee3\u4e25\u683c sub-6m gate\u3002

### Retained-mass diagnostics

| \u914d\u7f6e | mean | q10 | median | q90 |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(mass_rows)}

V17 selected \u7684 sub-6m \u7ed3\u679c\u4e3a **{v17_gate}**\u3002\u5168\u90e8\u6570\u503c\u53ca\u9010\u5b50\u96c6 realization records \u4fdd\u5b58\u5728 [\u6b63\u5f0f V17 \u7ed3\u679c](v17/outer_result_v17_reproduction.json)\u3002

## 5. \u95ee\u9898\u3001\u6700\u5c0f\u4fee\u590d\u4e0e\u9650\u5236

\u6570\u636e\u8def\u5f84\u548c\u6d4b\u8bd5 fixture \u4fee\u590d\u5728\u6b63\u5f0f prepare \u4e0e\u8bad\u7ec3\u524d\u5b8c\u6210\uff1bretained-mass \u62fc\u63a5\u4fee\u590d\u5728 25 \u540d\u6210\u5458\u8bad\u7ec3\u5b8c\u6210\u540e\u6267\u884c\u3002\u4e24\u9636\u6bb5\u5747\u53ea\u4fee\u6539 reproduction harness/tests\uff0c\u672a\u4fee\u6539 V9/V17 \u539f\u59cb\u6a21\u5757\u3001\u8bad\u7ec3\u635f\u5931\u3001seed\u3001split\u3001\u8bc4\u5206\u516c\u5f0f\u3001\u9009\u62e9\u89c4\u5219\u6216 gate\u3002

| \u95ee\u9898 | \u539f\u56e0\u4e0e\u5904\u7406 | \u8bc1\u636e |
| --- | --- | --- |
| sandbox exec \u542f\u52a8\u5931\u8d25 | helper_unknown_error/setup refresh \u53d1\u751f\u5728\u547d\u4ee4\u542f\u52a8\u524d\uff1b\u901a\u8fc7\u7ecf\u5ba1\u67e5\u7684\u672c\u5730\u6267\u884c\u7ee7\u7eed\uff0c\u4e0d\u4fee\u6539\u79d1\u5b66\u89c4\u5219 | [\u521d\u59cb\u5ba1\u8ba1](run_audit/INITIAL_EXECUTION_AUDIT.json) |
| pytest tmp_path \u6743\u9650\u4e0e\u65e5\u5fd7\u7f16\u7801\u5931\u8d25 | \u9ed8\u8ba4 Windows temp \u7236\u76ee\u5f55\u4e0d\u53ef\u8bfb\uff1b\u6539\u7528\u83b7\u51c6 model root \u4e0b fixture temp\uff1blogger/child \u4f7f\u7528 UTF-8\uff0c\u4fdd\u7559\u9996\u4efd\u90e8\u5206\u65e5\u5fd7 | [\u8fd0\u884c\u73af\u5883\u4fee\u590d](run_audit/PRECHECK_OPERATIONAL_REPAIR.json) |
| PYTEST_ADDOPTS \u4e2d Windows \u53cd\u659c\u6760\u88ab shlex \u541e\u6389 | basetemp \u6539\u4e3a as_posix\uff1b\u9519\u8bef\u8def\u5f84\u7ecf\u7edd\u5bf9\u8def\u5f84\u8fb9\u754c\u6838\u9a8c\u540e\u79fb\u52a8\u5230 model root\uff0c\u4fdd\u7559\u800c\u672a\u5220\u9664 | [\u8def\u5f84\u4e0e fixture \u4fee\u590d](run_audit/PRECHECK_DESIGN_FIXTURE_AND_PATH_REPAIR.json) |
| \u4e24\u4e2a\u539f registry \u6d4b\u8bd5\u5f15\u7528\u4e0d\u5b58\u5728\u7684\u65e7 design | \u6d4b\u8bd5\u663e\u5f0f\u7ed1\u5b9a\u672c\u8f6e\u5df2\u63d0\u4f9b\u7684 DESIGN_V9_REPRODUCTION.md \u4e0e\u540c\u4e00 TRAIN_DATA\uff1b\u539f registry hash/tamper/no-validation \u65ad\u8a00\u4fdd\u7559\uff1b\u65b0\u589e\u771f\u5b9e registry \u96c6\u6210\u6d4b\u8bd5 | [\u6d4b\u8bd5\u4fee\u590d](run_audit/PRECHECK_DESIGN_FIXTURE_AND_PATH_REPAIR.json) |
| reproduction harness \u7684\u8bad\u7ec3 CSV \u8def\u5f84\u9519\u8bef | TRAIN_DATA \u4ece package ROOT/data \u6539\u4e3a repository ROOT.parent/data\uff0c\u53ea\u6539\u8fd9\u4e00\u8f93\u5165\u7ed1\u5b9a\uff1b\u96c6\u6210\u6d4b\u8bd5\u5148\u5931\u8d25\u518d\u901a\u8fc7\uff1b\u5220\u9664\u8be5 assignment \u540e\u7684\u5168\u6587\u4ef6 AST \u9010\u9879\u4e00\u81f4 | [\u8f93\u5165\u8def\u5f84\u4fee\u590d\u4e0e AST \u8bc1\u660e](run_audit/PRECHECK_TRAINING_INPUT_PATH_REPAIR.json) |
| apply_patch \u62a5 reparse point\uff0c\u4ee5\u53ca\u5ba1\u8ba1\u8def\u5f84\u952e\u683c\u5f0f\u4e0d\u4e00\u81f4 | \u539f\u751f\u8def\u5f84\u68c0\u67e5\u786e\u8ba4\u76ee\u6807\u4e3a workspace \u5185\u666e\u901a\u6587\u4ef6\uff0c\u6539\u7528\u5e26\u8fb9\u754c\u6838\u9a8c\u7684\u6587\u4ef6\u5199\u5165\uff1b\u5931\u8d25\u952e\u67e5\u8be2\u672a\u53d1\u751f\u5199\u5165\uff0c\u968f\u540e\u4ec5\u89c4\u8303\u5316\u5ba1\u8ba1\u8def\u5f84\u952e | [\u540c\u4e00\u4fee\u590d\u8bb0\u5f55](run_audit/PRECHECK_DESIGN_FIXTURE_AND_PATH_REPAIR.json) |
| V17 retained-mass \u6700\u540e\u77ed\u6279\u62fc\u63a5\u5931\u8d25 | \u539f\u7edf\u8ba1\u6bb5\u62bd\u53d6\u4e3a\u79c1\u6709 helper\uff1b\u9010\u6279\u5c55\u5f00\u5168\u90e8 member-row scalar \u540e\u62fc\u63a5\uff0c\u4fdd\u6301 mean/q10/median/q90 \u5b9a\u4e49\u4e0d\u53d8\u30024066=127\u00d732+2 \u56de\u5f52\u5148\u590d\u73b0\u5931\u8d25\uff0c\u518d\u4e0e\u5b8c\u6574 25\u00d74066 \u53c2\u8003\u503c\u7cbe\u786e\u4e00\u81f4\uff1b\u975e\u8bca\u65ad\u90e8\u5206\u5168\u6587\u4ef6 AST \u4e0d\u53d8\uff1b\u5168\u90e8 checkpoint \u548c registry \u5b57\u8282\u54c8\u5e0c\u4e0d\u53d8\uff0c\u53ea\u91cd\u8dd1\u51bb\u7ed3\u8bc4\u4ef7 | [\u5931\u8d25\u4e0e\u4fee\u590d\u8ba1\u5212](run_audit/V17_RETAINED_MASS_FAILURE_AND_REPAIR_PLAN.json)\u3001[\u5b9e\u73b0\u4fee\u590d\u8bc1\u660e](run_audit/V17_RETAINED_MASS_IMPLEMENTATION_REPAIR.json) |
| \u539f V17 CUDA attention backward \u975e\u786e\u5b9a\u6027\u63d0\u793a | \u6309\u539f\u8fd0\u884c\u8bbe\u7f6e\u6267\u884c\u5e76\u4fdd\u7559 warning\uff0c\u672a\u901a\u8fc7\u6539\u7b97\u6cd5\u6216\u8bad\u7ec3\u8bbe\u7f6e\u8fdb\u884c\u4fee\u590d\uff1b\u56fa\u5b9a seed \u4e0d\u80fd\u636e\u6b64\u4fdd\u8bc1 GPU \u8bad\u7ec3\u9010\u4f4d\u590d\u73b0 | [V17 \u5b8c\u6574\u65e5\u5fd7](run_audit/{Path(stages['run-v17']['log_path']).name}) |

pytest \u4e2d\u95f4\u7ed3\u679c\u4f9d\u6b21\u5305\u62ec\u4e34\u65f6\u76ee\u5f55/\u65e5\u5fd7\u5931\u8d25\u30012 failed/14 passed\uff08\u65e7 design\uff09\u30013 failed/14 passed\uff08\u8bad\u7ec3\u8def\u5f84\uff09\uff0c\u8bad\u7ec3\u524d\u4e3a 17 passed\u3002\u8bad\u7ec3\u540e\u7684\u65b0\u56de\u5f52\u5148\u5f97\u5230 1 failed/18 passed\uff08\u6700\u540e width-2 batch\uff09\uff0c\u4fee\u590d\u540e\u4e3a 19 passed\u3002\u5b8c\u6574 attempt \u65e5\u5fd7\u548c JSON \u5747\u4fdd\u7559\u4e8e run_audit\u3002

\u6700\u7ec8\u4ec5 3 \u4e2a tracked \u6587\u4ef6\u5305\u542b\u672c\u8f6e\u6700\u5c0f\u5b9e\u73b0/\u6d4b\u8bd5\u4fee\u590d\uff1a{q('fhm_dapp_tim/fhm_dapp/historical_reproduction.py')}\u3001{q('tests/test_historical_reproduction.py')}\u3001{q('tests/test_uji_v9_experiment.py')}\u3002\u5176\u4f59 18 \u4e2a\u542f\u52a8\u65f6\u767b\u8bb0\u6587\u4ef6\u54c8\u5e0c\u4e0d\u53d8\uff0c7 \u4e2a\u6ce8\u518c\u7684 V9/V17 \u79d1\u5b66\u6a21\u5757\u4ecd\u4fdd\u6301\u539f\u54c8\u5e0c\u3002\u539f registry \u4fdd\u7559\u6ce8\u518c\u65f6 runner \u5feb\u7167 {q(diagnostic_repair['source_before_sha256'])}\uff1b\u8bca\u65ad\u4fee\u590d\u540e\u7684 runner \u4e3a {q(diagnostic_repair['source_after_sha256'])}\uff0c\u65e7/\u65b0\u54c8\u5e0c\u4e0e AST \u8bc1\u660e\u901a\u8fc7\u8ffd\u52a0\u5ba1\u8ba1\u660e\u786e\u5173\u8054\uff0c\u672a\u624b\u6539 registry\u3002\u6700\u7ec8\u660e\u7ec6\u89c1 [\u5b8c\u6574\u6838\u9a8c](run_audit/FINAL_EXECUTION_VERIFICATION.json)\u3002

## 6. \u7ed3\u8bba\u4e0e\u8fb9\u754c\u786e\u8ba4

\u6b63\u5f0f\u603b\u7ed3\u679c\u89c1 [REPRODUCTION_SUMMARY.json](REPRODUCTION_SUMMARY.json)\uff1aV9={v9_gate}\uff0cV17 selected={v17_gate}\uff0cboth_historical_baselines_reproduced={str(overall['both_historical_baselines_reproduced']).lower()}\uff0cnext_stage_authorized={str(overall['next_stage_authorized']).lower()}\u3002

- \u672c\u9636\u6bb5\u6ca1\u6709\u8fd0\u884c V18/V19 \u6a21\u578b\uff0c\u4e5f\u6ca1\u6709\u4f7f\u7528\u5176 outcome \u8c03\u53c2\u3002
- \u6ca1\u6709\u8bfb\u53d6\u6216\u9884\u6d4b data/UJIndoorLoc/validationData.csv\uff0cofficial validation predictions=0\u3002
- \u6ca1\u6709\u4f7f\u7528\u5916\u90e8\u6570\u636e\u96c6\uff0cexternal predictions=0\u3002
- \u6ca1\u6709\u6839\u636e outer outcome \u8c03\u53c2\u3001\u91cd\u9009 seed\u3001epoch \u6216 frozen law\u3002
- \u6ca1\u6709\u653e\u5bbd\u3001\u7ed5\u8fc7\u6216 waiver <6.0 m gate\uff1b\u672a\u6cbf\u7528 V19-R0 \u7684\u4efb\u4f55 gate waiver\u3002
- \u6ca1\u6709\u8bbe\u8ba1\u6216\u5b9e\u73b0\u4e0b\u4e00\u9636\u6bb5\u878d\u5408\u6a21\u578b\u3002
- \u6ca1\u6709\u6267\u884c git commit/push/pull/merge/rebase\uff0c\u4e5f\u6ca1\u6709\u521b\u5efa\u6216\u5220\u9664\u5206\u652f\uff1bbranch/HEAD \u4e0e\u542f\u52a8\u65f6\u4e00\u81f4\u3002
- \u6240\u6709\u65b0\u4ea7\u7269\u9650\u5b9a\u5728\u672c\u8f6e evidence/model roots\uff0c\u5931\u8d25\u8bc1\u636e\u548c checkpoint \u5747\u4fdd\u7559\u3002
- \u672c\u4efb\u52a1\u5230\u6b64\u7ed3\u675f\uff1b\u5373\u4f7f\u67d0\u4e2a\u57fa\u7ebf\u901a\u8fc7\uff0c\u4e5f\u4e0d\u7ee7\u7eed\u5b9e\u65bd\u540e\u7ee7\u9636\u6bb5\u3002
"""
    write_once(A / "FINAL_EXECUTION_VERIFICATION.json",
               json.dumps(final_audit,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    report_path = E / "RUN_REPORT_HISTORICAL_REPRODUCTION.md"
    write_once(report_path,report)
    print(json.dumps({"status":"REPORT_AND_FINAL_VERIFICATION_WRITTEN",
                      "report":str(report_path),"report_sha256":sha(report_path),
                      "v9_mae_m":v9["outer_coordinate_macro_mae_2d_m"],
                      "v17_selected_mae_m":v17["selected_coordinate_macro_mae_m"],
                      "next_stage_authorized":overall["next_stage_authorized"]},ensure_ascii=True))

if __name__ == "__main__":
    main()
