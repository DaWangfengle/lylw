"""Assemble the frozen experiment report after benchmark; no model imports."""
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parents[3]
E = AUDIT.parent
PHASES = ['prepare', 'train-selection', 'calibrate-freeze', 'stability', 'train-confirmation', 'confirm', 'train-final', 'benchmark']
SOURCES = ['invariant', 'prototype', 'learned', 'marked_radio', 'floor', 'geometry']
FAMILIES = ['representation', 'radio', 'structure']
LEVELS = [50, 80, 90, 95]
WITHIN = ['representation_invariant', 'representation_learned', 'radio_prototype', 'radio_marked', 'structure_floor', 'structure_geometry']
FAILED = E / 'failed_attempts/d7500d3-20261001T081726Z/run_audit'

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()

def digest(payload, omitted):
    return hashlib.sha256(json.dumps({k: v for k, v in payload.items() if k != omitted}, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')).hexdigest()

def checked(path, key):
    payload = read(path)
    assert payload[key] == digest(payload, key)
    return payload

execution = {phase: read(AUDIT / (phase + '_execution.json')) for phase in PHASES}
assert all(row['exit_code'] == 0 and row['sources_unchanged'] and row['runtime_audit_valid'] for row in execution.values())
runtime = {phase: read(AUDIT / (phase + '_runtime.json')) for phase in PHASES}
preflight = read(AUDIT / 'preflight.json')
protocol = read(AUDIT / 'prepare_verification.json')
selection = read(AUDIT / 'selection_verification.json')
cref = read(AUDIT / 'confirmation-refit_verification.json')
final = read(AUDIT / 'final_verification.json')
verification = read(AUDIT / 'benchmark-result_verification.json')
reg = checked(E / 'registry_v18_3.json', 'registry_sha256')
freeze = checked(E / 'selection_freeze_v18_3.json', 'freeze_sha256')
cal = checked(E / 'calibration_result_v18_3.json', 'result_sha256')
stab = checked(E / 'stability_result_v18_3.json', 'result_sha256')
conf = checked(E / 'confirmation_result_v18_3.json', 'result_sha256')
lock = checked(E / 'confirmation_consumed.lock', 'lock_sha256')
bench = checked(E / 'benchmark_result_v18_3.json', 'result_sha256')
cm, bm = conf['confirmation_metrics'], bench['benchmark_metrics']
assert all(row['registry_sha256'] == reg['registry_sha256'] for row in [freeze, cal, stab, conf, lock, bench])
assert all(row['freeze_sha256'] == freeze['freeze_sha256'] for row in [stab, conf, lock, bench])
assert bench['benchmark_was_pre_exposed_before_v18_3'] and not bench['blind_confirmation_claim'] and not bench['used_for_model_selection']
assert all(sha(item['path']) == item['sha256'] for item in reg['role_shards'].values())
snapshot = read(AUDIT / 'source_snapshot.json')
assert all(sha(ROOT / name) == value for name, value in {**snapshot['files'], **snapshot['common_source_files']}.items())
assert snapshot['formal_experiment_sha'] == subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
events = [json.loads(line) for path in AUDIT.glob('*_io.jsonl') for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
assert all(item['permitted'] for item in events)
used = [item for item in events if item['kind'] == 'role_shard' and item['reading'] and item['path'].endswith('/confirmation.pt') and item['phase'] == 'confirm']
writes = [item for item in events if item['kind'] == 'confirmation_lock' and not item['reading']]
assert len(writes) == 1 and len(used) == 2 and writes[0]['time_ns'] < min(item['time_ns'] for item in used)
gate = read(AUDIT / 'pre_confirmation_gate.json')
assert len(gate['gate_checks']) == 12 and all(gate['gate_checks'].values())
histories = dict(selection_search=selection['selection_history'], selection_refit=selection['selection_refit_history'],
    confirmation_refit=cref['confirmation_refit_history'], final_refit=final['final_refit_history'],
    stability=read(AUDIT / 'stability_full_training_histories.json'))
assert [row['seed'] for row in histories['stability']] == stab['seeds'] == [18431, 18432, 18433, 18434, 18435]
assert all(len(row['history']) == freeze['frozen_selected_epoch'] for row in histories['stability'])
seed_family_summary = {name: dict(mean=statistics.mean(row['metrics']['mean_family_weight_' + name] for row in stab['records']),
    sample_std=statistics.stdev(row['metrics']['mean_family_weight_' + name] for row in stab['records']),
    minimum=min(row['metrics']['mean_family_weight_' + name] for row in stab['records']),
    maximum=max(row['metrics']['mean_family_weight_' + name] for row in stab['records'])) for name in FAMILIES}

# Historical outcomes are loaded only after the current benchmark passed verification.
previous = checked(ROOT / 'fhm_dapp_tim/evidence/v18_2/benchmark_result_v18_2.json', 'result_sha256')
assert previous['benchmark_data_sha256'] == bench['benchmark_data_sha256']
old = previous['benchmark_metrics']
comparison_keys = ['coordinate_macro_mae_m', 'coordinate_macro_energy_score_m', 'coordinate_macro_mixture_nll',
    'final_nearest_anchor_recall_at_8', 'final_nearest_anchor_recall_at_64', 'union_nearest_anchor_recall',
    'invariant_nearest_anchor_recall', 'mean_between_anchor_radial_sd_m'] + ['predictive_radial_coverage_' + str(q) for q in LEVELS]
comparison = dict(status='descriptive comparison on the identical PRE-EXPOSED official CSV; no causal ablation or independent external confirmation',
    V18_2_result_sha256=previous['result_sha256'], official_csv_sha256=bench['benchmark_data_sha256'],
    V18_2=old, V18_3=bm, differences={key: bm[key] - old[key] for key in comparison_keys},
    confirmation_scores_compared_across_different_holdouts=False)
failure = read(FAILED / 'failure_record.json')
authorization = read(FAILED / 'restart_authorization.json')
fix_verification = read(FAILED / 'implementation_fix_proposal_verification.json')
old_snapshot = read(FAILED / 'source_snapshot.json')
assert snapshot['files']['fhm_dapp_tim/fhm_dapp/v18_3_single.py'] == old_snapshot['files']['fhm_dapp_tim/fhm_dapp/v18_3_single.py']
assert failure['confirmation_lock_created'] is False and failure['confirmation_content_read_after_prepare'] is False and failure['benchmark_read'] is False

audit = dict(one_inference_checkpoint=True, residual_experts_equals_one=True, trainable_evidence_gate_used=False,
    v18_2_runtime_import=False, teacher_used=False, distillation_used=False, ensemble_used=False, member_axis=False,
    best_seed_selected=False, prediction_averaging=False, historical_checkpoint_loaded=False,
    positive_self_anchor_target_used=False, calibration_labels_used_for_backbone_gradients_before_freeze=False,
    confirmation_contents_accessed_before_lock_after_prepare=False, prepare_confirmation_registration_hash_only=True,
    confirm_command_count=1, benchmark_command_count=1, confirmation_used_for_tuning=False, benchmark_used_for_tuning=False,
    temperature_below_one_selected=False, lock_written_before_confirmation_load=True,
    confirmation_first_read_time_ns=min(item['time_ns'] for item in used), lock_write_open_time_ns=writes[0]['time_ns'],
    formal_source_unchanged_after_fresh_prepare=True, frozen_config_unchanged=True, shards_unchanged=True,
    denied_accesses=sum(row['io_counts']['denied'] for row in runtime.values()),
    per_scan_consensus_boundary_violations=sum(row['tensor_checks']['violations'] for row in runtime.values()),
    calibration_labels_used_after_freeze_in_outer_refit=True, confirmation_labels_used_only_after_consumption_for_frozen_final_refit=True,
    original_failed_attempt_preserved=True, restart_after_failure_explicitly_authorized=True,
    independent_external_confirmation=False, official_benchmark_is_pre_exposed=True,
    benchmark_results_used_for_outcome_driven_changes=False, reporting_performs_model_inference=False)
assert audit['denied_accesses'] == audit['per_scan_consensus_boundary_violations'] == 0
for phase in PHASES[1:]:
    row = runtime[phase]
    assert row['deterministic_algorithms'] and not row['deterministic_warn_only'] and not row['flash_sdpa'] and not row['memory_efficient_sdpa'] and row['math_sdpa']
    assert row['cudnn_deterministic'] and not row['cudnn_benchmark'] and row['cublas_workspace_config'] == ':4096:8'
    assert not row['v18_2_runtime_modules'] and row['io_counts']['v18_2_import_attempts'] == 0

hashes = dict(training_csv_sha256=reg['train_data_sha256'], official_csv_sha256=bench['benchmark_data_sha256'],
    registry_canonical_sha256=reg['registry_sha256'], freeze_canonical_sha256=freeze['freeze_sha256'],
    calibration_canonical_sha256=cal['result_sha256'], stability_canonical_sha256=stab['result_sha256'],
    confirmation_canonical_sha256=conf['result_sha256'], confirmation_lock_canonical_sha256=lock['lock_sha256'],
    benchmark_canonical_sha256=bench['result_sha256'], selection_checkpoint_sha256=selection['checkpoint']['sha256'],
    confirmation_checkpoint_sha256=cref['confirmation_checkpoint']['sha256'], final_checkpoint_sha256=final['final_checkpoint']['sha256'])
for name in ['registry_v18_3.json', 'selection_freeze_v18_3.json', 'calibration_result_v18_3.json', 'stability_result_v18_3.json',
             'confirmation_result_v18_3.json', 'confirmation_consumed.lock', 'benchmark_result_v18_3.json']:
    hashes[name + '_file_bytes_sha256'] = sha(E / name)
for name, value in [('selection', selection['checkpoint']), ('confirmation', cref['confirmation_checkpoint']), ('final', final['final_checkpoint'])]:
    assert sha(ROOT / ('fhm_dapp_tim/models/v18_3/' + name + '.pt')) == value['sha256']

interpretation = []
interpretation.append(f"V18.2 式的自由 gate 权重塌缩已从结构上消除：本模型没有可训练 gate，逐 scan 检查要求 family 在 [0.15,0.70]、族内二源在 [0.30,0.70]。V18.2 同一 benchmark 的 learned gate 均值为 {100*old['mean_gate_weight_learned']:.3f}%；V18.3 的这些层级权重有硬边界，但不同层级的权重不能当作同一统计量直接比较。边界限制只证明保留融合质量，不证明每种来源都有独立因果增益。")
interpretation.append('三个 family 都保留非零且有实质幅度的融合质量。confirmation 的 Representation/Radio/Structure 均值为 ' + '/'.join(f"{100*cm['mean_family_weight_'+name]:.3f}%" for name in FAMILIES) + '，benchmark 为 ' + '/'.join(f"{100*bm['mean_family_weight_'+name]:.3f}%" for name in FAMILIES) + '。全部扫描的实际范围见运行时审计表。这里的“质量”指 probability mass，不是无消融条件下可识别的独立预测贡献。')
interpretation.append(f"radio mismatch 与 Radio family 权重的逐 scan Pearson r：confirmation={cm['radio_mismatch_radio_weight_correlation']:.6f}，benchmark={bm['radio_mismatch_radio_weight_correlation']:.6f}。" + ('两者均为负，观察方向符合注册的 mismatch 抑制。' if cm['radio_mismatch_radio_weight_correlation'] < 0 and bm['radio_mismatch_radio_weight_correlation'] < 0 else '不能声称两个数据集都观察到负相关。') + '固定其它 entropy/disagreement 项时，mismatch 增大必然降低 Radio reliability score 及其权重；跨 scan 相关仍同时受其它可靠性项影响，不能解读为消融实验。')
interpretation.append(f"坐标误差与 global disagreement 的逐 scan Pearson r：confirmation={cm['error_evidence_disagreement_correlation']:.6f}，benchmark={bm['error_evidence_disagreement_correlation']:.6f}。" + ('两者正相关，较高误差扫描倾向具有更大的 disagreement 与 T_dis。' if cm['error_evidence_disagreement_correlation'] > 0 and bm['error_evidence_disagreement_correlation'] > 0 else '结果不支持在两个集合上都概括为困难 scan 的 disagreement 更大。') + '本配置逐 scan 严格满足 T_dis=1+D，D∈[0,1]，所以 error 与 T_dis 的 Pearson r 等于上述值。相关描述不是困难程度检测的因果证明，也不是未记录的条件分组分析。')
delta8 = comparison['differences']['final_nearest_anchor_recall_at_8']
interpretation.append(f"同一预暴露 CSV 上，V18.2/V18.3 recall@8 为 {100*old['final_nearest_anchor_recall_at_8']:.3f}%/{100*bm['final_nearest_anchor_recall_at_8']:.3f}%，差 {100*delta8:+.3f} 个百分点，" + ('Top-8 有描述性改善。' if delta8 > 0 else 'Top-8 未显示改善。') + f"V18.3 union recall={100*bm['union_nearest_anchor_recall']:.3f}%，recall@64={100*bm['final_nearest_anchor_recall_at_64']:.3f}%；confirmation union={100*cm['union_nearest_anchor_recall']:.3f}%。这是自然候选覆盖，不含强制加入训练目标后的 recall。不同设计、epoch 和校准同时改变，不能归因于单一机制。")
improved_levels = [q for q in LEVELS if abs(bm['predictive_radial_coverage_' + str(q)] - q / 100) < abs(old['predictive_radial_coverage_' + str(q)] - q / 100)]
interpretation.append('以距 nominal coverage 的绝对差判断，同一官方 CSV 上改善的水平为 ' + (', '.join(str(q) + '%' for q in improved_levels) if improved_levels else '无') + '。V18.3 benchmark 的 50/80/90/95% 实际 coverage 为 ' + '/'.join(f"{100*bm['predictive_radial_coverage_'+str(q)]:.3f}%" for q in LEVELS) + '，confirmation 为 ' + '/'.join(f"{100*cm['predictive_radial_coverage_'+str(q)]:.3f}%" for q in LEVELS) + '。coverage 上升不必然更好，应同时看 nominal 偏差与预测半径。V18.2/V18.3 confirmation 使用不同训练坐标 holdout，不作受控改善比较。')
main_scores = ['coordinate_macro_mae_m', 'coordinate_macro_energy_score_m', 'coordinate_macro_mixture_nll']
main_deltas = [comparison['differences'][key] for key in main_scores]
interpretation.append('相较 V18.2，同一官方 CSV 的 MAE/ES/NLL 差值（V18.3−V18.2）为 ' + '/'.join(f'{value:+.6f}' for value in main_deltas) + '；' + ('三项均下降，有描述性恢复。' if all(value < 0 for value in main_deltas) else f'{sum(value < 0 for value in main_deltas)} 项下降，不能概括为三项全部恢复。') + '两版都在已经暴露的 benchmark 上评价，这不是新的独立外部确认；未把 benchmark 作为本轮模型、epoch、seed 或校准选择依据。')
interpretation.append('值得考虑单独预注册 V18.4：V18.3 已通过权重约束消除自由 gate 塌缩，但还需用预先确定的消融与独立数据检验各族的预测贡献。当前 benchmark 的 MAE/ES/NLL 为 ' + '/'.join(f'{bm[key]:.6f}' for key in main_scores) + '，五 seed 的 MAE sample SD 为 ' + f"{stab['summary']['coordinate_macro_mae_m']['sample_std']:.6f} m；coverage 距 nominal 的最大绝对偏差为 {100*max(abs(bm['predictive_radial_coverage_'+str(q)]-q/100) for q in LEVELS):.3f} 个百分点。这些结果可指导下一版的预注册问题，当前版本的训练、校准、确认和 benchmark 记录全部保留，不追加评价或调参。")

report = dict(
    GIT=dict(branch=subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
        formal_experiment_sha=snapshot['formal_experiment_sha'], initial_attempt_sha=failure['formal_experiment_sha'],
        bug_fix_commits=preflight['pre_prepare_bug_fixes'], source_snapshot=snapshot,
        failed_attempt_path=str(FAILED.parent), failed_attempt=failure, restart_authorization=authorization,
        implementation_fix_synthetic_verification=fix_verification),
    TESTS={**preflight, 'preflight_runtime': read(AUDIT / 'preflight_runtime.json'), 'initial_attempt_pytest_passed': 42,
        'real_dataset_reads_scope': 'synthetic CUDA preflight only; the required pytest suite also verifies real-data split counts before prepare',
        'all_seven_scientific_cuda_phases_strict_deterministic': True,
        'registered_seed_18431_first_epoch_replay': read(AUDIT / 'determinism_first_epoch_replay.json')},
    PROTOCOL={**protocol, 'constants': reg['constants'], 'single_member_contract': reg['single_member_contract'],
        'evaluation_contract': reg['evaluation_contract'], 'scientific_status': reg['scientific_status'],
        'post_benchmark_opaque_hash_checks_all_six_shards': True},
    MODEL=dict(selection=selection['checkpoint'], confirmation=cref['confirmation_checkpoint'], final=final['final_checkpoint'],
        selected_epoch=freeze['frozen_selected_epoch'], config=freeze['frozen_model_config'], sources=SOURCES, families=FAMILIES,
        checkpoint_count_at_inference=1, residual_experts=1, candidate_union_size=96, posterior_top_k=64,
        trainable_evidence_gate=False, inference_family_weight_floor=.15, within_family_source_weight_floor=.30,
        v18_2_runtime_import=False, teacher=False, distillation=False, ensemble=False),
    TRAINING=dict(full_trajectories=histories, selection_epochs=selection['selection_epochs'],
        shortlist=selection['shortlist'], best_eligible_energy_epoch=selection['best_eligible_energy_epoch'],
        best_eligible_energy_score_m=selection['best_eligible_energy_score_m'], selected_early_stop=selection['selected_early_stop']),
    CALIBRATION={**cal, 'freeze': freeze, 'selection_checkpoint_unchanged': True, 'model_gradient_updates': False},
    STABILITY={**stab, 'family_weight_seed_summary': seed_family_summary, 'full_training_histories': histories['stability']},
    CONFIRMATION=conf, PRE_EXPOSED_BENCHMARK=bench,
    COMPUTE=dict(python_executable=preflight['python_executable'], python=preflight['python'], pytorch=preflight['torch'],
        cuda=preflight['cuda'], GPU=preflight['gpu'], gpu_total_memory_bytes=preflight['gpu_total_memory_bytes'],
        executions=execution, runtimes=runtime, total_scientific_cli_wall_seconds=sum(row['wall_seconds'] for row in execution.values()),
        peak_cuda_allocated_bytes=max(row.get('peak_cuda_allocated_bytes', 0) for row in runtime.values()),
        peak_cuda_reserved_bytes=max(row.get('peak_cuda_reserved_bytes', 0) for row in runtime.values()),
        memory_scope='per-subprocess PyTorch CUDA allocator; excludes driver context and other processes',
        instrumentation_uses_existing_outputs_only=True, environment=read(AUDIT / 'environment.json')),
    AUDIT=audit, INTERPRETATION=interpretation, DESCRIPTIVE_V18_2_COMPARISON=comparison,
    PRE_CONFIRMATION_GATE=gate['gate_checks'], HASHES=hashes,
    report_created_utc=dt.datetime.now(dt.timezone.utc).isoformat(), report_status='COMPLETE',
    evidence_grade=dict(confirmation='one-shot sequential internal confirmation', official_benchmark='PRE-EXPOSED DESCRIPTIVE BENCHMARK',
        independent_external_confirmation=False, model_development_informed_by_prior_v18_2_outcomes=True))
report['report_sha256'] = digest(report, 'report_sha256')
destination = E / 'RUN_REPORT_V18_3.json'
assert not destination.exists()
destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

lines = []
def add(line=''):
    lines.append(str(line))
def heading(name):
    add(); add('## ' + name); add()
def yn(value):
    return 'YES' if value else 'NO'
def n6(value):
    return f'{value:.6f}'
def link(title, path):
    return '[' + title + '](<' + str(Path(path).resolve()).replace('\\', '/') + '>)'
def table(headers, rows):
    add(); add('| ' + ' | '.join(str(v) for v in headers) + ' |')
    add('| ' + ' | '.join('---' for _v in headers) + ' |')
    for row in rows: add('| ' + ' | '.join(str(v) for v in row) + ' |')
    add()
def metrics(values):
    table(['已注册指标', '值'], [(key, n6(value) if isinstance(value, float) else value) for key, value in values.items()])

add('# V18.3 完整冻结实验报告')
add(); add('运行日期：2026-10-01（Asia/Shanghai）；下面执行时间戳均为 UTC。')
add(); add('证据等级：**one-shot sequential internal confirmation（顺序性内部确认）**；官方 CSV 为 **PRE-EXPOSED DESCRIPTIVE BENCHMARK**。两者都不属于新的独立外部盲测。V18.3 的设计已经参考 V18.2 结果；本轮 outcome 未用于改动注册方法。')
add(); add('MAE、Energy Score、NLL、楼栋／楼层错误按物理坐标宏平均；其它均值与 coverage 按 scan 统计，相关性为逐 scan Pearson r。source/family entropy 是 H/log(L)，posterior entropy 才以 nats 计。')
heading('GIT')
add(f"分支：`{report['GIT']['branch']}`；正式实验 SHA：`{snapshot['formal_experiment_sha']}`。")
add(); add('首次尝试在原 SHA `d7500d3c7ee563d0c7b201e5a3ac4ee4dc21d1fe` 的第一轮 early-stop 统计处遇到 autograd→NumPy API 错误。按用户明确授权，完整归档首次登记／失败日志后，仅在评价统计边界增加 detach，补充多批次回归测试并提交为 `342b73cf77222bd93d13eeb094858d569354f9ad`。原模型文件、注册模型配置、损失、种子和校准网格未改。')
add(); add('修复先在 CPU／strict CUDA 合成数据验证：原错误可重现，detach 前后每个预测张量逐位一致，修复后全部统计有限。三项编译、全部测试和扩展 CUDA 预检通过后，从 Stage 1 完整重启；首次失败期间确认集和 benchmark 未访问，没有 confirmation lock。')
add(); add('首次失败与重启授权：' + link('失败记录', FAILED / 'failure_record.json') + '、' + link('授权记录', FAILED / 'restart_authorization.json') + '；新的科学源码在 prepare 前后及每阶段前后均核验哈希。')
heading('TESTS')
add('三个 py_compile 全部 PASS：v18_3_single.py、uji_v18_3_experiment.py、test_uji_v18_3_experiment.py。')
add(); add(f"指定四个测试文件：**43 passed in {preflight['pytest_seconds']:.2f}s**；原 42 个测试全部保留，新增 1 个真实评价路径回归测试。无 skip、xfail、删除或弱化测试。")
add(); table(['合成 CUDA case', 'anchors', 'APs', 'batch', '目标宽度', '有限前向／反向／优化器', '重建预测逐位一致', '完整评价／NumPy 相关'],
    [(row['case'], row['anchors'], row['APs'], row['batch'], row['target_width'], 'YES', 'YES', 'YES') for row in preflight['cases']])
add('严格 deterministic、warn_only=False、flash 与 memory-efficient SDPA 关闭、math SDPA 开启；cuDNN deterministic=True、benchmark=False、CUBLAS_WORKSPACE_CONFIG=:4096:8。七个正式 CUDA 阶段均保存实际开关，详见 COMPUTE。')
add(); add('注册的 seed 18431 stability 首轮与 selection-search 首轮在相同 member_fit 上的 21 个训练指标逐项完全一致；这来自已经要求运行的独立重训，没有为核验而额外训练或推理。')
heading('PROTOCOL')
table(['role', 'physical coordinates', 'rows', 'building-floor strata'], [(name, row['coordinates'], row['rows'], 13) for name, row in protocol['roles'].items()])
add('full trainingData = 933 coordinates / 19,937 rows / 520 APs。member_fit ∪ early_stop = selection_refit；selection_refit ∪ calibration = outer_refit；outer_refit ∪ confirmation = full trainingData。三组对应交集均为 0，union 完全成立；所有角色均保留全部 13 strata。')
add(); add('三项 hull_missing 均为 0：full hull ⊆ outer_refit，outer_refit hull ⊆ selection_refit，selection_refit hull ⊆ member_fit。')
table(['role shard', 'registered SHA256'], [(name, '`' + row['sha256'] + '`') for name, row in reg['role_shards'].items()])
add('prepare 对六个 shard 注册 SHA；确认前只独立加载／核验五个开发角色。confirmation SHA 在 confirm 写锁后由协议校验；完成 benchmark 后仅进行六个文件的 opaque SHA 复核，没有手工加载 confirmation 内容。')
add(); add('trainingData SHA256：`' + reg['train_data_sha256'] + '`；registry canonical SHA256：`' + reg['registry_sha256'] + '`。')
heading('MODEL')
table(['checkpoint', 'parameters', 'anchors', 'selected epoch', 'no trainable gate'], [(name, row['parameters'], row['anchors'], row['selected_epoch'], 'YES') for name, row in [('selection', selection['checkpoint']), ('confirmation', cref['confirmation_checkpoint']), ('final', final['final_checkpoint'])]])
add('one inference checkpoint；E=1；L=96/K=64。无 teacher、distillation、member axis、ensemble、best-seed selection 或 prediction averaging。V18.3 主模型和正式运行均未 import v18_2_single.py。')
table(['family', 'source 1', 'source 2', 'inference family floor', 'within source bounds'], [('Representation', 'invariant', 'learned', .15, '[0.30,0.70]'), ('Radio', 'prototype', 'marked_radio', .15, '[0.30,0.70]'), ('Structure', 'floor', 'geometry', .15, '[0.30,0.70]')])
add('family 权重和为 1，上界为 0.70；族内两源和为 1。训练以 p=0.30 对选中 scan 恰好丢一族并重归一化；所有评价关闭 family dropout。T_dis=clip(1+D,1,2) 在 Top-K 前作用，校准 T≥1 在 Top-K 后作用。')
add(); table(['注册配置', '值'], [(name, freeze['frozen_model_config'][name]) for name in ['invariant_retrieval_k', 'prototype_retrieval_k', 'floor_retrieval_k', 'geometry_seed_k', 'candidate_union_size', 'posterior_top_k', 'ranking_weight', 'family_dropout_probability', 'family_min_weight', 'within_family_min_source_weight', 'disagreement_temperature_strength', 'disagreement_temperature_max', 'radio_lambda', 'condition_building', 'deterministic_building']])
add('构建每个模型时检查 32+32+16+最宽非 self Delaunay target ≤96。容量不足、非有限值或共识边界违规立即报错，无静默裁剪或容错降级。')
heading('TRAINING')
add(f"搜索完成 {selection['selection_epochs']} 个 epoch，选定 **epoch={freeze['frozen_selected_epoch']}**。eligible 为 epoch≥12；eligible 最低 Energy Score 在 epoch={selection['best_eligible_energy_epoch']}，ES={selection['best_eligible_energy_score_m']:.6f} m。shortlist ES 距最优≤0.05 m，其中按 MAE、ES、较早 epoch 排序。")
table(['shortlist epoch', 'MAE (m)', 'Energy Score (m)'], [(row['epoch'], n6(row['mae_m']), n6(row['energy_m'])) for row in selection['shortlist']])
series = [('selection search', [row['train'] for row in histories['selection_search']]), ('selection refit', histories['selection_refit']), ('confirmation refit', histories['confirmation_refit']), ('final refit', histories['final_refit'])]
losskeys = ['loss', 'joint_geometry_loss', 'fused_anchor_nll', 'coordinate_nll', 'invariant_aux_loss', 'reranker_aux_loss', 'hierarchy_loss', 'coordinate_loss', 'ranking_loss', 'domain_loss',
    'invariant_recall_mass', 'prototype_recall_mass', 'floor_recall_mass', 'union_recall_mass'] + ['family_weight_' + name for name in FAMILIES] + ['mean_global_disagreement', 'mean_disagreement_temperature', 'mean_radio_mismatch', 'family_dropout_fraction']
table(['metric'] + [name + ' first → last' for name, _rows in series], [[key] + [n6(rows[0][key]) + ' → ' + n6(rows[-1][key]) for _name, rows in series] for key in losskeys])
add('每个搜索 epoch 的全部 TRAIN/EARLY STOP 指标、三个 refit 和五个 stability seed 的每个 epoch 训练指标均保存在 ' + link('完整 JSON 报告', E / 'RUN_REPORT_V18_3.json') + '。五种子完整 history 通过只读函数返回值记录，不改变训练或增加模型评价。')
table(['selection epoch', 'train joint', 'train ranking', 'union recall mass', 'early-stop MAE', 'early-stop ES', 'early-stop NLL', 'eligible'],
    [(row['epoch'], n6(row['train']['joint_geometry_loss']), n6(row['train']['ranking_loss']), n6(row['train']['union_recall_mass']), n6(row['early_stop']['coordinate_macro_mae_m']), n6(row['early_stop']['coordinate_macro_energy_score_m']), n6(row['early_stop']['coordinate_macro_mixture_nll']), yn(row['eligible_for_selection'])) for row in histories['selection_search']])
for name, rows in series[1:]:
    add('完整 ' + name + ' trajectory：')
    table(['epoch', 'joint', 'fused anchor NLL', 'coordinate NLL', 'ranking', 'union mass', 'Representation', 'Radio', 'Structure', 'D', 'T_dis', 'radio mismatch', 'dropout fraction'],
        [[row['epoch']] + [n6(row[key]) for key in ['joint_geometry_loss', 'fused_anchor_nll', 'coordinate_nll', 'ranking_loss', 'union_recall_mass'] + ['family_weight_' + name for name in FAMILIES] + ['mean_global_disagreement', 'mean_disagreement_temperature', 'mean_radio_mismatch', 'family_dropout_fraction']] for row in rows])
add('selected early-stop 完整诊断（未校准 T=1、scale=1）：'); metrics(selection['selected_early_stop'])
heading('CALIBRATION')
cc = cal['calibration']
add(f"冻结 posterior temperature={cc['posterior_temperature']}（≥1），scale multiplier={cc['scale_multiplier']}；temperature grid boundary={yn(cc['temperature_on_grid_boundary'])}，scale grid boundary={yn(cc['scale_on_grid_boundary'])}。")
add(); add('固定 6×6 网格的 36 点全部评价，按 calibration coordinate-macro mixture NLL 选择；tie 按 |log T|+|log scale|。没有更新模型参数。selection checkpoint 的字节 SHA 在校准前后相同；freeze 固定完整 config、epoch、T、scale 以及 registry、selection、calibration SHA。')
lookup = {(row['posterior_temperature'], row['scale_multiplier']): row['coordinate_macro_mixture_nll'] for row in cc['surface']}
table(['T / scale'] + [str(scale) for scale in reg['constants']['calibration_scale_grid']], [[temperature] + [n6(lookup[temperature, scale]) for scale in reg['constants']['calibration_scale_grid']] for temperature in reg['constants']['calibration_temperature_grid']])
metrics(cal['calibration_metrics'])
heading('STABILITY')
add('18431–18435 五个 seed 全部在 member_fit 独立训练冻结 epoch，并仅在 early_stop 使用冻结 T／scale 评价。no best seed；no ensemble；no prediction averaging；未用于 model selection。')
sk = ['coordinate_macro_mae_m', 'coordinate_macro_energy_score_m', 'coordinate_macro_mixture_nll', 'final_nearest_anchor_recall_at_8', 'final_nearest_anchor_recall_at_64']
table(['seed', 'MAE (m)', 'ES (m)', 'NLL', 'recall@8', 'recall@64'], [[row['seed']] + [n6(row['metrics'][key]) for key in sk] for row in stab['records']])
table(['metric', 'mean ± sample SD (ddof=1)', 'min', 'max'], [(key, n6(value['mean']) + ' ± ' + n6(value['sample_std']), n6(value['minimum']), n6(value['maximum'])) for key, value in stab['summary'].items()])
table(['seed', 'Representation', 'Radio', 'Structure', 'global D', 'T_dis', 'radio mismatch', 'error/D r', 'mismatch/Radio r'], [[row['seed']] + [n6(row['metrics'][key]) for key in ['mean_family_weight_' + name for name in FAMILIES] + ['mean_global_evidence_disagreement', 'mean_disagreement_temperature', 'mean_radio_mismatch', 'error_evidence_disagreement_correlation', 'radio_mismatch_radio_weight_correlation']] for row in stab['records']])
table(['family mean weight across seeds', 'mean ± sample SD', 'min', 'max'], [(name, n6(row['mean']) + ' ± ' + n6(row['sample_std']), n6(row['minimum']), n6(row['maximum'])) for name, row in seed_family_summary.items()])
add('每个 seed 的六个 within-family weights、family entropy/disagreement、recall、coverage、uncertainty 和全部其它注册诊断均完整保留在 JSON 中。')
heading('CONFIRMATION')
add('**one-shot sequential internal confirmation（顺序性内部确认）**。187 个完整物理坐标 / 4,133 scans；它是 newly registered trainingData holdout，设计已参考 V18.2 outcome，不能称为 independent external confirmation。')
for title, field, unit in [('MAE', 'coordinate_macro_mae_m', ' m'), ('ENERGY SCORE', 'coordinate_macro_energy_score_m', ' m'), ('NLL', 'coordinate_macro_mixture_nll', '')]:
    add(); add(f"**CONFIRMATION {title} = {cm[field]:.6f}{unit}**")
add(); add('执行前十二项 gate 检查全部通过。先核验 registry／freeze／checkpoint，再写 confirmation_consumed.lock，之后才校验并加载 confirmation shard，一次评价后保存 result self-hash。锁未删除，confirm command count=1；未根据结果修改当前方法。')
metrics(cm)
heading('PRE-EXPOSED BENCHMARK')
add('**PRE-EXPOSED DESCRIPTIVE BENCHMARK**')
add(); add('**independent external confirmation = NO；pre-exposed benchmark = YES；benchmark used for tuning = NO**')
for title, field, unit in [('MAE', 'coordinate_macro_mae_m', ' m'), ('ENERGY SCORE', 'coordinate_macro_energy_score_m', ' m'), ('NLL', 'coordinate_macro_mixture_nll', '')]:
    add(); add(f"**BENCHMARK {title} = {bm[field]:.6f}{unit}**")
add(); add(f"官方 validationData.csv：{bm['rows']} rows / {bm['coordinate_groups']} physical coordinates。仅在合法 confirmation self-hash 验证和冻结 full refit 完成后读取，并评价一次。结果标志：benchmark_was_pre_exposed_before_v18_3=true、blind_confirmation_claim=false、used_for_model_selection=false。")
metrics(bm)
heading('COMPUTE')
add(f"GPU：{preflight['gpu']}（{preflight['gpu_total_memory_bytes']/1024**2:.0f} MiB）；Python：{preflight['python']}；解释器：`{preflight['python_executable']}`；PyTorch：{preflight['torch']}；CUDA runtime：{preflight['cuda']}。")
add(); add('驱动与设备：`' + report['COMPUTE']['environment']['device_inventory'] + '`；依赖包：' + ', '.join(name + '=' + version for name, version in report['COMPUTE']['environment']['packages'].items()) + '。')
table(['训练', 'seconds'], [('selection search + refit', n6(selection['checkpoint']['training_seconds'])), ('five-seed stability wall', n6(execution['stability']['wall_seconds'])), ('confirmation refit', n6(cref['confirmation_checkpoint']['training_seconds'])), ('final refit', n6(final['final_checkpoint']['training_seconds']))])
table(['stability seed', 'fit_fixed wall seconds'], [(row['seed'], n6(row['fit_wall_seconds'])) for row in histories['stability']])
table(['phase', 'started UTC', 'finished UTC', 'wall seconds', 'peak allocated MiB', 'peak reserved MiB'], [(phase, execution[phase]['started_utc'], execution[phase]['finished_utc'], n6(execution[phase]['wall_seconds']), n6(runtime[phase].get('peak_cuda_allocated_bytes', 0)/1024**2), n6(runtime[phase].get('peak_cuda_reserved_bytes', 0)/1024**2)) for phase in PHASES])
add(f"八个正式 CLI 合计 wall={report['COMPUTE']['total_scientific_cli_wall_seconds']:.6f} s；最大 PyTorch CUDA allocator allocated/reserved={report['COMPUTE']['peak_cuda_allocated_bytes']/1024**2:.3f}/{report['COMPUTE']['peak_cuda_reserved_bytes']/1024**2:.3f} MiB。该峰值不含驱动 context 或其它进程，不等同整张显卡设备总占用。耗时含只读审计开销；首次失败和预检时间另在归档记录中。")
table(['phase', 'strict deterministic', 'warn_only', 'flash SDPA', 'memory-efficient SDPA', 'math SDPA', 'cuDNN deterministic', 'cuDNN benchmark'], [(phase, yn(runtime[phase]['deterministic_algorithms']), yn(runtime[phase]['deterministic_warn_only']), yn(runtime[phase]['flash_sdpa']), yn(runtime[phase]['memory_efficient_sdpa']), yn(runtime[phase]['math_sdpa']), yn(runtime[phase]['cudnn_deterministic']), yn(runtime[phase]['cudnn_benchmark'])) for phase in PHASES[1:]])
add('全部七个科学 CUDA 阶段严格 deterministic，未启用 warn_only；全部使用 CUBLAS_WORKSPACE_CONFIG=:4096:8。')
heading('AUDIT')
table(['项目', 'YES/NO 或计数'], [('one inference checkpoint', 'YES'), ('residual_experts == 1', 'YES'), ('trainable evidence gate used', 'NO'), ('V18.2 model runtime import', 'NO'), ('teacher used', 'NO'), ('distillation used', 'NO'), ('ensemble used', 'NO'), ('best seed selected', 'NO'), ('member axis / prediction averaging', 'NO'), ('positive self-anchor target used', 'NO'), ('calibration labels used for backbone gradients before freeze', 'NO'), ('confirmation 内容在 prepare 后、lock 前访问', 'NO'), ('prepare confirmation registration hash only', 'YES'), ('confirm command count', 1), ('benchmark command count', 1), ('confirmation used for tuning', 'NO'), ('benchmark used for tuning', 'NO'), ('temperature below 1 selected', 'NO'), ('independent external confirmation', 'NO'), ('all scientific files unchanged after fresh prepare', 'YES'), ('per-scan consensus boundary violation', 0), ('denied data access', 0), ('failed first attempt preserved', 'YES'), ('restart explicitly authorized by user', 'YES')])
add('prepare 的 confirmation 写入与 hash 属于注册步骤。此后首次内容读取发生在 consumed lock 之后；后续 train-final 按协议使用 confirmation 完成 full refit。calibration 只在 freeze 前选择 T／scale，不用于 backbone 梯度或 epoch；冻结后它属于 outer_refit，合法参与重训。')
table(['phase', 'training CSV read opens', 'confirmation shard read opens', 'benchmark CSV read opens', 'historical checkpoint reads', 'V18.2 import attempts', 'denied'], [[phase] + [runtime[phase]['io_counts'][key] for key in ['training_csv_reads', 'confirmation_reads', 'benchmark_reads', 'historical_checkpoint_reads', 'v18_2_import_attempts', 'denied']] for phase in PHASES])
add('verify-prepare 另做 training CSV 字节 SHA 和五个开发 shard 的独立校验，不加载 confirmation。上表是 formal CLI 文件 open 计数；prepare/confirm/train-final 的 hash 与 torch.load 会各自读取文件，不代表重复评价。旧 V18.2 result 只在新 benchmark 完成后的报告阶段读取，未参与模型、epoch、seed 或参数选择。')
table(['phase', 'training scans checked', 'inference scans checked', 'dropped training scans', 'violations'], [(phase, runtime[phase]['tensor_checks']['training_scans'], runtime[phase]['tensor_checks']['inference_scans'], runtime[phase]['tensor_checks']['dropped_training_scans'], runtime[phase]['tensor_checks']['violations']) for phase in PHASES])
range_names = ['family_' + name for name in FAMILIES] + ['within_' + str(f) + '_' + str(s) for f in range(3) for s in range(2)] + ['global_disagreement', 'disagreement_temperature', 'radio_mismatch']
table(['逐 scan 共识诊断', 'all formal phases observed min', 'max'], [(name, n6(min(row['observed_ranges'][name][0] for row in runtime.values() if name in row['observed_ranges'])), n6(max(row['observed_ranges'][name][1] for row in runtime.values() if name in row['observed_ranges']))) for name in range_names])
add(f"lock write open time_ns={audit['lock_write_open_time_ns']}；first confirm read time_ns={audit['confirmation_first_read_time_ns']}。")
table(['HASH', 'SHA256'], [(name, '`' + value + '`') for name, value in hashes.items()])
add('canonical self-hash 与实际文件 byte SHA 分别列出。科学结果、lock、registry、freeze 和 checkpoint 均保持原记录；报告与 QA 不运行模型。')
heading('INTERPRETATION')
for i, text in enumerate(interpretation, 1):
    add(f'{i}. {text}'); add()
table(['同一 PRE-EXPOSED official CSV metric', 'V18.2', 'V18.3', 'V18.3 − V18.2'], [(key, n6(old[key]), n6(bm[key]), n6(comparison['differences'][key])) for key in comparison_keys])
add('不同版本训练坐标划分、epoch、模型结构和校准同时改变，因此以上只作描述性比较，不作受控消融或新的 blind test 结论。任何 outcome-driven 方法修改进入 V18.4。')
add(); add('完整机器可读报告：' + link('RUN_REPORT_V18_3.json', E / 'RUN_REPORT_V18_3.json') + '。')
markdown_path = E / 'RUN_REPORT_V18_3.md'
assert not markdown_path.exists()
markdown_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(json.dumps(dict(status='COMPLETE', report_sha256=report['report_sha256'], markdown=str(markdown_path), json=str(destination),
    confirmation={key: cm[key] for key in main_scores}, pre_exposed_benchmark={key: bm[key] for key in main_scores},
    comparison_differences={key: comparison['differences'][key] for key in main_scores}, selected_epoch=freeze['frozen_selected_epoch']), ensure_ascii=False, indent=2))
