"""Report the actual blocked/failed attempt without inventing unreached results."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import statistics

AUDIT = Path(__file__).resolve().parent
E = AUDIT.parent
ROOT = AUDIT.parents[3]
PHASES = ['prepare', 'train-selection', 'calibrate-freeze', 'stability', 'train-confirmation']
FAMILIES = ['representation', 'radio', 'structure']
def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def digest(value): return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
preflight = read(AUDIT / 'preflight.json')
protocol = read(AUDIT / 'prepare_verification.json')
selection = read(AUDIT / 'selection_verification.json')
reg = read(E / 'registry_v18_3.json')
cal = read(E / 'calibration_result_v18_3.json')
freeze = read(E / 'selection_freeze_v18_3.json')
stab = read(E / 'stability_result_v18_3.json')
history = read(AUDIT / 'stability_full_training_histories.json')
capacity = read(AUDIT / 'capacity_diagnosis.json')
metadata_capacity = read(AUDIT / 'full_support_metadata_capacity.json')
exact_capacity = read(AUDIT / 'full_capacity_verification.json') if (AUDIT / 'full_capacity_verification.json').exists() else None
execution = {phase: read(AUDIT / (phase + '_execution.json')) for phase in PHASES}
runtime = {phase: read(AUDIT / (phase + '_runtime.json')) for phase in PHASES}
assert all(execution[phase]['exit_code'] == 0 for phase in PHASES[:4])
assert execution['train-confirmation']['exit_code'] != 0
assert not (E / 'confirmation_consumed.lock').exists() and not (E / 'confirmation_result_v18_3.json').exists()
assert not (E / 'benchmark_result_v18_3.json').exists()
assert all(row['io_counts']['benchmark_reads'] == 0 for row in runtime.values())
status = 'FAILED CLOSED: FULL SUPPORT EXCEEDS FROZEN CAPACITY' if exact_capacity and not exact_capacity['sufficient_full_capacity'] else 'BLOCKED: CAPACITY IMPLEMENTATION BUG; AUTHORIZATION PENDING'
interpretation = [
    '开发阶段没有可训练 gate；逐 scan family/source 权重边界检查通过。可确认结构防塌缩约束成立，不能据此声称新的 confirmation 或 benchmark 已验证改善。',
    '校准集三个 family 平均质量为 ' + '/'.join(f"{100*cal['calibration_metrics']['mean_family_weight_'+n]:.3f}%" for n in FAMILIES) + '，五 seed 的族权重范围完整保留。非零权重不等于已识别的独立因果贡献。',
    f"校准集 radio-mismatch/Radio-weight Pearson r={cal['calibration_metrics']['radio_mismatch_radio_weight_correlation']:.6f}；注册惩罚在固定其它可靠性项时单调抑制 Radio 权重。没有确认集或 benchmark 结果支持进一步判断。",
    f"校准集 error/disagreement Pearson r={cal['calibration_metrics']['error_evidence_disagreement_correlation']:.6f}，没有支持困难 scan 明显对应更高 disagreement 的证据。T_dis 按固定公式增加，但其对真实误差的诊断作用需区分数学定义与经验相关。",
    f"stability 的开发集 Top-8 mean={100*stab['summary']['final_nearest_anchor_recall_at_8']['mean']:.3f}%，Top-64 mean=100%。没有当前官方 benchmark，不能与 V18.2 判定同一 CSV 上的改善。",
    '校准集 50/80/90/95% nominal coverage 的实际值为 ' + '/'.join(f"{100*cal['calibration_metrics']['predictive_radial_coverage_'+str(q)]:.3f}%" for q in [50,80,90,95]) + '，区间偏宽。confirmation 与 benchmark 未运行，不能判断跨数据集 coverage 改善。',
    '本版 benchmark 未运行，不能判断相较 V18.2 是否恢复；没有加载旧 benchmark 结果进行当前比较，也未追加任何模型评价。',
    '进一步工作应先在正式 prepare 前核验全部训练支持的真实有效目标宽度，修正零权重 padding 计数错误。若 full support 实际需要超过 L=96，则变更 L 或 source quotas 属于新的方法合同，需另行注册，不能裁剪正目标或绕过容量检查。'
]
report = dict(report_status=status, completed_stages=PHASES[:4], failed_stage='train-confirmation',
    stages_not_run=['confirm','train-final','benchmark'],
    GIT=dict(branch='v18-3-structured-consensus', formal_experiment_sha=preflight['formal_experiment_sha'],
        bug_fix_commits=preflight['pre_prepare_bug_fixes'], scientific_source_snapshot=read(AUDIT / 'source_snapshot.json'),
        original_attempt_preserved='failed_attempts/d7500d3-20261001T081726Z'),
    TESTS=preflight, PROTOCOL=protocol,
    MODEL=dict(selection=selection['checkpoint'], confirmation=None, final=None, selected_epoch=17,
        config=freeze['frozen_model_config'], residual_experts=1, candidate_union_size=96, posterior_top_k=64,
        trainable_gate=False, teacher=False, distillation=False, ensemble=False, v18_2_runtime_import=False),
    TRAINING=dict(selection_search=selection['selection_history'], selection_refit=selection['selection_refit_history'],
        stability_full_histories=history, confirmation_refit=None, final_refit=None, shortlist=selection['shortlist']),
    CALIBRATION=dict(result=cal, freeze=freeze), STABILITY=stab,
    CONFIRMATION=dict(status='NOT RUN', command_count=0, metrics=None, lock_created=False,
        planned_grade='one-shot sequential internal confirmation', independent_external_confirmation=False),
    PRE_EXPOSED_BENCHMARK=dict(status='NOT RUN', command_count=0, metrics=None,
        planned_grade='PRE-EXPOSED DESCRIPTIVE BENCHMARK', used_for_model_selection=False, blind_confirmation_claim=False),
    COMPUTE=dict(environment=read(AUDIT / 'environment.json'), executions=execution, runtimes=runtime,
        scientific_cli_wall_seconds=sum(row['wall_seconds'] for row in execution.values())),
    AUDIT=dict(confirm_command_count=0, benchmark_command_count=0, confirmation_consumed=False,
        confirmation_shard_loaded_after_prepare=False, benchmark_loaded=False, model_config_changed=False,
        positive_self_target_used=False, positive_targets_pruned=False, primary_quotas_reduced=False,
        capacity_guard_bypassed=False, confirmation_used_for_tuning=False, benchmark_used_for_tuning=False,
        calibration_labels_used_for_backbone_gradients_before_freeze=False,
        formal_stages_use_historical_checkpoint=False, exact_coordinate_capacity_preflight_authorized=bool(exact_capacity),
        capacity_source_patch_applied=False, only_explicitly_authorized_detach_fix_applied=True),
    INTERPRETATION=interpretation, CAPACITY_DIAGNOSIS=capacity, FULL_SUPPORT_METADATA_PROBE=metadata_capacity,
    EXACT_FULL_CAPACITY=exact_capacity, CAPACITY_FIX_PROPOSAL=read(AUDIT / 'capacity_fix_proposal_verification.json'),
    report_created_utc=dt.datetime.now(dt.timezone.utc).isoformat())
report['report_sha256'] = digest(report)
(E / 'RUN_STATUS_V18_3.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

lines = ['# V18.3 执行记录：未完成', '', '**' + status + '**', '',
    'Stage 1–4 已完成；Stage 5 构建时中止；Stage 6–8 未运行。未生成或删除 confirmation lock，未消费确认集，未运行当前 benchmark。', '']
def add(line=''): lines.append(str(line))
def heading(name): add(); add('## ' + name); add()
def table(headers, rows):
    add(); add('| ' + ' | '.join(headers) + ' |'); add('| ' + ' | '.join('---' for _v in headers) + ' |')
    for row in rows: add('| ' + ' | '.join(str(v) for v in row) + ' |')
    add()
def metrics(values): table(['metric','value'], [(key, ('YES' if v else 'NO') if isinstance(v,bool) else f'{v:.6f}' if isinstance(v,float) else v) for key,v in values.items()])
heading('GIT'); add('分支 `v18-3-structured-consensus`；正式 SHA `' + preflight['formal_experiment_sha'] + '`。')
add('首次 API 失败按用户授权修复并提交为 `342b73c`，旧登记和日志完整归档；只在评价统计边界 detach，未改模型／损失／seed／epoch／校准网格。第二次容量失败后未改科学源码。')
heading('TESTS'); add('3 个 py_compile PASS；指定四个测试文件 **43 passed in 5.92s**。无删除、skip、xfail 或弱化测试。小模型与 520 AP CUDA 合成预检均完成有限前向／反向／优化器、精确 checkpoint 重建和完整评价。')
add('另外在合成 CPU/CUDA 中验证尚未应用的计数补丁：模型状态、正目标与预测值逐项相同；16 个有效邻居通过，17 个仍拒绝。')
heading('PROTOCOL')
table(['role','coordinates','rows','strata'], [(name,v['coordinates'],v['rows'],13) for name,v in protocol['roles'].items()])
add('933 坐标 / 19,937 rows / 520 APs；三项交集均为 0，三个 union 恒等式成立，所有角色保留 13 strata，三项 hull_missing 均为 0。')
table(['shard','registered SHA256'], [(name,v['sha256']) for name,v in reg['role_shards'].items()])
add('五个开发 shard 已独立加载并核验 SHA；confirmation 仅在 prepare 写入并固定注册 hash，之后从未手工加载，未作确认评价。')
heading('MODEL')
table(['checkpoint','parameters','anchors','epoch'], [('selection',selection['checkpoint']['parameters'],595,17),('confirmation','未生成','未构建',17),('final','未生成','未构建',17)])
add('单 checkpoint 合同，E=1，L=96/K=64；无 trainable gate、teacher、distillation、ensemble、member axis 或多 seed prediction averaging；正式运行未 import v18_2_single.py。family floor=0.15，族内 source ∈[0.30,0.70]，训练 dropout p=0.30，T_dis∈[1,2]。')
heading('TRAINING')
add('搜索 23 轮，eligible best ES epoch=15，shortlist 为 15/17/20，按 MAE 选 epoch 17；selection refit 完成 17 轮，五个 stability seed 均完成 17 轮。confirmation 尚未开始梯度训练。')
series = [('selection search',[row['train'] for row in selection['selection_history']]),('selection refit',selection['selection_refit_history'])]
keys = ['joint_geometry_loss','fused_anchor_nll','coordinate_nll','ranking_loss','invariant_recall_mass','prototype_recall_mass','floor_recall_mass','union_recall_mass'] + ['family_weight_'+n for n in FAMILIES] + ['mean_global_disagreement','mean_disagreement_temperature','mean_radio_mismatch','family_dropout_fraction']
table(['metric'] + [name+' first→last' for name,_r in series], [[key]+[f'{r[0][key]:.6f} → {r[-1][key]:.6f}' for _n,r in series] for key in keys])
add('完整搜索／refit／五 seed 每轮 trajectory 均保存在 RUN_STATUS_V18_3.json 与 run_audit 对应记录；没有额外重跑训练。')
heading('CALIBRATION')
cc=cal['calibration']; add(f"36 点全部评价；T={cc['posterior_temperature']}≥1，scale={cc['scale_multiplier']}；temperature boundary={cc['temperature_on_grid_boundary']}，scale boundary={cc['scale_on_grid_boundary']}。epoch/config/T/scale 和关联 SHA 已冻结；未更新 backbone。"); metrics(cal['calibration_metrics'])
heading('STABILITY')
sk=['coordinate_macro_mae_m','coordinate_macro_energy_score_m','coordinate_macro_mixture_nll','final_nearest_anchor_recall_at_8','final_nearest_anchor_recall_at_64']
table(['seed','MAE','ES','NLL','recall@8','recall@64'],[[r['seed']]+[f"{r['metrics'][k]:.6f}" for k in sk] for r in stab['records']])
table(['metric','mean±sample SD','min','max'],[(k,f"{v['mean']:.6f} ± {v['sample_std']:.6f}",f"{v['minimum']:.6f}",f"{v['maximum']:.6f}") for k,v in stab['summary'].items()])
table(['seed','Representation','Radio','Structure'],[[r['seed']]+[f"{r['metrics']['mean_family_weight_'+n]:.6f}" for n in FAMILIES] for r in stab['records']])
add('无 best seed selection，无 ensemble，无 prediction averaging；全部诊断与训练 history 保留。')
heading('CONFIRMATION'); add('**CONFIRMATION MAE / ENERGY SCORE / NLL = 未运行**。confirm count=0，lock 不存在，不能声称取得确认结果。计划证据等级为 one-shot sequential internal confirmation；independent external confirmation=NO。')
heading('PRE-EXPOSED BENCHMARK'); add('**BENCHMARK MAE / ENERGY SCORE / NLL = 未运行**。benchmark count=0，没有结果文件。计划等级仅为 PRE-EXPOSED DESCRIPTIVE BENCHMARK；independent external confirmation=NO；benchmark used for tuning=NO。')
heading('COMPUTE'); add(f"{preflight['gpu']} / Python {preflight['python']} / PyTorch {preflight['torch']} / CUDA {preflight['cuda']}；指定解释器 `{preflight['python_executable']}`。")
table(['phase','wall seconds','exit code','peak allocated MiB','peak reserved MiB'],[(p,f"{execution[p]['wall_seconds']:.6f}",execution[p]['exit_code'],f"{runtime[p].get('peak_cuda_allocated_bytes',0)/1024**2:.3f}",f"{runtime[p].get('peak_cuda_reserved_bytes',0)/1024**2:.3f}") for p in PHASES])
add('显存为各子进程 PyTorch allocator 峰值，不含驱动或其它进程。三个已完成 CUDA 阶段及失败阶段的实际开关均保留：strict deterministic=True，warn_only=False，flash/memory-efficient SDPA=False，math SDPA=True，cuDNN benchmark=False，deterministic=True。')
heading('AUDIT')
metrics(report['AUDIT'])
table(['role','padded width','max positive nonself targets','primary quotas','padded check capacity','actual positive capacity','L'],[(r['role'],r['padded_target_width'],r['maximum_positive_nonself_target_count'],80,r['guard_required_capacity'],r['positive_targets_required_capacity'],96) for r in capacity['roles']])
add('outer_refit 的 guard 把零权重 self/padding 位计入 17，误算 80+17=97；真实正目标数为 16，恰好 80+16=96。只更正计数的补丁已准备和验证但未应用。')
if exact_capacity:
    add('按用户新授权，仅解析坐标列、重建 loader 的 normalize→denormalize float32 流程和 full-refit 的原顺序／anchor 行均值；outer 坐标与原 shard 逐位相同，未加载 confirmation shard 或任何 RSS 列，未运行模型。')
    table(['static role','rows','anchors','padded width','positive width','required positive capacity','L'],[(r['role'],r['rows'],r['anchors'],r['padded_target_width'],r['maximum_positive_nonself_targets'],r['required_positive_target_capacity'],96) for r in exact_capacity['roles']])
else:
    add('登记坐标元数据的初步 full-support probe 得到 18 个有效邻居、所需 98；这不是实际按行数均值生成的 full checkpoint geometry，尚待用户授权做坐标列静态预检，不能当成已运行 train-final 的结果。')
heading('INTERPRETATION')
for i,text in enumerate(interpretation,1): add(f'{i}. {text}'); add()
add('本报告明确区分已完成、失败和未运行阶段。没有填补未取得的 confirmation／benchmark 指标，没有绕过容量 guard，也没有以新参数悄悄重跑。')
(E / 'RUN_STATUS_V18_3.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(dict(status=status, report_sha256=report['report_sha256'], completed=PHASES[:4], failed='train-confirmation', confirm_count=0, benchmark_count=0),ensure_ascii=False,indent=2))
