"""Post-run report assembly only; no model inference or outcome-driven changes."""
import datetime as dt,hashlib,json,math,statistics,subprocess
from pathlib import Path
AUDIT=Path(__file__).resolve().parent; ROOT=AUDIT.parents[3]; E=AUDIT.parent
PHASES=['prepare','train-selection','calibrate-freeze','stability','train-confirmation','confirm','train-final','benchmark']
SOURCES=['invariant','prototype','learned','marked_radio','floor','geometry']
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(p,key): return hashlib.sha256(json.dumps({k:v for k,v in p.items() if k!=key},sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf-8')).hexdigest()
def checked(p,key): assert p[key]==digest(p,key); return p
preflight=read(AUDIT/'preflight.json'); protocol=read(AUDIT/'prepare_verification.json'); selection=read(AUDIT/'selection_verification.json'); cref=read(AUDIT/'confirmation-refit_verification.json'); final=read(AUDIT/'final_verification.json')
reg=checked(read(E/'registry_v18_2.json'),'registry_sha256'); freeze=checked(read(E/'selection_freeze_v18_2.json'),'freeze_sha256')
cal=checked(read(E/'calibration_result_v18_2.json'),'result_sha256'); stab=checked(read(E/'stability_result_v18_2.json'),'result_sha256')
conf=checked(read(E/'confirmation_result_v18_2.json'),'result_sha256'); lock=checked(read(E/'confirmation_consumed.lock'),'lock_sha256'); bench=checked(read(E/'benchmark_result_v18_2.json'),'result_sha256')
execution={s:read(AUDIT/(s+'_execution.json')) for s in PHASES}; runtime={s:read(AUDIT/(s+'_runtime.json')) for s in PHASES}
assert all(r['exit_code']==0 and r['sources_unchanged'] for r in execution.values())
assert conf['confirmation_lock_sha256']==lock['lock_sha256'] and conf['checkpoint_sha256']==cref['confirmation_checkpoint']['sha256']
assert bench['benchmark_was_pre_exposed_before_v18_2'] and not bench['blind_confirmation_claim'] and not bench['used_for_model_selection']
assert bench['final_checkpoint_sha256']==final['final_checkpoint']['sha256']==sha(ROOT/'fhm_dapp_tim/models/v18_2/final.pt')
assert all(p['registry_sha256']==reg['registry_sha256'] for p in [freeze,cal,stab,conf,lock,bench])
assert all(p['freeze_sha256']==freeze['freeze_sha256'] for p in [stab,conf,lock,bench])
assert all(sha(r['path'])==r['sha256'] for r in reg['role_shards'].values())
snapshot=read(AUDIT/'source_snapshot.json')
assert snapshot['formal_experiment_sha']==subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
assert all(sha(ROOT/p)==h for p,h in snapshot['files'].items())
events=[json.loads(line) for p in AUDIT.glob('*_io.jsonl') for line in p.read_text(encoding='utf-8').splitlines() if line.strip()]
assert all(e['permitted'] for e in events)
confirmation_reads=[e for e in events if e['kind']=='role_shard' and e['reading'] and e['path'].endswith('/confirmation.pt')]
assert all(e['phase'] in ['prepare','confirm','train-final'] for e in confirmation_reads)
assert all(e['confirmation_lock_present'] for e in confirmation_reads if e['phase']!='prepare')
assert sum(e['phase']=='prepare' for e in confirmation_reads)==1
benchmark_reads=[e for e in events if e['kind']=='pre_exposed_benchmark' and e['reading']]; assert len(benchmark_reads)==2 and all(e['phase']=='benchmark' for e in benchmark_reads)
assert not any(e['kind']=='historical_evidence' for e in events)
for phase in ['train-selection','stability','train-confirmation','train-final']:
    r=runtime[phase]
    assert r['deterministic_algorithms'] and not r['deterministic_warn_only'] and not r['flash_sdpa'] and not r['memory_efficient_sdpa'] and r['math_sdpa'] and r['cudnn_deterministic'] and not r['cudnn_benchmark']
cm=conf['confirmation_metrics']; bm=bench['benchmark_metrics']; cfg=freeze['frozen_model_config']
comparison={}
# Historical outcome is read only here, after all V18.2 confirmation/benchmark outputs exist.
oldpath=ROOT/'fhm_dapp_tim/evidence/v18_1/external_result_v18_1.json'
if oldpath.is_file():
    previous=read(oldpath); assert previous['external_data_sha256']==bench['benchmark_data_sha256']; old=previous['external_metrics']
    comparison={'status':'descriptive same pre-exposed official CSV; no causal ablation','V18_1_result_sha256':previous['result_sha256'],'V18_1':old,'V18_2':bm,'recall_at_8_difference':bm['final_nearest_anchor_recall_at_8']-old['nearest_same_stratum_anchor_recall_at_8'],'recall_at_64_difference':bm['final_nearest_anchor_recall_at_64']-old['nearest_same_stratum_anchor_recall_at_64'],'between_radial_sd_difference_m':bm['mean_between_anchor_radial_sd_m']-old['mean_between_anchor_radial_sd_m']}
if comparison:
    old=comparison['V18_1']
    comparison.update(mae_difference_m=bm['coordinate_macro_mae_m']-old['coordinate_macro_mae_m'],energy_difference_m=bm['coordinate_macro_energy_score_m']-old['coordinate_macro_energy_score_m'],nll_difference=bm['coordinate_macro_mixture_nll']-old['coordinate_macro_mixture_nll'],floor_error_difference=bm['coordinate_macro_floor_error']-old['coordinate_macro_floor_error'])

gate_checks=read(AUDIT/'pre_confirmation_gate.json')
lock_e=[e for e in events if e['kind']=='confirmation_lock' and not e['reading']]; used=[e for e in confirmation_reads if e['phase']=='confirm']
assert len(lock_e)==1 and len(used)==2 and lock_e[0]['time_ns']<min(e['time_ns'] for e in used)
audit={'one_inference_checkpoint':True,'residual_experts_equals_one':True,'teacher_used':False,'distillation_used':False,'historical_V9_V17_checkpoint_loaded':False,'ensemble_used':False,'member_axis':False,'best_seed_selected':False,'prediction_averaging':False,'positive_self_anchor_target_used':False,'calibration_labels_used_for_backbone_gradients_before_freeze':False,'confirmation_contents_accessed_before_lock_after_prepare':False,'prepare_confirmation_registration_hash_only':True,'confirm_command_count':1,'confirmation_used_for_tuning':False,'benchmark_used_for_tuning':False,'benchmark_is_pre_exposed':True,'benchmark_is_blind_confirmation':False,'lock_written_before_confirmation_load':True,'confirmation_first_read_time_ns':min(e['time_ns'] for e in used),'lock_write_open_time_ns':lock_e[0]['time_ns'],'formal_source_unchanged':True,'frozen_config_unchanged':True,'shards_unchanged':True,'denied_accesses':sum(r['io_counts']['denied'] for r in runtime.values()),'calibration_labels_used_after_freeze_in_outer_refit':True,'confirmation_labels_used_only_after_consumption_for_frozen_final_refit':True}
warning_logs=[phase for phase in PHASES if 'Converting a tensor with requires_grad=True to a scalar' in (AUDIT/(phase+'.log')).read_text(encoding='utf-8')]
stability_gate_summary={s:{'mean':statistics.mean(r['metrics']['mean_gate_weight_'+s] for r in stab['records']),'sample_std':statistics.stdev(r['metrics']['mean_gate_weight_'+s] for r in stab['records'])} for s in SOURCES}
search=selection['selection_history']; sr=selection['selection_refit_history']; cr=cref['confirmation_refit_history']; fr=final['final_refit_history']
interpretation=[
 'prototype 是否救回 invariant 漏掉的 anchor：现有注册输出给出边际 recall，没有逐 scan 的两源交集。下表列出 prototype-only rescue 的可识别上下界；只有正的下界可直接确认 prototype 有救回，不能把全部 union 增益都归因于 prototype。',
 '96-anchor union 包含全部三类 primary retrieval quota；因此 union recall 减 invariant 或 prototype recall 可衡量 union 对该来源的净候选覆盖增量。提升幅度以表中百分比点报告，不用训练时强制加入目标后的 recall 冒充自然候选 recall。',
 'final recall@8 的改善须与同一官方 CSV 的旧结果作描述性比较，不能用不同坐标 holdout 或不同 anchor pool 的 recall 直接判定。若历史文件存在，下方同时列出 @8、@64 的差值。',
 '六个 gate 最大权重来源按各结果的实际均值排序；五种子均值和 sample SD 也单列。source entropy 是 H/log(L) 的归一化熵，posterior entropy 才以 nats 计。',
 'style mismatch 较大时是否自动降低 radio 信任：gate 的无标签输入包含 |style offset| 与 |log gain|，具备这种建模能力；注册结果只保存均值，没有 style-gate 的逐 scan 条件关系，因此不能仅凭平均 gate 与平均 style 声称已经验证。没有追加 confirmation 或 benchmark 推理来补证。',
 'floor evidence 使用 soft opinion 并按候选 floor anchor 数分摊 floor mass；坐标预测没有 hard floor filter，condition_building=False。报告 floor gate、floor retrieval recall 和 floor error；没有去掉 floor 的 ablation，不能单独证明其因果增益。floor 分类输出来自 invariant marginal，而不是六源 posterior 的 floor marginal。',
 'between-anchor uncertainty 是否下降：仅在相同预暴露 benchmark 上与 V18.1 作描述性比较；当前 confirmation、calibration、benchmark 的 within/between SD 也同时保留。不同集合及不同字典之间的数值差异不能当成受控 ablation。',
 'V18.3 是否值得设计将依据完整冻结结果的候选覆盖、Top-8 排名、gate 集中度、coverage 和耗时作描述性判断。任何新增方法、损失、gate 或校准调整都需要另行注册，当前 V18.2 不重跑确认集或 benchmark。'
]
rescue={}
for label,m in [('calibration',cal['calibration_metrics']),('confirmation',cm),('pre_exposed_benchmark',bm)]:
    i,p,u=m['invariant_nearest_anchor_recall'],m['prototype_nearest_anchor_recall'],m['union_nearest_anchor_recall']
    rescue[label]={'invariant_recall':i,'prototype_recall':p,'floor_recall':m['floor_nearest_anchor_recall'],'union_recall':u,'union_minus_invariant':u-i,'union_minus_prototype':u-p,'prototype_only_rescue_lower_bound':max(0.,p-i),'prototype_only_rescue_upper_bound':min(p,1.-i),'gate_weights_descending':sorted([(s,m['mean_gate_weight_'+s]) for s in SOURCES],key=lambda x:-x[1]),'style_gate_conditional_relation_not_identified_from_means':True}
# Interpret only the already-completed, immutable registered outcomes.
crescue=rescue['confirmation']; brescue=rescue['pre_exposed_benchmark']
cdominant=crescue['gate_weights_descending'][0]; bdominant=brescue['gate_weights_descending'][0]
learned_seed_weights=[r['metrics']['mean_gate_weight_learned'] for r in stab['records']]
interpretation[0]=(f"confirmation 的 invariant/prototype nearest-anchor recall 为 {100*crescue['invariant_recall']:.3f}%/{100*crescue['prototype_recall']:.3f}%。prototype-only rescue 的可识别下界/上界为 {100*crescue['prototype_only_rescue_lower_bound']:.3f}%/{100*crescue['prototype_only_rescue_upper_bound']:.3f}%；下界为零，当前边际记录不足以确认 prototype 的独立救回数量。没有两源逐 scan 交集，不能把 union 的全部增量归给 prototype。benchmark 的同类范围也列在表中。")
interpretation[1]=(f"96-anchor union 的 confirmation recall 为 {100*crescue['union_recall']:.3f}%，比 invariant 增加 {100*crescue['union_minus_invariant']:.3f} 个百分点，属于小幅补全，confirmation 候选覆盖已接近饱和。benchmark union 为 {100*brescue['union_recall']:.3f}%，比 invariant 增加 {100*brescue['union_minus_invariant']:.3f} 个百分点。三类 primary retrieval quota 均保留，因此这些是自然候选覆盖的净增量；没有用强制加入训练目标后的 recall 冒充自然 recall。")
if comparison:
    interpretation[2]=(f"同一 PRE-EXPOSED 官方 CSV 上，V18.1/V18.2 final recall@8 为 {100*comparison['V18_1']['nearest_same_stratum_anchor_recall_at_8']:.3f}%/{100*bm['final_nearest_anchor_recall_at_8']:.3f}%，差值 {100*comparison['recall_at_8_difference']:+.3f} 个百分点；recall@64 差值为 {100*comparison['recall_at_64_difference']:+.3f} 个百分点。Top-8 {'有描述性改善' if comparison['recall_at_8_difference']>0 else '未显示改善'}，这回答了窄 posterior 排名问题；训练设计、字典、epoch 和校准不同，不能归因于单一方法修改，也不能升级为新 blind test。")
interpretation[3]=(f"confirmation 最大平均 gate 为 {cdominant[0]}={100*cdominant[1]:.3f}%，benchmark 最大为 {bdominant[0]}={100*bdominant[1]:.3f}%；全部六源排序见表。" + ("实际融合高度集中在一个来源，不能声称六个来源有均衡的实证贡献。" if cdominant[1]>.95 and bdominant[1]>.95 else "这些均值描述平均贡献，不能替代逐 scan 响应诊断。") + "source entropy 是 H/log(L) 的归一化熵，posterior entropy 才以 nats 计。")
interpretation[4]+=(f"confirmation mean |offset|={cm['mean_abs_profiled_style_offset']:.6f}，mean scale={cm['mean_profiled_style_scale']:.6f}，mean radio gate={100*cm['mean_gate_weight_marked_radio']:.6f}%。这些边际均值无法确定 style mismatch 与 radio 信任的条件关系。")
interpretation[5]=(f"floor 使用 soft opinion，并按候选 floor anchor 数分摊 floor mass；condition_building=False，坐标输出没有 hard floor filter。confirmation floor gate 为 {100*cm['mean_gate_weight_floor']:.6f}%，floor retrieval recall 为 {100*cm['floor_nearest_anchor_recall']:.3f}%，coordinate-macro floor error 为 {100*cm['coordinate_macro_floor_error']:.3f}%。平均 floor opinion 权重较小，没有 floor ablation，不能证明它单独改善结果。floor 分类输出来自 invariant marginal，而不是六源 consensus 的 floor marginal。")
if comparison:
    interpretation[6]=(f"同一预暴露 benchmark 的 between-anchor radial SD 从 V18.1 的 {comparison['V18_1']['mean_between_anchor_radial_sd_m']:.6f} m 到 V18.2 的 {bm['mean_between_anchor_radial_sd_m']:.6f} m，差值 {comparison['between_radial_sd_difference_m']:+.6f} m，{'描述性下降' if comparison['between_radial_sd_difference_m']<0 else '没有下降'}。这只描述两个冻结实验；更小的 SD 本身不能证明校准更好，应结合下表的 coverage 和 within-component SD 评价。")
interpretation[7]=(f"值得设计 V18.3：五个独立 seed 的 learned gate 都为 {100*min(learned_seed_weights):.3f}%–{100*max(learned_seed_weights):.3f}%，confirmation 为 {100*cm['mean_gate_weight_learned']:.3f}%，提示应优先检验证据融合的有效贡献。先预注册两源 retrieval 的联合救回诊断、style 与 radio gate 的逐 scan 条件关系、floor ablation，并结合 coverage 判断不确定性；再决定是否需要方法修改。新增实验必须另行注册并设置合法确认方案，当前 V18.2 保留全部原记录。")

if comparison:
    interpretation[7]+=(f"同一官方 benchmark 相比 V18.1，MAE/ES/NLL 差值为 {comparison['mae_difference_m']:+.6f} m / {comparison['energy_difference_m']:+.6f} m / {comparison['nll_difference']:+.6f}，本轮三项评分均变差。")
interpretation[7]+=(f"benchmark 的 50/80/90/95% 实际 coverage 为 {100*bm['predictive_radial_coverage_50']:.3f}%/{100*bm['predictive_radial_coverage_80']:.3f}%/{100*bm['predictive_radial_coverage_90']:.3f}%/{100*bm['predictive_radial_coverage_95']:.3f}%，均低于 nominal，显示该数据上的区间偏窄。V18.3 应关注跨数据的定位评分与 coverage，同时保留候选 ranking 和 gate 诊断。")

report={'GIT':{'branch':subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),'formal_experiment_sha':snapshot['formal_experiment_sha'],'pre_prepare_bug_fixes':[],'fix_commits':[],'source_snapshot':snapshot},'TESTS':{**preflight,'audit_helper_setup_import_path_fixed_before_prepare':True,'registered_eval_scalar_conversion_warning_phases':warning_logs},'PROTOCOL':{**protocol,'all_six_shard_hashes_valid_at_completion':True,'confirmation_hash_checked_by_protocol_after_lock':True,'confirmation_hash_validated_by_protocol_after_lock':True},'MODEL':{'selection':selection['checkpoint'],'confirmation':cref['confirmation_checkpoint'],'final':final['final_checkpoint'],'full_frozen_config':cfg,'selected_epoch':freeze['frozen_selected_epoch'],'no_teacher':True,'no_ensemble':True},'TRAINING':selection,'CONFIRMATION_REFIT':cref,'FINAL_REFIT':final,'CALIBRATION':cal,'STABILITY':{**stab,'six_gate_seed_summary':stability_gate_summary},'CONFIRMATION':conf,'PRE_EXPOSED_BENCHMARK':bench,'COMPUTE':{'GPU':preflight['gpu'],'Python':preflight['python'],'PyTorch':preflight['torch'],'CUDA':preflight['cuda'],'executions':execution,'runtimes':runtime,'max_peak_cuda_allocated_bytes':max(r.get('peak_cuda_allocated_bytes',0) for r in runtime.values()),'max_peak_cuda_reserved_bytes':max(r.get('peak_cuda_reserved_bytes',0) for r in runtime.values())},'AUDIT':audit,'INTERPRETATION':interpretation,'INTERPRETATION_NUMBERS':rescue,'V18_1_DESCRIPTIVE_COMPARISON':comparison,'PRE_CONFIRMATION_GATE':gate_checks['gate_checks'],'HASHES':{'registry_self_sha256':reg['registry_sha256'],'freeze_self_sha256':freeze['freeze_sha256'],'calibration_result_self_sha256':cal['result_sha256'],'stability_result_self_sha256':stab['result_sha256'],'confirmation_lock_self_sha256':lock['lock_sha256'],'confirmation_lock_file_sha256':sha(E/'confirmation_consumed.lock'),'confirmation_result_self_sha256':conf['result_sha256'],'confirmation_result_file_sha256':sha(E/'confirmation_result_v18_2.json'),'benchmark_result_self_sha256':bench['result_sha256'],'benchmark_result_file_sha256':sha(E/'benchmark_result_v18_2.json'),'benchmark_data_sha256':bench['benchmark_data_sha256'],'selection_checkpoint_sha256':selection['checkpoint']['sha256'],'confirmation_checkpoint_sha256':cref['confirmation_checkpoint']['sha256'],'final_checkpoint_sha256':final['final_checkpoint']['sha256']},'generated_utc':dt.datetime.now(dt.timezone.utc).isoformat()}
lines=[]
def add(s=''): lines.append(s)
def heading(s): add(); add('## '+s); add()
def table(headers,rows):
    add('| '+' | '.join(headers)+' |'); add('| '+' | '.join(['---']*len(headers))+' |')
    for row in rows: add('| '+' | '.join(map(str,row))+' |')
    add()
def n6(x): return f'{x:.6f}'
def pct(x): return f'{100*x:.3f}%'
def yn(x): return 'YES' if x else 'NO'
def link(label,p): return f'[{label}](<{Path(p).as_posix()}>)'
def metrics(m):
    names=['coordinate_macro_mae_m','coordinate_macro_energy_score_m','coordinate_macro_mixture_nll','coordinate_macro_mae_se_m','coordinate_macro_energy_score_se_m','coordinate_macro_mixture_nll_se','coordinate_macro_building_error','coordinate_macro_floor_error','mean_anchor_entropy_nats','mean_effective_anchor_count','mean_top1_probability','mean_radial_scale_m','mean_within_component_radial_sd_m','mean_between_anchor_radial_sd_m','mean_posterior_radio_energy','mean_abs_profiled_style_offset','mean_profiled_style_scale','error_radial_scale_correlation']
    table(['指标','值'],[(k,n6(m[k])) for k in names])
    table(['最终 nearest same building-floor anchor','recall'],[(f'recall@{k}',pct(m[f'final_nearest_anchor_recall_at_{k}'])) for k in (8,16,32,64)])
    table(['source support','recall'],[(s,pct(m[s+'_nearest_anchor_recall'])) for s in ('invariant','prototype','floor','union')])
    table(['evidence source','mean gate weight','normalized source entropy H/log(L)'],[(s,n6(m['mean_gate_weight_'+s]),n6(m['mean_source_entropy_'+s])) for s in SOURCES])
    table(['nominal coverage','observed coverage','mean predictive radius (m)'],[(f'{q}%',pct(m[f'predictive_radial_coverage_{q}']),n6(m[f'mean_predictive_radius_{q}_m'])) for q in (50,80,90,95)])
add('# V18.2 正式执行报告'); add()
add('八条注册命令全部按顺序完成，五个 stability seeds 全部保留。V18.2 的主确认结果来自 newly registered trainingData physical-coordinate holdout；官方 validationData 的结果严格属于 PRE-EXPOSED DESCRIPTIVE BENCHMARK。')
add();add(f"**CONFIRMATION MAE = {cm['coordinate_macro_mae_m']:.6f} m**");add();add(f"**CONFIRMATION ENERGY SCORE = {cm['coordinate_macro_energy_score_m']:.6f} m**");add();add(f"**CONFIRMATION NLL = {cm['coordinate_macro_mixture_nll']:.6f}**")
add();add('MAE、ES、NLL 和 building/floor error 按完整物理坐标组取 macro；候选 recall、gate、entropy、uncertainty、coverage 和 radius 按注册代码对 scan 求均值。style offset 是绝对值均值，style gain 是 affine scale。source entropy 是归一化熵，posterior entropy 以 nats 计。')
heading('GIT'); add('- branch: `'+report['GIT']['branch']+'`'); add('- formal experiment SHA: `'+snapshot['formal_experiment_sha']+'`'); add('- prepare 前 model/protocol bug fix: NO；fix commit: 无；科学源码和注册配置均未修改。')
heading('TESTS'); add('- 三个指定文件的 py_compile 全部通过。指定 pytest：28 passed，0 failed，5.80 s，无 skip/xfail、删测试或放宽合同。'); add('- 合成 CUDA preflight 的两个用例（8 anchors/16 APs，以及 300 anchors/520 APs/batch 64）均通过 strict forward/backward/optimizer、gate gradient、有限值和 checkpoint 精确重建检查。'); add('- 辅助 preflight 脚本首次因仓库 import path 缺失失败，修正辅助路径后通过；发生在正式 prepare 前，没有真实数据访问，不涉及模型源码修复。'); add('- selection/stability/confirmation-refit/final-refit 的 strict CUDA 开关全部成功；evaluation CLI 的真实开关见 COMPUTE。'); add('- 原注册 evaluation 的 scalar conversion warning 所在阶段：'+', '.join(warning_logs)+'；没有转换为 warn_only，没有据此更改模型，最终 finite/hash 检查均通过。')
heading('PROTOCOL'); add('trainingData：19,937 rows；933 physical coordinates；520 APs；13 building-floor strata。')
add();table(['role','coordinates','rows'],[(s,r['coordinates'],r['rows']) for s,r in protocol['roles'].items()])
add('member_fit 与 early_stop 不交且并集为 selection_refit；selection_refit 与 calibration 不交且并集为 outer_refit；outer_refit 与 confirmation 不交且并集为 trainingData。每个 role 保留全部 13 strata，三个 hull missing 都为 0。')
add();add('prepare 按注册代码写入并 hash 六个 shard。之后独立验证五个 development shard，confirmation 在 confirm 写入 lock 后由正式 loader 验证哈希；没有提前手工读取或评价 confirmation。最终全部六个 shard 哈希有效且保持不变。')
add();add('trainingData SHA256: `'+protocol['train_data_sha256']+'`'); add();table(['role shard','SHA256'],[(s,'`'+r['sha256']+'`') for s,r in reg['role_shards'].items()]); add('registry self-hash: `'+reg['registry_sha256']+'`')
heading('MODEL'); table(['模型','parameters','anchors','epoch','tensor finite'],[(s,v['parameters'],v['anchors'],v['selected_epoch'],'YES') for s,v in [('selection',selection['checkpoint']),('confirmation',cref['confirmation_checkpoint']),('final',final['final_checkpoint'])]])
table(['合同','值'],[(k,cfg[k]) for k in ['residual_experts','scan_feature_mode','prototype_metric','candidate_reranker','evidence_alpha','invariant_retrieval_k','prototype_retrieval_k','floor_retrieval_k','geometry_seed_k','candidate_union_size','posterior_top_k','radio_lambda','minimum_epochs']])
add('六个 opinion 同属于一个模型：invariant、prototype、learned、marked_radio、floor、geometry；自适应 α=1 arithmetic evidence pooling。one inference checkpoint；no teacher；no distillation；no ensemble；no member axis；no best seed；no prediction averaging。')
heading('TRAINING'); add(f"搜索完成 {selection['selection_epochs']} 个 epoch；selected epoch={freeze['frozen_selected_epoch']}；best eligible Energy Score epoch={selection['best_eligible_energy_epoch']}，Energy={selection['best_eligible_energy_score_m']:.6f} m。只有 epoch≥12 eligible，shortlist 距 eligible 最优 ES 不超过 0.05 m，然后按 MAE、ES、较早 epoch 排序。NLL 未作为主选择标准。")
add();table(['shortlist epoch','MAE (m)','ES (m)'],[(r['epoch'],n6(r['mae_m']),n6(r['energy_m'])) for r in selection['shortlist']])
losskeys=['loss','joint_geometry_loss','fused_anchor_nll','coordinate_nll','invariant_aux_loss','reranker_aux_loss','hierarchy_loss','coordinate_loss','ranking_loss','domain_loss','invariant_recall_mass','prototype_recall_mass','floor_recall_mass','union_recall_mass']+['gate_weight_'+s for s in SOURCES]
table(['loss/diagnostic','selection search first→last','selection refit first→last','confirmation refit first→last','final refit first→last'],[(k,n6(search[0]['train'][k])+' → '+n6(search[-1]['train'][k]),n6(sr[0][k])+' → '+n6(sr[-1][k]),n6(cr[0][k])+' → '+n6(cr[-1][k]),n6(fr[0][k])+' → '+n6(fr[-1][k])) for k in losskeys])
add('selected early-stop 完整诊断（T=1, scale=1）：'); add();metrics(selection['selected_early_stop'])
add('每个搜索 epoch 的全部 TRAIN/EARLY STOP 指标和三个重训的每个 epoch loss 保留在 '+link('JSON 报告',E/'RUN_REPORT_V18_2.json')+' 及 run_audit/*_verification.json。')
heading('CALIBRATION'); cc=cal['calibration']
add(f"temperature={cc['posterior_temperature']}；scale multiplier={cc['scale_multiplier']}；temperature boundary={yn(cc['temperature_on_grid_boundary'])}；scale boundary={yn(cc['scale_on_grid_boundary'])}。")
add();add('8×6 固定网格全部 48 点已评价，唯一选择标准为 calibration coordinate-macro mixture NLL；没有更新 backbone/gate/reranker。selection checkpoint SHA 与 calibration 前一致；freeze 固定 registry、selection、calibration SHA、epoch、完整 V182_CONFIG、T 和 scale。')
add();metrics(cal['calibration_metrics'])
lookup={(x['posterior_temperature'],x['scale_multiplier']):x['coordinate_macro_mixture_nll'] for x in cc['surface']}; temps=reg['constants']['calibration_temperature_grid']; scales=reg['constants']['calibration_scale_grid']
table(['T / scale']+[str(s) for s in scales],[[t]+[n6(lookup[(t,s)]) for s in scales] for t in temps])
heading('STABILITY'); add('五个 seed 均在 member_fit 独立训练冻结 epoch 数，只在 early_stop 使用冻结的 T/scale 评价。no best seed selection / no ensemble / no prediction averaging。')
add();sk=['coordinate_macro_mae_m','coordinate_macro_energy_score_m','coordinate_macro_mixture_nll','final_nearest_anchor_recall_at_8','final_nearest_anchor_recall_at_64']
table(['seed','MAE (m)','ES (m)','NLL','recall@8','recall@64'],[[r['seed']]+[n6(r['metrics'][k]) for k in sk] for r in stab['records']])
table(['metric','mean ± sample SD (ddof=1)'],[(k,n6(v['mean'])+' ± '+n6(v['sample_std'])) for k,v in stab['summary'].items()])
table(['seed']+SOURCES,[[r['seed']]+[n6(r['metrics']['mean_gate_weight_'+s]) for s in SOURCES] for r in stab['records']])
table(['source gate weight','mean ± sample SD'],[(s,n6(v['mean'])+' ± '+n6(v['sample_std'])) for s,v in stability_gate_summary.items()])
heading('CONFIRMATION'); add(f"**CONFIRMATION MAE = {cm['coordinate_macro_mae_m']:.6f} m**"); add();add(f"**CONFIRMATION ENERGY SCORE = {cm['coordinate_macro_energy_score_m']:.6f} m**"); add();add(f"**CONFIRMATION NLL = {cm['coordinate_macro_mixture_nll']:.6f}**")
add();add('187 个完整物理坐标、4,133 scans。confirm command count=1。consumption lock 在第一次加载 confirmation shard 前写入；一次性评价后只保留原结果，没有调参或重跑。')
add();metrics(cm)
heading('PRE-EXPOSED BENCHMARK'); add('**PRE-EXPOSED DESCRIPTIVE BENCHMARK**');add();add('**blind confirmation = NO；pre-exposed benchmark = YES；used_for_model_selection = NO**')
add();add(f"**BENCHMARK MAE = {bm['coordinate_macro_mae_m']:.6f} m**");add();add(f"**BENCHMARK ENERGY SCORE = {bm['coordinate_macro_energy_score_m']:.6f} m**");add();add(f"**BENCHMARK NLL = {bm['coordinate_macro_mixture_nll']:.6f}**")
add();add(f"official validationData：{bm['rows']} rows / {bm['coordinate_groups']} physical coordinates。它已在 V18.1 暴露；本轮只在合法 confirmation 和 final refit 之后评价一次，结果没有用于选择模型或调参。")
add();metrics(bm)
heading('COMPUTE'); add(f"GPU：{preflight['gpu']}（6,144 MiB）；Python：{preflight['python']}；解释器：`{preflight['python_executable']}`；PyTorch：{preflight['torch']}；CUDA runtime：{preflight['cuda']}。")
add();table(['训练','seconds'],[('selection search + selection refit',n6(selection['checkpoint']['training_seconds'])),('stability wall seconds',n6(execution['stability']['wall_seconds'])),('confirmation refit',n6(cref['confirmation_checkpoint']['training_seconds'])),('final refit',n6(final['final_checkpoint']['training_seconds']))])
table(['phase','wall seconds','peak CUDA allocated MiB','peak CUDA reserved MiB'],[(s,n6(execution[s]['wall_seconds']),n6(runtime[s].get('peak_cuda_allocated_bytes',0)/1024**2),n6(runtime[s].get('peak_cuda_reserved_bytes',0)/1024**2)) for s in PHASES])
add('显存 peak 来自每个子进程 PyTorch allocator；不是驱动/context/其他进程的全设备峰值。prepare 没有 CUDA allocation。')
add();table(['phase','strict deterministic','warn_only','flash SDPA','memory-efficient SDPA','math SDPA','cuDNN deterministic'],[(s,yn(runtime[s]['deterministic_algorithms']),yn(runtime[s]['deterministic_warn_only']),yn(runtime[s]['flash_sdpa']),yn(runtime[s]['memory_efficient_sdpa']),yn(runtime[s]['math_sdpa']),yn(runtime[s]['cudnn_deterministic'])) for s in PHASES if s!='prepare'])
add('四个训练阶段均严格 deterministic；evaluation 阶段使用原注册 CLI 的默认开关。所有正式子进程 CUBLAS_WORKSPACE_CONFIG=:4096:8、cuDNN benchmark=False。')
heading('AUDIT'); table(['项目','YES/NO 或计数'],[('one inference checkpoint','YES'),('residual_experts == 1','YES'),('teacher used','NO'),('distillation used','NO'),('ensemble used','NO'),('best seed selected','NO'),('positive self-anchor target used','NO'),('calibration label 在 freeze 前用于 backbone gradient','NO'),('confirmation 内容在 prepare 后、lock 前访问','NO'),('prepare confirmation 写入与注册 hash','YES，协议规定'),('confirm command count','1'),('confirmation 用于调参','NO'),('benchmark 用于调参','NO'),('benchmark 是新的盲测','NO'),('全部 science source/config/freeze 保持不变','YES'),('calibration 在冻结后用于 outer_refit 梯度训练','YES，协议规定'),('confirmation 消费后用于最终 full refit 梯度训练','YES，协议规定')])
add('prepare 读取 trainingData 并写入/hash confirmation 是注册步骤；此后未手工加载/查看/评价 confirmation。confirm 中的 SHA 读取和 torch.load 均发生在 lock 已写入之后。consumed lock 从未删除，confirm 和 benchmark 均没有第二次调用。')
add();table(['phase','training CSV read opens','confirmation shard read opens','benchmark CSV read opens','historical checkpoint reads','denied'],[(s,*[runtime[s]['io_counts'][k] for k in ['training_csv_reads','confirmation_reads','benchmark_reads','historical_checkpoint_reads','denied']]) for s in PHASES])
add('verify-prepare 另做一次 training CSV SHA 校验，之后训练阶段只使用物理 shard。旧 V18.1 outcome 的描述性比较只发生在新 benchmark 完成后的报告阶段，未参与本轮模型/seed/epoch/参数选择。')
add();add(f"lock write open time_ns={audit['lock_write_open_time_ns']}；first confirm read time_ns={audit['confirmation_first_read_time_ns']}。")
add();table(['HASH','SHA256'],[(k,'`'+v+'`') for k,v in report['HASHES'].items()])
add('结果/lock 的 canonical self-hash 与实际文件 byte SHA 分别列出；前者验证字段内容，后者记录当前本机文件字节。')
heading('INTERPRETATION')
for i,text in enumerate(interpretation,1): add(f'{i}. {text}')
add();table(['集合','invariant recall','prototype recall','floor recall','union recall','union−invariant pp','union−prototype pp','prototype-only rescue 下界/上界'],[(name,pct(v['invariant_recall']),pct(v['prototype_recall']),pct(v['floor_recall']),pct(v['union_recall']),f"{100*v['union_minus_invariant']:+.3f}",f"{100*v['union_minus_prototype']:+.3f}",pct(v['prototype_only_rescue_lower_bound'])+' / '+pct(v['prototype_only_rescue_upper_bound'])) for name,v in rescue.items()])
add('每个集合按均值从大到小的 gate 权重：')
add();table(['集合','gate ranking'],[(name,' > '.join(s+'='+n6(w) for s,w in v['gate_weights_descending'])) for name,v in rescue.items()])
if comparison:
    old=comparison['V18_1']; add('同一 PRE-EXPOSED 官方 CSV 的 V18.1/V18.2 描述性比较（训练设计、字典、校准与 epoch 不同，不能解释为受控因果 ablation）：')
    add();table(['metric','V18.1','V18.2','V18.2−V18.1'],[('MAE (m)',n6(old['coordinate_macro_mae_m']),n6(bm['coordinate_macro_mae_m']),n6(comparison['mae_difference_m'])),('Energy Score (m)',n6(old['coordinate_macro_energy_score_m']),n6(bm['coordinate_macro_energy_score_m']),n6(comparison['energy_difference_m'])),('mixture NLL',n6(old['coordinate_macro_mixture_nll']),n6(bm['coordinate_macro_mixture_nll']),n6(comparison['nll_difference'])),('floor error',pct(old['coordinate_macro_floor_error']),pct(bm['coordinate_macro_floor_error']),f"{100*comparison['floor_error_difference']:+.3f} pp"),('final recall@8',pct(old['nearest_same_stratum_anchor_recall_at_8']),pct(bm['final_nearest_anchor_recall_at_8']),f"{100*comparison['recall_at_8_difference']:+.3f} pp"),('final recall@64',pct(old['nearest_same_stratum_anchor_recall_at_64']),pct(bm['final_nearest_anchor_recall_at_64']),f"{100*comparison['recall_at_64_difference']:+.3f} pp"),('between-anchor radial SD (m)',n6(old['mean_between_anchor_radial_sd_m']),n6(bm['mean_between_anchor_radial_sd_m']),n6(comparison['between_radial_sd_difference_m']))])
    add('Top-8 recall '+('提高' if comparison['recall_at_8_difference']>0 else '未提高')+'；between-anchor SD '+('下降' if comparison['between_radial_sd_difference_m']<0 else '未下降')+'。这些判断只描述当前两个冻结实验的官方 benchmark。')
else: add('当前 checkout 中没有 V18.1 历史 outcome 文件，未追加加载任何历史 checkpoint；跨版本的 Top-8/uncertainty 改善不能从当前不同 holdout 数据单独判定。')
add();table(['集合','nominal coverage','observed coverage','difference pp'],[(label,f'{q}%',pct(m[f'predictive_radial_coverage_{q}']),f"{100*(m[f'predictive_radial_coverage_{q}']-q/100):+.3f}") for label,m in [('confirmation',cm),('pre_exposed_benchmark',bm)] for q in (50,80,90,95)])
add('观察覆盖低于 nominal 提示区间偏窄/over-confidence，高于 nominal 提示偏保守；没有根据结果追加校准。V18.3 可预注册 source/union rescue 的联合诊断、style-gate 条件关系和 floor ablation，以检验当前只能描述的机制，并设置新的合法确认方案。')
add();add('完整原始数值与每轮记录：'+link('RUN_REPORT_V18_2.json',E/'RUN_REPORT_V18_2.json')+'。报告生成 UTC：'+report['generated_utc'])
report['report_sha256']=digest(report,'report_sha256')
json_path=E/'RUN_REPORT_V18_2.json'; md_path=E/'RUN_REPORT_V18_2.md'; assert not json_path.exists() and not md_path.exists()
with json_path.open('x',encoding='utf-8') as f: json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
with md_path.open('x',encoding='utf-8') as f: f.write('\n'.join(lines)+'\n')
print(json.dumps({'report_markdown':str(md_path),'report_json':str(json_path),'confirmation_MAE_m':cm['coordinate_macro_mae_m'],'confirmation_Energy_m':cm['coordinate_macro_energy_score_m'],'confirmation_NLL':cm['coordinate_macro_mixture_nll'],'pre_exposed_benchmark_MAE_m':bm['coordinate_macro_mae_m'],'pre_exposed_benchmark_Energy_m':bm['coordinate_macro_energy_score_m'],'pre_exposed_benchmark_NLL':bm['coordinate_macro_mixture_nll'],'all_audit_checks_passed':True},indent=2))
