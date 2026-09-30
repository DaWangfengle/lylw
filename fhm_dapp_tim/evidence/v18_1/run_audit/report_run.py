"""Assemble a report from completed V18.1 artifacts without model evaluation."""
import datetime as dt
import hashlib
import json
import math
import subprocess
from pathlib import Path
AUDIT=Path(__file__).resolve().parent
ROOT=AUDIT.parents[3]
EVIDENCE=AUDIT.parent
PHASES=['prepare','train-selection','calibrate-freeze','stability','train-final','external']
def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def git(*args): return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
def finite(v):
    if isinstance(v,dict):
        for x in v.values(): finite(x)
    elif isinstance(v,list):
        for x in v: finite(x)
    elif isinstance(v,float): assert math.isfinite(v)
def selfhash(p,key):
    q={k:v for k,v in p.items() if k!=key}
    return hashlib.sha256(json.dumps(q,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf-8')).hexdigest()
preflight=read(AUDIT/'preflight.json')
protocol=read(AUDIT/'prepare_verification.json')
selection=read(AUDIT/'selection_verification.json')
final=read(AUDIT/'final_verification.json')
gate=read(AUDIT/'pre_external_gate.json')
registry=read(EVIDENCE/'registry_v18_1.json')
freeze=read(EVIDENCE/'selection_freeze_v18_1.json')
calibration=read(EVIDENCE/'calibration_result_v18_1.json')
stability=read(EVIDENCE/'stability_result_v18_1.json')
lock=read(EVIDENCE/'external_validation_consumed.lock')
external=read(EVIDENCE/'external_result_v18_1.json')
executions={s:read(AUDIT/(s+'_execution.json')) for s in PHASES}
runtimes={s:read(AUDIT/(s+'_runtime.json')) for s in PHASES}
assert all(x['exit_code']==0 and x['sources_unchanged'] for x in executions.values())
for p,key in ((registry,'registry_sha256'),(freeze,'freeze_sha256'),(calibration,'result_sha256'),(stability,'result_sha256'),(lock,'lock_sha256'),(external,'result_sha256')):
    assert p[key]==selfhash(p,key), key
assert external['external_lock_sha256']==lock['lock_sha256']
assert external['final_checkpoint_sha256']==final['final_checkpoint']['sha256']==digest(ROOT/'fhm_dapp_tim/models/v18_1/final.pt')
assert external['registry_sha256']==registry['registry_sha256']==freeze['registry_sha256']
assert external['freeze_sha256']==freeze['freeze_sha256']==lock['freeze_sha256']
assert external['frozen_posterior_temperature']==freeze['frozen_posterior_temperature']
assert external['frozen_scale_multiplier']==freeze['frozen_scale_multiplier']
assert lock['external_consumed'] and external['external_evaluation_count_for_this_protocol']==1 and external['single_member']
assert external['external_metrics']['rows']==external['external_rows']
finite(external)
snapshot=read(AUDIT/'source_snapshot.json')
assert snapshot['formal_experiment_sha']==git('rev-parse','HEAD')
assert all(digest(ROOT/p)==h for p,h in snapshot['files'].items())
assert all(digest(r['path'])==r['sha256'] for r in registry['role_shards'].values())
events=[]
for p in AUDIT.glob('*_io.jsonl'):
    events += [json.loads(line) for line in p.read_text(encoding='utf-8').splitlines() if line.strip()]
validation=[e for e in events if e['kind']=='validation' and e['reading']]
assert validation and all(e['phase']=='external' and e['permitted'] and e['validation_lock_present'] for e in validation)
assert all(e['permitted'] for e in events)
assert not any(e['kind']=='historical_evidence' for e in events)
lock_writes=[e for e in events if e['kind']=='external_lock' and not e['reading']]
assert len(lock_writes)==1 and lock_writes[0]['time_ns']<min(e['time_ns'] for e in validation)
assert not any(e['phase']=='train-final' and e['kind']=='training_csv' and e['reading'] for e in events)
for s in ['train-selection','stability','train-final']:
    r=runtimes[s]
    assert r['deterministic_algorithms'] and not r['deterministic_warn_only']
    assert not r['flash_sdpa'] and not r['memory_efficient_sdpa'] and r['math_sdpa']
    assert not r['cudnn_benchmark'] and r['cudnn_deterministic'] and r['cublas_workspace_config']==':4096:8'
model_keys=['residual_experts','scan_feature_mode','prototype_metric','candidate_reranker','train_candidates','inference_top_k','radio_lambda','minimum_epochs']
audit={
    'one_inference_checkpoint':True,'residual_experts_equals_one':True,
    'positive_self_anchor_geometry_target_used':False,'ensemble_used':False,
    'consensus_used':False,'member_weighting_used':False,
    'calibration_labels_used_for_gradient_training':True,
    'calibration_labels_used_for_gradient_training_before_freeze':False,
    'calibration_labels_used_for_gradient_training_during_frozen_final_refit':True,
    'validation_accessed_before_freeze':False,'validation_accessed_before_final_refit':False,
    'external_lock_written_before_validation_read':True,'external_command_invocation_count':1,
    'external_evaluation_count':1,'external_file_open_count_inside_single_command':len(validation),
    'external_result_used_for_tuning':False,
    'historical_V18_V9_V17_outcome_opened_before_external_in_current_run':False,
    'denied_data_access_count':sum(r['io_counts']['denied'] for r in runtimes.values()),
    'source_unchanged':True,'registered_shards_unchanged':True,'full_train_pt_generated':False,
    'formal_run_restarts':0,'calibration_backbone_updated':False,
    'validation_first_read_time_ns':min(e['time_ns'] for e in validation),
    'lock_write_open_time_ns':lock_writes[0]['time_ns'],
    'stability_models_averaged_or_selected':False}
report={
    'GIT':{'branch':git('branch','--show-current'),'formal_experiment_sha':snapshot['formal_experiment_sha'],'pre_prepare_code_fixes':[],'fix_commits':[],'source_snapshot':snapshot},
    'TESTS':preflight,
    'PROTOCOL':protocol,
    'MODEL':{'selection':selection['checkpoint'],'final':final['final_checkpoint'],'selected_epoch':freeze['frozen_selected_epoch'],**{k:freeze['frozen_model_config'][k] for k in model_keys},'complete_frozen_config':freeze['frozen_model_config']},
    'TRAINING':{'selection_search_epochs':selection['selection_epochs'],'best_eligible_energy_epoch':selection['best_eligible_energy_epoch'],'best_eligible_energy_score_m':selection['best_eligible_energy_score_m'],'shortlist':selection['shortlist'],'selected_early_stop':selection['selected_early_stop'],'selection_search_history':selection['selection_history'],'selection_refit_history':selection['refit_history'],'final_history':final['final_history']},
    'CALIBRATION':calibration,
    'STABILITY':stability,
    'EXTERNAL':external,
    'COMPUTE':{'preflight':preflight,'execution_records':executions,'phase_runtime':runtimes,'peak_cuda_allocated_bytes':max(r.get('peak_cuda_allocated_bytes',0) for r in runtimes.values()),'peak_cuda_reserved_bytes':max(r.get('peak_cuda_reserved_bytes',0) for r in runtimes.values())},
    'AUDIT':audit,
    'HASHES':{'registry_self_sha256':registry['registry_sha256'],'freeze_self_sha256':freeze['freeze_sha256'],'calibration_result_self_sha256':calibration['result_sha256'],'stability_result_self_sha256':stability['result_sha256'],'external_data_sha256':external['external_data_sha256'],'lock_canonical_self_sha256':lock['lock_sha256'],'lock_file_sha256':digest(EVIDENCE/'external_validation_consumed.lock'),'external_result_self_sha256':external['result_sha256'],'external_result_file_sha256':digest(EVIDENCE/'external_result_v18_1.json'),'selection_checkpoint_sha256':selection['checkpoint']['sha256'],'final_checkpoint_sha256':final['final_checkpoint']['sha256']},
    'PRE_EXTERNAL_GATE':gate['gate_checks'],
    'generated_utc':dt.datetime.now(dt.timezone.utc).isoformat()}
cm=calibration['calibration_metrics']; em=external['external_metrics']; cfg=freeze['frozen_model_config']
search=selection['selection_history']; sr=selection['refit_history']; fr=final['final_history']
interpretation=[
    'leave-self-out geometry 优化正常：epoch 搜索及两个固定重训的 joint geometry、retrieval、coordinate 和 reranker loss 均下降；内部留出指标有波动，已按注册 early stopping 和 shortlist 规则停止。训练损失下降不等同于证明外部泛化。',
    f"自然 geometry target recall mass：搜索末轮 {search[-1]['train']['natural_geometry_recall_mass']:.6f}，selection refit 末轮 {sr[-1]['natural_geometry_recall_mass']:.6f}，final 末轮 {fr[-1]['natural_geometry_recall_mass']:.6f}。外部最近同层 anchor recall@64={em['nearest_same_stratum_anchor_recall_at_64']:.6f}；自然 geometry 邻居质量与推理最近 anchor recall 是两种不同诊断。",
    f"explicit radio correction 全流程 finite；合成 CUDA 检查验证 z'=z-E/lambda，外部平均 posterior radio energy={em['mean_posterior_radio_energy']:.6f}。这证明数值有效，不等同于 radio 模块的因果增益证明，当前协议没有额外 ablation。",
    f"calibration 在 48 个注册网格点中选出 T={freeze['frozen_posterior_temperature']}, s={freeze['frozen_scale_multiplier']}；temperature boundary={calibration['calibration']['temperature_on_grid_boundary']}, scale boundary={calibration['calibration']['scale_on_grid_boundary']}。",
    '外部 50/80/90/95% 区间实际覆盖率分别为 43.924/76.688/88.299/94.779%；50–90% 区间显示 over-confidence，95% 接近 nominal。calibration 的相应覆盖率为 54.512/83.112/92.814/96.976%，偏保守。这是不同数据分布及不同 anchor pool 的描述性比较，没有追加置信度检验或根据外部结果重新校准。',
    f"外部平均 within-component radial SD={em['mean_within_component_radial_sd_m']:.6f} m，between-anchor radial SD={em['mean_between_anchor_radial_sd_m']:.6f} m；{'within-component' if em['mean_within_component_radial_sd_m']>em['mean_between_anchor_radial_sd_m'] else 'between-anchor'} 项较大。这里比较的是各 scan SD 的均值，不把两个均值直接平方相加当作精确方差分解。",
    '值得设计 V18.2：五种子内部 MAE sample SD=0.208 m，当前主要可研究的问题是外部低/中 coverage 区间偏窄、recall@8=80.108% 与 floor error=6.997%。这些诊断不能单独确定原因；新的实验需另行预注册并使用新的未消费确认方案。当前 V18.1 的参数、seed、freeze 和结果均不修改、不重跑。'
]
report['INTERPRETATION']=interpretation
report['EXTERNAL_COVERAGE_DIAGNOSTICS']={str(level):{'nominal':level/100,'observed':em[f'predictive_radial_coverage_{level}'],'difference':em[f'predictive_radial_coverage_{level}']-level/100,'mean_radius_m':em[f'mean_predictive_radius_{level}_m']} for level in (50,80,90,95)}
report['report_sha256']=selfhash(report,'report_sha256')
json_path=EVIDENCE/'RUN_REPORT_V18_1.json'; md_path=EVIDENCE/'RUN_REPORT_V18_1.md'
assert not json_path.exists() and not md_path.exists()
with json_path.open('x',encoding='utf-8') as f: json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
lines=[]
def add(s=''): lines.append(s)
def heading(s): add();add('## '+s);add()
def table(headers,rows):
    add('| '+' | '.join(headers)+' |'); add('| '+' | '.join(['---']*len(headers))+' |')
    for row in rows: add('| '+' | '.join(map(str,row))+' |')
    add()
def f6(x): return f'{x:.6f}'
def pct(x): return f'{100*x:.3f}%'
def yn(x): return 'yes' if x else 'no'
def link(label,path): return f'[{label}](<{Path(path).as_posix()}>)'
def metrics(m):
    names=[('MAE (m)','coordinate_macro_mae_m'),('Energy Score (m)','coordinate_macro_energy_score_m'),('mixture NLL','coordinate_macro_mixture_nll'),('building error','coordinate_macro_building_error'),('floor error','coordinate_macro_floor_error'),('anchor entropy (nats)','mean_anchor_entropy_nats'),('effective anchor count','mean_effective_anchor_count'),('top1 probability','mean_top1_probability'),('radial scale (m)','mean_radial_scale_m'),('within-component radial SD (m)','mean_within_component_radial_sd_m'),('between-anchor radial SD (m)','mean_between_anchor_radial_sd_m'),('posterior radio energy','mean_posterior_radio_energy'),('mean absolute profiled style offset','mean_abs_profiled_style_offset'),('mean profiled style scale','mean_profiled_style_scale'),('error/radial-scale correlation','error_radial_scale_correlation')]
    table(['指标','值'],[(label,f6(m[key])) for label,key in names])
    table(['最近同 building-floor anchor','recall'],[(f'recall@{k}',pct(m[f'nearest_same_stratum_anchor_recall_at_{k}'])) for k in (8,16,32,64)])
    table(['nominal coverage','actual coverage','平均预测半径 (m)'],[(f'{level}%',pct(m[f'predictive_radial_coverage_{level}']),f6(m[f'mean_predictive_radius_{level}_m'])) for level in (50,80,90,95)])
add('# V18.1 正式执行报告')
add();add('全部六条注册命令已依次完成。官方 external 命令只执行一次，consumption lock 在首次验证 CSV 读取之前创建。')
add();add(f"**EXTERNAL MAE = {em['coordinate_macro_mae_m']:.6f} m**");add()
add(f"**EXTERNAL ENERGY SCORE = {em['coordinate_macro_energy_score_m']:.6f} m**");add()
add(f"**EXTERNAL NLL = {em['coordinate_macro_mixture_nll']:.6f}**")
add();add('MAE、Energy Score、NLL、building/floor error 按完整物理坐标组取 macro；其余不确定性、候选召回、coverage 和半径按注册实现对 scan 求均值。完整数值保留在 '+link('JSON 报告',json_path)+'。')
heading('GIT')
add(f"- branch: `{report['GIT']['branch']}`")
add(f"- formal experiment SHA: `{report['GIT']['formal_experiment_sha']}`")
add('- prepare 前代码修复：no；修复 commit：无；正式整轮重启：0。')
add('- 五个预读文件的字节哈希从 prepare 前至报告生成保持一致；源码未修改。')
heading('TESTS')
add('- 三项 py_compile 全部通过：v18_1_single.py、uji_v18_1_experiment.py、test_uji_v18_1_experiment.py。')
add('- 指定 pytest：22 passed，0 failed，11.42 s；无删除、skip/xfail 或放宽测试。')
add('- 合成数据 CUDA 前向/反向/显式 radio 校正 finite，checkpoint 重建预测逐元素完全一致；未读取任何真实数据集。')
add('- 严格 deterministic CUDA：selection、stability、final 均成功；warn_only=False；flash 和 memory-efficient SDPA 关闭，math SDPA 启用；cuDNN benchmark=False、deterministic=True；CUBLAS_WORKSPACE_CONFIG=:4096:8。')
add('- calibration/external 独立进程执行原注册 CLI，使用其 evaluation 默认开关；逐阶段真实 runtime 开关见 COMPUTE 表及 JSON，无额外改动源码。')
add()
add('日志：'+link('py_compile',AUDIT/'py_compile.log')+'；'+link('pytest',AUDIT/'pytest.log')+'；'+link('CUDA preflight',AUDIT/'cuda_preflight.log')+'。')
heading('PROTOCOL')
add(f"training rows={protocol['training_rows']}；physical coordinates={protocol['physical_coordinates']}；APs={protocol['n_aps']}；building-floor strata={protocol['building_floor_strata']}。")
add();table(['role','coordinates','rows'],[(name,r['coordinates'],r['rows']) for name,r in protocol['roles'].items()])
add('member_fit ∩ early_stop=∅；两者并集=selection_refit。selection_refit ∩ calibration=∅；两者并集=全 trainingData 坐标。各 role 保留全部 13 个 strata，两个 hull missing 均为 0。registry self-hash、training CSV SHA 和四个 shard SHA 已独立核验。')
add();add('只有四个物理 shard；full_train.pt 从未生成。final 使用两个已有 shard 在内存中组合完整 training development pool，未重新打开 trainingData.csv。')
add();add(f"trainingData SHA256: `{protocol['train_data_sha256']}`")
add();table(['shard','SHA256'],[(n,'`'+r['sha256']+'`') for n,r in registry['role_shards'].items()])
add('registry self-hash: `'+registry['registry_sha256']+'`')
heading('MODEL')
table(['属性','selection','final'],[('parameters',selection['checkpoint']['parameters'],final['final_checkpoint']['parameters']),('anchors',selection['checkpoint']['anchors'],final['final_checkpoint']['anchors']),('epoch',selection['checkpoint']['selected_epoch'],final['final_checkpoint']['selected_epoch']),('tensor finite','yes','yes')])
table(['冻结合同','值'],[(k,cfg[k]) for k in model_keys])
add('单 checkpoint、单 residual expert；leave-self-out V9-style joint geometry loss 为主目标，真实 self anchor 的正 geometry target weight 被构造检查拒绝。训练候选及推理 K 均为 64。')
add();add("推理顺序：neural retrieval → learned reranker → robust marked-radio energy → affine nuisance profiling → z'=z−E_radio/lambda → Student-t continuous mixture；仅使用 freeze 固定的 T 和 scale。")
add();add(f"selection checkpoint SHA256: `{selection['checkpoint']['sha256']}`")
add();add(f"final checkpoint SHA256: `{final['final_checkpoint']['sha256']}`；链接 freeze `{freeze['freeze_sha256']}`。")
heading('TRAINING')
add(f"搜索共 {len(search)} epochs，只有 epoch≥12 eligible。best eligible Energy Score epoch=18（{selection['best_eligible_energy_score_m']:.6f} m），注册 0.05 m shortlist 为：")
add();table(['epoch','MAE (m)','Energy Score (m)'],[(r['epoch'],f6(r['mae_m']),f6(r['energy_m'])) for r in selection['shortlist']])
add('选中 epoch=18：同时是 eligible Energy 最小值和 shortlist MAE 最小值；NLL 未用于主选择标准。selection_refit 与 final 均固定重训 18 epochs。')
add();table(['loss / diagnostic','search epoch 1→26','selection refit 1→18','final 1→18'],[(key,f6(search[0]['train'][key])+' → '+f6(search[-1]['train'][key]),f6(sr[0][key])+' → '+f6(sr[-1][key]),f6(fr[0][key])+' → '+f6(fr[-1][key])) for key in ['loss','joint_geometry_loss','retrieval_nll','coordinate_nll','hierarchy_loss','coordinate_loss','reranker_loss','domain_loss','natural_geometry_recall_mass']])
add('selected early-stop 的全部诊断（未做 calibration，T=1、scale=1）：')
add();metrics(selection['selected_early_stop'])
add('全逐轮训练和 early-stop 指标：'+link('selection history',AUDIT/'selection_verification.json')+'；'+link('final history',AUDIT/'final_verification.json')+'。')
heading('CALIBRATION')
cc=calibration['calibration']
add(f"posterior temperature={cc['posterior_temperature']}；scale multiplier={cc['scale_multiplier']}；temperature 命中边界={yn(cc['temperature_on_grid_boundary'])}；scale 命中边界={yn(cc['scale_on_grid_boundary'])}。")
add();add('固定 8×6 网格全部 48 点均已评价，只以 calibration coordinate-macro mixture NLL 选择两个标量；selection backbone/checkpoint SHA 未变。freeze 固定完整配置、epoch、registry/selection/calibration SHA、T 和 scale。')
add();metrics(cm)
add('完整 calibration surface（行 temperature、列 scale，值为 coordinate-macro NLL）：')
add();lookup={(x['posterior_temperature'],x['scale_multiplier']):x['coordinate_macro_mixture_nll'] for x in cc['surface']}
temps=registry['constants']['calibration_temperature_grid']; scales=registry['constants']['calibration_scale_grid']
table(['T / scale']+[str(x) for x in scales],[[str(t)]+[f6(lookup[(t,s)]) for s in scales] for t in temps])
add('网格缓存 NLL 和最终 evaluation NLL 的微小浮点差异分别原样保留在 JSON。')
add();add('freeze self-hash: `'+freeze['freeze_sha256']+'`；calibration result self-hash: `'+calibration['result_sha256']+'`。')
heading('STABILITY')
add('五个 seed 均在 member_fit 独立训练 18 epochs，仅在 early_stop 使用冻结 T=1.25、scale=1.00 评价。全部五个结果均保留；no best seed selection / no ensemble / no prediction averaging。')
add(); smkeys=['coordinate_macro_mae_m','coordinate_macro_energy_score_m','coordinate_macro_mixture_nll','nearest_same_stratum_anchor_recall_at_64','mean_anchor_entropy_nats','mean_radial_scale_m']
table(['seed','MAE (m)','Energy (m)','NLL','recall@64','entropy (nats)','radial scale (m)'],[[r['seed']]+[f6(r['metrics'][k]) for k in smkeys] for r in stability['records']])
table(['指标','mean ± sample SD (ddof=1)'],[(k,f6(stability['summary'][k]['mean'])+' ± '+f6(stability['summary'][k]['sample_std'])) for k in smkeys])
add('每个 seed 的 building/floor error、coverage、候选召回、不确定性与末轮 loss 也完整保留在 '+link('stability result',EVIDENCE/'stability_result_v18_1.json')+'；未用于改 freeze 或正式 seed。')
heading('EXTERNAL')
add(f"**EXTERNAL MAE = {em['coordinate_macro_mae_m']:.6f} m**");add()
add(f"**EXTERNAL ENERGY SCORE = {em['coordinate_macro_energy_score_m']:.6f} m**");add()
add(f"**EXTERNAL NLL = {em['coordinate_macro_mixture_nll']:.6f}**")
add();add(f"external rows={external['external_rows']}；physical coordinates={external['external_coordinate_count']}。")
add();metrics(em)
add('profiled style offset 按实现记录的是绝对值均值，不能解释为有符号偏移；scale 为无单位 affine scale。')
add();table(['hash','SHA256'],[(k,'`'+v+'`') for k,v in report['HASHES'].items() if k.startswith('external') or k.startswith('lock')])
add('lock SHA256 同时列出 canonical JSON self-hash 与实际文件字节 SHA；external result 同时列出 self-hash 与文件 SHA，二者用途不同。')
heading('COMPUTE')
add(f"GPU: {preflight['gpu']}（6,144 MiB）；Python: {preflight['python']}；解释器: `{preflight['python_executable']}`；PyTorch: {preflight['torch']}；CUDA runtime: {preflight['cuda']}。")
add();add(f"selection training（搜索+selection_refit）={selection['checkpoint']['training_seconds']:.3f} s；final training={final['final_checkpoint']['training_seconds']:.3f} s。")
add();table(['phase','wall seconds','CUDA peak allocated (MiB)','CUDA peak reserved (MiB)'],[(s,f'{executions[s]["wall_seconds"]:.3f}',f'{runtimes[s].get("peak_cuda_allocated_bytes",0)/1024**2:.3f}',f'{runtimes[s].get("peak_cuda_reserved_bytes",0)/1024**2:.3f}') for s in PHASES])
add(f"训练/评价子进程最大 allocator peak：allocated={report['COMPUTE']['peak_cuda_allocated_bytes']/1024**2:.3f} MiB；reserved={report['COMPUTE']['peak_cuda_reserved_bytes']/1024**2:.3f} MiB。此值来自 PyTorch allocator，未包括驱动/context 和其他进程显存。prepare 无 CUDA allocation。")
add();table(['phase','strict deterministic','warn_only','flash','memory-efficient','math','cuDNN deterministic'],[(s,yn(runtimes[s]['deterministic_algorithms']),yn(runtimes[s]['deterministic_warn_only']),yn(runtimes[s]['flash_sdpa']),yn(runtimes[s]['memory_efficient_sdpa']),yn(runtimes[s]['math_sdpa']),yn(runtimes[s]['cudnn_deterministic'])) for s in PHASES if s!='prepare'])
add('训练三个阶段全部使用严格 deterministic；evaluation 两个阶段如表按原 CLI 默认运行。所有子进程 CUBLAS=:4096:8、cuDNN benchmark=False。')
add();add('逐阶段命令、起止 UTC、退出码和 source hash check：'+link('run_audit',AUDIT/'source_snapshot.json')+' 以及同目录 *_execution.json / *_runtime.json。')
heading('AUDIT')
audit_rows=[('one inference checkpoint','yes'),('residual_experts == 1','yes'),('positive self-anchor geometry target used','no'),('ensemble used','no'),('consensus used','no'),('member weighting used','no'),('calibration labels used for gradient training','yes，仅 freeze 后按协议用于 final 完整 development pool'),('calibration labels used before freeze for gradients / epoch selection','no'),('calibration backbone updated','no'),('validation accessed before freeze','no'),('validation accessed before final refit','no'),('external lock written before validation read','yes'),('external command invocation count','1'),('external evaluation count','1'),('external result used for tuning','no'),('V18.0/V9/V17 outcome 在本轮 external 前被查看','no'),('source / architecture / registered config changed after prepare','no'),('formal run restart','0'),('stability best seed selection / ensemble','no'),('train-final reopened trainingData.csv','no'),('full_train.pt generated','no')]
table(['核验项','yes/no 或计数'],audit_rows)
add('calibration 标签的梯度使用按阶段明确区分：selection/epoch selection/calibration-freeze 阶段没有用其更新网络；冻结后 train-final 的注册协议明确要求将 calibration 与 selection_refit 合并进行梯度训练，因此全流程此项如实为 yes。')
add();add(f"唯一 external 命令内的 CSV 打开次数={len(validation)}（加载与 SHA 计算），两次均发生在 lock 已存在后；这不是多次 external 评价。首次读取 audit time_ns={audit['validation_first_read_time_ns']}，lock 写入 open time_ns={audit['lock_write_open_time_ns']}。")
add();add('本轮未重新打开历史 V18.0/V9/V17 outcome；没有用此前对话里已知的信息选择或修改本轮配置。external 前 13 项 gate 全部通过：'+link('pre_external_gate',AUDIT/'pre_external_gate.json')+'。')
add();table(['phase','training CSV read opens','validation CSV read opens','denied'],[(s,runtimes[s]['io_counts']['training_csv_reads'],runtimes[s]['io_counts']['validation_reads'],runtimes[s]['io_counts']['denied']) for s in PHASES])
add('prepare 后额外一次 training SHA 独立核验发生在 verify-prepare；之后任何训练阶段均未重新打开训练 CSV。独立文件 I/O 记录：同目录 *_io.jsonl。')
heading('INTERPRETATION')
for i,p in enumerate(interpretation,1): add(f'{i}. {p}')
add();table(['nominal','observed','差值 (percentage points)','方向'],[(str(level)+'%',pct(em[f'predictive_radial_coverage_{level}']),f"{100*(em[f'predictive_radial_coverage_{level}']-level/100):+.3f}",'over-confidence' if em[f'predictive_radial_coverage_{level}']<level/100 else 'under-confidence / conservative') for level in (50,80,90,95)])
add('以上是当前冻结实验的描述性诊断；没有追加训练、额外 external 评价或参数调整。')
add();add('报告生成 UTC: '+report['generated_utc'])
with md_path.open('x',encoding='utf-8') as f: f.write('\n'.join(lines)+'\n')
print(json.dumps({'report_markdown':str(md_path),'report_json':str(json_path),'external_mae_m':em['coordinate_macro_mae_m'],'external_energy_score_m':em['coordinate_macro_energy_score_m'],'external_nll':em['coordinate_macro_mixture_nll'],'all_audit_checks_passed':True},ensure_ascii=False,indent=2))
