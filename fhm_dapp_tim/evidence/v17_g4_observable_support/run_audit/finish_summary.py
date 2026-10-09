"""Create one short G4 execution summary solely from saved outcomes and logs."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
from collections import defaultdict
import json,hashlib,math,numpy as np
root=Path(__file__).resolve().parents[4];audit=Path(__file__).resolve().parent;base=audit.parent
load=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=load(base/'manifest_v17_g4.json');cal=load(base/'calibration_v17_g4.json');conf=load(base/'confirmation_v17_g4.json')
context=load(audit/'INITIAL_CONTEXT.json');prep=load(audit/'preparation_review_attempt01.stdout.log')
records=[]
for p in sorted(audit.glob('*.json')):
 obj=load(p)
 if 'stage' not in obj or 'exit_code' not in obj or 'stdout_sha256' not in obj:continue
 for stream in ('stdout','stderr'):assert sha(root/obj[stream])==obj[stream+'_sha256'],p.name
 access=audit/(obj['stage']+'.access.json');trace=load(access)
 assert sha(access)==obj['access_guard_sha256']
 assert not trace['denied_attempts'] and not trace['prior_G1_G2_G3_artifact_open_counts']
 assert all(Path(v).name.lower()=='uts_training.csv' for v in trace['python_data_open_counts'])
 records.append(obj)
records.sort(key=lambda x:x['started_utc'])
assert sum(r['stage']=='confirm_attempt01' for r in records)==1
assert next(r for r in records if r['stage']=='check_after_confirm_attempt02')['exit_code']==0
final=load(audit/'FINAL_INTEGRITY_REVIEW.json');assert final['verified']
assert conf['selected_policy']==cal['selected_policy']
policy=cal['selected_policy']['policy'];selection=cal['selected_policy'];gate=conf['confirmation_gate']
zone=timezone(timedelta(hours=8));start=datetime.fromisoformat(context['started_utc']);end=datetime.fromisoformat(next(r for r in records if r['stage']=='check_after_confirm_attempt02')['ended_utc'])
fmt=lambda x:'N/A' if x is None else f'{float(x):.9f}'
lines=['# V17-G4 执行摘要','',f"完成：5/5 FIT-only 成员、CAL 自动冻结、一次内部 CONFIRM；最终 check=VERIFIED。科学结论：**{conf['decision']}**。",'',
 f"Asia/Shanghai：{start.astimezone(zone).isoformat()} → {end.astimezone(zone).isoformat()}；历时 {(end-start).total_seconds()/60:.3f} 分钟。",
 f"Python {prep['python_version'].split('|')[0].strip()}；{prep['python_executable']}；Torch {prep['versions']['torch']} / CUDA {prep['cuda']}；{prep['gpu']}（6GB）。",
 f"NumPy {prep['versions']['numpy']} / pandas {prep['versions']['pandas']} / SciPy {prep['versions']['scipy']} / pytest {prep['versions']['pytest']}。",'',
 '## 冻结输入与分区','',
 '只读取 UTS_training.csv。角色均按同一物理坐标的全部扫描分组；三个两两交集为 0，各角色覆盖同一 16 个编码楼层。FIT 的原始物理键与 R0 中心化 float32/四位键一一对应。',
 '门控只使用扫描 RSS、5 成员基线后验、FIT 锚点密度；真实查询位置仅用于离线 MAE/ES。CAL 阈值按坐标组平均支持度的 q25/q50/q75 固定计算，CONFIRM 不重新计算阈值。','',
 '| 角色 | 坐标 | 行 | 排序键 SHA256（紧凑 JSON） |','|---|---:|---:|---|']
for role in ('fit','calibration','confirmation'):lines.append(f"| {role} | {manifest['role_groups'][role]} | {manifest['role_rows'][role]} | {prep['role_sorted_keys_sha256'][role]} |")
lines+=['',f"训练 CSV SHA256：{manifest['source_sha256']['training_csv']}",f"G4 源码 SHA256：{manifest['source_sha256']['g4_code']}",f"最终 G4 测试 SHA256：{prep['current_test_sha256']}",
 f"manifest 自哈希：{manifest['manifest_sha256']}；文件 SHA256：{sha(base/'manifest_v17_g4.json')}",
 '完整算法、训练、评分及设计哈希绑定在 manifest。E5/L192/K192/alpha1，gamma=0.70、10m；原始 R0 模型/损失/优化器/增强及五个固定 epochs 未改。','',
 '## 五成员','',
 '| 成员 | seed | epoch | checkpoint SHA256 |','|---|---:|---:|---|']
for i,(seed,epoch,h) in enumerate(zip(manifest['seeds'],manifest['epochs'],cal['member_sha256']),1):
 p=root/f'fhm_dapp_tim/models/v17_g4_observable_support/member_{i:02d}_seed{seed}.pt';assert sha(p)==h
 lines.append(f'| {i} | {seed} | {epoch} | {h} |')
assert len(set(cal['member_sha256']))==5
lines+=['','五名成员均从 FIT 5,791 行/945 锚点完整训练，未重启或复用旧 G3 权重。训练阶段只给原始 R0 loader/loss 传入 rows[fit]；CAL/CONFIRM 标签未进入训练。锚点/原型/原点的一致性由 calibration 的 _load_models 严格逐位核验。五份 .pt 保留本地。','',
 '## CAL 与冻结选择','',
 '| CAL 方法 | Macro MAE m | Macro ES m | Row MAE m | Row ES m |','|---|---:|---:|---:|---:|']
for name,m in cal['metrics'].items():lines.append(f"| {name} | {fmt(m['mae_2d_m'])} | {fmt(m['energy_score_2d_m'])} | {fmt(m['row_mae_m'])} | {fmt(m['row_es_m'])} |")
lines+=['',f"自动冻结：**{policy}**；阈值 {selection['threshold']}；满足 CAL 改善≥0.05m、ES 增量≤0.02m 的候选：{selection['feasible_policies']}。",
 f"预声明门控阈值：{json.dumps(selection['thresholds'],sort_keys=True)}。placebo 不参与选择。",
 f"calibration 自哈希：{cal['result_sha256']}；文件 SHA256：{sha(base/'calibration_v17_g4.json')}。",'',
 '## 唯一内部 CONFIRM','',
 '| CONFIRM 方法 | Macro MAE m | Macro ES m | Row MAE m | Row ES m |','|---|---:|---:|---:|---:|']
for name,m in conf['metrics'].items():lines.append(f"| {name} | {fmt(m['mae_2d_m'])} | {fmt(m['energy_score_2d_m'])} | {fmt(m['row_mae_m'])} | {fmt(m['row_es_m'])} |")
lines+=['',f"差值统一为冻结候选−基线：MAE {fmt(gate['delta_mae_m'])}m；ES {fmt(gate['delta_es_m'])}m。",'','| CONFIRM 固定条件 | 数值 | 是否通过 |','|---|---:|---|',
 f"| 候选非 baseline | {policy} | {'PASS' if policy!='baseline' else 'FAIL'} |",
 f"| MAE 差≤−0.15m | {fmt(gate['delta_mae_m'])} | {'PASS' if gate['delta_mae_m']<=-.15 else 'FAIL'} |",
 f"| MAE 同时上界<0 | {fmt(gate['mae_simultaneous_upper_m'])} | {'PASS' if policy!='baseline' and gate['mae_simultaneous_upper_m']<0 else 'FAIL'} |",
 f"| ES 同时上界≤0 | {fmt(gate['es_simultaneous_upper_m'])} | {'PASS' if policy!='baseline' and gate['es_simultaneous_upper_m']<=0 else 'N/A' if policy=='baseline' else 'FAIL'} |"]
if policy=='baseline':lines+=['','CAL 回退 baseline；差向量严格为 0，未伪造 studentized 区间。JSON 的上界 0 是分支占位，bootstrap 推断为 N/A，自动 NO_INTERNAL_PROMOTION。']
else:lines+=['',f"原 R3 studentized maxT：5000 次物理坐标配对 resampling，family=MAE+ES，95% 单侧同时上界，seed={manifest['bootstrap_seed']}。不是逐扫描独立重采样。"]
lines+=['',f"confirmation 自哈希：{conf['result_sha256']}；文件 SHA256：{sha(base/'confirmation_v17_g4.json')}。",f"锁 SHA256：{sha(base/'internal_confirmation.lock')}；只执行 1 次 confirm，保留锁，未清锁或重评。",'',
 '## 描述性机制与竞争解释','',
 'gain=baseline MAE−global G1 MAE，正值表示 G1 收益。以下 Spearman 仅描述方向，不视为独立显著性或因果。','',
 '| 特征 vs G1 gain | CAL Spearman | CONFIRM Spearman |','|---|---:|---:|',
 f"| 后验期望 log 支持度 | {fmt(cal['observable_support_vs_fixed_g1_gain_spearman'])} | {fmt(conf['observable_support_vs_fixed_g1_gain_spearman'])} |"]
for key in ('entropy','visible_aps'):lines.append(f"| {key} | {fmt(cal['competing_observable_controls'][key]['spearman_vs_g1_gain'])} | {fmt(conf['competing_observable_controls'][key]['spearman_vs_g1_gain'])} |")
lines+=['','| 组/对照（MAE） | 改善坐标 | 恶化坐标 | 不变 | 差值中位数 m |','|---|---:|---:|---:|---:|']
for role,obj in [('CAL',cal),('CONFIRM',conf)]:
 for method in dict.fromkeys(('g1',policy)):
  delta=np.asarray(obj['coordinate_vectors'][method]['mae'])-np.asarray(obj['coordinate_vectors']['baseline']['mae'])
  lines.append(f"| {role}/{method}−baseline | {int((delta<0).sum())} | {int((delta>0).sum())} | {int((delta==0).sum())} | {fmt(np.median(delta))} |")
lines+=['','| CONFIRM 编码楼宇/楼层 | 坐标 | Baseline MAE m | 冻结候选 MAE m | 差 m | Global G1 差 m |','|---|---:|---:|---:|---:|---:|']
groups=defaultdict(list)
for i,key in enumerate(conf['coordinate_keys']):groups[(key[0],key[1])].append(i)
for (b,f),ix in sorted(groups.items()):
 baseline=np.asarray(conf['coordinate_vectors']['baseline']['mae'])[ix].mean();chosen=np.asarray(conf['coordinate_vectors'][policy]['mae'])[ix].mean();glob=np.asarray(conf['coordinate_vectors']['g1']['mae'])[ix].mean()
 lines.append(f'| B{b}/F{f} | {len(ix)} | {fmt(baseline)} | {fmt(chosen)} | {fmt(chosen-baseline)} | {fmt(glob-baseline)} |')
lines+=['','楼层为适配器词典序编码，不能把编码直接当物理楼层；分层数值仅读取冻结 JSON 的同序坐标向量描述计算，不重新评分、选门槛或追加推断。','',
 '## 错误、修复与限制','',
 '- pytest_attempt01：43 passed/5 setup errors。新增日志包装器未建 basetemp 父目录；只补 mkdir，保存 PYTEST_TMP_PARENT_FIX.json/.diff。',
 '- pytest_attempt02：47 passed/1 failed。新 G4 测试的错误文本 regex 不符；只改匹配并加强锁后不得调用 prepare 的回归，保存 G4_TEST_LOCK_MESSAGE_FIX.json/.diff。',
 '- 最终 compile_attempt02、pytest_attempt03（48 passed）、preflight 全通过后才 prepare；G4 实验源码未改，manifest 后未改源码/测试/设计。全部原失败 stdout/stderr/hash 留存。']
lines+=['- check_after_confirm_attempt01 因 Markdown 表格行序与 JSON 回读顺序不一致失败；只重排本轮派生报告表格，原报告快照和 REPORT_TABLE_ORDER_FIX.json/.diff 保留。数值、正文、源码/测试/manifest/权重/CAL/CONFIRM/锁均未变；只重跑只读 check。渲染器源码在冻结后未改。']
warn=[]
for r in records:
 stderr=(root/r['stderr']).read_text(encoding='utf-8',errors='replace')
 if stderr.strip():warn.append((r['stage'],stderr))
attention=[s for s,text in warn if 'Memory Efficient attention' in text]
lines.append(f'- CUDA：{len(attention)} 个训练日志记录 Memory Efficient attention backward 非确定性警告；固定 seed/CUBLAS 配置不保证逐位重训相同，保存实际 checkpoint 哈希，未更换训练算法。')
other=[s for s,text in warn if 'Memory Efficient attention' not in text]
lines.append(f'- 其他非空 stderr：{other}；完整内容在对应原始 stderr 日志。')
lines+=['- G1/G2/G3 历史科学代码及原始证据 SHA256 与开始快照一致，G1/G3 STOP 保持；既有权重未加载。命令中的 Python open 轨迹仅见 UTS_training.csv，没有其他提供测试或旧结果输入；这不是 OS 全局访问监控。',
 '- 本轮仅为已公开 G3 历史背景下的 UTS 训练文件内部探索；5 成员共享 FIT 和评估坐标，不是5次独立实验，不能与G3 25成员绝对MAE直接排名。同楼层空间依赖降低有效样本量。',
 '- 后验支持度可能代理熵/原始误差，相关性不能证明稀疏因果；任何内部正结果也不等于独立外部泛化、发表许可或 Q1 完整证据链。不训练下一模型，不读取其他测试，不执行 Git 操作。','',
 '## 指定命令退出码与时长','',
 '每条同名 .json 保存 UTC 起止、退出码及 stdout/stderr/access SHA256。失败未删除；训练、CAL 和 CONFIRM 未重复。','',
 '| 阶段 | 退出码 | 秒数 |','|---|---:|---:|']
for r in records:lines.append(f"| {r['stage']} | {r['exit_code']} | {r['runtime_seconds']:.3f} |")
lines+=['',f"训练五命令合计 {sum(r['runtime_seconds'] for r in records if r['stage'].startswith('train_member_'))/60:.3f} 分钟；记录 {len(records)} 条命令、{len(records)} 份访问轨迹。",'']
with (audit/'EXECUTION_SUMMARY.md').open('x',encoding='utf-8',newline='\n') as f:f.write('\n'.join(lines))
print(json.dumps({'status':'SUMMARY_WRITTEN','decision':conf['decision'],'summary_sha256':sha(audit/'EXECUTION_SUMMARY.md'),'summary_lines':len(lines)},ensure_ascii=False))
