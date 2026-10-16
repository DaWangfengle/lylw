"""Render the complete saved experiment and independent audit as Markdown."""
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
AUDIT = Path(__file__).resolve().parent
NAMES = {
    'v17_single_raw':('Raw','原基线骨干'),
    'v17_single_temp_t06':('T=.6','固定骨干/固定温度'),
    'v17_single_temp_fit':('T_CAL','固定骨干/CAL 1参数'),
    'v17_single_g13_historical_cal':('G13_hist','固定骨干/CAL原代理2系数'),
    'v17_single_g13':('G13_NLL','固定骨干/CAL连续NLL2系数'),
    'v17_single_unified_frozen':('Unified_frozen','固定骨干/CAL门控'),
    'unified_joint_no_evidence':('Joint_no_ev','联合骨干/推理关证据'),
    'unified_joint':('Joint','FIT真实联合训练'),
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    results = {d:read(ROOT/f'fhm_dapp_tim/evidence/msep_single_{d}_full_v1/RESULT.json') for d in ('tuji','uts','uji')}
    audit = read(AUDIT/'FULL_RESULT_AUDIT.json')
    data = read(AUDIT/'DATA_AUDIT.json')
    pairs = read(AUDIT/'PAIRED_SITE_ANALYSIS.json')
    execution = read(AUDIT/'EXECUTION_RECORD.json')
    freeze = read(AUDIT/'METHOD_FREEZE.json')
    env = read(AUDIT/'run_logs/environment.stdout.log')
    parameters = {}
    for d,r in results.items():
        checkpoint_path=ROOT/f'fhm_dapp_tim/evidence/msep_single_{d}_full_v1/CHECKPOINTS.pt'
        cp=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
        assert all(torch.isfinite(v).all() for group in ('base_model','joint_model','frozen_head','joint_head')
                   for v in cp[group].values() if torch.is_tensor(v))
        parameters[d]={'checkpoint_sha256':hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                       'frozen_head':{k:v.tolist() for k,v in cp['frozen_head'].items()},
                       'joint_head':{k:v.tolist() for k,v in cp['joint_head'].items()},
                       'all_saved_model_and_head_tensors_finite':True,
                       'joint_loss_supervision':'2D XY mixture NLL + point risk + KL/gate penalty; no explicit building/floor categorical loss',
                       'joint_and_no_evidence_metrics_identical':r['test_metrics']['unified_joint']==r['test_metrics']['unified_joint_no_evidence']}
        assert parameters[d]['checkpoint_sha256']==r['checkpoint_sha256']
        if parameters[d]['joint_head']['theta']==[0.,0.]:
            assert parameters[d]['joint_and_no_evidence_metrics_identical']
        if cp['g13_historical_coefficients']==[0.,0.]:
            assert r['test_metrics']['v17_single_g13_historical_cal']==r['test_metrics']['v17_single_temp_t06']
    lines = ['# MSEP-Loc 单成员锚点后验三阶段实验实测报告', '',
             f"生成时间：{datetime.datetime.now().astimezone().isoformat()}。", '',
             '三域真实训练与一次正式 TEST 评分已完成。所有八组、负结果及逐位置伤害均保留。没有使用官方提供测试文件，没有执行 Git 操作；本报告不选择部署策略，不执行后续模型或集成试验。', '',
             '这是当前四角色站点留出的新实验，原始训练语料和部分物理位置此前有历史研究暴露。不是全新外部盲测，不是空间块留出；同域近邻及扫描/设备相关性可能降低有效独立样本数。不能与历史 E5、5.7m/4.58m 的不同 TEST 拆分直接比较，也不能证明 JCR Q1 外部泛化。', '',
             '## 环境、冻结与命令', '',
             f"分支：`{freeze['branch_ref']}`；起始源码引用：`{freeze['source_revision']}`。Python `{env['executable']}`，{env['python'].split('|')[0].strip()}；PyTorch {env['torch']}，CUDA {env['cuda_runtime']}，{env['gpu']}（{env['gpu_memory_bytes']/1024**3:.2f} GiB）；NumPy {env['numpy']}，SciPy {env['scipy']}，pandas {env['pandas']}，pytest {env['pytest']}。", '',
             f"正式方法于 {freeze['frozen_at']} 冻结，`METHOD_FREEZE.json` SHA256 `{hashlib.sha256((AUDIT/'METHOD_FREEZE.json').read_bytes()).hexdigest()}`。三域均 seed=17301、split_seed=20261016、全 FIT 锚点、batch=64、base最多80轮（DEV早停）、joint固定10轮、ES两流各32次。没有降低 batch、延长训练、温度扩网格或选择 TEST 赢家。", '',
             '每域 `TEST_INVOCATION.lock` 在所有权重/校准参数保存之后、首次 TEST 评分之前排他创建，绑定源代码和 checkpoint SHA。每种方法每域仅一次正式评分。smoke为1轮base+1轮joint、batch32、两流各8次ES，全部只评价DEV；其checkpoint无TEST站点键，RESULT无test_metrics。', '',
             '|命令记录标签|退出码|耗时(s)|', '|---|---:|---:|']
    for command in execution['commands']:
        lines.append(f"|{command['label']}|{command['exit_code']}|{command['duration_seconds']:.3f}|")
    lines += ['', '所有实际参数、开始结束时间、stdout/stderr 路径及日志 SHA256 见 `EXECUTION_RECORD.json`。原始日志保留，不把回归演示的失败删除。命令前缀均为指定 Python：', '']
    for command in execution['commands']:
        if command['label'].startswith(('smoke_', 'full_')) and '-m' in command['command']:
            lines += ['```powershell', '& "' + command['command'][0] + '" ' + ' '.join(command['command'][1:]), '```', '']
    lines += ['**明确的命令例外**：没有执行旧 `v18_4_datasets --help`。该入口忽略参数而实际加载四域训练数据，TUJI默认还解码测试变量；已只读检查入口并运行安全的 MSEP CLI `--help`，exit=0。不能把被替代的命令写成已通过。', '',
              '原指定测试9 passed；限制变量的合成回归在修复前真实产生2 failed/10 passed，修复后最终12 passed。日志扩展引入的一处花括号编译错误也保留；修复后编译/测试以及三域冒烟均成功。只改新MSEP study和新测试，未改后验数学算子、方法常数、历史科学源码或数据。完整原始失败及源码diff见 `FIX_RECORD.md`。', '',
              '## 数据和四角色隔离', '',
              '按(building,floor,raw XY四位)分组，各楼层固定PRNG划分约70% FIT/10% DEV/10% CAL/10% TEST（逐层floor取整，剩余全部FIT）。每个物理位置全部扫描只属于一角色。此处原始坐标来自canonical float32适配器；UJI投影坐标已有float32量化，未声称厘米以下几何精度。', '',
              '|域|FIT站点/扫描|DEV站点/扫描|CAL站点/扫描|TEST站点/扫描|AP总数/FIT模型AP|', '|---|---:|---:|---:|---:|---:|']
    for d,r in results.items():
        role = r['role_count']
        cells = [f"{role[x]['sites']}/{role[x]['rows']}" for x in ('fit','dev','cal','test')]
        lines.append(f"|{d.upper()}|"+'|'.join(cells)+f"|{r['input_ap_count']}/{r['fit_seen_ap_count']}|")
    lines += ['', '三域角色交集均为空，原始/居中坐标分组均一一对应，FIT站点与anchor ID一一对应；非FIT sample_anchor全部为-1。G13观测memory仅调用原FIT RSS与FIT anchor IDs构建，model AP mask来自FIT原型，设备/用户类别不进入后验门控。真实XY/楼层仅用于训练/校准损失、DEV早停和评分诊断，推断API不接收query真值。', '',
              'TUJI五个MAT物理容器被打开及哈希，但MSEP调用限定 `variable_names=("results_train",)`，返回字典拒绝results_test；不能声称容器所有测试字节从未被打开。没有加载UTS_test、UJI validationData或Tampere。', '',
              '|域|FIT每站扫描分布(count:站点数)|至少2扫描站点|', '|---|---|---:|']
    for d,v in data.items():
        lines.append(f"|{d.upper()}|`{json.dumps(v['fit_scan_count_distribution'],ensure_ascii=False)}`|{v['fit_sites_with_at_least_two_scans']}|")
    lines += ['', 'RSS>-0.999才算模型检测；-1可能同时表示缺失或裁剪至-110dBm的有效弱信号，无法完美区分。重复扫描不等于独立采样；FIT统计没有估计设备/会话独立性。', '',
              '输入完整SHA和每建筑楼层、设备的角色计数及role-key SHA见 `DATA_AUDIT.json`；三域正式结果的角色SHA已与该独立预检匹配。输入SHA：', '']
    for d,v in data.items():
        for source,digest in v['input_sha256'].items(): lines.append(f'- {d.upper()} `{source}`：`{digest}`')
    lines += ['', '## 八组方法和训练条件', '', '|简称|JSON策略键|训练/干预类型|', '|---|---|---|']
    for method,(alias,kind) in NAMES.items(): lines.append(f'|{alias}|`{method}`|{kind}|')
    lines += ['', '前六组使用同一base骨干、所有相同FIT候选、同一连续分量中心和Student-t尺度；只改锚点概率。统一冻结头的全局系数与G13_NLL完全相同，只在CAL学习4→2线性可靠性门控（10参数）。G13_hist是单域单成员、原24轮Dirac ES+.3点风险+.1KL代理，不是原E5双域pooled的历史系数；G13_NLL改用连续Student-t NLL+.3点风险+.05KL，须作为不同校准目标解释。', '',
              'Joint从同一选中base开始，在FIT对query自身anchor排除后，实际更新编码器、检索keys、连续Student-t头和可靠性后验，门控/系数均在FIT训练。原G13观测统计固定。**Joint_no_ev不是独立训练的原始模型**：它仅在同一共同训练骨干上关闭观测证据，保留T=.6。比较Joint与Joint_no_ev隔离的是该共同骨干的推理证据开关。Joint与固定骨干方法的对比同时含有骨干与分量变化，不能只归因概率修正。', '',
              '原V17同建筑邻居损失保留，仍可能跨楼层；domain adversarial权重固定为0。单成员R3原子评分neural_scale=1、prototype_weight=32/3，默认全FIT支持；并非历史R0 top24/.75配置，也不是原E5 Top192实验的严格数值复现。', '',
              'Joint的leave-own-anchor-out不等于完全未见站点meta-learning：预训练base和全FIT观测统计原本已见这些FIT查询。真正的四角色站点隔离用于DEV/CAL/TEST，不将FIT内部自排除夸大为端到端独立外部评价。', '',
              'CAL温度优化90步、lr=.045、T∈[.25,3]；G13_hist 24轮、Adam .05、每轮累积所有CAL站点等权梯度、批块12；G13_NLL 100步、Adam .04；frozen gate 100步、Adam .025、.01门控惩罚。Joint原定10轮、骨干lr3e-5/头lr1e-3、KL .02、门控惩罚 .002。全部保持冻结源码，不按TEST调参。', '',
              '|域|base实际轮数/选中轮|选中DEV MAE(m)|joint轮数|CAL T|G13_hist VIS/RSS|G13_NLL VIS/RSS|', '|---|---:|---:|---:|---:|---|---|']
    for d,r in results.items():
        c,b = r['calibration'],r['base_training']
        hist='/'.join(f'{v:.6f}' for v in c['g13_historical_surrogate_coefficients'])
        nll='/'.join(f'{v:.6f}' for v in c['g13_full_nll_coefficients'])
        lines.append(f"|{d.upper()}|{len(b['history'])}/{b['best_epoch']}|{b['best_dev_mae']:.6f}|{len(r['joint_fit_history'])}|{c['fitted_temperature']:.6f}|{hist}|{nll}|")
    lines += ['', '最终联合后验全局theta（直接只读冻结checkpoint）：' + '；'.join(
        f"{d.upper()} VIS/RSS={p['joint_head']['theta']}" for d,p in parameters.items()) + '。完整theta、gate weight/bias及checkpoint SHA见 `PARAMETER_REVIEW.json`，未对原始RESULT/权重作补写。', '']
    lines += ['', 'base遵循原实现：需比记录best改善>.01m才更新best，在epoch≥20且距best≥patience12时早停；否则到80轮上限，不因曲线继续下降追加预算。FIT基本训练和joint按扫描DataLoader，CAL目标按站点等权。只有一个训练seed，不能声称已估计训练随机性。', '',
              '## 一次TEST：完整性能表', '',
              'MAE、ES、P90/P95均先站点内RSS扫描均值，再站点等权。下表楼宇楼层joint accuracy按RESULT原字段是**扫描加权**；下一表的全部诊断是独立重算的**站点等权**，两者不混用。所有距离单位m。', '',
              '|域|方法|类型|站点宏MAE|站点宏ES|站点P90|站点P95|楼宇楼层准确率(扫描%)|', '|---|---|---|---:|---:|---:|---:|---:|']
    for d,r in results.items():
        for method,(alias,kind) in NAMES.items():
            m=r['test_metrics'][method]
            lines.append(f"|{d.upper()}|{alias}|{kind}|{m['site_macro_mae_m']:.6f}|{m['site_macro_es_m']:.6f}|{m['site_p90_mae_m']:.6f}|{m['site_p95_mae_m']:.6f}|{100*m['building_floor_joint_accuracy']:.3f}|")
    lines += ['', '## 概率/候选诊断（均为事后真值辅助）', '',
              '点预测对所有建筑/楼层的连续分量求混合均值，类别准确率由建筑-楼层概率边缘的argmax另算；都未用真楼层筛选。TUJI canonical数据只有一个建筑/楼层类别，100%类别准确率在该域是平凡结果。', '',
              '正确邻域=同建筑同楼层且锚点XY距离≤5m。所有策略全FIT支持，所以覆盖不会因概率修正而改变。Top1/Top10是概率排序后的邻域命中；温度正变换不改变候选排序，三种温度控制逐位置Top1/Top10已验证完全相同。', '',
              '|域|方法|站点joint准确率%|近邻覆盖%|近邻质量|Top1近邻%|Top10近邻%|熵(nats)|最近连续中心oracle(m)|真楼层概率|', '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for d,a in audit.items():
        for method,(alias,kind) in NAMES.items():
            m=a['summaries'][method]['site_macro_diagnostics']
            lines.append(f"|{d.upper()}|{alias}|{100*m['joint_correct']:.3f}|{100*m['coverage']:.3f}|{m['mass']:.6f}|{100*m['top1_near']:.3f}|{100*m['top10_near']:.3f}|{m['entropy']:.6f}|{m['nearest_center']:.6f}|{m['floor']:.6f}|")
    lines += ['', '最近连续中心误差是用真实XY在实际Student-t分量中心中取最小，代码不要求该分量所属楼层正确。这是不可部署的oracle诊断，既不是任意凸混合点决策误差下界，也不证明概率已给正确空间模式足够质量。邻域质量、覆盖、Top-k、真实楼层概率均不能作为部署门控输入。', '',
              '## 逐位置配对和伤害', '',
              '`PAIRED_SITE_ANALYSIS.json`保存每域全部28对策略的同序site keys、逐位置MAE/ES差值、改善/恶化数、最大伤害与改善。这里没有新增p值或选择晋级门槛，均为描述性结果。差值=候选−参照，负数改善。', '']
    focus = [('v17_single_temp_fit','v17_single_g13_historical_cal'),
             ('v17_single_temp_fit','v17_single_g13'),
             ('v17_single_g13','v17_single_unified_frozen'),
             ('v17_single_temp_fit','unified_joint'),
             ('v17_single_unified_frozen','unified_joint'),
             ('unified_joint_no_evidence','unified_joint')]
    lines += ['|域|候选−参照|宏MAE差|宏ES差|改善/恶化/相同站点|', '|---|---|---:|---:|---:|']
    for d in results:
        for left,right in focus:
            # itertools enumeration follows NAMES order.
            key=f'{right} minus {left}'
            comparison=pairs[d]['comparisons'][key]
            lines.append(f"|{d.upper()}|{NAMES[right][0]}−{NAMES[left][0]}|{comparison['mae_difference_m']:+.6f}|{comparison['es_difference_m']:+.6f}|{comparison['improved_sites']}/{comparison['worsened_sites']}/{comparison['unchanged_sites']}|")
    lines += ['', '主要新候选Joint相对冻结Unified_frozen的最大伤害（每域保留前三，不只报告最好位置）：', '',
              '|域|物理站点|参照MAE|Joint MAE|MAE差|ES差|', '|---|---|---:|---:|---:|---:|']
    for d in results:
        comparison=pairs[d]['comparisons']['unified_joint minus v17_single_unified_frozen']
        for harm in comparison['worst_harms'][:3]:
            lines.append(f"|{d.upper()}|`{harm['site']}`|{harm['reference_mae']:.6f}|{harm['candidate_mae']:.6f}|{harm['delta_mae_m']:+.6f}|{harm['delta_es_m']:+.6f}|")
    lines += ['', '## 第一性原理机制判断', '']
    for d,a in audit.items():
        raw=a['summaries']['v17_single_raw']; temp=a['summaries']['v17_single_temp_fit']
        hist=a['summaries']['v17_single_g13_historical_cal']; nll=a['summaries']['v17_single_g13']
        frozen=a['summaries']['v17_single_unified_frozen']; joint=a['summaries']['unified_joint']; no=a['summaries']['unified_joint_no_evidence']
        diag=raw['site_macro_diagnostics']
        lines += [f"### {d.upper()}", '',
                  f"**候选支持与排序**：全FIT同层5m邻域的站点宏覆盖为{100*diag['coverage']:.3f}%，Raw Top1/Top10命中{100*diag['top1_near']:.3f}%/{100*diag['top10_near']:.3f}%，邻域质量{diag['mass']:.6f}。因本轮支持=all，不能归咎Top192排除；覆盖不足指真实FIT几何不足，而不是连续模型必然无法定位。Raw最近连续中心oracle为{diag['nearest_center']:.6f}m，部署均值MAE为{raw['site_macro_mae_m']:.6f}m；两者差距描述概率/点决策与可用中心之间的空间错配，不是因果识别。", '',
                  f"Top10−Top1相差{100*(diag['top10_near']-diag['top1_near']):.3f}个百分点，表示这部分扫描的5m近邻在Top10但没有成为Top1（先站点内平均）。这是有可用近邻仍排序不优的直接描述，不能等同于所有混合均值误差由Top1引起。", '',
                  f"**固定骨干观测的增量**：T_CAL MAE/ES={temp['site_macro_mae_m']:.6f}/{temp['site_macro_es_m']:.6f}，G13_hist={hist['site_macro_mae_m']:.6f}/{hist['site_macro_es_m']:.6f}，G13_NLL={nll['site_macro_mae_m']:.6f}/{nll['site_macro_es_m']:.6f}，Unified_frozen={frozen['site_macro_mae_m']:.6f}/{frozen['site_macro_es_m']:.6f}。相同连续分量下的变化只能来自概率动作；但G13统计和神经网络使用同一RSS，不能称独立新传感信息。CAL目标和自由参数量不同，无法只凭这组结果证明门控机制或真实隐含anchor概率校准。", '',
                  f"**联合学习与连续分量**：Joint MAE/ES={joint['site_macro_mae_m']:.6f}/{joint['site_macro_es_m']:.6f}，P95={joint['site_p95_mae_m']:.6f}；同骨干关闭证据后={no['site_macro_mae_m']:.6f}/{no['site_macro_es_m']:.6f}。Joint最近连续中心oracle={joint['site_macro_diagnostics']['nearest_center']:.6f}m，相比固定base={diag['nearest_center']:.6f}m，骨干/头变化与概率变化并存；不据此宣称连续Student-t分量足够或不足已经得到唯一证实。", '']
        delta=pairs[d]['comparisons']['unified_joint minus v17_single_unified_frozen']
        conclusion=('MAE和ES均下降' if delta['mae_difference_m']<0 and delta['es_difference_m']<0 else
                    'MAE下降而ES未下降，存在精度/分布评分权衡' if delta['mae_difference_m']<0 else
                    'MAE未改善，不能将其称为超越冻结可靠性校准')
        lines += [f"Joint相对Unified_frozen：{conclusion}；{delta['improved_sites']}个位置改善、{delta['worsened_sites']}个位置恶化。即使宏平均有利，也必须结合上表伤害和尾部，不视为所有位置可靠性改善。", '']
        lines += [f"尾部变化（Joint−Unified_frozen）：P95 {joint['site_p95_mae_m']-frozen['site_p95_mae_m']:+.6f}m，P99 {joint['site_p99_mae_m']-frozen['site_p99_mae_m']:+.6f}m，最坏站点 {joint['worst_site_mae_m']-frozen['worst_site_mae_m']:+.6f}m。正值就是对应尾部恶化，不能因平均改善予以豁免。", '']
        if results[d]['calibration']['g13_full_nll_coefficients'] == [0.,0.]:
            lines += ['**冻结门控的结构性退化**：CAL-NLL将两个全局theta都拟合为0。因为eta=theta×(1+.75 tanh(gate))且frozen阶段theta固定，任意门控都无法产生证据修正，其损失对gate的证据梯度为0。G13_NLL、Unified_frozen、T=.6的全部保存指标精确相同。不能把该对照当作一个实际生效的动态门控，也不能由此证明AP观测特征没有信息。参数为0是本次预设优化/约束的合法结果，本轮不通过重置系数或改目标修复科学负结果。', '']
        if parameters[d]['joint_head']['theta']==[0.,0.]:
            lines += ['**联合后验也退化为无观测证据**：最终FIT联合训练theta=(0,0)，即使gate权重本身非零，也被全局零系数完全关闭。Joint与Joint_no_ev的每项TEST指标精确相同。因此该域的任何骨干变化都不能称为最终观测证据改善；完整参数已另存只读提取记录。', '']
        if joint['site_macro_mae_m'] > raw['site_macro_mae_m'] or joint['site_macro_es_m'] > raw['site_macro_es_m']:
            lines += [f"相对Raw，Joint宏MAE差{joint['site_macro_mae_m']-raw['site_macro_mae_m']:+.6f}m、ES差{joint['site_macro_es_m']-raw['site_macro_es_m']:+.6f}m；扫描加权楼宇楼层准确率差{100*(joint['row_building_floor_joint_accuracy']-raw['row_building_floor_joint_accuracy']):+.3f}个百分点。存在不利指标，不能称为无代价提高。", '']
    lines += ['**本轮能与不能识别的内容**：', '',
              '- 全支持控制和同分量概率修正可检验排序/分配是否有可利用空间，但不能把低oracle中心误差直接变为可部署性能，也不能由宏平均将少数严重错误掩盖。',
              '- 纯温度保持Top1/Top10顺序，MAE/ES变化是质量集中程度与混合均值的变化；G13改变顺序才涉及额外FIT统计证据的重排。观测增益不等于新增独立信息。',
              '- Joint同时更新表示、检索与连续分量，缺少独立训练的no-evidence骨干、固定联合中心的额外反事实，无法唯一分离表示改善、中心改善与观测门控贡献。本轮不补做这些新模型。',
              '- 联合阶段的mixture_loss只监督二维XY，没有继续使用base training_loss中的建筑/楼层分类项；同XY跨楼层模式的权重可能在XY目标下不可辨识。UTS类别准确率下降、UJI从96.383%降至56.489%与该目标缺少楼层约束一致，但没有单因素对照，不能声称已证明唯一原因。二维MAE接近不代表楼层可靠性保持。',
              '- 位置均值用于二维平方风险更自然，非欧氏MAE的理论最优动作；所有方法共享同类混合均值动作，未在TEST挑选中位数等解码器。',
              '- ES为df4独立坐标Student-t连续混合的MC近似，种子由seed+900+batch_start固定，两流各32次，方法间共用随机数组；小ES差异没有MC误差条，不能当独立显著性。G13_hist训练的离散代理ES不是最终连续ES。',
              '- 未证明真实隐含锚点概率校准；单seed、每域不足140个TEST站点、随机站点而非空间块隔离、历史语料暴露和扫描相关性都限制结论。', '',
              '不启动新模型、集成扩员、温度/门控搜索或任何官方外部测试。后续若考虑进入核心机制，应预注册独立位置/采集与公平训练预算，在新未观察评估中直接对照T_CAL及固定骨干G13；本轮结果本身不授予部署或正式晋级。', '',
              '就本轮实测而言，UTS已经构成Joint相对Raw宏MAE、ES及类别准确率恶化的反例，所以不能把单成员统一联合后验描述为三域一致可靠的替代。固定骨干CAL校准、联合训练和最终点决策是不同干预，必须按完整表格分别判断；不据TEST数值回头更改任何一种方法。', '',
              '如后续只选一个机制做独立验证，优先检验“联合微调时保留FIT建筑/楼层监督”这一单因素，其余后验、预算和组件固定；当前仅指出目标可辨识性问题，不在已观察TEST上补跑或挑权重。', '',
              '## 只读核验与交付', '',
              '|域|站点/扫描|JSON重汇总最大绝对差|checkpoint SHA256|', '|---|---:|---:|---|']
    for d,a in audit.items(): lines.append(f"|{d.upper()}|{a['sites']}/{a['scans']}|{a['maximum_reaggregation_absolute_difference']:.3g}|`{a['checkpoint_sha256']}`|")
    lines += ['', '独立审计只读RESULT/SITE_DIAGNOSTICS/checkpoint，没有再次前向TEST：四角色与预检SHA匹配；八策略站点序列/扫描数相同；保存的站点平均重新计算宏MAE/ES、P90/P95及扫描加权诊断均匹配（浮点32/64汇总差≤2e-5m）。SITE文件没有逐扫描原始向量，因此没有声称独立从逐扫描重算。所有JSON可解析且无NaN/Inf。', '',
              '同base六策略的oracle中心向量完全相同，同joint两策略也相同；源码确认每组内复用同一次提取的完整center/scale张量。joint checkpoint的encoder/anchor_keys/residual_head均实际变化，固定anchor几何/原型/AP mask保持不变。冻结头全局theta与G13_NLL相同。原126个历史科学模块及全部输入文件在最终完整性核验中与冻结前哈希比较。', '',
              '没有强制所有CUDA内核确定性，固定seed不能保证跨机器/运行逐位相同。成功科学命令stderr是否为空及全部退出码以原日志为准；没有把回归演示失败或编译错误当成通过。耗时为本机命令墙钟，不是纯GPU活跃算力。', '',
              '三域正式目录（均含RESULT.json、SITE_DIAGNOSTICS.json、CHECKPOINTS.pt和TEST_INVOCATION.lock）：', '']
    for d,path in freeze['formal_outputs'].items():
        lines += [f'- {d.upper()}：`{ROOT/path}`；checkpoint：`{ROOT/path/"CHECKPOINTS.pt"}`。']
    lines += ['', '公共执行目录：`'+str(AUDIT)+'`。包含METHOD_FREEZE、DATA_AUDIT、SMOKE/FULL_RESULT_AUDIT、PAIRED_SITE_ANALYSIS、FIX_RECORD、完整执行记录与日志。所有权重留本地，用户另行安排产物上传。', '']
    path=AUDIT/'RUN_REPORT_MSEP_SINGLE_POSTERIOR.md'
    with path.open('x',encoding='utf-8') as handle: handle.write('\n'.join(lines))
    with (AUDIT/'PARAMETER_REVIEW.json').open('x',encoding='utf-8') as handle:
        json.dump(parameters,handle,ensure_ascii=False,indent=2,allow_nan=False)
    print(path)


if __name__ == '__main__':
    main()
