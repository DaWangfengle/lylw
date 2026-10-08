import sys
sys.dont_write_bytecode=True
import datetime, hashlib, json, re
from pathlib import Path
ROOT=Path.cwd(); EVIDENCE=ROOT/'fhm_dapp_tim/evidence/v17_g2_mechanism'; AUDIT=EVIDENCE/'run_audit'
def load(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def write_new(p,value):
    with p.open('x',encoding='utf-8',newline='\n') as f: json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False); f.write('\n')
def table(headers,rows): return '\n'.join(['| '+' | '.join(map(str,headers))+' |','| '+' | '.join('---' for x in headers)+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def main():
    now=datetime.datetime.now(datetime.timezone.utc)
    initial=load(AUDIT/'INITIAL_EXECUTION_AUDIT.json'); failure=load(AUDIT/'REPORT_RENDER_ORDER_FAILURE.json'); hand=load(AUDIT/'INDEPENDENT_COORDINATE_REVIEW.json'); residual=load(AUDIT/'CORRELATION_RESIDUAL_REVIEW.json')
    stages=[]; warnings=[]; traces=[]
    for name in ('compile','pytest','preflight','analyze','check','independent-review'):
        record=load(AUDIT/(name+'_attempt01.json'))
        for channel in ('stdout','stderr'):
            path=Path(record[channel+'_path'])
            assert sha(path)==record[channel+'_sha256']
            for line in path.read_text(encoding='utf-8',errors='replace').splitlines():
                if 'Warning' in line or 'warning' in line: warnings.append({'stage':name,'channel':channel,'line':line})
        trace=load(Path(record['access_trace_path']))
        assert sha(Path(record['access_trace_path']))==record['access_trace_sha256']
        assert trace['guard_active_before_module_import'] and not trace['blocked_accesses']
        assert trace['checkpoint_or_official_file_open_attempts']==0
        assert all('v17_g2_mechanism' in x.lower() for x in trace['repository_writes'])
        traces.append({'stage':name,'trace_sha256':record['access_trace_sha256'],'checkpoint_or_official_file_open_attempts':0,'blocked_accesses':0,'repository_read_count':sum(trace['repository_reads'].values()),'repository_write_count':sum(trace['repository_writes'].values()),'descriptor_open_events_without_path':trace['descriptor_open_events_without_path']})
        stages.append(record)
    assert [r['exit_code'] for r in stages[:5]]==[0,0,0,0,1]
    for a,b in zip(stages,stages[1:]): assert a['ended_utc']<=b['started_utc']
    test_stdout=Path(stages[1]['stdout_path']).read_text(encoding='utf-8')
    assert re.search(r'\b19 passed\b',test_stdout)
    assert len(list(AUDIT.glob('analyze_attempt*.json')))==1
    assert (AUDIT/'ANALYZE_COMMAND_STARTED.json').exists()
    for name,entry in initial['protected_historical_text_assets'].items(): assert sha(ROOT/name)==entry['sha256'],name
    for name,entry in initial['protected_model_metadata'].items():
        p=ROOT/name
        assert p.stat().st_size==entry['bytes'] and p.stat().st_mtime_ns==entry['mtime_ns'],name
    input_after={name:sha(ROOT/name) for name in initial['input_sha256']}
    assert input_after==initial['input_sha256']
    source_after={name:sha(ROOT/name) for name in initial['initial_G2_sources']}
    assert all(source_after[name]==entry['sha256'] for name,entry in initial['initial_G2_sources'].items())
    assert sha(EVIDENCE/'G2_MECHANISM_AUDIT.json')==failure['immutable_output_sha256']
    assert sha(EVIDENCE/'RUN_REPORT_V17_G2.md')==failure['immutable_report_sha256']
    result=load(EVIDENCE/'G2_MECHANISM_AUDIT.json')
    canonical=dict(result); canonical.pop('result_sha256')
    assert hashlib.sha256(json.dumps(canonical,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()==result['result_sha256']
    assert result['g1_formal_outcome_unchanged']['status']=='STOP_GEOMETRY_DENSITY_MECHANISM'
    assert result['g1_formal_outcome_unchanged']['new_confirmation_performed'] is False
    assert result['outer_test_predictions']==result['official_validation_predictions']==result['external_predictions']==0
    artifacts={p.relative_to(ROOT).as_posix():sha(p) for p in [EVIDENCE/'G2_MECHANISM_AUDIT.json',EVIDENCE/'RUN_REPORT_V17_G2.md',AUDIT/'INDEPENDENT_COORDINATE_REVIEW.json',AUDIT/'CORRELATION_RESIDUAL_REVIEW.json',AUDIT/'REPORT_RENDER_ORDER_FAILURE.json',AUDIT/'REPORT_RENDER_ORDER_FAILURE.diff',AUDIT/'PROPOSED_RENDER_ORDER_SOURCE_FIX.diff',AUDIT/'PROPOSED_RENDER_ORDER_REGRESSION_TEST.diff',AUDIT/'PROPOSED_RENDER_ORDER_REPAIR.json']}
    tests={'status':'ANALYZE_COMPLETE_OFFICIAL_CHECK_FAILED_STOP','recorded_utc':now.isoformat(),'recorded_local_time':now.astimezone(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'client_timezone':'Asia/Shanghai','environment':{k:initial[k] for k in ('python_executable','python_version','platform','versions')},'execution_mode':'CPU only; CUDA_VISIBLE_DEVICES=-1; no checkpoints loaded','prescribed_commands_completed_in_order':True,'formal_pipeline_all_exit_codes_zero':False,'test_counts':{'passed':19,'failed':0,'skipped':0,'xfailed':0},'runtime_seconds_prescribed_commands':sum(x['runtime_seconds'] for x in stages[:5]),'runtime_seconds_including_independent_review':sum(x['runtime_seconds'] for x in stages),'wall_clock_seconds_since_initial_audit':(now-datetime.datetime.fromisoformat(initial['recorded_utc'])).total_seconds(),'stage_records':stages,'runtime_warnings':warnings,'operational_exceptions':initial['operational_exceptions'],'known_failure':failure,'known_numerical_interpretation_limitations':residual['limitations'],'new_G2_source_or_test_repairs_applied':False,'analyze_execution_count':1,'check_execution_count':1,'Git_commands_executed':0,'artifact_sha256':artifacts}
    write_new(AUDIT/'TESTS_AND_EXECUTION.json',tests)
    review={'status':'IMMUTABLE_INPUTS_AND_COORDINATE_VALUES_VERIFIED_OFFICIAL_CHECK_FAILED','recorded_utc':now.isoformat(),'official_check_passed':False,'formal_execution_stopped':True,'official_failure_preserved':failure,'G1_input_sha256_before':initial['input_sha256'],'G1_input_sha256_after':input_after,'all_G1_input_bytes_unchanged':True,'G2_source_design_tests_protocol_sha256_before':{k:v['sha256'] for k,v in initial['initial_G2_sources'].items()},'G2_source_design_tests_protocol_sha256_after':source_after,'all_G2_sources_unchanged':True,'G2_output_sha256':sha(EVIDENCE/'G2_MECHANISM_AUDIT.json'),'G2_result_self_sha256':result['result_sha256'],'G2_report_sha256':sha(EVIDENCE/'RUN_REPORT_V17_G2.md'),'G2_outputs_not_overwritten':True,'protected_historical_text_assets_unchanged':len(initial['protected_historical_text_assets']),'protected_model_metadata_unchanged':len(initial['protected_model_metadata']),'model_verification_scope':'Size and mtime equality only; checkpoint content not opened and no weights loaded. Prior G1 checkpoint SHA256 records were not recalculated.','G1_original_stop_unchanged':True,'G1_original_gamma':0.7,'G1_original_gain_m':result['roles']['confirmation']['macro_gain_m'],'G1_original_familywise_MAE_upper_m':result['g1_formal_outcome_unchanged']['confirmation_mae_familywise_upper_m'],'role_count_and_no_overlap_proof':{k:hand[k] for k in ('selection_refit','roles','calibration_confirmation_coordinate_overlap','calibration_confirmation_row_overlap','development_groups')},'hand_check_count':hand['hand_check_count'],'hand_check_records':hand['hand_checked_coordinates'],'file_access_trace_reviews':traces,'new_models_trained':0,'checkpoint_files_opened_or_loaded':0,'new_model_predictions':0,'official_validation_file_opens':0,'official_validation_predictions':0,'outer_test_predictions':0,'external_predictions':0,'new_confirmation_performed':False,'new_PASS_or_waiver':False,'unidentifiable_partial_rank_proxy_limitation':residual['limitations'],'proposal_not_applied':True,'Git_commands_executed':0,'artifact_sha256':artifacts,'tests_and_execution_sha256':sha(AUDIT/'TESTS_AND_EXECUTION.json')}
    write_new(AUDIT/'G2_INTEGRITY_REVIEW.json',review)
    cal=result['roles']['calibration']; conf=result['roles']['confirmation']
    paragraphs=[
        '# G2 执行审计与研究解释',
        f'记录时间（Asia/Shanghai）：{now.astimezone(datetime.timezone(datetime.timedelta(hours=8))).isoformat()}',
        '**执行状态：唯一一次 analyze 成功，但正式 check 失败，按指令 STOP。** 本解释另存于审计目录，未覆盖正式 G2 JSON、报告或 G1 结果。',
        '## 执行与失败事实',
        '指定五条命令已依次执行。编译、19 项测试、preflight 和 analyze 退出码均为 0；check 退出码为 1。根因是 JSON 字典排序导致相关性表格的六行渲染顺序改变。全部报告行和数值一致，但不把独立审计通过宣称为正式 check 通过。原始失败日志、差异、未应用补丁及隔离渲染证明均保留。',
        f'指定命令累计运行 {tests["runtime_seconds_prescribed_commands"]:.3f} 秒；加独立坐标复核为 {tests["runtime_seconds_including_independent_review"]:.3f} 秒。运行时 warning 日志匹配数为 {len(warnings)}；另记录一次审计工具写入异常、报告顺序 bug 和下述 partial rank 可识别性限制。',
        '## 数据、角色和独立复核',
        '595 个 selection_refit 训练坐标 / 12,717 行；calibration 72 坐标 / 1,529 行；confirmation 79 坐标 / 1,625 行。两侧物理坐标及行索引交集均为 0。按精确 (BUILDINGID, FLOOR, LONGITUDE, LATITUDE) 键排序，不能用 CSV 行序代替。全部 151 坐标的 10m 支持数、d1/d3/d5、楼宇楼层训练锚点数、RSS 行数和原始 G1 MAE/ES 已独立核对；每个原始变体的向量均值都复原官方 G1 macro 指标。',
        '手工复核位置在 analyze 前固定为每侧排序后的首个、中间、末个，共六个。使用仅坐标列的原始 CSV 读取、原始 V17 角色构造、独立复写的固定哈希 split、Python math.hypot 距离与原始 G1 JSON 索引，不调用 G2 geometry_features 或 risk_diagnostics。完整坐标、原始行号、10m 内所有训练键及最近五个锚点见 INDEPENDENT_COORDINATE_REVIEW.json。',
        table(['角色','排序索引（0起）','精确物理坐标键','RSS行数','10m训练锚点数','d1 / d3 / d5 (m)'],[[x['role'],x['sorted_vector_index_zero_based'],str(x['coordinate_key']),x['source_RSS_row_count'],x['training_anchors_within_10m_count'],' / '.join(f'{v:.6f}' for v in x['independently_calculated_d1_d3_d5_m'])] for x in hand['hand_checked_coordinates']]),
        '## 正向与反向机制证据',
        '**存在与局部稀疏支持相关的描述性迹象，但不能证明几何因果，也不支持普遍改善或新 PASS。** 10m 邻居数与收益的负相关、d3/d5 与收益的正相关，在两侧方向一致，并在 baseline MAE + 楼宇楼层 rank 控制后保留。最近锚点 d1 的调整后相关接近零：不能把全部距离代理都说成稳定有效。RSS 行数在确认组的调整后关系也接近零。',
        table(['代理','Cal Pearson','Conf Pearson','Cal Spearman','Conf Spearman','Cal partial rank','Conf partial rank'],[[proxy,f'{cal["associations"][proxy]["pearson_with_gain"]:+.6f}',f'{conf["associations"][proxy]["pearson_with_gain"]:+.6f}',f'{cal["associations"][proxy]["spearman_with_gain"]:+.6f}',f'{conf["associations"][proxy]["spearman_with_gain"]:+.6f}',('N/A' if proxy=='training_anchor_count_same_stratum' else f'{cal["associations"][proxy]["partial_rank_after_baseline_and_floor_building"]:+.6f}'),('N/A' if proxy=='training_anchor_count_same_stratum' else f'{conf["associations"][proxy]["partial_rank_after_baseline_and_floor_building"]:+.6f}')] for proxy in ('training_anchors_within_10m','nearest_training_anchor_m','training_anchor_d3_m','training_anchor_d5_m','query_rss_row_count','training_anchor_count_same_stratum')]),
        'baseline MAE 与收益的 Pearson 为 calibration +0.647500、confirmation +0.449294。收益定义为 baseline−G1，baseline 同时参与收益定义；相关性包含数学耦合和回归均值风险。控制风险与楼层后保留关联，仍不足以识别因果；两侧都已公开，因此不能把一致方向当独立重复确认。',
        '本楼宇楼层训练锚点总数是楼宇楼层的确定函数。控制楼宇楼层后其 rank 残差为机器精度下的零，partial rank 不可识别，应解释为 N/A。原始 G2 JSON 的 +0.000154490205 / +0.000012652926 是投影浮点残差产生的微小值，原样保留，但不作为真实调整后效应解释。其他可识别代理的 Pearson、Spearman 和 QR partial rank 均与 G2 数值复核一致。',
        '## 收益集中与恶化位置',
        table(['诊断','Calibration','Confirmation'],[['改善 / 恶化 / 不变位置',f'{cal["improved_groups"]} / {cal["worsened_groups"]} / {cal["unchanged_groups"]}',f'{conf["improved_groups"]} / {conf["worsened_groups"]} / {conf["unchanged_groups"]}'],['MAE 平均收益 (m)',f'{cal["macro_gain_m"]:+.9f}',f'{conf["macro_gain_m"]:+.9f}'],['MAE 中位收益 (m)',f'{cal["median_gain_m"]:+.9f}',f'{conf["median_gain_m"]:+.9f}'],['ES 平均收益 (m)',f'{cal["macro_es_gain_m"]:+.9f}',f'{conf["macro_es_gain_m"]:+.9f}'],['前五收益和 / 全体净收益',f'{cal["net_gain_concentration_top_five"]:.6f}',f'{conf["net_gain_concentration_top_five"]:.6f}'],['RSS行数加权的坐标误差收益 (m)',f'{cal["row_count_weighted_mean_coordinate_error_gain_m"]:+.9f}',f'{conf["row_count_weighted_mean_coordinate_error_gain_m"]:+.9f}']]),
        table(['删除收益最大位置数','Cal 剩余平均收益 (m)','Conf 剩余平均收益 (m)'],[[a['removed_top_gains'],f'{a["remaining_gain_m"]:+.9f}',f'{b["remaining_gain_m"]:+.9f}'] for a,b in zip(cal['top_gain_influence'],conf['top_gain_influence'])]),
        'calibration 删除一个最大收益位置后，剩余收益已为负；confirmation 删除五个后为 −0.024540757m。其 39/79 位置改善、40/79 恶化，中位收益接近零且为负。整体正均值由少量尾部位置贡献，不能替代对恶化位置的解释，也不能把这种事后删除敏感性当作新模型成绩。',
        '## 原始误差与密度四分位（描述性）',
        '四分位按特征排序划分为近等量组，同值用物理键确定顺序。不同箱可能共享边界值；它们不是学习出的线上阈值。',
    ]
    for feature in ('support_density_quartiles','baseline_error_quartiles'):
        paragraphs.append(table([feature+' / role','Q','n','特征范围','Baseline MAE (m)','平均收益 (m)','中位收益 (m)','改善位置'],[[role,x['quartile'],x['size'],f'{x["feature_min"]:.6f}..{x["feature_max"]:.6f}',f'{x["baseline_mae_m"]:.6f}',f'{x["mean_improvement_m"]:+.6f}',f'{x["median_improvement_m"]:+.6f}',x['improved_coordinates']] for role,data in [('calibration',cal),('confirmation',conf)] for x in data[feature]]))
    paragraphs += [
        '## 完整性、哈希与限制',
        table(['文件','SHA256'],[[name,digest] for name,digest in initial['input_sha256'].items()]+[[p.name,sha(p)] for p in (EVIDENCE/'G2_MECHANISM_AUDIT.json',EVIDENCE/'RUN_REPORT_V17_G2.md',AUDIT/'TESTS_AND_EXECUTION.json',AUDIT/'G2_INTEGRITY_REVIEW.json')]),
        f'{len(initial["protected_historical_text_assets"])} 项历史文本资产 SHA256 未变，{len(initial["protected_model_metadata"])} 项模型文件大小和 mtime 未变。未打开权重内容或做新的权重 SHA256，不把元数据检查冒称为内容哈希复核。所有执行/复核访问轨迹均未尝试读取权重或官方验证文件，历史写入被守卫禁止。G2 源码、测试、设计和协议也保留初始字节。',
        'G1 gamma=0.70、正式 STOP、原始 MAE familywise 上界 +0.012352838m 均未变。G2 没有重新确认、没有推断 p-value、没有新 PASS 或 waiver。用户真坐标只用于离线诊断，部署时不可观测，不能据此构造线上密度门控。没有训练、加载模型、模型推断、outer/official/external 预测或 Git 命令。',
        '当前证据只能提出局部几何与尾部纠错的探索性线索。最近锚点单一解释缺乏调整后的稳定支持，收益集中也限制普适性结论。若要继续研究，必须另行独立预注册假设并使用真正新鲜、兼容的样本；本轮不启动新架构或新评估。历史 5.725985m 使用不同训练锚点和 outer cohort，不与这里做同测试集优劣比较。未完成外部泛化、因果识别或 Q1 完整证据链。',
        '正式 check 的实现失败仍待用户授权的最小非科学性修复例外。源代码与回归补丁只作为未应用审计方案保存；不会私自修改、覆盖结果或重跑分析。',
    ]
    interpretation=AUDIT/'G2_EXECUTION_INTERPRETATION.md'
    with interpretation.open('x',encoding='utf-8',newline='\n') as f: f.write('\n\n'.join(paragraphs)+'\n')
    final={'status':'REQUIRED_AUDITS_WRITTEN_FORMAL_CHECK_FAIL_PRESERVED','recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'formal_check_passed':False,'analyze_execution_count':1,'hand_checked_coordinates':6,'artifact_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in (EVIDENCE/'G2_MECHANISM_AUDIT.json',EVIDENCE/'RUN_REPORT_V17_G2.md',AUDIT/'TESTS_AND_EXECUTION.json',AUDIT/'G2_INTEGRITY_REVIEW.json',interpretation)},'no_result_or_source_overwrite':True,'no_Git_operations':True}
    write_new(AUDIT/'FINAL_AUDIT_STATE.json',final)
    print(json.dumps(final,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
