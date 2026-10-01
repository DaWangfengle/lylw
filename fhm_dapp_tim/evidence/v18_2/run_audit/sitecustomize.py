"""Audit V18.2 data access and CUDA runtime without modifying computations."""
import atexit,json,os,sys,time
from pathlib import Path
phase=os.environ.get('V182_AUDIT_PHASE'); folder=os.environ.get('V182_AUDIT_DIR')
if phase and folder:
    folder=Path(folder); root=Path(os.environ['V182_REPO_ROOT'])
    evidence=root/'fhm_dapp_tim/evidence/v18_2'; lock=evidence/'confirmation_consumed.lock'
    counts={'training_csv_reads':0,'confirmation_reads':0,'benchmark_reads':0,'historical_checkpoint_reads':0,'denied':0}
    log=(folder/(phase+'_io.jsonl')).open('a',encoding='utf-8')
    allowed={
      'prepare':{'member_fit','early_stop','selection_refit','calibration','outer_refit','confirmation'},
      'verify-prepare':{'member_fit','early_stop','selection_refit','calibration','outer_refit'},
      'train-selection':{'member_fit','early_stop','selection_refit'},
      'calibrate-freeze':{'calibration'},'stability':{'member_fit','early_stop'},
      'train-confirmation':{'outer_refit'},'confirm':{'confirmation'},
      'train-final':{'outer_refit','confirmation'}}
    def observe(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)): return
        path=os.path.normcase(os.path.abspath(os.fsdecode(args[0]))).replace('\\','/')
        mode,flags=args[1],args[2]
        reading=('r' in mode or '+' in mode) if isinstance(mode,str) else not (flags & os.O_WRONLY)
        kind=None; permitted=True; present=None; purpose=None
        if path.endswith('/trainingdata.csv'):
            kind='training_csv'
            if reading: counts['training_csv_reads']+=1; permitted=phase in {'prepare','verify-prepare'}
        elif path.endswith('/validationdata.csv'):
            kind='pre_exposed_benchmark'
            if reading:
                counts['benchmark_reads']+=1
                permitted=phase=='benchmark' and (folder/'train-final_execution.json').is_file() and (evidence/'confirmation_result_v18_2.json').is_file()
        elif '/role_shards/' in path and path.endswith('.pt'):
            kind='role_shard'
            if reading:
                role=Path(path).stem
                permitted='/evidence/v18_2/' in path and role in allowed.get(phase,set())
                if role=='confirmation':
                    counts['confirmation_reads']+=1; present=lock.is_file()
                    if phase=='prepare': purpose='registered_prepare_hash_only'
                    elif phase=='confirm': permitted=permitted and present; purpose='one_shot_confirmation'
                    elif phase=='train-final': permitted=permitted and present and (evidence/'confirmation_result_v18_2.json').is_file(); purpose='frozen_final_refit_after_confirmation'
                    else: permitted=False
        elif '/models/' in path and path.endswith(('.pt','.pth','.ckpt')):
            kind='checkpoint'
            if reading and '/models/v18_2/' not in path:
                counts['historical_checkpoint_reads']+=1; permitted=False
        elif '/evidence/' in path and '/evidence/v18_2/' not in path and path.endswith('.json') and reading:
            kind='historical_evidence'; permitted=False
        elif path.endswith('/confirmation_consumed.lock'): kind='confirmation_lock'
        if kind is None: return
        log.write(json.dumps({'time_ns':time.time_ns(),'phase':phase,'kind':kind,'path':path,'mode':mode,'reading':reading,'permitted':permitted,'confirmation_lock_present':present,'purpose':purpose})+'\n'); log.flush()
        if not permitted: counts['denied']+=1; raise PermissionError('V18.2 audit denied '+kind+' in '+phase)
    sys.addaudithook(observe)
    def finish():
        record={'phase':phase,'io_counts':counts}; t=sys.modules.get('torch')
        if t is not None:
            try:
                record.update(torch_version=t.__version__,cuda_version=t.version.cuda,deterministic_algorithms=t.are_deterministic_algorithms_enabled(),deterministic_warn_only=t.is_deterministic_algorithms_warn_only_enabled(),flash_sdpa=t.backends.cuda.flash_sdp_enabled(),memory_efficient_sdpa=t.backends.cuda.mem_efficient_sdp_enabled(),math_sdpa=t.backends.cuda.math_sdp_enabled(),cudnn_benchmark=t.backends.cudnn.benchmark,cudnn_deterministic=t.backends.cudnn.deterministic,cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
                if t.cuda.is_initialized(): record.update(gpu=t.cuda.get_device_name(0),peak_cuda_allocated_bytes=t.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=t.cuda.max_memory_reserved())
            except Exception as e: record['runtime_audit_error']=str(e)
        (folder/(phase+'_runtime.json')).write_text(json.dumps(record,indent=2),encoding='utf-8'); log.close()
    atexit.register(finish)
