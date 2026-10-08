import atexit, collections, json, os, sys
_root=os.environ.get('G2_AUDIT_ROOT')
_trace=os.environ.get('G2_ACCESS_TRACE')
if _root and _trace:
    _root=os.path.normcase(os.path.abspath(_root))
    _allowed=os.path.join(_root,'fhm_dapp_tim','evidence','v17_g2_mechanism')
    _reads=collections.Counter(); _writes=collections.Counter(); _blocked=[]; _fd_count=0
    def _inside(p,d): return p==d or p.startswith(d+os.sep)
    def _hook(event,args):
        global _fd_count
        if event not in {'open','os.remove','os.rename','os.rmdir','shutil.rmtree'}: return
        raw=args[0]
        if not isinstance(raw,(str,bytes,os.PathLike)):
            if event=='open': _fd_count+=1
            return
        p=os.path.normcase(os.path.abspath(os.fsdecode(raw)))
        forbidden_input=os.path.basename(p)=='validationdata.csv' or os.path.splitext(p)[1] in {'.pt','.pth','.ckpt','.onnx','.safetensors'}
        if event=='open':
            mode,flags=args[1],args[2]
            writing=bool(isinstance(mode,str) and any(c in mode for c in 'wax+'))
            if isinstance(flags,int): writing=writing or bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
            if forbidden_input or (writing and _inside(p,_root) and not _inside(p,_allowed)):
                _blocked.append({'event':event,'path':p,'writing':writing})
                raise RuntimeError('G2 boundary blocked file access: '+p)
            if _inside(p,_root): (_writes if writing else _reads)[os.path.relpath(p,_root)]+=1
        elif _inside(p,_root) and not _inside(p,_allowed):
            _blocked.append({'event':event,'path':p})
            raise RuntimeError('G2 boundary blocked historical filesystem mutation: '+p)
    sys.addaudithook(_hook)
    def _finish():
        record={'status':'READ_ONLY_ACCESS_GUARD_COMPLETE','guard_active_before_module_import':True,'blocked_accesses':_blocked,'repository_reads':dict(_reads),'repository_writes':dict(_writes),'descriptor_open_events_without_path':_fd_count,'checkpoint_or_official_file_open_attempts':len([x for x in _blocked if os.path.basename(x['path'])=='validationdata.csv' or os.path.splitext(x['path'])[1] in {'.pt','.pth','.ckpt','.onnx','.safetensors'}]),'scope':'CPython audit events, supplemented by source-call review and immutable file comparisons.'}
        with open(_trace,'x',encoding='utf-8',newline='\n') as f: json.dump(record,f,indent=2,ensure_ascii=False,allow_nan=False); f.write('\n')
    atexit.register(_finish)
