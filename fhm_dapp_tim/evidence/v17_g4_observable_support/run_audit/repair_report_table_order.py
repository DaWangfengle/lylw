"""Repair only the derived G4 Markdown table order; preserve all frozen inputs."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,difflib
from fhm_dapp_tim.fhm_dapp import uji_v17_g4_observable_support as g4
root=g4.REPO;base=g4.EVIDENCE;audit=base/'run_audit'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
immutable=[base/'manifest_v17_g4.json',base/'calibration_v17_g4.json',base/'internal_confirmation.lock',base/'confirmation_v17_g4.json',Path(g4.__file__),root/'tests/test_uji_v17_g4_observable_support.py',base/'DESIGN_V17_G4.md']
before={p.relative_to(root).as_posix():sha(p) for p in immutable}
manifest=json.loads(g4.MANIFEST.read_text(encoding='utf-8'))
assert g4._source_hashes()==manifest['source_sha256']
cal=json.loads(g4.CALIBRATION.read_text(encoding='utf-8'));conf=json.loads(g4.CONFIRMATION.read_text(encoding='utf-8'))
old=g4.REPORT.read_bytes();oldtext=old.decode('utf-8');canonical=g4.make_report(cal,conf)
assert hashlib.sha256(old).hexdigest()=='3764a5e6a218902de29c86fab01824a1c5d20336c4b5d6036bdfe975d7d2bdf8'
assert hashlib.sha256(canonical.encode('utf-8')).hexdigest()=='319337740dc92c02a79afe7f8fd8f8b05cc702c1ff68cbc5d826bad4615b29d9'
oldlines=oldtext.splitlines(True);newlines=canonical.splitlines(True)
assert sorted(x for x in oldlines if x.startswith('|'))==sorted(x for x in newlines if x.startswith('|'))
assert [x for x in oldlines if not x.startswith('|')]==[x for x in newlines if not x.startswith('|')]
variant_order=tuple(dict.fromkeys(('baseline','g1','placebo',cal['selected_policy']['policy'])))
original=dict(conf);original['metrics']={name:conf['metrics'][name] for name in variant_order}
assert g4.make_report(cal,original)==oldtext
assert g4.make_report(json.loads(json.dumps(cal,sort_keys=True)),json.loads(json.dumps(conf,sort_keys=True)))==canonical
with (audit/'RUN_REPORT_V17_G4_before_table_order_fix.md').open('xb') as f:f.write(old)
with (audit/'REPORT_TABLE_ORDER_FIX.diff').open('x',encoding='utf-8',newline='\n') as f:f.write(''.join(difflib.unified_diff(oldlines,newlines,fromfile='RUN_REPORT_V17_G4.md original',tofile='RUN_REPORT_V17_G4.md canonical JSON order')))
g4.REPORT.write_bytes(canonical.encode('utf-8'))
assert g4.REPORT.read_text(encoding='utf-8')==g4.make_report(cal,conf)
after={p.relative_to(root).as_posix():sha(p) for p in immutable};assert before==after
record={'recorded_utc':datetime.now(timezone.utc).isoformat(),'failed_stage':'check_after_confirm_attempt01','cause':'Original writer uses baseline,g1,placebo,selected insertion order; sorted JSON decoder changes metrics key order','fix':'Only rearrange derived current G4 Markdown table to frozen JSON readback order; preserve exact original report and diff','report_before_sha256':hashlib.sha256(old).hexdigest(),'report_after_sha256':sha(g4.REPORT),'all_numeric_values_and_non_table_text_identical':True,'serialization_roundtrip_regression_passed':True,'frozen_files_before_sha256':before,'frozen_files_after_sha256':after,'experiment_source_modified':False,'manifest_models_calibration_confirmation_lock_or_results_modified':False,'calibration_or_confirmation_rerun':False,'scientific_decision':conf['decision'],'source_render_iteration_bug_not_changed_after_manifest':True}
with (audit/'REPORT_TABLE_ORDER_FIX.json').open('x',encoding='utf-8') as f:json.dump(record,f,indent=2);f.write('\n')
print(json.dumps({'status':'DERIVED_REPORT_ORDER_REPAIRED','frozen_files_unchanged':True,'report_before_sha256':record['report_before_sha256'],'report_after_sha256':record['report_after_sha256'],'scientific_decision':conf['decision']}))
