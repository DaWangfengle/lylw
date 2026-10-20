"""Retain complete raw command streams without changing scientific commands."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

def now():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

if __name__ == '__main__':
    label, *command = sys.argv[1:]
    assert command and '/' not in label and '\\' not in label
    record_path = HERE / 'EXECUTION_RECORD.json'
    if record_path.exists():
        record = json.loads(record_path.read_text(encoding='utf-8'))
    else:
        record = {'started_at': now(), 'cwd': str(ROOT), 'python': sys.executable,
                  'instruction_source': 'V17_SPATIAL_COMPOSITION_CODEX_RUN.txt',
                  'requested_attachment_missing': 'D:/liulanqi/V17_SPATIAL_COMPOSITION_CODEX_RUN (1).txt',
                  'commands': []}
    assert not any(c['label'] == label for c in record['commands']), label
    stdout, stderr = HERE / f'{label}.stdout.log', HERE / f'{label}.stderr.log'
    entry = {'label': label, 'command': command, 'cwd': str(ROOT), 'started_at': now(),
             'stdout': stdout.name, 'stderr': stderr.name, 'status': 'RUNNING'}
    record['commands'].append(entry)
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    started = time.perf_counter()
    with stdout.open('xb') as out, stderr.open('xb') as err:
        process = subprocess.run(command, cwd=ROOT, stdout=out, stderr=err)
    entry.update(status='COMPLETE', exit_code=process.returncode, ended_at=now(),
                 elapsed_seconds=time.perf_counter() - started,
                 stdout_sha256=sha(stdout), stderr_sha256=sha(stderr))
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(entry, ensure_ascii=False), flush=True)
    sys.exit(process.returncode)
