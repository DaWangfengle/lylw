"""Read-only status summary; never launches training or predictions."""
import json
from pathlib import Path
p=Path(__file__).resolve().parent
completed=set()
failed=[]
interrupted=[]
for f in sorted(p.glob("train-member-*_attempt*.json")):
    s=json.loads(f.read_text(encoding="utf-8"))
    if s["exit_code"] == 0:
        completed.add(s["member_index"])
    elif s["exit_code"] is None:
        interrupted.append({"member":s["member_index"],"attempt":s["attempt"]})
    else:
        failed.append({"member":s["member_index"],"attempt":s["attempt"],"exit_code":s["exit_code"]})
active=[]
for f in sorted(p.glob("train-member-*_attempt*.stdout.log")):
    if f.with_name(f.name.replace(".stdout.log",".json")).exists():
        continue
    progress=[]
    for line in f.read_text(encoding="utf-8",errors="replace").splitlines():
        if line.startswith('{"refit_epoch"'):
            s=json.loads(line)
            progress.append({"epoch":s["refit_epoch"],"total_epochs":s["refit_total_epochs"]})
    active.append({"log":f.name,"last_progress":progress[-1] if progress else None})
print(json.dumps({"completed_members":len(completed),"completed_indices":sorted(completed),
                  "active":active,"failed_members":failed,"interrupted_attempts":interrupted,
                  "calibration_exists":(p.parent/"calibration_v17_g1.json").exists(),
                  "confirmation_exists":(p.parent/"confirmation_v17_g1.json").exists()}))
