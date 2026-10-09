"""Pre-manifest G6-only repairs; retains original bytes and unified diff."""
from pathlib import Path
import difflib
import json
import sys
ROOT = Path(__file__).resolve().parents[4]
AUDIT = Path(__file__).resolve().parent
CODE = ROOT / "fhm_dapp_tim/fhm_dapp/uji_v17_g6_calibration.py"
TEST = ROOT / "tests/test_uji_v17_g6_calibration.py"
assert not (AUDIT.parent / "manifest_v17_g6.json").exists()
def read(path):
    return path.read_bytes().decode("utf-8").replace("\r\n", "\n")
def write(path, text):
    ending = "\r\n" if b"\r\n" in path.read_bytes() else "\n"
    path.write_bytes(text.replace("\n", ending).encode("utf-8"))
REGRESSIONS = r'''

def test_original_double_sites_cannot_merge_during_float32_adapter(monkeypatch):
    scans = [
        SimpleNamespace(coords=np.array([100000000., 1.], dtype=np.float64),
                        measured=np.array([[11., -65.], [12., -75.]])),
        SimpleNamespace(coords=np.array([100000001., 1.], dtype=np.float64),
                        measured=np.array([[11., -65.], [12., -75.]])),
    ]
    monkeypatch.setattr(g6, "loadmat", lambda *a, **k: {
        "results_train": SimpleNamespace(Wifi=scans)})
    with pytest.raises(g6.G6Error, match="float32.*physical site"):
        g6.load_tuji_train_variable_only()


def test_report_is_identical_after_sorted_json_roundtrip():
    record = {
        "selected_policy": {"policy": "T0.8_S0.75"},
        "candidate_metrics": {
            "T1_S1": {"mae_2d_m": 5., "energy_score_2d_m": 4.},
            "T0.8_S0.75": {"mae_2d_m": 4.7, "energy_score_2d_m": 3.8},
        },
        "gate": {"mae_delta_m": -.3, "es_delta_m": -.2,
                 "mae_upper_m": -.1, "es_upper_m": -.05, "passed": True},
        "decision": "PROMISING_JOINT_CALIBRATION", "result_sha256": "synthetic",
    }
    assert g6.make_report(record) == g6.make_report(json.loads(
        json.dumps(record, sort_keys=True)))


def test_actual_role_scoring_retains_all_candidate_reliability(monkeypatch):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
            self.n_anchors = 210
            self.encoder = torch.nn.Identity()
            self.register_buffer("centres", torch.column_stack((
                torch.linspace(-8, 8, self.n_anchors),
                torch.zeros(self.n_anchors))))
        def retrieval_logits_from_embedding(self, embedding):
            return -self.centres[:, 0].square()[None, :].expand(len(embedding), -1)/10
        def _components(self, embedding, anchor_ix):
            loc = self.centres[None, :, :].expand(len(embedding), -1, -1)
            return loc, torch.ones_like(loc)
    monkeypatch.setattr(g6.r2b, "_prototype_distance",
                        lambda ref, rss: torch.zeros(len(rss), ref.n_anchors))
    data = {"keys": [(0, 0, 2., 1.), (0, 0, 2., 1.), (0, 0, 3., 1.)],
            "raw_xy": np.array([[2., 1.], [2., 1.], [3., 1.]]),
            "phone": np.zeros(3, dtype=np.int64), "rss": torch.zeros(3, 2)}
    scored = g6._score_role("calibration", [Model() for _ in range(5)],
                           data, np.arange(3), np.zeros(2))
    names = {g6.candidate_name(*pair) for pair in g6.CANDIDATES}
    assert set(scored["reliability"]) == names
    assert scored["coordinate_groups"] == 2
    for name in names:
        diag = scored["reliability"][name]
        assert len(diag["coordinate_vectors"]["predicted_radius"]) == 2
        assert len(diag["coordinate_vectors"]["observed_error"]) == 2
        assert set(diag["coverage"]) == {"0.5", "0.8", "0.9"}
        assert set(diag["macro_credible_radius_m"]) == {"0.5", "0.8", "0.9"}
    for t in g6.T_VALUES:
        mae = [scored["coordinate_vectors"][g6.candidate_name(t, s)]["mae"]
               for s in g6.S_VALUES]
        assert mae[0] == mae[1] == mae[2]
'''
if sys.argv[1] == "tests":
    for path, name in ((CODE, "G6_CODE_BEFORE_FIX.txt"), (TEST, "G6_TEST_BEFORE_FIX.txt")):
        with (AUDIT / name).open("xb") as stream:
            stream.write(path.read_bytes())
    write(TEST, read(TEST) + REGRESSIONS)
    print("Added three targeted regressions; scientific source remains original.")
elif sys.argv[1] == "fix":
    text = read(CODE)
    replacements = [
        ('    xy=np.empty((len(rows),2),dtype=np.float32)\n',
         '    xy=np.empty((len(rows),2),dtype=np.float32)\n    original_xy=np.empty((len(rows),2),dtype=np.float64)\n'),
        ('        xy[i]=np.asarray(scan.coords,dtype=np.float32).reshape(2)\n',
         '        original_xy[i]=np.asarray(scan.coords,dtype=np.float64).reshape(2)\n        xy[i]=original_xy[i].astype(np.float32)\n'),
        ('    data.validate()\n    return data\n',
         '''    data.validate()
    original_keys=location_keys(original_xy,data.building,data.floor)
    canonical_keys=location_keys(xy,data.building,data.floor)
    forward={};reverse={}
    for original,canonical in zip(original_keys,canonical_keys):
        if ((original in forward and forward[original]!=canonical)
            or (canonical in reverse and reverse[canonical]!=original)):
            raise G6Error("float32 conversion collapses or divides an original TUJI physical site")
        forward[original]=canonical;reverse[canonical]=original
    data.metadata.update({
        "original_to_float32_site_bijection":True,
        "original_physical_site_count":len(forward),
        "float32_physical_site_count":len(reverse),
        "float32_key_value_changes":sum(a!=b for a,b in forward.items()),
    })
    return data
'''),
        ('            "stats":{"n_aps":rss.shape[1],"n_buildings":1,"n_floors":1,\n',
         '''            "coordinate_identity_audit":{
                key:canonical.metadata[key] for key in (
                    "original_to_float32_site_bijection",
                    "original_physical_site_count","float32_physical_site_count",
                    "float32_key_value_changes")
            },
            "stats":{"n_aps":rss.shape[1],"n_buildings":1,"n_floors":1,
'''),
        ('        "physical_site_overlap":0,\n',
         '        "physical_site_overlap":0,\n        "coordinate_identity_audit":{**data["coordinate_identity_audit"],\n                                     "all_role_centred_bijection":True},\n'),
        ('        "role_device_rows":info["role_device_rows"],\n',
         '        "role_device_rows":info["role_device_rows"],\n        "coordinate_identity_audit":info["coordinate_identity_audit"],\n'),
        ('''                         "coverage":{str(level):torch.empty(n,dtype=torch.float64)
                                     for level in COVERAGE_LEVELS}}
                      for k in names if k in (BASE,(selected or {}).get("policy"))}''',
         '''                         "coverage":{str(level):torch.empty(n,dtype=torch.float64)
                                     for level in COVERAGE_LEVELS},
                         "credible_radius":{str(level):torch.empty(n,dtype=torch.float64)
                                     for level in COVERAGE_LEVELS}}
                      for k in names}'''),
        ('                        threshold=torch.quantile(distances,level,dim=1)\n',
         '                        threshold=torch.quantile(distances,level,dim=1)\n                        diag["credible_radius"][str(level)][start:end]=threshold.double().cpu()\n'),
        ('''                    for level,vec in diag["coverage"].items()},
            }
''',
         '''                    for level,vec in diag["coverage"].items()},
                "macro_credible_radius_m":{
                    level:float(r3._metric_record(vec,group_ids,len(group_keys))[0]["coordinate_macro_m"])
                    for level,vec in diag["credible_radius"].items()},
                "coordinate_vectors":{
                    "predicted_radius":r3._metric_record(diag["predicted_radius"],group_ids,len(group_keys))[1].tolist(),
                    "observed_error":r3._metric_record(diag["observed_error"],group_ids,len(group_keys))[1].tolist(),
                    "coverage":{level:r3._metric_record(vec,group_ids,len(group_keys))[1].tolist()
                                for level,vec in diag["coverage"].items()},
                    "credible_radius":{level:r3._metric_record(vec,group_ids,len(group_keys))[1].tolist()
                                       for level,vec in diag["credible_radius"].items()},
                },
            }
'''),
        ('# G6 | TUJI1 training-variable-only two-axis distribution calibration',
         '# G6 | TUJI1 TRAIN-variable-only two-axis distribution calibration'),
        ('    for name,m in result["candidate_metrics"].items():\n',
         '    for name in sorted(result["candidate_metrics"]):\n        m=result["candidate_metrics"][name]\n'),
    ]
    for old, new in replacements:
        if text.count(old) != 1:
            raise RuntimeError("unexpected repair match count: " + repr(old))
        text = text.replace(old, new)
    write(CODE, text)
    diff = []
    for path, before in ((CODE, AUDIT/"G6_CODE_BEFORE_FIX.txt"), (TEST, AUDIT/"G6_TEST_BEFORE_FIX.txt")):
        diff.extend(difflib.unified_diff(
            read(before).splitlines(keepends=True), read(path).splitlines(keepends=True),
            fromfile="before/"+path.relative_to(ROOT).as_posix(),
            tofile="after/"+path.relative_to(ROOT).as_posix()))
    with (AUDIT/"PRE_MANIFEST_REPAIRS.diff").open("x",encoding="utf-8",newline="\n") as stream:
        stream.writelines(diff)
    print("G6-only repairs applied before manifest; full original bytes and diff retained.")
else:
    raise ValueError("mode must be tests or fix")
