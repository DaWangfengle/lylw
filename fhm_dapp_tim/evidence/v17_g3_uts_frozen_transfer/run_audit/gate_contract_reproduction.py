import inspect, json
from fhm_dapp_tim.fhm_dapp import uji_v17_g3_uts as g3
assert "gate = g1.confirm_gate(" in inspect.getsource(g3._evaluate_test_once)
# Synthetic counterexample only: candidate -0.20m, placebo -0.10m versus baseline.
original=g3.g1.confirm_gate(-0.20,-0.01,-0.10,-0.02,-0.10)
required=(-0.20<=-0.15 and -0.01<0 and -0.02<=0 and -0.10>0)
print(json.dumps({"synthetic_only":True,"original_G3_gate_passed":original["passed"],"fixed_G3_design_required_passed":required,"placebo_minus_baseline_m":-0.10}),flush=True)
assert original["passed"]==required, "G3 implementation accepts improving placebo contrary to fixed design"
