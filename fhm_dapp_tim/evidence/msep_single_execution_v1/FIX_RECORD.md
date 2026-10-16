# MSEP 单成员实验修复记录

全部修复发生于三域正式训练及 TEST 评分之前；未修改旧科学模块、旧 checkpoint、原始数据或协议方法常数。

## 1. TUJI 训练变量读取边界

静态审查发现旧 v18_4_datasets._tuji_scans 无 variable_names，连带反序列化 results_test。新 MSEP _load 保留原 canonical adapter 的 AP、RSS、坐标和类别转换，仅在调用期间限定 scipy loadmat(variable_names=("results_train",))，并拒绝返回 results_test。原 adapter 文件逐字不变。

原始指定单测 9 passed。补合成回归后，修复前 2 failed / 10 passed，明确证明旧路径未限制变量且不拒绝测试变量。修复后以及最终冒烟前测试均为 12 passed。所有失败原文见 run_logs/pytest_loader_regression_before_fix.stdout.log。

## 2. 冻结证据和一次评价保护

在首次 TEST 评分前保存 CHECKPOINTS.pt、全部校准温度/系数、head，并排他写入 TEST_INVOCATION.lock，绑定 checkpoint 和新模块 SHA256。原脚本在评分后才保存且未保存温度和 G13 系数；本改动不影响任何模型计算。增加逐 epoch 阶段日志、角色 SHA、FIT seen AP 数及评价概率有限/归一断言。

添加阶段日志时引入一处右花括号缺失；compile_safe_loader 已在任何训练前以 SyntaxError 退出，随后修复并重新编译/测试通过。原始编译失败日志保留。

## 3. 旧数据模块 --help 的命令例外

v18_4_datasets.main 不解析参数，直接加载四套训练数据，且 TUJI 默认解码测试变量。为遵守本轮三套训练数据边界，该 --help 命令未运行；通过只读源码审查确认 API，并运行真正支持 argparse 的 msep_single_study --help（exit 0）。这是一项明确记录的命令替代，不能声称原 --help 已通过。没有修改旧模块以影响历史 manifest。

## 4. 科学定义

T/s、目标函数、分区、种子、训练上限、候选支持、ES 抽样均保持原实现与指令。smoke 为 DEV-only，未用于选择方法。固定骨干六策略共享同一次提取的连续中心/尺度，联合训练两策略共享联合骨干的连续中心/尺度。正式参数见 METHOD_FREEZE.json。

## 最小源码/测试差异

```diff
--- original/fhm_dapp_tim/fhm_dapp/msep_single_study.py
+++ fhm_dapp_tim/fhm_dapp/msep_single_study.py
@@ -12,7 +12,10 @@
 import json
 import math
 import random
+import time
+from datetime import datetime
 from pathlib import Path
+from unittest.mock import patch
 
 import numpy as np
 import torch
@@ -54,7 +57,22 @@
 def _load(dataset, root):
     kw = {"data_root": root} if root else {}
     if dataset == "tuji":
-        return load_tuji1("train", **kw)[0]
+        # Preserve canonical AP/category/coordinate handling without decoding the
+        # co-located official-test variable in the legacy MAT adapter.
+        from fhm_dapp_tim.fhm_dapp import v18_4_datasets as ds
+        original_loadmat = ds.loadmat
+
+        def train_only_loadmat(*args, **kwargs):
+            kwargs["variable_names"] = ("results_train",)
+            payload = original_loadmat(*args, **kwargs)
+            if "results_test" in payload:
+                raise RuntimeError("TUJI results_test was unexpectedly decoded")
+            if "results_train" not in payload:
+                raise RuntimeError("TUJI results_train is missing")
+            return payload
+
+        with patch.object(ds, "loadmat", train_only_loadmat):
+            return load_tuji1("train", **kw)[0]
     if dataset == "uts":
         return load_uts("train", **kw)[0]
     if dataset == "uji":
@@ -260,6 +278,7 @@
             total += float(terms["loss"].detach()); batches += 1
         score = _dev_mae(model, obs, rss, xy, keys, roles["dev"], k=k, device=device, batch=batch)
         history.append({"epoch": epoch, "fit_loss": total/batches, "dev_mae": score})
+        print(json.dumps({"stage":"base", **history[-1]}), flush=True)
         if score < best - .01:
             best, best_epoch, best_state = score, epoch, copy.deepcopy(model.state_dict())
         if epoch >= 20 and epoch-best_epoch >= patience:
@@ -309,6 +328,7 @@
             opt.step(); head.clamp_()
             total += float(loss.detach()); steps += 1
         history.append({"epoch": epoch, "fit_leave_self_loss": total/max(steps,1)})
+        print(json.dumps({"stage":"joint", **history[-1]}), flush=True)
     model.eval(); head.eval()
     return model, head, history
 
@@ -339,6 +359,9 @@
                 else:
                     prob = posterior(scores,z,mode=spec["mode"],
                                      temperature=spec.get("temperature",.6),theta=spec.get("theta",(0.,0.)))
+                if (not torch.isfinite(prob).all() or (prob < 0).any()
+                    or not torch.allclose(prob.sum(1), torch.ones_like(prob[:, 0]), atol=1e-5, rtol=1e-5)):
+                    raise RuntimeError(f"invalid normalized posterior: {name}")
                 point = (prob[:, :, None]*loc).sum(1)
                 err = torch.linalg.vector_norm(point-y,dim=1)
                 es = energy_score_mc(prob,loc,scales,y,draws=draws,seed=seed+start)
@@ -400,6 +423,7 @@
 
 
 def run(args):
+    started = time.perf_counter()
     output = Path(args.output).resolve()
     if output.exists():
         raise FileExistsError(f"refusing to overwrite existing result directory: {output}")
@@ -416,12 +440,19 @@
     k = None if args.support=="all" else min(args.k,n)
     if len(set(keys[int(x)] for x in roles["fit"])) != n:
         raise AssertionError("FIT anchor IDs do not biject with site keys")
+    print(json.dumps({"stage":"fit_ready", "dataset":args.dataset,
+                      "device":str(device), "fit_anchors":n,
+                      "roles":{r:{"sites":len(set(keys[int(i)] for i in rows)),
+                                  "scans":len(rows)} for r,rows in roles.items()}}), flush=True)
     base_history = _train_base(base,rss,xy,sample_anchor,device_ids,neighbors,nw,obs,
                                roles,keys,device=device,epochs=args.base_epochs,
                                patience=args.patience,batch=args.batch,k=k,seed=args.seed)
     cal = _pack(base,obs,rss,roles["cal"],xy,keys,k=k,device=device,batch_size=args.batch)
+    print(json.dumps({"stage":"calibration_started"}), flush=True)
     fitted_temperature = _fit_temperature(cal)
+    print(json.dumps({"stage":"temperature_fitted", "temperature":fitted_temperature}), flush=True)
     theta_hist = _fit_g13_historical_surrogate(cal)
+    print(json.dumps({"stage":"historical_g13_fitted", "theta":theta_hist}), flush=True)
     theta = _fit_g13(cal)
     frozen_head = _fit_unified_frozen(cal,theta)
     del cal
@@ -436,6 +467,28 @@
              "v17_single_g13": {"mode":"g13","theta":theta},
              "v17_single_unified_frozen": {"mode":"unified","head":frozen_head}}
     evaluation_role = "dev" if args.smoke else "test"
+    # Persist every learned choice before the first held-out score. A failed
+    # evaluation retains the frozen checkpoint and an irreversible invocation.
+    torch.save({"base_model":base.state_dict(),"joint_model":joint.state_dict(),
+                "frozen_head":frozen_head.state_dict(),"joint_head":joint_head.state_dict(),
+                "fit_anchor_coords":memory["coords"],"fit_anchor_building":memory["building"],
+                "fit_anchor_floor":memory["floor"],"reference_origin":origin,
+                "ap_ids":data.ap_ids,"role_site_keys":{r:[keys[int(i)] for i in v] for r,v in roles.items() if (not args.smoke or r!="test")},
+                "observation_memory":{name:v.cpu() for name,v in obs.items()},
+                "calibrated_temperature":fitted_temperature,
+                "g13_historical_coefficients":theta_hist,"g13_nll_coefficients":theta,
+                "config":vars(args)}, output/"CHECKPOINTS.pt")
+    checkpoint_hash = hashlib.sha256((output/"CHECKPOINTS.pt").read_bytes()).hexdigest()
+    if not args.smoke:
+        lock = {"created_at":datetime.now().astimezone().isoformat(),
+                "evaluation_role":"test", "checkpoint_sha256":checkpoint_hash,
+                "source_sha256":{Path(__file__).name:hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
+                                 "msep_single_posterior.py":hashlib.sha256(Path(__file__).with_name("msep_single_posterior.py").read_bytes()).hexdigest()},
+                "all_learned_choices_frozen":True, "config":vars(args)}
+        with (output/"TEST_INVOCATION.lock").open("x", encoding="utf-8") as handle:
+            json.dump(lock, handle, ensure_ascii=False, indent=2, allow_nan=False)
+    print(json.dumps({"stage":"evaluation_started", "evaluation_role":evaluation_role,
+                      "checkpoint_sha256":checkpoint_hash}), flush=True)
     site_diagnostics = {}
     base_result = _evaluate(modes,base,obs,rss,xy,keys,roles[evaluation_role],memory,
                             k=k,device=device,batch=args.batch,draws=args.es_draws,
@@ -457,6 +510,8 @@
         "role_isolation":"physical site (building,floor,XY rounded to 4 decimals)",
         "role_count": {role:{"rows":len(idx),"sites":len(set(keys[int(i)] for i in idx))} for role,idx in roles.items()},
         "input_ap_count":len(data.ap_ids), "fit_anchor_count":int(n),
+        "fit_seen_ap_count":int(obs["seen_ap_mask"].sum()),
+        "role_site_key_sha256":{role:hashlib.sha256(json.dumps(sorted(set(keys[int(i)] for i in rows)),separators=(",",":")).encode()).hexdigest() for role,rows in roles.items()},
         "fit_scan_count_quantiles":np.quantile(memory["counts"].cpu().numpy(),[0,.5,.9,1]).tolist(),
         "base_training":base_history,"joint_fit_history":joint_history,
         "calibration":{"fitted_temperature":fitted_temperature,"g13_historical_surrogate_coefficients":theta_hist,
@@ -474,19 +529,14 @@
             "A true neighborhood is an audit-only label-derived quantity; never an inference input.",
         ],
         "config":vars(args),
+        "checkpoint_sha256":checkpoint_hash,
+        "elapsed_seconds":time.perf_counter()-started,
     }
     (output/"RESULT.json").write_text(json.dumps(_json_safe(result),ensure_ascii=False,indent=2),encoding="utf-8")
     (output/"SITE_DIAGNOSTICS.json").write_text(
         json.dumps(_json_safe({"evaluation_role":evaluation_role,
                                 "audit_only_not_inference":True,
                                 "by_model":site_diagnostics}),ensure_ascii=False),encoding="utf-8")
-    torch.save({"base_model":base.state_dict(),"joint_model":joint.state_dict(),
-                "frozen_head":frozen_head.state_dict(),"joint_head":joint_head.state_dict(),
-                "fit_anchor_coords":memory["coords"],"fit_anchor_building":memory["building"],
-                "fit_anchor_floor":memory["floor"],"reference_origin":origin,
-                "ap_ids":data.ap_ids,"role_site_keys":{k:[keys[int(i)] for i in v] for k,v in roles.items() if (not args.smoke or k!="test")},
-                "observation_memory":{k:v.cpu() for k,v in obs.items()},"config":vars(args)},
-               output/"CHECKPOINTS.pt")
     metric_key = "dev_smoke_metrics" if args.smoke else "test_metrics"
     print(json.dumps({"output":str(output),"evaluation_role":evaluation_role,
                       metric_key:result[metric_key]},ensure_ascii=False,indent=2))
@@ -517,4 +567,4 @@
 
 
 if __name__ == "__main__":
-    main()+    main()

```

```diff
--- original/tests/test_msep_single_posterior.py
+++ tests/test_msep_single_posterior.py
@@ -119,4 +119,56 @@
     objective.backward()
     for grad in (scores.grad,centers.grad,head.theta.grad,head.gate.weight.grad):
         assert grad is not None and torch.isfinite(grad).all()
-        assert torch.linalg.vector_norm(grad) > 0+        assert torch.linalg.vector_norm(grad) > 0
+
+
+def test_tuji_adapter_requests_train_variable_only(monkeypatch):
+    from types import SimpleNamespace
+    from fhm_dapp_tim.fhm_dapp import msep_single_study as study
+    from fhm_dapp_tim.fhm_dapp import v18_4_datasets as ds
+
+    calls = []
+
+    def fake_loadmat(path, **kwargs):
+        calls.append(kwargs)
+        assert kwargs.get("variable_names") == ("results_train",)
+        return {"results_train": SimpleNamespace(Wifi=np.asarray([], dtype=object))}
+
+    def fake_adapter(split, **kwargs):
+        assert split == "train"
+        for filename in ds.TUJI_DEVICE_FILES.values():
+            payload = ds.loadmat(filename, squeeze_me=True, struct_as_record=False)
+            assert "results_test" not in payload
+        return "canonical_train_data", None
+
+    monkeypatch.setattr(ds, "loadmat", fake_loadmat)
+    monkeypatch.setattr(study, "load_tuji1", fake_adapter)
+    assert study._load("tuji", None) == "canonical_train_data"
+    assert len(calls) == 5
+    assert ds.loadmat is fake_loadmat  # Temporary restriction was restored.
+
+
+def test_tuji_adapter_rejects_unexpected_test_variable(monkeypatch):
+    from fhm_dapp_tim.fhm_dapp import msep_single_study as study
+    from fhm_dapp_tim.fhm_dapp import v18_4_datasets as ds
+
+    monkeypatch.setattr(ds, "loadmat", lambda *a, **k: {"results_train": object(), "results_test": object()})
+
+    def fake_adapter(*args, **kwargs):
+        ds.loadmat("synthetic.mat")
+        return None, None
+
+    monkeypatch.setattr(study, "load_tuji1", fake_adapter)
+    with pytest.raises(RuntimeError, match="results_test"):
+        study._load("tuji", None)
+
+
+def test_mixture_nll_matches_independent_student_t_density():
+    weights = torch.tensor([[.2, .8]], dtype=torch.float64)
+    centers = torch.tensor([[[0., 2.], [3., -1.]]], dtype=torch.float64)
+    scales = torch.tensor([[[.7, 1.2], [2., .4]]], dtype=torch.float64)
+    target = torch.tensor([[1., .5]], dtype=torch.float64)
+    lp = torch.distributions.StudentT(4., centers, scales).log_prob(target[:, None]).sum(-1)
+    expected = -torch.logsumexp(weights.log() + lp, dim=1).mean()
+    actual = mixture_loss(weights, centers, scales, target, mae_weight=0.)
+    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)

```
