"""Dataset-adaptive V17 spatial operators. No training or TEST scoring."""
from __future__ import annotations
import hashlib, itertools, json, math, csv, argparse, platform, sys, time
from pathlib import Path
import numpy as np
import torch
from . import msep_v17_spatial_composition_v1 as comp
from . import msep_v17_spatial_operators_v1 as op
from . import msep_v17_spatial_structure_v1 as s3

REPO=s3.REPO
DEFAULT_INPUT=s3.DEFAULT_INPUT
DEFAULT_CHECKPOINT_ROOT=comp.DEFAULT_CHECKPOINT_ROOT
DEFAULT_OUTPUT=REPO/"fhm_dapp_tim/evidence/v17_dataset_adaptive_joint_fit_iter3"
ARMS=("C","D","E")
TEST_COUNTS={"internal_test_rows_scored":0,"official_test_rows_scored":0}
# Spatial parameters use separate, operator-specific FIT-derived scale pools.
CUTOFF_QUANTILES=("p50","p75","p90","p95","p100")
S1_CUTOFF_QUANTILES=("p75","p90","p95","p100")
KERNEL_EDGE_QUANTILES=("p50","p75","p90")
CALIBRATION_RELATIVE_MULTIPLIERS=(0.75,1.0,1.25)
CALIBRATION_EXPANSION_MULTIPLIERS=(0.50,1.50,0.625,1.75,2.0)
TEMPERATURE_BOUNDS=(0.25,3.0)
THETA_BOUNDS=(0.0,1.5)
ALPHA_PRIORS=(0.0,0.05,0.10,0.20,0.35,0.65)
ALPHA_OBS=(0.0,0.10,0.25)
BETAS=(0.0,0.05,0.10,0.15,0.20,0.35,0.50,0.65,0.80)
LAMBDAS=(0.02,0.05)
BF_CE_WEIGHT=0.25
W1_WEIGHT=0.10
BF_CE_RELATIVE_TOLERANCE=0.02
BF_CE_ABSOLUTE_TOLERANCE=0.005
BF_CE_NORMALIZATION_FLOOR=0.05
MAE_NON_DEGRADATION_TOLERANCE_M=1e-6
MIN_S2_RETAINED_EDGE_FRACTION=0.45
SELECTION_OBJECTIVE_FORMULA=(
    "MAE/MAE_clean + 0.10*(W1-W1_clean)/max(W1_clean,1e-9) + "
    "0.25*max(0,BFCE-BFCE_clean)/max(BFCE_clean,0.05) + "
    "lambda*(NLL-NLL_clean)/max(abs(NLL_clean),1)"
)
NEAR_OPTIMAL_OBJECTIVE_TOLERANCE=0.01
QUANTILES=(0.00,0.01,0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.99,1.00)


def _selected_datasets(dataset="all"):
    if dataset == "all":
        return tuple(s3.DATASETS)
    if dataset not in s3.DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; expected 'all' or one of {s3.DATASETS}")
    return (dataset,)

def _json_safe(value):
    if isinstance(value,dict): return {str(k):_json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [_json_safe(v) for v in value]
    if isinstance(value,np.ndarray): return _json_safe(value.tolist())
    if isinstance(value,np.generic): return _json_safe(value.item())
    if isinstance(value,float) and not math.isfinite(value):
        return "Infinity" if value>0 else ("-Infinity" if value<0 else "NaN")
    return value

def _write_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8") as f:
        json.dump(_json_safe(value),f,ensure_ascii=False,indent=2,allow_nan=False); f.write("\n")


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def _hash_anchor_order(coords,buildings,floors):
    h=hashlib.sha256()
    for a in (np.asarray(coords,dtype="<f8"),np.asarray(buildings,dtype="<i8"),np.asarray(floors,dtype="<i8")):
        h.update(str(a.shape).encode()); h.update(a.tobytes(order="C"))
    return h.hexdigest()

def _quantiles(values):
    a=np.asarray(values,dtype=np.float64); a=a[np.isfinite(a)]
    if not len(a):
        return {f"p{int(q*100):02d}":None for q in QUANTILES}|{"mean":None,"std":None,"count":0}
    out={f"p{int(q*100):02d}":float(np.quantile(a,q)) for q in QUANTILES}
    out.update(mean=float(a.mean()),std=float(a.std()),count=int(len(a)))
    return out

def _threshold_counts(values):
    a=np.asarray(values,dtype=np.float64)
    return {f"gt_{m}m":{"count":int((a>m).sum()),"fraction":float((a>m).mean()) if len(a) else None}
            for m in (5,10,15,20,30,40)}

def _pairwise_fit_geometry(coords,buildings,floors,edges):
    xy=np.asarray(coords,dtype=np.float64); b=np.asarray(buildings,dtype=np.int64); f=np.asarray(floors,dtype=np.int64)
    if xy.ndim!=2 or xy.shape[1]!=2 or len(xy)!=len(b) or len(xy)!=len(f) or not np.isfinite(xy).all():
        raise ValueError("invalid FIT anchor geometry")
    strata={}
    for i,key in enumerate(zip(b.tolist(),f.tolist())): strata.setdefault(tuple(key),[]).append(i)
    kth={k:[] for k in (1,2,4,8)}; zero_pairs=0; groups={}
    for key,members in sorted(strata.items()):
        block=xy[members]; m=len(members)
        _,duplicate_counts=np.unique(block,axis=0,return_counts=True)
        zeros=int(sum(int(count)*(int(count)-1)//2 for count in duplicate_counts)); zero_pairs+=zeros
        group_kth={k:[] for k in (1,2,4,8)}
        # Exact nearest neighbors, chunked to avoid a temporary [m,m,2] array
        # for large building-floor strata. Memory is O(chunk*m), not O(m^2*2).
        chunk_size=256
        for start in range(0,m,chunk_size):
            stop=min(start+chunk_size,m)
            delta=block[start:stop,None,:]-block[None,:,:]
            dist=np.sqrt(np.maximum((delta*delta).sum(-1),0.0))
            dist[np.arange(stop-start),np.arange(start,stop)]=np.inf
            sorted_dist=np.sort(dist,axis=1)
            for k in (1,2,4,8):
                if k<=m-1:
                    vals=sorted_dist[:,k-1]
                    vals=vals[np.isfinite(vals)]
                    group_kth[k].extend(vals.tolist()); kth[k].extend(vals.tolist())
        groups[f"{key[0]}::{key[1]}"]={"anchors":m,"zero_distance_pairs":zeros,
            "kth_neighbor":{str(k):_quantiles(group_kth[k]) for k in (1,2,4,8)}}
    e=np.asarray(torch.as_tensor(edges).detach().cpu(),dtype=np.int64).reshape(-1,2)
    if len(e):
        lengths=np.linalg.norm(xy[e[:,0]]-xy[e[:,1]],axis=1)
        if np.any(b[e[:,0]]!=b[e[:,1]]) or np.any(f[e[:,0]]!=f[e[:,1]]):
            raise RuntimeError("cross-building/floor edge in frozen union-kNN graph")
        if len({tuple(sorted(map(int,x))) for x in e.tolist()})!=len(e): raise RuntimeError("duplicate edge")
    else: lengths=np.asarray([],dtype=np.float64)
    result={"anchors":int(len(xy)),"building_floor_strata":len(strata),
        "zero_distance_pairs_within_building_floor":int(zero_pairs),
        "kth_neighbor_distance_m":{str(k):_quantiles(v) for k,v in kth.items()},
        "kth_neighbor_threshold_counts":{str(k):_threshold_counts(v) for k,v in kth.items()},
        "base_union_knn_edge_distance_m":_quantiles(lengths),
        "base_union_knn_edge_threshold_counts":_threshold_counts(lengths),
        "base_union_knn_edges":int(len(e)),
        "base_union_knn_edge_isolated_anchor_count":int(len(xy)-len(set(e.reshape(-1).tolist()))),
        "groups":groups,"distance_semantics":"FIT anchor k-neighbors are within same building/floor; union-kNN edge lengths are reported separately."}
    result["exp_neg_distance_over_8_at_quantiles"]={
        f"FIT_k{k}_nearest_neighbor":{key:(float(math.exp(-value/8.0)) if value is not None else None)
            for key,value in result["kth_neighbor_distance_m"][str(k)].items() if key.startswith("p")}
        for k in (1,2,4,8)}
    result["exp_neg_distance_over_8_at_quantiles"]["base_union_knn_edges"]={
        key:(float(math.exp(-value/8.0)) if value is not None else None)
        for key,value in result["base_union_knn_edge_distance_m"].items() if key.startswith("p")}
    result["exp_neg_distance_over_8_at_quantiles"]["note"]="Descriptive only; normalized operator weights also depend on graph topology and Sinkhorn."
    for key,group in result["groups"].items():
        bb,ff=(int(x) for x in key.split("::",1))
        mask=(b[e[:,0]]==bb)&(f[e[:,0]]==ff)&(b[e[:,1]]==bb)&(f[e[:,1]]==ff) if len(e) else np.zeros(0,dtype=bool)
        vals=lengths[mask] if len(e) else np.empty(0,dtype=np.float64)
        group["union_knn_edge_count"]=int(len(vals))
        group["union_knn_edge_distance_m"]=_quantiles(vals)
        group["union_knn_edge_threshold_counts"]=_threshold_counts(vals)
    return result

def build_cutoff_graph(coords,buildings,floors,base_edges,r_max_m,tau_graph_m,*,sinkhorn_iterations=20000,tolerance=1e-10):
    """Build the symmetric doubly-stochastic cutoff graph in O(N+E) memory.

    The old dense NxN balancing matrix was mathematically sparse (identity plus
    retained union-kNN edges) but allocated several dense arrays. Here the same
    symmetric Sinkhorn scaling is computed from endpoint accumulations, avoiding
    O(N^2) memory while preserving identical graph semantics.
    """
    xy=np.asarray(coords,dtype=np.float64); b=np.asarray(buildings,dtype=np.int64); f=np.asarray(floors,dtype=np.int64)
    if xy.ndim!=2 or xy.shape[1]!=2 or len(b)!=len(xy) or len(f)!=len(xy) or len(xy)<1 or not np.isfinite(xy).all():
        raise ValueError("invalid/nonfinite coordinates or building-floor arrays")
    e=np.asarray(torch.as_tensor(base_edges).detach().cpu(),dtype=np.int64).reshape(-1,2)
    if not (float(r_max_m)==math.inf or (math.isfinite(float(r_max_m)) and float(r_max_m)>0)):
        raise ValueError("R_max must be positive or +inf")
    if not (float(tau_graph_m)==math.inf or (math.isfinite(float(tau_graph_m)) and float(tau_graph_m)>0)):
        raise ValueError("tau_graph must be positive or +inf")
    if int(sinkhorn_iterations)<1 or not (math.isfinite(float(tolerance)) and float(tolerance)>0):
        raise ValueError("Sinkhorn iteration budget and tolerance must be positive")
    n=len(xy)
    if len(e):
        if np.any(e<0) or np.any(e>=n): raise ValueError("edge endpoint outside canonical anchor range")
        e=np.unique(np.sort(e,axis=1),axis=0)
        if np.any(e[:,0]==e[:,1]): raise ValueError("self-edges are represented by the mandatory diagonal")
        if np.any(b[e[:,0]]!=b[e[:,1]]) or np.any(f[e[:,0]]!=f[e[:,1]]):
            raise ValueError("cross-BF edge rejected")
        lengths=np.linalg.norm(xy[e[:,0]]-xy[e[:,1]],axis=1)
        keep=lengths<=float(r_max_m); kept=e[keep]; d=lengths[keep]; removed=int((~keep).sum())
    else:
        kept=np.empty((0,2),dtype=np.int64); d=np.empty(0,dtype=np.float64); removed=0
    if len(kept):
        exponent=np.zeros_like(d) if float(tau_graph_m)==math.inf else d/float(tau_graph_m)
        edge_kernel=np.maximum(np.exp(-np.minimum(exponent,745.0)),np.finfo(np.float64).tiny)
        ei,ej=kept[:,0],kept[:,1]
    else:
        edge_kernel=np.empty(0,dtype=np.float64); ei=ej=np.empty(0,dtype=np.int64)
    scale=np.ones(n,dtype=np.float64); converged=False; iterations=0
    for iterations in range(1,int(sinkhorn_iterations)+1):
        neighbor=np.zeros(n,dtype=np.float64)
        if len(kept):
            np.add.at(neighbor,ei,edge_kernel*scale[ej])
            np.add.at(neighbor,ej,edge_kernel*scale[ei])
        mass=scale*(scale+neighbor)  # A has unit diagonal plus retained weighted edges.
        if not np.isfinite(mass).all() or np.any(mass<=0):
            raise RuntimeError("invalid Sinkhorn row mass")
        updated=scale/np.sqrt(mass)
        if float(np.max(np.abs(updated-scale)))<=float(tolerance):
            scale=updated; converged=True; break
        scale=updated
    diagonal=scale*scale
    edge_values=scale[ei]*edge_kernel*scale[ej] if len(kept) else np.empty(0,dtype=np.float64)
    row_mass=diagonal.copy()
    if len(kept):
        np.add.at(row_mass,ei,edge_values); np.add.at(row_mass,ej,edge_values)
    re=float(np.max(np.abs(row_mass-1.0))); ce=re  # The constructed operator is exactly symmetric.
    if not np.isfinite(diagonal).all() or not np.isfinite(edge_values).all():
        raise RuntimeError("nonfinite balanced graph")
    if max(re,ce)>2e-6:
        raise RuntimeError(f"Sinkhorn residual too large: {re},{ce}; converged={converged}, iterations={iterations}")
    rr=np.concatenate((np.arange(n,dtype=np.int64),ei,ej))
    cc=np.concatenate((np.arange(n,dtype=np.int64),ej,ei))
    vv=np.concatenate((diagonal,edge_values,edge_values))
    graph=torch.sparse_coo_tensor(torch.as_tensor(np.stack([rr,cc]),dtype=torch.long),
        torch.as_tensor(vv,dtype=torch.float32),(n,n)).coalesce()
    # Check the actual float32 operator consumed by the model, not just float64 intermediates.
    actual_rows=torch.sparse.sum(graph,dim=1).to_dense().cpu().numpy()
    actual_cols=torch.sparse.sum(graph,dim=0).to_dense().cpu().numpy()
    actual_re=float(np.max(np.abs(actual_rows-1.0))); actual_ce=float(np.max(np.abs(actual_cols-1.0)))
    if max(actual_re,actual_ce)>2e-6:
        raise RuntimeError(f"float32 graph residual too large: {actual_re},{actual_ce}")
    deg=np.bincount(kept.reshape(-1),minlength=n) if len(kept) else np.zeros(n,dtype=np.int64)
    diag={"r_max_s1_m":float(r_max_m),"tau_graph_m":float(tau_graph_m),"base_edges":int(len(e)),
        "retained_edges":int(len(kept)),"removed_edges":removed,
        "retained_fraction":float(len(kept)/len(e)) if len(e) else 0.0,
        "degree_min":int(deg.min()),"degree_median":float(np.median(deg)),"degree_max":int(deg.max()),
        "zero_neighbor_nodes":int((deg==0).sum()),"sinkhorn_converged":converged,
        "sinkhorn_iterations":iterations,"sinkhorn_row_error":actual_re,"sinkhorn_col_error":actual_ce,
        "retained_edge_distance_m":_quantiles(d),"retained_edge_threshold_counts":_threshold_counts(d)}
    parent=np.arange(n,dtype=np.int64)
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]; x=parent[x]
        return x
    for i,j in kept:
        a,b0=find(int(i)),find(int(j))
        if a!=b0: parent[a]=b0
    diag["connected_components"]=len({find(i) for i in range(n)})
    diag["isolated_component_count"]=int((deg==0).sum())
    return graph,torch.as_tensor(kept.copy(),dtype=torch.long),diag

def normalized_virtual_endpoint_weights(edges,coords,tau_virt_m):
    e=np.asarray(torch.as_tensor(edges).detach().cpu(),dtype=np.int64).reshape(-1,2); xy=np.asarray(coords,dtype=np.float64); n=len(xy)
    if xy.ndim!=2 or xy.shape[1]!=2 or not np.isfinite(xy).all():
        raise ValueError("virtual-edge coordinates must be finite with shape [N,2]")
    if not (float(tau_virt_m)==math.inf or (math.isfinite(float(tau_virt_m)) and tau_virt_m>0)): raise ValueError("tau_virt must be positive or +inf")
    if len(e):
        e=np.sort(e,axis=1)
        if np.any(e<0) or np.any(e>=n): raise ValueError("virtual edge endpoint outside canonical anchor range")
        if len({tuple(x) for x in e.tolist()})!=len(e) or np.any(e[:,0]==e[:,1]): raise ValueError("virtual edges must be unique undirected non-self edges")
        d=np.linalg.norm(xy[e[:,0]]-xy[e[:,1]],axis=1)
        k=np.ones(len(e)) if float(tau_virt_m)==math.inf else np.exp(-np.minimum(d/float(tau_virt_m),745.0))
        k=np.maximum(k,np.finfo(np.float64).tiny); deg=np.bincount(e.reshape(-1),minlength=n); den=np.zeros(n)
        np.add.at(den,e[:,0],k); np.add.at(den,e[:,1],k)
        wi=k/den[e[:,0]]; wj=k/den[e[:,1]]
        endpoint_sum=np.zeros(n,dtype=np.float64)
        np.add.at(endpoint_sum,e[:,0],wi); np.add.at(endpoint_sum,e[:,1],wj)
        active=deg>0
        if not np.allclose(endpoint_sum[active],1.0,atol=1e-10,rtol=1e-10):
            raise RuntimeError("virtual endpoint weights do not conserve mass")
    else:
        d=k=wi=wj=np.empty(0); deg=np.zeros(n,dtype=np.int64)
    return {"edges":e,"lengths_m":d,"kernel":k,"degree":deg,"weight_i":wi,"weight_j":wj,
        "effective_virtual_mass_by_parent":np.bincount(np.concatenate([e[:,0],e[:,1]]) if len(e) else np.array([],dtype=int),
            weights=np.concatenate([wi,wj]) if len(e) else np.array([],dtype=float),minlength=n).tolist()}

def weighted_cell_mixture(parent_logits,virtual_logits,edges,endpoint_weights,beta,parent_centers,parent_scales,virtual_centers,virtual_scales):
    if not 0<=float(beta)<1: raise ValueError("beta must be in [0,1)")
    if parent_logits.ndim!=2: raise ValueError("parent logits must be [B,N]")
    batch,n=parent_logits.shape; e=torch.as_tensor(edges,dtype=torch.long,device=parent_logits.device).reshape(-1,2); ne=len(e)
    if ne == 0 and float(beta) > 0:
        raise ValueError("positive beta requires at least one retained S2 edge")
    if virtual_logits.shape!=(batch,ne): raise ValueError("virtual logits/edge count mismatch")
    if parent_centers.shape!=(batch,n,2) or parent_scales.shape!=parent_centers.shape: raise ValueError("parent components shape mismatch")
    if virtual_centers.shape!=(batch,ne,2) or virtual_scales.shape!=virtual_centers.shape: raise ValueError("virtual components shape mismatch")
    if not torch.isfinite(parent_logits).all() or not torch.isfinite(virtual_logits).all(): raise ValueError("nonfinite mixture logits")
    if not torch.isfinite(parent_centers).all() or not torch.isfinite(virtual_centers).all(): raise ValueError("nonfinite component centers")
    if not torch.isfinite(parent_scales).all() or not torch.isfinite(virtual_scales).all(): raise ValueError("nonfinite component scales")
    if bool((parent_scales<=0).any()) or bool((virtual_scales<=0).any()): raise ValueError("component scales must be positive")
    wi=torch.as_tensor(endpoint_weights["weight_i"],dtype=parent_logits.dtype,device=parent_logits.device)
    wj=torch.as_tensor(endpoint_weights["weight_j"],dtype=parent_logits.dtype,device=parent_logits.device)
    degree=torch.as_tensor(endpoint_weights["degree"],dtype=torch.long,device=parent_logits.device)
    incident=torch.zeros(n,dtype=parent_logits.dtype,device=parent_logits.device)
    if ne:
        incident.scatter_add_(0,e[:,0],wi); incident.scatter_add_(0,e[:,1],wj)
    if not torch.allclose(incident,(degree>0).to(parent_logits.dtype),atol=2e-6,rtol=2e-6):
        raise ValueError("per-parent virtual measure is not normalized")
    pmass=torch.where(degree>0,parent_logits.new_full((n,),1.0-float(beta)),parent_logits.new_ones((n,)))
    measure_total=pmass.sum()+float(beta)*(wi+wj).sum()
    if not torch.allclose(measure_total,parent_logits.new_tensor(float(n)),atol=2e-5,rtol=2e-5):
        raise ValueError("physical-plus-virtual pre-normalization measure is not conserved")
    pterms=parent_logits+pmass.log()[None]
    if float(beta)==0.0 or ne==0:
        q=torch.softmax(parent_logits,dim=1); ids=torch.arange(n,device=parent_logits.device)[None].expand(batch,-1)
        return {"q":q,"xy":parent_centers,"sigma":parent_scales,"ids":ids}
    emass=float(beta)*(wi+wj)
    eterms=virtual_logits+emass.clamp_min(torch.finfo(parent_logits.dtype).tiny).log()[None]
    q=torch.softmax(torch.cat([pterms,eterms],dim=1),dim=1)
    ids=torch.cat([torch.arange(n,device=e.device),e[:,0]])[None].expand(batch,-1)
    return {"q":q,"xy":torch.cat([parent_centers,virtual_centers],dim=1),
        "sigma":torch.cat([parent_scales,virtual_scales],dim=1),"ids":ids}


def _unique_positive_scales(values):
    ordered=sorted(float(x) for x in values if x is not None and math.isfinite(float(x)) and float(x)>0)
    out=[]
    for value in ordered:
        if not out or abs(value-out[-1])>1e-8*max(1.0,value,out[-1]): out.append(value)
    return out


def _geometry_candidate_pools(diag):
    """Separate cutoff radii from kernel temperatures using only FIT geometry."""
    kth=diag.get("kth_neighbor_distance_m",{})
    edge=diag.get("base_union_knn_edge_distance_m",{})
    edge_count=int(diag.get("base_union_knn_edges",edge.get("count",0)) or 0)
    if edge_count<=0:
        raise RuntimeError("cannot derive cutoff radii without a nonempty FIT union-kNN edge set")
    cutoff_by_quantile={}
    for q in CUTOFF_QUANTILES:
        value=edge.get(q)
        if value is None or not math.isfinite(float(value)) or float(value)<=0:
            raise RuntimeError(f"missing positive FIT union-kNN edge quantile {q}; refusing to guess cutoff radii")
        cutoff_by_quantile[q]=float(value)
    ordered=[cutoff_by_quantile[q] for q in CUTOFF_QUANTILES]
    if any(right+1e-10*max(1.0,left,right)<left for left,right in zip(ordered,ordered[1:])):
        raise RuntimeError("FIT union-kNN edge quantiles are not monotone")
    k_medians=[]
    for k in (1,2,4,8):
        value=kth.get(str(k),{}).get("p50")
        if value is not None and math.isfinite(float(value)) and float(value)>0:
            k_medians.append(float(value))
    if not k_medians:
        raise RuntimeError("FIT-only nearest-neighbor medians are unavailable; refusing to guess kernel temperatures")
    kernel_values=_unique_positive_scales(
        k_medians+[cutoff_by_quantile[q] for q in KERNEL_EDGE_QUANTILES]
    )
    if len(kernel_values)<3:
        raise RuntimeError(
            f"FIT geometry yields only {len(kernel_values)} distinct finite kernel scales; "
            "cannot construct three distinguishable temperature levels"
        )
    cutoff_values=_unique_positive_scales([cutoff_by_quantile[q] for q in CUTOFF_QUANTILES])
    if not cutoff_values:
        raise RuntimeError("no positive FIT edge-quantile cutoff radii")
    return {
        "r_max_s1_m":cutoff_values,
        "r_max_s2_m":cutoff_values,
        "tau_graph_m":kernel_values,
        "tau_virt_m":kernel_values,
    }


def _ranked_scale_levels(values,count=4):
    """Pick deterministic, evenly spaced ranks from a FIT-derived scale pool."""
    values=_unique_positive_scales(values)
    if len(values)<count:
        raise RuntimeError(f"need at least {count} distinct FIT-derived scale values; got {len(values)}")
    indices=np.unique(np.round(np.linspace(0,len(values)-1,count)).astype(int))
    selected=[values[int(i)] for i in indices]
    if len(selected)!=count:
        raise RuntimeError("could not create the requested distinct scale quantile levels")
    return selected


def geometry_candidates(diag):
    pools=_geometry_candidate_pools(diag)
    edge=diag["base_union_knn_edge_distance_m"]
    cutoff_by_quantile={q:float(edge[q]) for q in CUTOFF_QUANTILES}
    r1_levels=[(q,cutoff_by_quantile[q]) for q in S1_CUTOFF_QUANTILES]
    tau_levels=_ranked_scale_levels(pools["tau_graph_m"],3)
    tau_virt_low=float(cutoff_by_quantile["p50"])
    tau_virt_high=float(cutoff_by_quantile["p90"])
    out=[]; seen=set()
    def add(r1_q,r1,r2_q,r2,tau_graph,tau_virt,role):
        r1=float(r1); tau_graph=float(tau_graph); r2=float(r2); tau_virt=float(tau_virt)
        tol=1e-10*max(1.0,r1,r2)
        if r2>r1+tol:
            raise RuntimeError("candidate violates R_max_S2 <= R_max_S1")
        if r1 not in pools["r_max_s1_m"] or r2 not in pools["r_max_s2_m"]:
            raise RuntimeError("cutoff radius is not an observed FIT union-kNN edge quantile")
        if tau_graph not in pools["tau_graph_m"]:
            raise RuntimeError("tau_graph is not an observed FIT neighbor/edge scale")
        if not math.isinf(tau_virt) and tau_virt not in pools["tau_virt_m"]:
            raise RuntimeError("finite tau_virt is not an observed FIT neighbor/edge scale")
        key=(r1,tau_graph,r2,tau_virt)
        if key in seen:
            return
        seen.add(key)
        out.append({
            "geometry_id":f"fit_quantile_geometry_{len(out):02d}",
            "r_max_s1_m":r1,"tau_graph_m":tau_graph,"r_max_s2_m":r2,
            "tau_virt_m":tau_virt,"role":role,
            "r_max_s1_quantile":r1_q,"r_max_s2_quantile":r2_q,
            "r_max_s2_le_r_max_s1":True,
        })
    # S1 uses P75/P90/P95/P100. S2 uses P50..P100. Kernel temperatures use
    # neighbor P50 and edge P50/P75/P90 values, never cutoff-derived multipliers.
    for r1_q,r1 in r1_levels:
        r1_quantile_index=CUTOFF_QUANTILES.index(r1_q)
        eligible=[(q,cutoff_by_quantile[q]) for q in CUTOFF_QUANTILES[:r1_quantile_index+1]]
        if not eligible:
            raise RuntimeError(f"no eligible S2 edge quantile for S1 radius {r1_q}")
        r2_low=eligible[0]
        r2_mid=eligible[len(eligible)//2]
        r2_high=eligible[-1]
        combinations=(
            (tau_levels[0],r2_low,math.inf,"low_S2_tau_virt_infinity_control"),
            (tau_levels[1],r2_mid,tau_virt_low,"mid_S2_finite_tau_edge_p50"),
            (tau_levels[2],r2_high,tau_virt_high,"high_S2_finite_tau_edge_p90"),
        )
        for tau_graph,(r2_q,r2),tau_virt,role in combinations:
            add(r1_q,r1,r2_q,r2,tau_graph,tau_virt,
                f"S1_{r1_q}__S2_{r2_q}__{role}")
    if not 12<=len(out)<=16:
        raise RuntimeError(
            f"FIT edge/neighbor quantiles produced {len(out)} unique spatial configurations; expected 12-16"
        )
    if not any(math.isinf(row["tau_virt_m"]) for row in out):
        raise RuntimeError("spatial candidate design omitted tau_virt=+inf control")
    if any(row["r_max_s2_m"]>row["r_max_s1_m"] for row in out):
        raise RuntimeError("spatial candidate violates R_max_S2 <= R_max_S1")
    return out,{
        "by_parameter_m":{
            "r_max_s1_m":pools["r_max_s1_m"],
            "r_max_s2_m":pools["r_max_s2_m"],
            "tau_graph_m":pools["tau_graph_m"],
            "tau_virt_m":pools["tau_virt_m"],
        },
        "cutoff_quantile_values_m":cutoff_by_quantile,
        "scale_derivation":(
            "R_max_S1 uses FIT union-kNN edge P75/P90/P95/P100; "
            "R_max_S2 uses FIT union-kNN edge P50/P75/P90/P95/P100; "
            "kernel temperatures use FIT k=1/2/4/8 neighbor P50 plus edge P50/P75/P90; "
            "no cross-pool multipliers"
        ),
        "geometry_candidate_count":len(out),
        "r_max_s2_le_r_max_s1":True,
        "tau_virt_infinity_control_count":sum(math.isinf(row["tau_virt_m"]) for row in out),
    }


def _validate_geometry_support(coords,base_edges,geometries,
                               min_s2_fraction=MIN_S2_RETAINED_EDGE_FRACTION):
    """Validate actual retained edge support for every proposed FIT geometry."""
    xy=np.asarray(coords,dtype=np.float64)
    edges=np.asarray(torch.as_tensor(base_edges).detach().cpu(),dtype=np.int64).reshape(-1,2)
    if xy.ndim!=2 or xy.shape[1]!=2 or len(xy)<1 or not np.isfinite(xy).all():
        raise RuntimeError("invalid coordinates while validating geometry support")
    if len(edges)==0:
        raise RuntimeError("base union-kNN graph has no edges; spatial search is undefined")
    if not math.isfinite(float(min_s2_fraction)) or not 0.0<float(min_s2_fraction)<=1.0:
        raise RuntimeError("minimum S2 retained-edge fraction must lie in (0,1]")
    if not 12<=len(geometries)<=16:
        raise RuntimeError(f"expected 12-16 spatial geometries, got {len(geometries)}")
    if np.any(edges<0) or np.any(edges>=len(xy)) or np.any(edges[:,0]==edges[:,1]):
        raise RuntimeError("invalid base union-kNN edge indices during support validation")
    if len({tuple(sorted(map(int,e))) for e in edges.tolist()})!=len(edges):
        raise RuntimeError("duplicate base union-kNN edges during support validation")
    lengths=np.linalg.norm(xy[edges[:,0]]-xy[edges[:,1]],axis=1)
    if not np.isfinite(lengths).all():
        raise RuntimeError("nonfinite base union-kNN edge lengths")
    quantiles=_quantiles(lengths)
    allowed_s1=[float(quantiles[q]) for q in S1_CUTOFF_QUANTILES]
    allowed_s2=[float(quantiles[q]) for q in CUTOFF_QUANTILES]
    def is_quantile_radius(value,allowed):
        value=float(value)
        return any(abs(value-candidate)<=1e-8*max(1.0,abs(value),abs(candidate))
                   for candidate in allowed)
    minimum_s2_edges=max(1,int(math.ceil(float(min_s2_fraction)*len(edges)-1e-12)))
    result={}
    for geom in geometries:
        geometry_id=str(geom.get("geometry_id","<missing>"))
        r1=float(geom["r_max_s1_m"]); r2=float(geom["r_max_s2_m"])
        if not math.isfinite(r1) or not math.isfinite(r2) or r1<=0 or r2<=0:
            raise RuntimeError(f"{geometry_id}: cutoff radii must be finite and positive")
        tol=1e-10*max(1.0,r1,r2)
        if r2>r1+tol:
            raise RuntimeError(f"{geometry_id}: R_max_S2 exceeds R_max_S1")
        if not is_quantile_radius(r1,allowed_s1):
            raise RuntimeError(f"{geometry_id}: R_max_S1 is not FIT edge P75/P90/P95/P100")
        if not is_quantile_radius(r2,allowed_s2):
            raise RuntimeError(f"{geometry_id}: R_max_S2 is not FIT edge P50/P75/P90/P95/P100")
        retained_s1=int(np.count_nonzero(lengths<=r1))
        retained_s2=int(np.count_nonzero(lengths<=r2))
        if retained_s1<=0:
            raise RuntimeError(f"{geometry_id}: S1 cutoff retained no base graph edges")
        if retained_s2<minimum_s2_edges:
            raise RuntimeError(
                f"{geometry_id}: S2 retained {retained_s2}/{len(edges)} edges; "
                f"requires at least {minimum_s2_edges} ({min_s2_fraction:.0%})"
            )
        result[geometry_id]={
            "base_edges":int(len(edges)),
            "retained_s1_edges":retained_s1,
            "retained_s1_fraction":float(retained_s1/len(edges)),
            "retained_virtual_edges":retained_s2,
            "retained_virtual_fraction":float(retained_s2/len(edges)),
            "minimum_required_virtual_edges":int(minimum_s2_edges),
            "minimum_required_virtual_fraction":float(min_s2_fraction),
        }
    return result


def _validate_kernel_scales(pools,geometries):
    """Reject stale or hand-edited temperature values at both preflight and run time."""
    if not 12<=len(geometries)<=16:
        raise RuntimeError(f"expected 12-16 spatial geometries, got {len(geometries)}")
    graph_pool=[float(x) for x in pools["tau_graph_m"]]
    virt_pool=[float(x) for x in pools["tau_virt_m"]]
    def matches(value,values):
        return any(math.isclose(value,x,rel_tol=1e-8,abs_tol=1e-10) for x in values)
    for geom in geometries:
        gid=str(geom.get("geometry_id","<missing>"))
        tau_graph=float(geom["tau_graph_m"])
        tau_virt=float(geom["tau_virt_m"])
        if not math.isfinite(tau_graph) or tau_graph<=0 or not matches(tau_graph,graph_pool):
            raise RuntimeError(f"{gid}: tau_graph is not a current FIT neighbor/edge P50 scale")
        if math.isinf(tau_virt):
            if tau_virt<0:
                raise RuntimeError(f"{gid}: tau_virt must be +infinity, not -infinity")
        elif not math.isfinite(tau_virt) or tau_virt<=0 or not matches(tau_virt,virt_pool):
            raise RuntimeError(f"{gid}: finite tau_virt is not a current FIT neighbor/edge P50/P75/P90 scale")
    return True


def _subset_virtual_pack(pack,edge_ids):
    out=dict(pack); idx=torch.as_tensor(edge_ids,dtype=torch.long,device=pack["scores"].device)
    for name in ("virtual_scores","virtual_z","virtual_xy","virtual_sigma","geometric_xy"):
        if name in pack: out[name]=pack[name].index_select(1,idx)
    return out

def _target_class_ids(pack,ctx):
    pairs=ctx["class_pairs"].to(pack["scores"].device)
    target=torch.stack([pack["building"].long(),pack["floor"].long()],dim=1)
    match=(target[:,None,:]==pairs[None,:,:]).all(-1)
    if not bool(match.any(dim=1).all()): raise RuntimeError("target BF class absent from FIT anchor classes")
    return match.long().argmax(dim=1)

def _local_relative_grid(center,lower,upper,*,counterpart_scale=None):
    """Three-point local grid centered on the exact dataset/checkpoint baseline."""
    center=float(center); lower=float(lower); upper=float(upper)
    if not math.isfinite(center) or center<lower-1e-12 or center>upper+1e-12:
        raise ValueError(f"baseline calibration center {center} is outside [{lower},{upper}]")
    if abs(center)<=1e-12:
        scale=abs(float(counterpart_scale or 0.0))
        if scale<=1e-12: scale=(upper-lower)/4.0
        values=[center,center+0.25*scale,center+0.50*scale]
    else:
        values=[]
        for multiplier in CALIBRATION_RELATIVE_MULTIPLIERS+CALIBRATION_EXPANSION_MULTIPLIERS:
            value=center*multiplier
            if lower-1e-12<=value<=upper+1e-12: values.append(float(value))
        values=sorted(set(values+[center]))
        below=sorted((x for x in values if x<center),reverse=True)
        above=sorted(x for x in values if x>center)
        if below and above: values=[below[0],center,above[0]]
        elif below: values=[below[min(1,len(below)-1)],below[0],center]
        elif above: values=[center,above[0],above[min(1,len(above)-1)]]
    values=sorted(set(float(x) for x in values))
    if len(values)!=3 or not any(x==center for x in values):
        raise RuntimeError(f"could not form a 3-point local grid around baseline {center}: {values}")
    return values


def _calibration_candidates(baseline_config):
    """Joint calibration grid centered on this dataset/checkpoint's clean baseline."""
    theta=baseline_config.get("theta")
    if not isinstance(theta,(list,tuple)) or len(theta)!=2:
        raise ValueError("baseline_config must contain theta=[theta_v, theta_s]")
    temperature=float(baseline_config["temperature"])
    theta_v=float(theta[0]); theta_s=float(theta[1])
    temperatures=_local_relative_grid(temperature,*TEMPERATURE_BOUNDS)
    theta_vs=_local_relative_grid(theta_v,*THETA_BOUNDS,counterpart_scale=theta_s)
    theta_ss=_local_relative_grid(theta_s,*THETA_BOUNDS,counterpart_scale=theta_v)
    return [dict(candidate_id=f"cal_{i:05d}",temperature=float(t),theta_v=float(tv),theta_s=float(ts),
        alpha_prior=float(ap),alpha_obs=float(ao),beta=float(beta))
        for i,(t,tv,ts,ap,ao,beta) in enumerate(
            itertools.product(temperatures,theta_vs,theta_ss,ALPHA_PRIORS,ALPHA_OBS,BETAS))]


def _calibration_grid_spec(baseline_config):
    theta_v,theta_s=(float(x) for x in baseline_config["theta"])
    return {"baseline_center":{"temperature":float(baseline_config["temperature"]),
            "theta_v":theta_v,"theta_s":theta_s,
            "baseline_source":baseline_config.get("baseline_source","current_dataset_checkpoint_clean_baseline")},
        "relative_multipliers":list(CALIBRATION_RELATIVE_MULTIPLIERS),
        "expansion_multipliers_for_parameter_boundaries":list(CALIBRATION_EXPANSION_MULTIPLIERS),
        "temperature_bounds":list(TEMPERATURE_BOUNDS),"theta_bounds":list(THETA_BOUNDS),
        "temperature_values":_local_relative_grid(float(baseline_config["temperature"]),*TEMPERATURE_BOUNDS),
        "theta_v_values":_local_relative_grid(theta_v,*THETA_BOUNDS,counterpart_scale=theta_s),
        "theta_s_values":_local_relative_grid(theta_s,*THETA_BOUNDS,counterpart_scale=theta_v),
        "alpha_prior_values":list(ALPHA_PRIORS),"alpha_obs_values":list(ALPHA_OBS),
        "beta_values":list(BETAS),"candidate_count":len(_calibration_candidates(baseline_config))}


def _calibration_candidate_count():
    return 3*3*3*len(ALPHA_PRIORS)*len(ALPHA_OBS)*len(BETAS)


@torch.no_grad()
def evaluate_calibration_grid(pack_all,ctx,graph,s2_edges,geom,candidates,*,endpoint_cache=None,chunk_size=2,row_chunk_size=128):
    """Evaluate candidates with sparse graph multiplication shared across all beta variants."""
    device=pack_all["scores"].device
    coords=ctx["memory"]["coords"].detach().cpu().numpy()
    cached_graph=graph.to(device).coalesce()
    ep=endpoint_cache if endpoint_cache is not None else normalized_virtual_endpoint_weights(s2_edges,coords,geom["tau_virt_m"])
    all_edges=np.asarray(ctx["edges"].detach().cpu(),dtype=np.int64).reshape(-1,2)
    lookup={tuple(sorted(map(int,e))):i for i,e in enumerate(all_edges.tolist())}
    edge_ids=[lookup[tuple(map(int,e))] for e in ep["edges"].tolist()]
    pack=_subset_virtual_pack(pack_all,edge_ids)
    edges=torch.as_tensor(ep["edges"],dtype=torch.long,device=device)
    endpoint={"weight_i":torch.as_tensor(ep["weight_i"],dtype=pack["scores"].dtype,device=device),
              "weight_j":torch.as_tensor(ep["weight_j"],dtype=pack["scores"].dtype,device=device),
              "degree":torch.as_tensor(ep["degree"],dtype=torch.long,device=device)}
    n=len(ctx["memory"]["coords"])
    class_ids=torch.cat([ctx["classes"].to(device).long(),ctx["classes"].to(device).long()[edges[:,0]]])
    centers=torch.cat([pack["xy"],pack["virtual_xy"]],dim=1)
    scales=torch.cat([pack["sigma"],pack["virtual_sigma"]],dim=1)
    ab=ctx["memory"]["building"].to(device).long()
    af=ctx["memory"]["floor"].to(device).long()
    component_ids=torch.cat([torch.arange(n,device=device),edges[:,0]])
    valid=((ab[component_ids][None,:]==pack["building"].long()[:,None])&
           (af[component_ids][None,:]==pack["floor"].long()[:,None]))
    lp=s3.student_t_logpdf(centers,scales,pack["target"]).masked_fill(~valid,-torch.inf)
    keys=pack["keys"]; unique_sites=list(dict.fromkeys(keys)); site_map={k:i for i,k in enumerate(unique_sites)}
    site_ids=torch.as_tensor([site_map[k] for k in keys],dtype=torch.long,device=device)
    site_counts=np.bincount(site_ids.cpu().numpy(),minlength=len(unique_sites)).astype(np.float64)

    # beta is absent from parent and virtual logits, so group all beta variants
    # by the five parameters that actually feed those logits.
    base_fields=("temperature","theta_v","theta_s","alpha_prior","alpha_obs")
    grouped={}
    for idx,candidate in enumerate(candidates):
        signature=tuple(float(candidate[name]) for name in base_fields)
        grouped.setdefault(signature,[]).append((idx,candidate))
    groups=list(grouped.items())
    beta_values=sorted({float(row["beta"]) for row in candidates})
    beta_index={value:i for i,value in enumerate(beta_values)}
    beta_tensor=torch.as_tensor(beta_values,dtype=pack["scores"].dtype,device=device)
    output=[None]*len(candidates)

    for start in range(0,len(groups),chunk_size):
        group_chunk=groups[start:start+chunk_size]
        pars=torch.as_tensor([signature for signature,_ in group_chunk],dtype=pack["scores"].dtype,device=device)
        c=len(group_chunk); k=len(beta_values)
        sums={name:torch.zeros((c,k),dtype=pars.dtype,device=device) for name in
              ("coordinate_mae_m","transport_w1_m","bf_accuracy","bf_cross_entropy","joint_nll")}
        site_sums=torch.zeros((c,k,len(unique_sites)),dtype=pars.dtype,device=device)
        site_bf_correct=torch.zeros((c,k,len(unique_sites)),dtype=torch.int64,device=device)
        for rs in range(0,len(pack["scores"]),row_chunk_size):
            re=min(rs+row_chunk_size,len(pack["scores"])); sl=slice(rs,re)
            scores=pack["scores"][sl]; z=pack["z"][sl]; bsz=len(scores)
            temp=pars[:,0,None,None]; tv=pars[:,1,None,None]; ts=pars[:,2,None,None]
            ap=pars[:,3,None,None]; ao=pars[:,4,None,None]
            prior=scores[None]/temp
            obs=z[None,:,:,0]*tv+z[None,:,:,1]*ts
            # Two sparse graph multiplies per parameter chunk and row chunk, regardless of beta count.
            pn=torch.sparse.mm(cached_graph,prior.reshape(c*bsz,n).T).T.reshape(c,bsz,n)
            on=torch.sparse.mm(cached_graph,obs.reshape(c*bsz,n).T).T.reshape(c,bsz,n)
            base=prior+obs
            parent=(1-ap)*prior+ap*pn+(1-ao)*obs+ao*on
            vbase=(pack["virtual_scores"][sl][None]/temp+
                   pack["virtual_z"][sl][None,:,:,0]*tv+pack["virtual_z"][sl][None,:,:,1]*ts)
            if len(edges):
                delta=parent-base
                v_logits=vbase+.5*(delta.index_select(2,edges[:,0])+delta.index_select(2,edges[:,1]))
            else:
                v_logits=vbase

            # All beta values are broadcast only after parent/v_logits exist.
            deg=endpoint["degree"]
            pmass=torch.where(deg[None,:]>0,1.0-beta_tensor[:,None],
                              torch.ones((k,n),dtype=pars.dtype,device=device))
            pterms=parent[:,None,:,:]+pmass[None,:,None,:].log()
            if len(edges):
                emass=beta_tensor[:,None]*(endpoint["weight_i"]+endpoint["weight_j"])[None,:]
                eterms=v_logits[:,None,:,:]+emass.clamp_min(torch.finfo(pars.dtype).tiny).log()[None,:,None,:]
                eterms=eterms.masked_fill(beta_tensor[None,:,None,None]==0.0,-torch.inf)
                q=torch.softmax(torch.cat([pterms,eterms],dim=-1),dim=-1)
            else:
                q=torch.softmax(pterms,dim=-1)

            sub_centers=centers[sl]; sub_lp=lp[sl]; target=pack["target"][sl]
            target_cls=_target_class_ids({**pack,"building":pack["building"][sl],
                                          "floor":pack["floor"][sl],"scores":pack["scores"][sl]},ctx)
            flat_q=q.reshape(c*k*bsz,-1)
            masses=torch.zeros((c*k*bsz,len(ctx["class_pairs"])),dtype=q.dtype,device=device)
            masses.scatter_add_(1,class_ids[None,:].expand(c*k*bsz,-1),flat_q)
            masses=masses.reshape(c,k,bsz,-1)
            correct=(masses.argmax(-1)==target_cls[None,None,:]).to(q.dtype)
            true_mass=masses.gather(3,target_cls[None,None,:,None].expand(c,k,bsz,1)).squeeze(-1)
            bfce=-true_mass.clamp_min(torch.finfo(q.dtype).tiny).log()
            point=torch.einsum("ckbm,bmd->ckbd",q,sub_centers)
            err=torch.linalg.vector_norm(point-target[None,None,:,:],dim=-1)
            dist=torch.linalg.vector_norm(sub_centers-target[:,None,:],dim=-1)
            risk=(q*dist[None,None,:,:]).sum(-1)
            logq=q.clamp_min(torch.finfo(q.dtype).tiny).log()
            valid_nll=(class_ids[None,:]==target_cls[:,None])[None,None,:,:]&(q>0)
            nll=-torch.logsumexp((logq+sub_lp[None,None,:,:]).masked_fill(~valid_nll,-torch.inf),dim=-1)
            wt=pack["weight"][sl][None,None,:]
            for name,val in (("coordinate_mae_m",err),("transport_w1_m",risk),
                             ("bf_accuracy",correct),("bf_cross_entropy",bfce),("joint_nll",nll)):
                sums[name]+=(val*wt).sum(-1)
            site_sums.reshape(c*k,len(unique_sites)).scatter_add_(
                1,site_ids[sl][None,:].expand(c*k,-1),err.reshape(c*k,bsz))
            site_bf_correct.reshape(c*k,len(unique_sites)).scatter_add_(
                1,site_ids[sl][None,:].expand(c*k,-1),correct.to(torch.int64).reshape(c*k,bsz))

        site_means=site_sums/torch.as_tensor(site_counts,dtype=pars.dtype,device=device)[None,None,:]
        p95=torch.quantile(site_means,.95,dim=2).cpu().tolist()
        p99=torch.quantile(site_means,.99,dim=2).cpu().tolist()
        worst=site_means.max(2).values.cpu().tolist()
        sums["bf_accuracy"]=(site_bf_correct.to(torch.float64)/
            torch.as_tensor(site_counts,dtype=torch.float64,device=device)[None,None,:]).mean(2)
        vals={name:value.cpu().tolist() for name,value in sums.items()}
        for ci,(_,members) in enumerate(group_chunk):
            for original_index,candidate in members:
                bi=beta_index[float(candidate["beta"])]
                row=dict(candidate)
                row.update({name:float(values[ci][bi]) for name,values in vals.items()})
                row.update(site_p95_mae_m=float(p95[ci][bi]),site_p99_mae_m=float(p99[ci][bi]),
                    worst_site_mae_m=float(worst[ci][bi]),n_fit_rows=len(pack["scores"]),
                    n_fit_sites=len(unique_sites))
                output[original_index]=row
    rows_out=[row for row in output if row is not None]
    return rows_out if len(edges) else [row for row in rows_out if float(row["beta"])==0.0]


def _primary(r):
    """Primary tie-break metric: point-coordinate MAE only."""
    return float(r["coordinate_mae_m"])


def _clean_baseline(rows):
    clean=[r for r in rows if r["alpha_prior"]==0 and r["alpha_obs"]==0 and r["beta"]==0]
    if not clean: raise RuntimeError("joint grid omitted clean-base alpha=0,beta=0 boundary")
    return min(clean,key=lambda r:(_primary(r),float(r["transport_w1_m"]),
        float(r["joint_nll"]),r["temperature"],r["theta_v"],r["theta_s"]))


def _add_objectives(rows,base):
    maebase=max(float(base["coordinate_mae_m"]),1e-9)
    w1base=max(float(base["transport_w1_m"]),1e-9)
    nllbase=float(base["joint_nll"])
    nscale=max(abs(nllbase),1.0)
    bfbase=float(base["bf_cross_entropy"])
    bfdenom=max(bfbase,BF_CE_NORMALIZATION_FLOOR)
    accbase=float(base["bf_accuracy"])
    maetol=MAE_NON_DEGRADATION_TOLERANCE_M
    for r in rows:
        mae=float(r["coordinate_mae_m"])
        w1=float(r["transport_w1_m"])
        bfce=float(r["bf_cross_entropy"])
        nll=float(r["joint_nll"])
        r["primary_coordinate_mae_m"]=mae
        r["normalized_primary"]=mae/maebase
        r["normalized_w1_delta"]=(w1-float(base["transport_w1_m"]))/w1base
        r["normalized_joint_nll_delta"]=(nll-nllbase)/nscale
        r["normalized_bf_cross_entropy"]=bfce/bfdenom
        # Keep the signed delta for diagnostics; the objective uses only the
        # one-sided penalty so a BF-CE improvement cannot compensate for MAE loss.
        r["normalized_bf_cross_entropy_delta"]=(bfce-bfbase)/bfdenom
        r["normalized_bf_cross_entropy_penalty"]=max(0.0,bfce-bfbase)/bfdenom
        r["negative_bf_accuracy"]=-float(r["bf_accuracy"])
        r["mae_non_degradation"]=bool(
            math.isfinite(mae) and mae<=float(base["coordinate_mae_m"])+maetol
        )
        r["bf_accuracy_non_degradation"]=bool(
            math.isfinite(float(r["bf_accuracy"])) and float(r["bf_accuracy"])>=accbase-1e-12
        )
        r["bf_cross_entropy_guard"]=bool(
            math.isfinite(bfce) and bfce<=bfbase*(1.0+BF_CE_RELATIVE_TOLERANCE)+BF_CE_ABSOLUTE_TOLERANCE
        )
        r["feasible"]=bool(
            r["mae_non_degradation"] and r["bf_accuracy_non_degradation"] and r["bf_cross_entropy_guard"]
        )
        for lam in LAMBDAS:
            r[f"selection_objective_lambda_{lam:.2f}"]=(
                r["normalized_primary"]
                + W1_WEIGHT*r["normalized_w1_delta"]
                + BF_CE_WEIGHT*r["normalized_bf_cross_entropy_penalty"]
                + lam*r["normalized_joint_nll_delta"]
            )
    return rows


def _is_clean_anchor_candidate(row,baseline_config):
    if row.get("candidate_id")=="clean_baseline_reference":
        return False
    return (
        float(row.get("alpha_prior",math.nan))==0.0
        and float(row.get("alpha_obs",math.nan))==0.0
        and float(row.get("beta",math.nan))==0.0
        and math.isclose(float(row.get("temperature",math.nan)),
                         float(baseline_config["temperature"]),rel_tol=0.0,abs_tol=1e-12)
        and math.isclose(float(row.get("theta_v",math.nan)),
                         float(baseline_config["theta"][0]),rel_tol=0.0,abs_tol=1e-12)
        and math.isclose(float(row.get("theta_s",math.nan)),
                         float(baseline_config["theta"][1]),rel_tol=0.0,abs_tol=1e-12)
    )


def _assert_clean_anchor_feasible(rows,baseline_config,expected_geometry_count):
    anchors=[r for r in rows if _is_clean_anchor_candidate(r,baseline_config)]
    if len(anchors)!=int(expected_geometry_count):
        raise RuntimeError(
            f"expected one exact clean calibration anchor per geometry ({expected_geometry_count}); found {len(anchors)}"
        )
    anchor_geometry_ids={str(r.get("geometry_id")) for r in anchors}
    if len(anchor_geometry_ids)!=int(expected_geometry_count):
        raise RuntimeError(
            f"exact clean calibration anchors do not cover distinct geometries: "
            f"{len(anchor_geometry_ids)} unique geometry IDs for {expected_geometry_count} geometries"
        )
    baseline_refs=[r for r in rows if r.get("candidate_id")=="clean_baseline_reference"]
    if len(baseline_refs)!=1:
        raise RuntimeError(f"expected one clean baseline reference row, found {len(baseline_refs)}")
    failed=[r for r in anchors if not r.get("feasible",False)]
    if failed:
        baseline=next((x for x in rows if x.get("candidate_id")=="clean_baseline_reference"),None)
        details=[{
            "candidate_id":r.get("candidate_id"),
            "mae_non_degradation":r.get("mae_non_degradation"),
            "bf_accuracy_non_degradation":r.get("bf_accuracy_non_degradation"),
            "bf_cross_entropy_guard":r.get("bf_cross_entropy_guard"),
            "mae_delta_m":(
                float(r["coordinate_mae_m"])-float(baseline["coordinate_mae_m"])
                if baseline is not None else None
            ),
        } for r in failed[:3]]
        raise RuntimeError(
            "exact clean calibration anchor is not feasible; stop rather than relaxing the MAE/BF guards: "
            +json.dumps(details,ensure_ascii=False)
        )


def _choose(rows,lam,kind="joint"):
    objective_key=f"selection_objective_lambda_{lam:.2f}"
    metric_fields=("coordinate_mae_m","transport_w1_m","bf_accuracy","bf_cross_entropy","joint_nll",
                   "site_p95_mae_m","site_p99_mae_m","worst_site_mae_m")
    rs=[r for r in rows if r.get("candidate_id")!="clean_baseline_reference" and r.get("feasible") and
        math.isfinite(float(r.get(objective_key,math.inf))) and
        all(math.isfinite(float(r.get(field,math.nan))) for field in metric_fields)]
    if kind=="single_S1": rs=[r for r in rs if r["beta"]==0 and (r["alpha_prior"]!=0 or r["alpha_obs"]!=0)]
    if kind=="single_S2": rs=[r for r in rs if r["alpha_prior"]==0 and r["alpha_obs"]==0 and r["beta"]>0]
    return min(rs,key=lambda r:(r[objective_key],_primary(r),r["transport_w1_m"],r["joint_nll"],
        r["temperature"],r["theta_v"],r["theta_s"],r["alpha_prior"],r["alpha_obs"],r["beta"],
        r["geometry_id"])) if rs else None
def _near_optimal_sensitivity(rows):
    """Summarize CAL-fit near-optimal parameter ambiguity without CAL-eval access."""
    parameter_fields=("r_max_s1_m","tau_graph_m","r_max_s2_m","tau_virt_m","temperature",
                      "theta_v","theta_s","alpha_prior","alpha_obs","beta")
    result={}
    for lam in LAMBDAS:
        best=_choose(rows,lam)
        if best is None: raise RuntimeError(f"no feasible candidate for near-optimal lambda={lam}")
        objective_key=f"selection_objective_lambda_{lam:.2f}"
        best_value=float(best[objective_key])
        near=[r for r in rows if r.get("candidate_id")!="clean_baseline_reference" and r.get("feasible") and
              math.isfinite(float(r[objective_key])) and
              float(r[objective_key])<=best_value+NEAR_OPTIMAL_OBJECTIVE_TOLERANCE+1e-12]
        near.sort(key=lambda r:(float(r[objective_key]),_primary(r),r["transport_w1_m"],r["joint_nll"],
            r["temperature"],r["theta_v"],r["theta_s"],r["alpha_prior"],
            r["alpha_obs"],r["beta"],r["geometry_id"]))
        ranges={}
        for field in parameter_fields:
            vals=np.asarray([float(r[field]) for r in near],dtype=np.float64)
            finite=vals[np.isfinite(vals)]
            ranges[field]={"finite_count":int(len(finite)),
                "unbounded_count":int(len(vals)-len(finite)),
                "min":float(finite.min()) if len(finite) else None,
                "median":float(np.median(finite)) if len(finite) else None,
                "max":float(finite.max()) if len(finite) else None}
        result[f"lambda_{lam:.2f}"]={"objective_tolerance_absolute":NEAR_OPTIMAL_OBJECTIVE_TOLERANCE,
            "best_objective":best_value,"best_candidate_id":best.get("candidate_id"),
            "near_optimal_candidate_count":len(near),
            "near_optimal_geometry_count":len({r["geometry_id"] for r in near}),
            "parameter_ranges":ranges,
            "top_candidate_ids":[r.get("candidate_id") for r in near[:25]]}
    return result


def _config(r):
    return {"geometry_id":r["geometry_id"],"R_max_S1_m":float(r["r_max_s1_m"]),"tau_graph_m":float(r["tau_graph_m"]),
        "R_max_S2_m":float(r["r_max_s2_m"]),"tau_virt_m":float(r["tau_virt_m"]),
        "temperature":float(r["temperature"]),"theta":[float(r["theta_v"]),float(r["theta_s"])],
        "alpha_prior":float(r["alpha_prior"]),"alpha_obs":float(r["alpha_obs"]),"beta":float(r["beta"]),
        "gamma_space":0.0,"selection_objective":{f"lambda_{lam:.2f}":float(r[f"selection_objective_lambda_{lam:.2f}"]) for lam in LAMBDAS},
        "normalized_primary":float(r["normalized_primary"]),
        "normalized_w1_delta":float(r["normalized_w1_delta"]),
        "normalized_joint_nll_delta":float(r["normalized_joint_nll_delta"]),
        "normalized_bf_cross_entropy":float(r["normalized_bf_cross_entropy"]),
        "normalized_bf_cross_entropy_delta":float(r["normalized_bf_cross_entropy_delta"]),
        "normalized_bf_cross_entropy_penalty":float(r["normalized_bf_cross_entropy_penalty"]),
        "mae_non_degradation":bool(r["mae_non_degradation"]),
        "bf_accuracy_non_degradation":bool(r["bf_accuracy_non_degradation"]),
        "bf_cross_entropy_guard":bool(r["bf_cross_entropy_guard"]),
        "fit_metrics":{k:float(r[k]) for k in ("coordinate_mae_m","transport_w1_m","bf_accuracy",
        "bf_cross_entropy","joint_nll","site_p95_mae_m","site_p99_mae_m","worst_site_mae_m")}}
def _csv_write(path,rows):
    rows=list(rows); fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore"); w.writeheader()
        for r in rows: w.writerow({k:(json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list,tuple)) else v) for k,v in r.items()})
def _csv_read(path):
    with Path(path).open(encoding="utf-8-sig",newline="") as f: return list(csv.DictReader(f))
def _safe_rows(ctx,rows,role):
    if role not in ("CAL-fit","CAL-eval"): raise RuntimeError(f"unauthorized row role {role}")
    allowed=set(map(int,ctx["cal_fit"] if role=="CAL-fit" else ctx["cal_eval"])); arr=np.asarray(rows,dtype=np.int64)
    if any(int(i) not in allowed for i in arr): raise RuntimeError(f"{role} rows escaped the assigned CAL split")
    if set(map(int,arr))&set(map(int,ctx["roles"]["test"])): raise RuntimeError("TEST row guard blocked pack construction")
    return arr

def _make_probability(pack_all,ctx,config):
    coords=ctx["memory"]["coords"].detach().cpu().numpy().astype(np.float64,copy=False); b=ctx["memory"]["building"].detach().cpu().numpy()
    f=ctx["memory"]["floor"].detach().cpu().numpy(); all_edges=np.asarray(ctx["edges"].detach().cpu(),dtype=np.int64).reshape(-1,2)
    graph,_,gdiag=build_cutoff_graph(coords,b,f,all_edges,config["R_max_S1_m"],config["tau_graph_m"])
    lengths=np.linalg.norm(coords[all_edges[:,0]]-coords[all_edges[:,1]],axis=1) if len(all_edges) else np.empty(0)
    e=all_edges[lengths<=float(config["R_max_S2_m"])] if len(all_edges) else all_edges
    ep=normalized_virtual_endpoint_weights(e,coords,config["tau_virt_m"])
    lookup={tuple(sorted(map(int,x))):i for i,x in enumerate(all_edges.tolist())}
    pack=_subset_virtual_pack(pack_all,[lookup[tuple(sorted(map(int,x)))] for x in ep["edges"].tolist()])
    edges=torch.as_tensor(ep["edges"],dtype=torch.long,device=pack["scores"].device)
    weights={k:torch.as_tensor(ep[k],device=pack["scores"].device) for k in ("weight_i","weight_j","degree")}
    theta=torch.as_tensor(config["theta"],dtype=pack["scores"].dtype,device=pack["scores"].device)
    before,logits=op.full_logit_graph_diffusion(pack["scores"],pack["z"],theta,config["temperature"],
        graph.to(pack["scores"].device),config["alpha_prior"],config["alpha_obs"])
    v=op.combine_retrieval_and_g13_logits(pack["virtual_scores"],pack["virtual_z"],config["temperature"],theta)
    if len(edges): v=comp.propagate_s1_delta_to_virtual_logits(v,logits,before,edges)
    mix=weighted_cell_mixture(logits,v,edges,weights,config["beta"],pack["xy"],pack["sigma"],pack["virtual_xy"],pack["virtual_sigma"])
    return mix,pack,gdiag
def _eval_config(pack,ctx,config,seed):
    mix,sub,gdiag=_make_probability(pack,ctx,config)
    metrics,sites=s3.score_mixture(sub,ctx,mix["q"],centers=mix["xy"],scales=mix["sigma"],ids=mix["ids"],seed=seed+900)
    cls=ctx["classes"].to(mix["q"].device)[mix["ids"]]
    masses=torch.zeros((len(mix["q"]),len(ctx["class_pairs"])),dtype=mix["q"].dtype,device=mix["q"].device)
    masses.scatter_add_(1,cls,mix["q"]); target_cls=_target_class_ids(sub,ctx)
    tm=masses.gather(1,target_cls[:,None]).squeeze(1)
    metrics["bf_cross_entropy"]=float(((-tm.clamp_min(torch.finfo(mix["q"].dtype).tiny).log())*sub["weight"]).sum().cpu())
    metrics["geometry_diagnostics"]=gdiag
    return {"config":config,"metrics":metrics,"site_diagnostics":sites}
def _old_combo(dataset,seed,arm,checkpoint_sha256):
    path=comp.DEFAULT_OUTPUT/"runs"/f"{dataset}_seed{seed}"/arm/"CAL_FIT_SELECTION.json"
    if not path.is_file(): return None
    s=_read_json(path)
    if s.get("checkpoint_sha256")!=checkpoint_sha256: return None
    try:
        clean=s["joint_calibrated_base"]; s1=s["structure1"]["1D_global"]; combo=s["combined"]["1D_global"]["2A"]
        return {"geometry_id":"legacy_unbounded_8m","R_max_S1_m":math.inf,"tau_graph_m":8.,
            "R_max_S2_m":math.inf,"tau_virt_m":math.inf,"temperature":float(clean["temperature"]),
            "theta":[float(x) for x in clean["theta"]],"alpha_prior":float(s1["alpha_prior"]),
            "alpha_obs":float(s1["alpha_obs"]),"beta":float(combo["beta"]),"gamma_space":0.0,
            "legacy_selection_sha256":s3.sha(path)}
    except (KeyError,TypeError,ValueError): return None


def _group_indices(buildings,floors):
    groups={}
    for i,key in enumerate(zip(np.asarray(buildings).tolist(),np.asarray(floors).tolist())):
        groups.setdefault(tuple(key),[]).append(i)
    return groups

def _attach_virtual_measure(config,geometry_diagnostics):
    diag=geometry_diagnostics[config["geometry_id"]]["S2"]
    parent_measure=diag["effective_virtual_mass_by_parent"]
    beta=float(config["beta"])
    config["retained_virtual_edges"]=int(diag["retained_virtual_edges"])
    config["beta_weighted_virtual_mass_by_parent"]=[beta*float(x) for x in parent_measure]
    config["physical_measure_by_parent"]=[(1.0-beta if float(x)>0 else 1.0) for x in parent_measure]
    config["pre_normalization_virtual_measure_total"]=float(beta*sum(parent_measure))
    config["pre_normalization_physical_measure_total"]=float(sum(config["physical_measure_by_parent"]))
    config["pre_normalization_total_measure"]=config["pre_normalization_virtual_measure_total"]+config["pre_normalization_physical_measure_total"]
    if not math.isclose(config["pre_normalization_total_measure"],float(len(parent_measure)),rel_tol=1e-8,abs_tol=1e-8):
        raise RuntimeError("selected S2 physical-plus-virtual measure is not conserved")
    return config

def _legacy_clean_config(dataset,seed,arm,fitpack,ctx,checkpoint_sha256):
    path=comp.DEFAULT_OUTPUT/"runs"/f"{dataset}_seed{seed}"/arm/"CAL_FIT_SELECTION.json"
    saved=_read_json(path) if path.is_file() else {}
    clean=saved.get("joint_calibrated_base")
    if clean is not None and saved.get("checkpoint_sha256")==checkpoint_sha256:
        source="frozen_prior_CAL_FIT_SELECTION_same_checkpoint"
    else:
        clean,_,_=s3.grid_calibrate(fitpack,ctx)
        source="recomputed_current_checkpoint_CAL_fit_only"
    return {"geometry_id":"legacy_unbounded_8m","R_max_S1_m":math.inf,"tau_graph_m":8.0,
        "R_max_S2_m":math.inf,"tau_virt_m":math.inf,"temperature":float(clean["temperature"]),
        "theta":[float(x) for x in clean["theta"]],"alpha_prior":0.0,"alpha_obs":0.0,"beta":0.0,
        "gamma_space":0.0,"baseline_source":source}

def _baseline_row(config,metrics):
    m=metrics["metrics"]
    row={"candidate_id":"clean_baseline_reference","geometry_id":"legacy_unbounded_8m",
        "r_max_s1_m":math.inf,"tau_graph_m":8.0,"r_max_s2_m":math.inf,"tau_virt_m":math.inf,
        "temperature":float(config["temperature"]),"theta_v":float(config["theta"][0]),
        "theta_s":float(config["theta"][1]),"alpha_prior":0.0,"alpha_obs":0.0,"beta":0.0,
        "coordinate_mae_m":float(m["macro_mae_full_m"]),
        "transport_w1_m":float(m["macro_transport_risk_m"]),
        "bf_accuracy":float(m["macro_bf_accuracy"]),"bf_cross_entropy":float(m["bf_cross_entropy"]),
        "joint_nll":float(m["macro_joint_nll"]),"site_p95_mae_m":float(m["site_p95_full_m"]),
        "site_p99_mae_m":float(m["site_p99_full_m"]),"worst_site_mae_m":float(m["worst_site_mae_m"]),
        "baseline_source":config["baseline_source"]}
    return row

def preflight(args):
    """Build FIT-only geometry diagnostics and candidate scales; no labels are scored."""
    out=Path(args.output).resolve()
    diagnostics={}; spaces={}
    for ds in s3.DATASETS:
      for seed in s3.SEEDS:
        ctx=s3.reconstruct(ds,seed,Path(args.input_root).resolve(),torch.device("cpu"))
        fit=_safe_rows(ctx,ctx["cal_fit"],"CAL-fit")
        ev=_safe_rows(ctx,ctx["cal_eval"],"CAL-eval")
        xy=ctx["memory"]["coords"].detach().cpu().numpy()
        b=ctx["memory"]["building"].detach().cpu().numpy()
        f=ctx["memory"]["floor"].detach().cpu().numpy()
        d=_pairwise_fit_geometry(xy,b,f,ctx["edges"]); key=f"{ds}_seed{seed}"
        d.update(dataset=ds,seed=int(seed),cal_fit_rows=len(fit),cal_eval_rows=len(ev))
        diagnostics[key]=d
        geoms,pools=geometry_candidates(d)
        _validate_kernel_scales(pools["by_parameter_m"],geoms)
        support=_validate_geometry_support(xy,ctx["edges"],geoms)
        for geom in geoms: geom["support_diagnostics"]=support[geom["geometry_id"]]
        spaces[key]={"dataset":ds,"seed":int(seed),
          "geometry_candidate_pools_m":pools["by_parameter_m"],
          "cutoff_quantile_values_m":pools["cutoff_quantile_values_m"],
          "geometry_candidates":geoms,
          "calibration_candidate_count":_calibration_candidate_count(),
          "gamma_space":0.0,"lambda_2d_nll":list(LAMBDAS),"bf_ce_weight":BF_CE_WEIGHT,
          "w1_weight":W1_WEIGHT,"selection_objective_formula":SELECTION_OBJECTIVE_FORMULA,
          "mae_non_degradation_tolerance_m":MAE_NON_DEGRADATION_TOLERANCE_M,
          "candidate_count_per_run":len(geoms)*_calibration_candidate_count()}
    _write_json(out/"FIT_NEIGHBOR_DISTANCE_DIAGNOSTICS.json",{
        "source":"FIT anchor dictionary only","datasets_and_seeds":diagnostics,**TEST_COUNTS})
    md=["# FIT-only Neighbor Distance Diagnostics","",
        "Nearest-neighbor distances are computed among distinct anchors on the same building and floor. Union-kNN edge lengths are reported separately.",""]
    for key,d in diagnostics.items():
        md += [f"## {key}","",f"- FIT anchors: {d['anchors']}",
          f"- Building-floor strata: {d['building_floor_strata']}",
          f"- Zero-distance pairs: {d['zero_distance_pairs_within_building_floor']}",
          f"- Existing union-kNN edges: {d['base_union_knn_edges']}",
          f"- Union-kNN edge quantiles: {json.dumps(d['base_union_knn_edge_distance_m'],ensure_ascii=False)}"]
        for k,v in d["kth_neighbor_distance_m"].items():
            md.append(f"- k={k} neighbor distances: {json.dumps(v,ensure_ascii=False)}")
        md.append(f"- k-neighbor threshold counts: {json.dumps(d['kth_neighbor_threshold_counts'],ensure_ascii=False)}")
        md.append(f"- exp(-d/8m) at distance quantiles: {json.dumps(d['exp_neg_distance_over_8_at_quantiles'],ensure_ascii=False)}")
        md.append("")
    (out/"FIT_NEIGHBOR_DISTANCE_DIAGNOSTICS.md").write_text("\n".join(md),encoding="utf-8")
    _write_json(out/"SPATIAL_SCALE_SEARCH_SPACE.json",{
      "geometry_design":"12 deterministic FIT-derived spatial configurations per dataset/seed; cutoff radii use actual union-kNN edge P50/P75/P90/P95/P100 only; S1 uses P75/P90/P95/P100; S2 retains at least 45% of base edges; includes tau_virt=+inf controls",
      "selection_objective_formula":SELECTION_OBJECTIVE_FORMULA,
      "w1_objective_weight":W1_WEIGHT,
      "mae_non_degradation_tolerance_m":MAE_NON_DEGRADATION_TOLERANCE_M,
      "outer_geometry_candidates":spaces,
      "calibration_grid":{
        "temperature":"three-point relative neighborhood around dataset/seed/checkpoint clean baseline; legal bounds [0.25,3.0]",
        "theta_v":"three-point relative neighborhood around dataset/seed/checkpoint clean baseline; legal bounds [0.0,1.5]",
        "theta_s":"three-point relative neighborhood around dataset/seed/checkpoint clean baseline; legal bounds [0.0,1.5]",
        "relative_multipliers":list(CALIBRATION_RELATIVE_MULTIPLIERS),
        "alpha_prior":list(ALPHA_PRIORS),"alpha_obs":list(ALPHA_OBS),"beta":list(BETAS),
        "baseline_center_resolved_at_run_time":True},
      "inner_joint_grid_size":_calibration_candidate_count(),"gamma_space_fixed":0.0,
      "lambda_2d_nll":list(LAMBDAS),"bf_ce_weight":BF_CE_WEIGHT,"w1_weight":W1_WEIGHT,
      "mae_non_degradation_tolerance_m":MAE_NON_DEGRADATION_TOLERANCE_M,
      "selection_objective_formula":SELECTION_OBJECTIVE_FORMULA,
      "bf_ce_normalization_floor":BF_CE_NORMALIZATION_FLOOR,
      "bf_ce_guard_relative_tolerance":BF_CE_RELATIVE_TOLERANCE,
      "bf_ce_guard_absolute_tolerance":BF_CE_ABSOLUTE_TOLERANCE,**TEST_COUNTS})
    print(json.dumps({"stage":"preflight_complete","output":str(out),
      "geometry_sets":len(spaces),**TEST_COUNTS},ensure_ascii=False),flush=True)


def run(args):
    """Jointly fit spatial scales and calibration parameters on CAL-fit, then report CAL-eval."""
    out=Path(args.output).resolve()
    search=_read_json(out/"SPATIAL_SCALE_SEARCH_SPACE.json")["outer_geometry_candidates"]
    reports=[]
    datasets=_selected_datasets(getattr(args,"dataset","all"))
    for ds in datasets:
      for seed in s3.SEEDS:
        ctx=s3.reconstruct(ds,seed,Path(args.input_root).resolve(),torch.device(args.device))
        fit=_safe_rows(ctx,ctx["cal_fit"],"CAL-fit")
        ev=_safe_rows(ctx,ctx["cal_eval"],"CAL-eval")
        coords=ctx["memory"]["coords"].detach().cpu().numpy().astype(np.float64,copy=False)
        b=ctx["memory"]["building"].detach().cpu().numpy()
        f=ctx["memory"]["floor"].detach().cpu().numpy()
        base_edges=np.asarray(ctx["edges"].detach().cpu(),dtype=np.int64).reshape(-1,2)
        runtime_diag=_pairwise_fit_geometry(coords,b,f,base_edges)
        runtime_pools=_geometry_candidate_pools(runtime_diag)
        for arm in ARMS:
            rd=out/"runs"/f"{ds}_seed{seed}"/arm
            rd.mkdir(parents=True,exist_ok=True)
            sp=rd/"CAL_FIT_SELECTION.json"; rp=rd/"CAL_EVAL_REPORT.json"
            ckpt=comp._checkpoint_path(Path(args.checkpoint_root).resolve(),ds,seed,arm)
            ckpt_sha=s3.sha(ckpt)
            model,payload,native_alpha=comp._load_model(ctx,Path(args.checkpoint_root).resolve(),ds,seed,arm,args.device)
            fitpack=s3.make_pack(ctx,model,fit,virtual_gamma=0.0)
            baseline_config=_legacy_clean_config(ds,seed,arm,fitpack,ctx,ckpt_sha)
            inner=_calibration_candidates(baseline_config)
            calibration_grid=_calibration_grid_spec(baseline_config)
            geoms=search[f"{ds}_seed{seed}"]["geometry_candidates"]
            _validate_kernel_scales(runtime_pools,geoms)
            support_by_geometry=_validate_geometry_support(coords,base_edges,geoms)
            rows_all=[]; geometry_diagnostics={}
            for geom in geoms:
                support=support_by_geometry[geom["geometry_id"]]
                if float(geom["r_max_s2_m"])>float(geom["r_max_s1_m"]):
                    raise RuntimeError("preflight geometry violates R_max_S2 <= R_max_S1")
                graph,kept_s1,gdiag=build_cutoff_graph(coords,b,f,base_edges,geom["r_max_s1_m"],geom["tau_graph_m"])
                if int(gdiag["retained_edges"])<=0 or int(len(kept_s1))!=int(support["retained_s1_edges"]):
                    raise RuntimeError(f"{geom['geometry_id']}: S1 edge support differs from preflight validation")
                lens=np.linalg.norm(coords[base_edges[:,0]]-coords[base_edges[:,1]],axis=1) if len(base_edges) else np.empty(0)
                s2=base_edges[lens<=float(geom["r_max_s2_m"])] if len(base_edges) else base_edges
                if len(s2)<int(support["minimum_required_virtual_edges"]) or int(len(s2))!=int(support["retained_virtual_edges"]):
                    raise RuntimeError(f"{geom['geometry_id']}: S2 edge support below validated minimum")
                epdiag=normalized_virtual_endpoint_weights(s2,coords,geom["tau_virt_m"])
                s2lens=np.linalg.norm(coords[s2[:,0]]-coords[s2[:,1]],axis=1) if len(s2) else np.empty(0)
                gdiag["S2"]={"r_max_s2_m":float(geom["r_max_s2_m"]),"tau_virt_m":float(geom["tau_virt_m"]),
                    "base_edges":int(len(base_edges)),"retained_virtual_edges":int(len(s2)),
                    "removed_virtual_edges":int(len(base_edges)-len(s2)),
                    "retained_fraction":float(len(s2)/len(base_edges)) if len(base_edges) else 0.0,
                    "distance_m":_quantiles(s2lens),"distance_threshold_counts":_threshold_counts(s2lens),
                    "zero_virtual_neighbor_nodes":int((epdiag["degree"]==0).sum()),
                    "effective_virtual_mass_by_parent":epdiag["effective_virtual_mass_by_parent"],
                    "effective_virtual_mass_quantiles":_quantiles(epdiag["effective_virtual_mass_by_parent"])}
                gdiag["preflight_support_validation"]=support
                geometry_diagnostics[geom["geometry_id"]]=gdiag
                candidates=[dict(x,candidate_id=f'{geom["geometry_id"]}__{x["candidate_id"]}',
                    geometry_id=geom["geometry_id"],r_max_s1_m=geom["r_max_s1_m"],
                    tau_graph_m=geom["tau_graph_m"],r_max_s2_m=geom["r_max_s2_m"],
                    tau_virt_m=geom["tau_virt_m"],geometry_role=geom["role"]) for x in inner]
                rows=evaluate_calibration_grid(fitpack,ctx,graph.to(args.device),s2,geom,candidates,endpoint_cache=epdiag)
                for r in rows:
                    r.update(dataset=ds,seed=int(seed),backbone=arm,checkpoint_sha256=ckpt_sha,
                        split_hash=s3.key_sha(ctx["keys"][int(i)] for i in fit))
                rows_all.extend(rows)
            baseline_eval=_eval_config(fitpack,ctx,baseline_config,seed)
            base=_baseline_row(baseline_config,baseline_eval)
            base.update(dataset=ds,seed=int(seed),backbone=arm,checkpoint_sha256=ckpt_sha,
                split_hash=s3.key_sha(ctx["keys"][int(i)] for i in fit),geometry_role="clean_baseline_reference")
            rows_all.append(base)
            rows_all=_add_objectives(rows_all,base)
            _assert_clean_anchor_feasible(rows_all,baseline_config,len(geoms))
            metric_fields=("coordinate_mae_m","transport_w1_m","bf_accuracy","bf_cross_entropy","joint_nll",
                "site_p95_mae_m","site_p99_mae_m","worst_site_mae_m")
            finite_count=sum(all(math.isfinite(float(r.get(k,math.nan))) for k in metric_fields) for r in rows_all)
            chosen={}; s1={}; s2sel={}
            for lam in LAMBDAS:
                r=_choose(rows_all,lam)
                if r is None: raise RuntimeError(f"no feasible candidate for lambda={lam}")
                chosen[f"lambda_{lam:.2f}"]=_config(r)
                r1=_choose(rows_all,lam,"single_S1"); s1[f"lambda_{lam:.2f}"]=_config(r1) if r1 else None
                r2=_choose(rows_all,lam,"single_S2"); s2sel[f"lambda_{lam:.2f}"]=_config(r2) if r2 else None
            clean=_config(base)  # reporting-only legacy reference, not a geometry search candidate
            for config in chosen.values(): _attach_virtual_measure(config,geometry_diagnostics)
            for config in list(s1.values())+list(s2sel.values()):
                if config is not None: _attach_virtual_measure(config,geometry_diagnostics)
            near_optimal=_near_optimal_sensitivity(rows_all)
            selection={"status":"CAL_FIT_SELECTED","dataset":ds,"seed":int(seed),"backbone":arm,
                "checkpoint_sha256":ckpt_sha,"selected_epoch":int(payload["selected_epoch"]),
                "native_E_alpha":native_alpha,"selection_role":"CAL-fit",
                "cal_eval_not_used_for_selection":True,"gamma_space":0.0,"gamma_candidate_count":0,
                "joint_calibrated_base":clean,"baseline_reference_source":baseline_config["baseline_source"],
                "selected_joint":chosen,"single_S1":s1,"single_S2":s2sel,
                "near_optimal_sensitivity":near_optimal,
                "near_optimal_objective_tolerance_absolute":NEAR_OPTIMAL_OBJECTIVE_TOLERANCE,
                "finite_candidate_metric_count":int(finite_count),
                "nonfinite_candidate_metric_count":int(len(rows_all)-finite_count),
                "lambda_2d_nll":list(LAMBDAS),"bf_ce_weight":BF_CE_WEIGHT,"w1_weight":W1_WEIGHT,
                "candidate_count":len(rows_all),"geometry_count":len(geoms),"inner_grid_count":len(inner),
                "calibration_grid":calibration_grid,
                "geometry_diagnostics":geometry_diagnostics,
                "fit_split_sha256":s3.key_sha(ctx["keys"][int(i)] for i in fit),
                "eval_split_sha256":s3.key_sha(ctx["keys"][int(i)] for i in ev),
                "selection_objective_formula":SELECTION_OBJECTIVE_FORMULA,
                "w1_objective_weight":W1_WEIGHT,
                "mae_non_degradation_guard":(
                    f"candidate coordinate MAE <= clean CAL-fit baseline + "
                    f"{MAE_NON_DEGRADATION_TOLERANCE_M:g} m"
                ),
                "bf_accuracy_guard":"candidate macro BF accuracy >= clean CAL-fit baseline",
                "bf_cross_entropy_guard":"candidate BF CE <= clean baseline * 1.02 + 0.005",
                "bf_cross_entropy_objective_penalty":"max(0, candidate BF-CE - clean BF-CE) / max(clean BF-CE, 0.05)",
                "bf_cross_entropy_normalization_denominator":"max(clean BF-CE, 0.05)",
                **TEST_COUNTS}
            _write_json(sp,selection)
            evalpack=s3.make_pack(ctx,model,ev,virtual_gamma=0.0)
            configs={"joint_lambda_0.02":chosen["lambda_0.02"],
                "joint_lambda_0.05":chosen["lambda_0.05"],"joint_calibrated_base":clean}
            for lam_key in ("lambda_0.02","lambda_0.05"):
                if s1[lam_key]: configs[f"single_S1_{lam_key}"]=s1[lam_key]
                if s2sel[lam_key]: configs[f"single_S2_{lam_key}"]=s2sel[lam_key]
            old=_old_combo(ds,seed,arm,ckpt_sha)
            if old: configs["old_combo_1D_2A"]=old
            metrics={name:_eval_config(evalpack,ctx,config,seed) for name,config in configs.items()}
            report={"status":"OK","dataset":ds,"seed":int(seed),"backbone":arm,
                "checkpoint_sha256":ckpt_sha,"selection_sha256":s3.sha(sp),
                "selection_role":"CAL-fit","evaluation_role":"CAL-eval",
                "cal_eval_used_for_selection":False,"metrics":metrics,
                "old_combo_available":old is not None,"gamma_space":0.0,**TEST_COUNTS}
            _write_json(rp,report)
            _csv_write(rd/"CANDIDATE_METRICS.csv",rows_all)
            reports.append(report)
            print(json.dumps({"stage":"run_complete","dataset":ds,"seed":seed,"backbone":arm,
                "candidate_count":len(rows_all),"selected_lambda_0.02":chosen["lambda_0.02"],
                "selected_lambda_0.05":chosen["lambda_0.05"],**TEST_COUNTS},ensure_ascii=False),flush=True)
    aggregate(argparse.Namespace(output=str(out)))


def _pareto(rows,metrics):
    vals=[]
    for r in rows:
        try: v=[float(r[m]) for m in metrics]
        except (KeyError,ValueError): continue
        if all(math.isfinite(x) for x in v): vals.append((r,v))
    ordered=sorted(vals,key=lambda x:tuple(x[1])); front=[]
    for row,v in ordered:
        if any(all(a<=b+1e-12 for a,b in zip(w,v)) for _,w in front): continue
        front=[(r,w) for r,w in front if not all(a<=b+1e-12 for a,b in zip(v,w))]
        front.append((row,v))
    return [r for r,_ in front]

def aggregate(args):
    out=Path(args.output).resolve(); reports=[]; selections=[]; files=[]
    (out/"PARETO_FRONTIER.csv").unlink(missing_ok=True)
    for p in sorted((out/"runs").glob("*/[CDE]/CAL_EVAL_REPORT.json")): reports.append(_read_json(p))
    for p in sorted((out/"runs").glob("*/[CDE]/CAL_FIT_SELECTION.json")): selections.append(_read_json(p))
    for p in sorted((out/"runs").glob("*/[CDE]/CANDIDATE_METRICS.csv")): files.append(p)
    if not reports: raise RuntimeError("no CAL-eval reports found")
    with (out/"CANDIDATE_METRICS.csv").open("w",newline="",encoding="utf-8-sig") as target:
        writer=None
        for p in files:
            with p.open(encoding="utf-8-sig",newline="") as source:
                reader=csv.DictReader(source)
                if writer is None: writer=csv.DictWriter(target,fieldnames=reader.fieldnames,extrasaction="ignore"); writer.writeheader()
                writer.writerows(reader)
            feasible=[r for r in _csv_read(p) if str(r.get("feasible","")).lower()=="true"]
            pareto=_pareto(feasible,["coordinate_mae_m","transport_w1_m","bf_cross_entropy","joint_nll",
                "site_p95_mae_m","site_p99_mae_m","worst_site_mae_m","negative_bf_accuracy"])
            if pareto:
                if not (out/"PARETO_FRONTIER.csv").exists(): _csv_write(out/"PARETO_FRONTIER.csv",pareto)
                else:
                    old=_csv_read(out/"PARETO_FRONTIER.csv"); old.extend(pareto); _csv_write(out/"PARETO_FRONTIER.csv",old)
    if not (out/"PARETO_FRONTIER.csv").exists(): _csv_write(out/"PARETO_FRONTIER.csv",[])
    sensitivity_rows=[]
    for selection in selections:
        for lambda_key,summary in selection.get("near_optimal_sensitivity",{}).items():
            row={"dataset":selection.get("dataset"),"seed":selection.get("seed"),
                "backbone":selection.get("backbone"),"lambda_2d_nll":lambda_key,
                "objective_tolerance_absolute":summary.get("objective_tolerance_absolute"),
                "best_objective":summary.get("best_objective"),
                "best_candidate_id":summary.get("best_candidate_id"),
                "near_optimal_candidate_count":summary.get("near_optimal_candidate_count"),
                "near_optimal_geometry_count":summary.get("near_optimal_geometry_count"),
                "top_candidate_ids":summary.get("top_candidate_ids")}
            for parameter,stats in summary.get("parameter_ranges",{}).items():
                for stat,value in stats.items(): row[f"{parameter}_{stat}"]=value
            sensitivity_rows.append(row)
    _csv_write(out/"NEAR_OPTIMAL_SENSITIVITY.csv",sensitivity_rows)
    md=["# V17 Dataset-Adaptive Spatial Joint Fit: CAL-eval Report","",
      "Parameters are selected on CAL-fit before CAL-eval. CAL-eval is reporting-only. Internal/official TEST scores: 0. Gamma-space is fixed at 0.",
      "Near-optimal sensitivity is computed on CAL-fit only, using the predeclared absolute objective tolerance 0.01; see NEAR_OPTIMAL_SENSITIVITY.csv.","",
      "| Dataset | Seed | Backbone | Method | MAE m | ES m | Joint NLL | BF accuracy | BF CE | P95 site | P99 site | Worst site |",
      "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in reports:
        for name,e in r["metrics"].items():
            m=e["metrics"]; md.append(f"| {r['dataset']} | {r['seed']} | {r['backbone']} | {name} | "
              f"{m.get('macro_mae_full_m',float('nan')):.4f} | {m.get('macro_es_m',float('nan')):.4f} | "
              f"{m.get('macro_joint_nll',float('nan')):.4f} | {m.get('macro_bf_accuracy',float('nan')):.4f} | "
              f"{m.get('bf_cross_entropy',float('nan')):.4f} | {m.get('site_p95_full_m',float('nan')):.4f} | "
              f"{m.get('site_p99_full_m',float('nan')):.4f} | {m.get('worst_site_mae_m',float('nan')):.4f} |")
    md += ["","## Paired deltas versus clean baseline","",
      "| Dataset | Seed | Backbone | Method | Δ MAE m | Δ ES m | Δ Joint NLL | Δ BF accuracy | Δ P95 m | Δ P99 m | Δ Worst-site m | Δ MAE vs old combo m |",
      "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in reports:
        clean=r["metrics"].get("joint_calibrated_base",{}).get("metrics",{})
        oldm=r["metrics"].get("old_combo_1D_2A",{}).get("metrics",{})
        for name,entry in r["metrics"].items():
            m=entry["metrics"]
            def delta(field, reference=clean):
                return m.get(field,float("nan"))-reference.get(field,float("nan")) if field in m and field in reference else float("nan")
            old_delta=delta("macro_mae_full_m",oldm) if oldm else float("nan")
            md.append(f"| {r['dataset']} | {r['seed']} | {r['backbone']} | {name} | "
              f"{delta('macro_mae_full_m'):.4f} | {delta('macro_es_m'):.4f} | {delta('macro_joint_nll'):.4f} | "
              f"{delta('macro_bf_accuracy'):.4f} | {delta('site_p95_full_m'):.4f} | {delta('site_p99_full_m'):.4f} | "
              f"{delta('worst_site_mae_m'):.4f} | {old_delta:.4f} |")
    md += ["","## Paired deltas versus old combo","",
      "| Dataset | Seed | Backbone | Method | Δ MAE m | Δ ES m | Δ Joint NLL | Δ BF accuracy | Δ P95 m | Δ P99 m | Δ Worst-site m |",
      "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in reports:
        oldm=r["metrics"].get("old_combo_1D_2A",{}).get("metrics",{})
        if not oldm: continue
        for name,entry in r["metrics"].items():
            m=entry["metrics"]
            def old_delta(field):
                return m.get(field,float("nan"))-oldm.get(field,float("nan")) if field in m and field in oldm else float("nan")
            md.append(f"| {r['dataset']} | {r['seed']} | {r['backbone']} | {name} | "
              f"{old_delta('macro_mae_full_m'):.4f} | {old_delta('macro_es_m'):.4f} | {old_delta('macro_joint_nll'):.4f} | "
              f"{old_delta('macro_bf_accuracy'):.4f} | {old_delta('site_p95_full_m'):.4f} | "
              f"{old_delta('site_p99_full_m'):.4f} | {old_delta('worst_site_mae_m'):.4f} |")
    md += ["","## Spatial parameter stability across seeds","",
      "| Dataset | Backbone | Lambda | Seed-specific (R_S1, tau_graph, R_S2, tau_virt) | Mean/std over finite values |",
      "|---|---|---:|---|---|"]
    for ds in sorted({x["dataset"] for x in selections}):
      for arm in sorted({x["backbone"] for x in selections if x["dataset"]==ds}):
        group=[x for x in selections if x["dataset"]==ds and x["backbone"]==arm]
        for lam in ("lambda_0.02","lambda_0.05"):
          configs=[(x["seed"],x["selected_joint"][lam]) for x in group if lam in x.get("selected_joint",{})]
          fields=("R_max_S1_m","tau_graph_m","R_max_S2_m","tau_virt_m")
          per_seed="; ".join(f"seed {seed}: "+",".join(f"{float(c[k]):.4g}" for k in fields) for seed,c in configs)
          summaries=[]
          for k in fields:
            vals=np.asarray([float(c[k]) for _,c in configs],dtype=float); vals=vals[np.isfinite(vals)]
            summaries.append(f"{k} mean={vals.mean():.4g}, sd={vals.std():.4g}" if len(vals) else f"{k} n/a")
          md.append(f"| {ds} | {arm} | {lam} | {per_seed} | {'; '.join(summaries)} |")
    md += ["","## Integrity","",f"- Completed run reports: {len(reports)}",f"- Candidate files: {len(files)}",
      "- Gamma-space fixed at zero.","- Internal TEST scored: 0.","- Official TEST scored: 0.",
      "- Historical old combo is evaluated only when its frozen CAL-fit selection is available."]
    (out/"CAL_EVAL_REPORT.md").write_text("\n".join(md)+"\n",encoding="utf-8")
    expected=len(s3.DATASETS)*len(s3.SEEDS)*len(ARMS)
    status="PASS" if len(reports)==expected and len(selections)==expected and len(files)==expected and len(sensitivity_rows)==expected*len(LAMBDAS) else "INCOMPLETE"
    _write_json(out/"JOINT_FIT_SELECTION.json",{"status":status,"runs":selections,
      "cal_eval_report_count":len(reports),"expected_run_count":expected,
      "selection_role":"CAL-fit","evaluation_role":"CAL-eval","gamma_space":0.0,
      "lambda_2d_nll":list(LAMBDAS),"bf_ce_weight":BF_CE_WEIGHT,"w1_weight":W1_WEIGHT,
      "mae_non_degradation_tolerance_m":MAE_NON_DEGRADATION_TOLERANCE_M,
      "selection_objective_formula":SELECTION_OBJECTIVE_FORMULA,
      "bf_ce_normalization_floor":BF_CE_NORMALIZATION_FLOOR,
      "bf_ce_guard_relative_tolerance":BF_CE_RELATIVE_TOLERANCE,
      "bf_ce_guard_absolute_tolerance":BF_CE_ABSOLUTE_TOLERANCE,**TEST_COUNTS})
    _write_json(out/"INTEGRITY_REVIEW.json",{"status":status,"phase":"aggregate",
      "completed_runs":len(reports),"expected_runs":expected,"cal_eval_used_for_selection":False,
      "gamma_space":0.0,**TEST_COUNTS})
    (out/"RUN_REPORT_V17_DATASET_ADAPTIVE_JOINT_FIT.md").write_text(
      "# V17 Dataset-Adaptive Spatial Joint Fit\n\n"
      "See FIT_NEIGHBOR_DISTANCE_DIAGNOSTICS.md for FIT-only geometry; SPATIAL_SCALE_SEARCH_SPACE.json for candidate design; "
      "JOINT_FIT_SELECTION.json for selected parameters and per-run near-optimal sensitivity; "
      "CANDIDATE_METRICS.csv, PARETO_FRONTIER.csv and NEAR_OPTIMAL_SENSITIVITY.csv for candidate diagnostics; "
      "CAL_EVAL_REPORT.md for reporting-only CAL-eval.\n\n"
      f"Completed run reports: {len(reports)} / {len(s3.DATASETS)*len(s3.SEEDS)*len(ARMS)}. Gamma-space=0; TEST scores=0.\n",encoding="utf-8")
    print(json.dumps({"stage":"aggregate_complete","runs":len(reports),**TEST_COUNTS},ensure_ascii=False),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    for name in ("preflight","run","all","aggregate"):
        q=sub.add_parser(name)
        q.add_argument("--input-root",default=str(DEFAULT_INPUT))
        q.add_argument("--checkpoint-root",default=str(DEFAULT_CHECKPOINT_ROOT))
        q.add_argument("--output",default=str(DEFAULT_OUTPUT))
        q.add_argument("--device",choices=("cpu","cuda"),default="cuda")
        if name=="run": q.add_argument("--dataset",choices=("all",*s3.DATASETS),default="all")
    args=p.parse_args(); torch.set_num_threads(4)
    if args.command=="preflight": preflight(args)
    elif args.command=="run": run(args)
    elif args.command=="aggregate": aggregate(argparse.Namespace(output=args.output))
    else: preflight(args); run(args)


if __name__=="__main__":
    main()
