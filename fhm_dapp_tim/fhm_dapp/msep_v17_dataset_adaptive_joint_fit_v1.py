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
DEFAULT_OUTPUT=REPO/"fhm_dapp_tim/evidence/v17_dataset_adaptive_joint_fit_v1"
ARMS=("C","D","E")
TEST_COUNTS={"internal_test_rows_scored":0,"official_test_rows_scored":0}
TEMPERATURES=(0.25,0.5,1.0,2.0,3.0)
THETAS=(0.0,0.75,1.5)
ALPHAS=(0.0,0.1,0.35,0.65,1.0)
BETAS=(0.0,0.05,0.10,0.15,0.20,0.35,0.50,0.65,0.80)
LAMBDAS=(0.02,0.05)
BF_CE_RELATIVE_TOLERANCE=0.01
NEAR_OPTIMAL_OBJECTIVE_TOLERANCE=0.01
QUANTILES=(0.00,0.01,0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.99,1.00)

def _json_safe(value):
    if isinstance(value,dict): return {str(k):_json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [_json_safe(v) for v in value]
    if isinstance(value,np.ndarray): return _json_safe(value.tolist())
    if isinstance(value,np.generic): return _json_safe(value.item())
    if isinstance(value,float) and not math.isfinite(value):
        return "Infinity" if value>0 else ("-Infinity" if value<0 else "NaN")
    return value

def _stable_hash(value):
    payload=json.dumps(_json_safe(value),sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()

def _write_json(path,value,exclusive=False):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x" if exclusive else "w",encoding="utf-8") as f:
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
    return {"anchors":int(len(xy)),"building_floor_strata":len(strata),
        "zero_distance_pairs_within_building_floor":int(zero_pairs),
        "kth_neighbor_distance_m":{str(k):_quantiles(v) for k,v in kth.items()},
        "kth_neighbor_threshold_counts":{str(k):_threshold_counts(v) for k,v in kth.items()},
        "base_union_knn_edge_distance_m":_quantiles(lengths),
        "base_union_knn_edge_threshold_counts":_threshold_counts(lengths),
        "base_union_knn_edges":int(len(e)),
        "base_union_knn_edge_isolated_anchor_count":int(len(xy)-len(set(e.reshape(-1).tolist()))),
        "groups":groups,"distance_semantics":"FIT anchor k-neighbors are within same building/floor; union-kNN edge lengths are reported separately."}

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
    return graph,torch.as_tensor(kept.copy(),dtype=torch.long),{
        "r_max_s1_m":float(r_max_m),"tau_graph_m":float(tau_graph_m),"base_edges":int(len(e)),
        "retained_edges":int(len(kept)),"removed_edges":removed,
        "retained_fraction":float(len(kept)/len(e)) if len(e) else 0.0,
        "degree_min":int(deg.min()),"degree_median":float(np.median(deg)),"degree_max":int(deg.max()),
        "zero_neighbor_nodes":int((deg==0).sum()),"sinkhorn_converged":converged,
        "sinkhorn_iterations":iterations,"sinkhorn_row_error":actual_re,"sinkhorn_col_error":actual_ce,
        "retained_edge_distance_m":_quantiles(d),"retained_edge_threshold_counts":_threshold_counts(d)}

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


def _sample_scale_pool(values,required=(),limit=8):
    all_values=_unique_positive_scales(values)
    chosen=_unique_positive_scales(required)
    if len(chosen)>limit:
        chosen=[chosen[int(round(i*(len(chosen)-1)/(limit-1)))] for i in range(limit)]
        chosen=_unique_positive_scales(chosen)
    remaining=[x for x in all_values if not any(abs(x-y)<=1e-8*max(1.0,x,y) for y in chosen)]
    need=max(0,limit-len(chosen))
    if need and remaining:
        ids=np.unique(np.round(np.linspace(0,len(remaining)-1,min(need,len(remaining)))).astype(int))
        chosen.extend(remaining[int(i)] for i in ids)
    return _unique_positive_scales(chosen)[:limit]


def _geometry_candidate_pools(diag):
    # Geometry is derived only from the FIT anchor dictionary. Use both the
    # within-building/floor k-NN distributions and the frozen union-kNN edge
    # lengths; cutoff pools emphasize edge quantiles while kernel pools retain
    # the k=1/2/4/8 physical spacing scales.
    kth=diag["kth_neighbor_distance_m"]
    edge=diag.get("base_union_knn_edge_distance_m",{})
    quantile_names=("p50","p75","p90","p95","p99")
    k_medians=[kth[str(k)].get("p50") for k in (1,2,4,8)]
    k_all=[kth[str(k)].get(q) for k in (1,2,4,8) for q in quantile_names]
    edge_all=[edge.get(q) for q in quantile_names]
    physical=[1.,2.,5.,8.,10.,15.,20.,30.,40.,60.]
    required_k=[x for x in k_medians if x is not None]
    required_edge=[edge.get(q) for q in ("p50","p75","p90","p95") if edge.get(q) is not None]
    if not required_k:
        raise RuntimeError("FIT-only nearest-neighbor geometry is unavailable; refusing to guess spatial scales")
    pools={
      "r_max_s1_m":_sample_scale_pool(k_all+edge_all+physical,required=required_k),
      "tau_graph_m":_sample_scale_pool(k_all+[edge.get(q) for q in ("p50","p75","p90")]+physical,
                                      required=required_k),
      "r_max_s2_m":_sample_scale_pool(edge_all+k_all+physical,
                                      required=required_edge or required_k),
      "tau_virt_m":_sample_scale_pool(edge_all+k_all+physical,
                                      required=(required_edge[:3]+required_k[:4])[:8] or required_k),
    }
    for name,values in pools.items():
        if not values: raise RuntimeError(f"no finite FIT-derived scale candidates for {name}")
    return pools


def _geometry_value_pool(diag):
    pools=_geometry_candidate_pools(diag)
    # Compact backward-compatible summary, preserving every k median while
    # surfacing edge-length and upper-tail scales in the remaining slots.
    all_values=[]
    for values in pools.values(): all_values.extend(values)
    k_medians=[diag["kth_neighbor_distance_m"][str(k)].get("p50") for k in (1,2,4,8)]
    return _sample_scale_pool(all_values,required=[x for x in k_medians if x is not None],limit=8)


def geometry_candidates(diag):
    pools=_geometry_candidate_pools(diag)
    shared=_geometry_value_pool(diag)
    out=[{"geometry_id":"legacy_unbounded_8m","r_max_s1_m":math.inf,"tau_graph_m":8.,
          "r_max_s2_m":math.inf,"tau_virt_m":math.inf,"role":"legacy_reference"}]
    # Eight-point deterministic Latin-hypercube design: each parameter gets its
    # own FIT-derived pool, and every pool is traversed as evenly as its size
    # allows. This avoids coupling cutoff values to kernel temperatures.
    multipliers=(1,3,5,7); offsets=(0,1,2,3)
    names=("r_max_s1_m","tau_graph_m","r_max_s2_m","tau_virt_m")
    for i in range(8):
        row={"geometry_id":f"fit_lhs_{i:02d}","role":"FIT_neighbor_and_union_edge_LHS"}
        for j,name in enumerate(names):
            rank=(i*multipliers[j]+offsets[j])%8
            values=pools[name]
            row[name]=float(values[min(len(values)-1,int(rank*len(values)/8))])
        out.append(row)
    # Defensive deduplication is scientifically relevant here: tiny datasets
    # may have repeated quantiles, and repeated geometry tuples waste a full
    # inner calibration sweep.
    unique=[]; seen=set()
    for row in out:
        key=tuple(row[k] for k in names)
        if key not in seen: unique.append(row); seen.add(key)
    return unique,{"shared_union_m":shared,"by_parameter_m":pools}

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

def _calibration_candidates():
    return [dict(candidate_id=f"cal_{i:05d}",temperature=float(t),theta_v=float(tv),theta_s=float(ts),
        alpha_prior=float(ap),alpha_obs=float(ao),beta=float(beta))
        for i,(t,tv,ts,ap,ao,beta) in enumerate(itertools.product(TEMPERATURES,THETAS,THETAS,ALPHAS,ALPHAS,BETAS))]

@torch.no_grad()
def evaluate_calibration_grid(pack_all,ctx,graph,s2_edges,geom,candidates,*,endpoint_cache=None,chunk_size=8,row_chunk_size=256):
    device=pack_all["scores"].device; coords=ctx["memory"]["coords"].detach().cpu().numpy()
    cached_graph=graph.to(device).coalesce()
    ep=endpoint_cache if endpoint_cache is not None else normalized_virtual_endpoint_weights(s2_edges,coords,geom["tau_virt_m"])
    all_edges=np.asarray(ctx["edges"].detach().cpu(),dtype=np.int64).reshape(-1,2)
    lookup={tuple(sorted(map(int,e))):i for i,e in enumerate(all_edges.tolist())}
    edge_ids=[lookup[tuple(map(int,e))] for e in ep["edges"].tolist()]
    pack=_subset_virtual_pack(pack_all,edge_ids); edges=torch.as_tensor(ep["edges"],dtype=torch.long,device=device)
    endpoint={"weight_i":torch.as_tensor(ep["weight_i"],device=device),"weight_j":torch.as_tensor(ep["weight_j"],device=device),
              "degree":torch.as_tensor(ep["degree"],device=device)}
    n=len(ctx["memory"]["coords"]); class_ids=torch.cat([ctx["classes"].to(device).long(),
        ctx["classes"].to(device).long()[edges[:,0]]])
    centers=torch.cat([pack["xy"],pack["virtual_xy"]],dim=1); scales=torch.cat([pack["sigma"],pack["virtual_sigma"]],dim=1)
    ab=ctx["memory"]["building"].to(device).long(); af=ctx["memory"]["floor"].to(device).long()
    component_ids=torch.cat([torch.arange(n,device=device),edges[:,0]])
    valid=((ab[component_ids][None,:]==pack["building"].long()[:,None])&
           (af[component_ids][None,:]==pack["floor"].long()[:,None]))
    lp=s3.student_t_logpdf(centers,scales,pack["target"]).masked_fill(~valid,-torch.inf)
    keys=pack["keys"]; unique_sites=list(dict.fromkeys(keys)); site_map={k:i for i,k in enumerate(unique_sites)}
    site_ids=torch.as_tensor([site_map[k] for k in keys],dtype=torch.long,device=device)
    site_counts=np.bincount(site_ids.cpu().numpy(),minlength=len(unique_sites)).astype(np.float64)
    rows_out=[]
    for start in range(0,len(candidates),chunk_size):
        chunk=candidates[start:start+chunk_size]
        pars=torch.as_tensor([[r["temperature"],r["theta_v"],r["theta_s"],r["alpha_prior"],r["alpha_obs"],r["beta"]]
                              for r in chunk],dtype=pack["scores"].dtype,device=device)
        c=len(chunk); sums={k:torch.zeros(c,dtype=pars.dtype,device=device) for k in
            ("coordinate_mae_m","transport_w1_m","bf_accuracy","bf_cross_entropy","joint_nll")}
        site_sums=torch.zeros((c,len(unique_sites)),dtype=pars.dtype,device=device)
        for rs in range(0,len(pack["scores"]),row_chunk_size):
            re=min(rs+row_chunk_size,len(pack["scores"])); sl=slice(rs,re)
            scores=pack["scores"][sl]; z=pack["z"][sl]; bsz=len(scores)
            temp=pars[:,0,None,None]; tv=pars[:,1,None,None]; ts=pars[:,2,None,None]
            ap=pars[:,3,None,None]; ao=pars[:,4,None,None]; beta=pars[:,5]
            prior=scores[None]/temp; obs=z[None,:,:,0]*tv+z[None,:,:,1]*ts
            pn=torch.sparse.mm(cached_graph,prior.reshape(c*bsz,n).T).T.reshape(c,bsz,n)
            on=torch.sparse.mm(cached_graph,obs.reshape(c*bsz,n).T).T.reshape(c,bsz,n)
            base=prior+obs
            parent=(1-ap)*prior+ap*pn+(1-ao)*obs+ao*on
            vbase=(pack["virtual_scores"][sl][None]/temp+
                   pack["virtual_z"][sl][None,:,:,0]*tv+pack["virtual_z"][sl][None,:,:,1]*ts)
            if len(edges):
                delta=parent-base
                v_logits=vbase+.5*(delta.index_select(2,edges[:,0])+delta.index_select(2,edges[:,1]))
            else: v_logits=vbase
            deg=endpoint["degree"]
            pmass=torch.where(deg[None,:]>0,1-beta[:,None],torch.ones((c,n),dtype=pars.dtype,device=device))
            pterms=parent+pmass.log()[:,None,:]
            if len(edges):
                emass=beta[:,None]*(endpoint["weight_i"]+endpoint["weight_j"])[None,:]
                eterms=v_logits+emass.clamp_min(torch.finfo(pars.dtype).tiny).log()[:,None,:]
                eterms=eterms.masked_fill(beta[:,None,None]==0.,-torch.inf)
                q=torch.softmax(torch.cat([pterms,eterms],dim=-1),dim=-1)
            else: q=torch.softmax(pterms,dim=-1)
            sub_centers=centers[sl]; sub_lp=lp[sl]; target=pack["target"][sl]
            target_cls=_target_class_ids({**pack,"building":pack["building"][sl],"floor":pack["floor"][sl],
                                          "scores":pack["scores"][sl]},ctx)
            masses=torch.zeros((c*bsz,len(ctx["class_pairs"])),dtype=q.dtype,device=device)
            masses.scatter_add_(1,class_ids[None,:].expand(c*bsz,-1),q.reshape(c*bsz,-1))
            masses=masses.reshape(c,bsz,-1); correct=(masses.argmax(-1)==target_cls[None]).to(q.dtype)
            true_mass=masses.gather(2,target_cls[None,:,None].expand(c,bsz,1)).squeeze(-1)
            bfce=-true_mass.clamp_min(torch.finfo(q.dtype).tiny).log()
            point=torch.einsum("cbm,bmd->cbd",q,sub_centers)
            err=torch.linalg.vector_norm(point-target[None],dim=-1)
            dist=torch.linalg.vector_norm(sub_centers-target[:,None,:],dim=-1)
            risk=(q*dist[None]).sum(-1); logq=q.clamp_min(torch.finfo(q.dtype).tiny).log()
            valid_nll=(class_ids[None,:]==target_cls[:,None])[None]&(q>0)
            nll=-torch.logsumexp((logq+sub_lp[None]).masked_fill(~valid_nll,-torch.inf),dim=-1)
            wt=pack["weight"][sl][None]
            for key,val in (("coordinate_mae_m",err),("transport_w1_m",risk),("bf_accuracy",correct),
                            ("bf_cross_entropy",bfce),("joint_nll",nll)): sums[key]+=(val*wt).sum(1)
            site_sums.scatter_add_(1,site_ids[sl][None,:].expand(c,-1),err)
        site_means=site_sums/torch.as_tensor(site_counts,dtype=pars.dtype,device=device)[None,:]
        p95=torch.quantile(site_means,.95,dim=1).cpu().tolist()
        p99=torch.quantile(site_means,.99,dim=1).cpu().tolist(); worst=site_means.max(1).values.cpu().tolist()
        vals={k:v.cpu().tolist() for k,v in sums.items()}
        for i,candidate in enumerate(chunk):
            row=dict(candidate); row.update({k:float(v[i]) for k,v in vals.items()})
            row.update(site_p95_mae_m=float(p95[i]),site_p99_mae_m=float(p99[i]),
                worst_site_mae_m=float(worst[i]),n_fit_rows=len(pack["scores"]),n_fit_sites=len(unique_sites))
            rows_out.append(row)
    return rows_out

def _primary(r): return float(r["coordinate_mae_m"])+.30*float(r["transport_w1_m"])
def _clean_baseline(rows):
    clean=[r for r in rows if r["alpha_prior"]==0 and r["alpha_obs"]==0 and r["beta"]==0]
    if not clean: raise RuntimeError("joint grid omitted clean-base alpha=0,beta=0 boundary")
    return min(clean,key=lambda r:(_primary(r),r["bf_cross_entropy"],r["joint_nll"],r["temperature"],r["theta_v"],r["theta_s"]))
def _add_objectives(rows,base):
    pbase=max(_primary(base),1e-9); nscale=max(abs(float(base["joint_nll"])),1.0)
    bfbase=float(base["bf_cross_entropy"]); accbase=float(base["bf_accuracy"])
    for r in rows:
        r["primary_mae_plus_030_w1_m"]=_primary(r); r["normalized_primary"]=_primary(r)/pbase
        r["normalized_joint_nll_delta"]=(float(r["joint_nll"])-float(base["joint_nll"]))/nscale
        r["normalized_bf_cross_entropy"]=float(r["bf_cross_entropy"])/max(bfbase,1e-6)
        r["negative_bf_accuracy"]=-float(r["bf_accuracy"])
        r["bf_accuracy_non_degradation"]=float(r["bf_accuracy"])>=accbase-1e-12
        r["bf_cross_entropy_guard"]=float(r["bf_cross_entropy"])<=bfbase*1.01+1e-8
        r["feasible"]=bool(r["bf_accuracy_non_degradation"] and r["bf_cross_entropy_guard"])
        for lam in LAMBDAS: r[f"selection_objective_lambda_{lam:.2f}"]=r["normalized_primary"]+lam*r["normalized_joint_nll_delta"]
    return rows
def _choose(rows,lam,kind="joint"):
    objective_key=f"selection_objective_lambda_{lam:.2f}"
    metric_fields=("coordinate_mae_m","transport_w1_m","bf_accuracy","bf_cross_entropy","joint_nll",
                   "site_p95_mae_m","site_p99_mae_m","worst_site_mae_m")
    rs=[r for r in rows if r.get("feasible") and
        math.isfinite(float(r.get(objective_key,math.inf))) and
        all(math.isfinite(float(r.get(field,math.nan))) for field in metric_fields)]
    if kind=="single_S1": rs=[r for r in rs if r["beta"]==0 and (r["alpha_prior"]!=0 or r["alpha_obs"]!=0)]
    if kind=="single_S2": rs=[r for r in rs if r["alpha_prior"]==0 and r["alpha_obs"]==0 and r["beta"]>0]
    return min(rs,key=lambda r:(r[objective_key],r["normalized_bf_cross_entropy"],
        _primary(r),r["temperature"],r["theta_v"],r["theta_s"],r["alpha_prior"],r["alpha_obs"],r["beta"],r["geometry_id"])) if rs else None
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
        near=[r for r in rows if r.get("feasible") and
              math.isfinite(float(r[objective_key])) and
              float(r[objective_key])<=best_value+NEAR_OPTIMAL_OBJECTIVE_TOLERANCE+1e-12]
        near.sort(key=lambda r:(float(r[objective_key]),r["normalized_bf_cross_entropy"],
            _primary(r),r["temperature"],r["theta_v"],r["theta_s"],r["alpha_prior"],
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
        "normalized_primary":float(r["normalized_primary"]),"normalized_joint_nll_delta":float(r["normalized_joint_nll_delta"]),
        "normalized_bf_cross_entropy":float(r["normalized_bf_cross_entropy"]),
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
    if any(int(i) not in allowed for i in arr): raise RuntimeError(f"{role} rows escaped locked split")
    if set(map(int,arr))&set(map(int,ctx["roles"]["test"])): raise RuntimeError("TEST row guard blocked pack construction")
    return arr

def _verify_selection_context(selection,ctx):
    """Recompute the split and anchor hashes from the currently reconstructed context."""
    fit_hash=s3.key_sha(ctx["keys"][int(i)] for i in ctx["cal_fit"])
    eval_hash=s3.key_sha(ctx["keys"][int(i)] for i in ctx["cal_eval"])
    coords=ctx["memory"]["coords"].detach().cpu().numpy()
    buildings=ctx["memory"]["building"].detach().cpu().numpy()
    floors=ctx["memory"]["floor"].detach().cpu().numpy()
    anchor_hash=_hash_anchor_order(coords,buildings,floors)
    actual={"fit_split_sha256":fit_hash,"eval_split_sha256":eval_hash,"anchor_order_sha256":anchor_hash}
    failed=[key for key,value in actual.items() if selection.get(key)!=value]
    if failed:
        raise RuntimeError(f"selection context hashes differ from reconstructed data: {failed}")
    return actual


def _verify_locked_selection(selection_path,lock_path,manifest_path,input_root):
    selection=_read_json(selection_path); lock=_read_json(lock_path)
    unhashed=dict(selection); config_hash=unhashed.pop("parameter_config_sha256",None)
    run_root=Path(input_root)/f"{selection['dataset']}_seed{selection['seed']}"
    input_hash_ok=all((run_root/name).is_file() and s3.sha(run_root/name)==digest
        for name,digest in selection.get("input_run_file_sha256",{}).items())
    test_zero=all(selection.get(k)==0 and lock.get(k)==0 for k in TEST_COUNTS)
    checks={
      "selection_sha256":lock.get("selection_sha256")==s3.sha(selection_path),
      "parameter_config_sha256":bool(config_hash) and config_hash==_stable_hash(unhashed)
          and lock.get("parameter_config_sha256")==config_hash,
      "algorithm_code_sha256":selection.get("algorithm_code_sha256")==s3.sha(Path(__file__))
          and lock.get("algorithm_code_sha256")==selection.get("algorithm_code_sha256"),
      "source_manifest_sha256":selection.get("source_manifest_sha256")==s3.sha(manifest_path)
          and lock.get("source_manifest_sha256")==selection.get("source_manifest_sha256"),
      "checkpoint_sha256":lock.get("checkpoint_sha256")==selection.get("checkpoint_sha256"),
      "split_hashes":all(lock.get(k)==selection.get(sk) for k,sk in (
          ("fit_split_sha256","fit_split_sha256"),("eval_split_sha256","eval_split_sha256"),
          ("anchor_order_sha256","anchor_order_sha256"))),
      "input_hashes":bool(selection.get("input_run_file_sha256")) and input_hash_ok
          and lock.get("input_run_file_sha256")==selection.get("input_run_file_sha256"),
      "selection_frozen":lock.get("selection_frozen_before_cal_eval") is True,
      "test_rows_zero":test_zero,
      "selection_role":selection.get("selection_role")=="CAL-fit"
          and lock.get("selection_role")=="CAL-fit" and lock.get("evaluation_role")=="CAL-eval",
    }
    failed=[k for k,v in checks.items() if not v]
    if failed: raise RuntimeError(f"CAL-eval lock verification failed; refusing CAL-eval access: {failed}")
    return selection

def _make_probability(pack_all,ctx,config):
    coords=ctx["memory"]["coords"].detach().cpu().numpy(); b=ctx["memory"]["building"].detach().cpu().numpy()
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

_pairwise_fit_geometry_basic = _pairwise_fit_geometry
def _pairwise_fit_geometry(coords,buildings,floors,edges):
    result=_pairwise_fit_geometry_basic(coords,buildings,floors,edges)
    xy=np.asarray(coords,dtype=np.float64); b=np.asarray(buildings,dtype=np.int64); f=np.asarray(floors,dtype=np.int64)
    e=np.asarray(torch.as_tensor(edges).detach().cpu(),dtype=np.int64).reshape(-1,2)
    if len(e):
        ed=np.linalg.norm(xy[e[:,0]]-xy[e[:,1]],axis=1)
        exp_edge={k:(float(math.exp(-v/8.0)) if v is not None else None)
                  for k,v in result["base_union_knn_edge_distance_m"].items() if k.startswith("p")}
    else:
        ed=np.asarray([],dtype=np.float64); exp_edge={}
    result["exp_neg_distance_over_8_at_quantiles"]={
        f"FIT_k{k}_nearest_neighbor":{key:(float(math.exp(-value/8.0)) if value is not None else None)
            for key,value in result["kth_neighbor_distance_m"][str(k)].items() if key.startswith("p")}
        for k in (1,2,4,8)}
    result["exp_neg_distance_over_8_at_quantiles"]["base_union_knn_edges"]=exp_edge
    result["exp_neg_distance_over_8_at_quantiles"]["note"]=(
        "Kernel values are descriptive only; actual normalized operator weights also depend on graph topology and Sinkhorn.")
    for key,group in result["groups"].items():
        try: bb,ff=(int(x) for x in key.split("::",1))
        except ValueError: continue
        mask=(b[e[:,0]]==bb)&(f[e[:,0]]==ff)&(b[e[:,1]]==bb)&(f[e[:,1]]==ff) if len(e) else np.zeros(0,dtype=bool)
        vals=ed[mask] if len(e) else np.asarray([],dtype=np.float64)
        group["union_knn_edge_count"]=int(len(vals))
        group["union_knn_edge_distance_m"]=_quantiles(vals)
        group["union_knn_edge_threshold_counts"]=_threshold_counts(vals)
    return result

_build_cutoff_graph_basic = build_cutoff_graph
def build_cutoff_graph(coords,buildings,floors,base_edges,r_max_m,tau_graph_m,**kwargs):
    graph,kept,diag=_build_cutoff_graph_basic(coords,buildings,floors,base_edges,r_max_m,tau_graph_m,**kwargs)
    n=len(coords); parent=list(range(n))
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]; x=parent[x]
        return x
    for i,j in np.asarray(kept.detach().cpu(),dtype=np.int64).reshape(-1,2):
        a,b=find(int(i)),find(int(j))
        if a!=b: parent[a]=b
    roots={find(i) for i in range(n)}
    diag["connected_components"]=len(roots)
    diag["isolated_component_count"]=int(diag["zero_neighbor_nodes"])
    return graph,kept,diag

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
    started=time.perf_counter()
    if args.device=="cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable; rerun with --device cpu")
    out=Path(args.output).resolve()
    if out.exists() and any(out.iterdir()) and not (out/"EXECUTION_MANIFEST.json").is_file(): raise RuntimeError("nonempty output root without manifest")
    comp.preflight(args); manifest_path=out/"EXECUTION_MANIFEST.json"; manifest=_read_json(manifest_path)
    diagnostics={}; spaces={}
    for ds in s3.DATASETS:
      for seed in s3.SEEDS:
        ctx=s3.reconstruct(ds,seed,Path(args.input_root).resolve(),torch.device("cpu"))
        _safe_rows(ctx,ctx["cal_fit"],"CAL-fit"); _safe_rows(ctx,ctx["cal_eval"],"CAL-eval")
        xy=ctx["memory"]["coords"].detach().cpu().numpy(); b=ctx["memory"]["building"].detach().cpu().numpy()
        f=ctx["memory"]["floor"].detach().cpu().numpy(); d=_pairwise_fit_geometry(xy,b,f,ctx["edges"]); key=f"{ds}_seed{seed}"
        d.update(dataset=ds,seed=int(seed),cal_fit_rows=len(ctx["cal_fit"]),cal_eval_rows=len(ctx["cal_eval"]),
          cal_fit_key_sha256=s3.key_sha(ctx["keys"][int(i)] for i in ctx["cal_fit"]),
          cal_eval_key_sha256=s3.key_sha(ctx["keys"][int(i)] for i in ctx["cal_eval"]),
          anchor_order_sha256=_hash_anchor_order(xy,b,f))
        diagnostics[key]=d; geoms,pools=geometry_candidates(d)
        spaces[key]={"dataset":ds,"seed":int(seed),"geometry_value_pool_m":pools["shared_union_m"],
          "geometry_candidate_pools_m":pools["by_parameter_m"],"geometry_candidates":geoms,
          "calibration_candidate_count":len(_calibration_candidates()),"gamma_space":{"locked":0.0,"searched":False},
          "lambda_2d_nll":list(LAMBDAS),"bf_ce_relative_tolerance":BF_CE_RELATIVE_TOLERANCE,
          "candidate_count_per_run":len(geoms)*len(_calibration_candidates())}
    _write_json(out/"FIT_NEIGHBOR_DISTANCE_DIAGNOSTICS.json",{"status":"PASS","source":"FIT anchor dictionary only",
      "datasets_and_seeds":diagnostics,**TEST_COUNTS})
    md=["# FIT-only Neighbor Distance Diagnostics","","Nearest-neighbor distances use distinct anchors on the same building and floor. Union-kNN edge lengths are reported separately.",""]
    for key,d in diagnostics.items():
        md += [f"## {key}","",f"- FIT anchors: {d['anchors']}",f"- Building-floor strata: {d['building_floor_strata']}",
          f"- Zero-distance pairs: {d['zero_distance_pairs_within_building_floor']}",
          f"- Existing union-kNN edges: {d['base_union_knn_edges']}",
          f"- Union-kNN edge quantiles: {json.dumps(d['base_union_knn_edge_distance_m'],ensure_ascii=False)}"]
        for k,v in d["kth_neighbor_distance_m"].items():
            md.append(f"- k={k} neighbor distances: {json.dumps(v,ensure_ascii=False)}")
        md.append(f"- k-neighbor threshold counts: {json.dumps(d['kth_neighbor_threshold_counts'],ensure_ascii=False)}")
        md.append(f"- exp(-d/8m) at distance quantiles: {json.dumps(d['exp_neg_distance_over_8_at_quantiles'],ensure_ascii=False)}")
        md.append("- Building/floor-stratified neighbor and union-kNN edge distributions are in the JSON file.")
        md.append("")
    (out/"FIT_NEIGHBOR_DISTANCE_DIAGNOSTICS.md").write_text("\n".join(md),encoding="utf-8")
    _write_json(out/"SPATIAL_SCALE_SEARCH_SPACE.json",{"status":"PASS",
      "geometry_design":"8-point deterministic Latin-hypercube with separate R_S1/tau_graph/R_S2/tau_virt pools derived from FIT k=1/2/4/8 quantiles, union-kNN edge-length quantiles, and fixed physical scales",
      "outer_geometry_candidates":spaces,"calibration_grid":{"temperature":TEMPERATURES,"theta_v":THETAS,"theta_s":THETAS,
      "alpha_prior":ALPHAS,"alpha_obs":ALPHAS,"beta":BETAS},"inner_joint_grid_size":len(_calibration_candidates()),
      "gamma_space_fixed":0.0,"search_method":"joint vectorized grid for every cached geometry; no sequential freezing",
      "near_optimal_objective_tolerance_absolute":NEAR_OPTIMAL_OBJECTIVE_TOLERANCE,
      "lambda_2d_nll":list(LAMBDAS),**TEST_COUNTS})
    manifest.update({"study":"V17 dataset-adaptive spatial joint fit",
      "fit_neighbor_diagnostics_sha256":s3.sha(out/"FIT_NEIGHBOR_DISTANCE_DIAGNOSTICS.json"),
      "spatial_search_space_sha256":s3.sha(out/"SPATIAL_SCALE_SEARCH_SPACE.json"),
      "spatial_parameters_fit_per_dataset_backbone_seed":True,"gamma_space":{"locked":0.0,"searched":False},
      "selection_protocol":{"primary":"Coordinate MAE + 0.30 * component-center W1",
      "normalized_objective":"normalized primary + lambda * normalized joint-NLL delta",
      "lambda_2d_nll":list(LAMBDAS),"BF CE relative tolerance":BF_CE_RELATIVE_TOLERANCE,
      "near_optimal_objective_tolerance_absolute":NEAR_OPTIMAL_OBJECTIVE_TOLERANCE,
      "hard_constraint":"macro BF accuracy must not fall below same-checkpoint CAL-fit clean baseline"},
      "preflight_elapsed_seconds":time.perf_counter()-started,**TEST_COUNTS})
    _write_json(manifest_path,manifest)
    _write_json(out/"INTEGRITY_REVIEW.json",{"status":"PASS","phase":"preflight","test_rows_scored":0,
      "FIT_only_geometry_verified":True,"internal_test_rows_scored":0,"official_test_rows_scored":0})
    print(json.dumps({"stage":"preflight_complete","output":str(out),"geometry_sets":len(spaces),**TEST_COUNTS},ensure_ascii=False),flush=True)

def smoke(args):
    """Small CAL-fit-only end-to-end check; never builds or scores CAL-eval."""
    out=Path(args.output).resolve()
    manifest_path=out/"EXECUTION_MANIFEST.json"
    search_path=out/"SPATIAL_SCALE_SEARCH_SPACE.json"
    if not manifest_path.is_file() or not search_path.is_file():
        raise FileNotFoundError("run preflight first; EXECUTION_MANIFEST.json and SPATIAL_SCALE_SEARCH_SPACE.json are required")
    manifest=_read_json(manifest_path)
    comp._verify_source_manifest(manifest)
    ds=s3.DATASETS[0]; seed=s3.SEEDS[0]; arm="D"
    input_root=Path(args.input_root).resolve(); checkpoint_root=Path(args.checkpoint_root).resolve()
    ctx=s3.reconstruct(ds,seed,input_root,torch.device(args.device))
    fit=_safe_rows(ctx,ctx["cal_fit"],"CAL-fit")
    checkpoint_path=comp._checkpoint_path(checkpoint_root,ds,seed,arm)
    relative=str(checkpoint_path.relative_to(checkpoint_root)).replace("\\","/")
    checkpoint_sha=s3.sha(checkpoint_path)
    if manifest.get("checkpoint_sha256",{}).get(relative)!=checkpoint_sha:
        raise RuntimeError("smoke checkpoint SHA does not match preflight manifest")
    model,payload,native_alpha=comp._load_model(ctx,checkpoint_root,ds,seed,arm,args.device)
    fitpack=s3.make_pack(ctx,model,fit,virtual_gamma=0.0)
    geometry_set=_read_json(search_path)["outer_geometry_candidates"][f"{ds}_seed{seed}"]["geometry_candidates"]
    geom=next(x for x in geometry_set if x.get("role")!="legacy_reference")
    coords=ctx["memory"]["coords"].detach().cpu().numpy()
    buildings=ctx["memory"]["building"].detach().cpu().numpy()
    floors=ctx["memory"]["floor"].detach().cpu().numpy()
    all_edges=np.asarray(ctx["edges"].detach().cpu(),dtype=np.int64).reshape(-1,2)
    graph,_,gdiag=build_cutoff_graph(coords,buildings,floors,all_edges,geom["r_max_s1_m"],geom["tau_graph_m"])
    edge_lengths=np.linalg.norm(coords[all_edges[:,0]]-coords[all_edges[:,1]],axis=1) if len(all_edges) else np.empty(0)
    s2_edges=all_edges[edge_lengths<=float(geom["r_max_s2_m"])] if len(all_edges) else all_edges
    ep=normalized_virtual_endpoint_weights(s2_edges,coords,geom["tau_virt_m"])
    grid=_calibration_candidates()
    clean=next(x for x in grid if x["alpha_prior"]==0.0 and x["alpha_obs"]==0.0 and x["beta"]==0.0)
    single_s1=next(x for x in grid if x["alpha_prior"]>0.0 and x["alpha_obs"]==0.0 and x["beta"]==0.0)
    single_s2=next(x for x in grid if x["alpha_prior"]==0.0 and x["alpha_obs"]==0.0 and x["beta"]>0.0)
    selected=[dict(x,geometry_id=geom["geometry_id"],r_max_s1_m=geom["r_max_s1_m"],
        tau_graph_m=geom["tau_graph_m"],r_max_s2_m=geom["r_max_s2_m"],
        tau_virt_m=geom["tau_virt_m"],geometry_role=geom["role"])
        for x in (clean,single_s1,single_s2)]
    rows=evaluate_calibration_grid(fitpack,ctx,graph.to(args.device),s2_edges,geom,selected,endpoint_cache=ep)
    required=("coordinate_mae_m","transport_w1_m","bf_accuracy","bf_cross_entropy","joint_nll",
              "site_p95_mae_m","site_p99_mae_m","worst_site_mae_m")
    if len(rows)!=3 or any(not math.isfinite(float(row[k])) for row in rows for k in required):
        raise RuntimeError("CAL-fit smoke evaluation produced missing/nonfinite metrics")
    endpoint_sums=np.zeros(len(coords),dtype=np.float64)
    if len(ep["edges"]):
        np.add.at(endpoint_sums,ep["edges"][:,0],ep["weight_i"])
        np.add.at(endpoint_sums,ep["edges"][:,1],ep["weight_j"])
    active=ep["degree"]>0
    if not np.allclose(endpoint_sums[active],1.0,atol=1e-10,rtol=1e-10):
        raise RuntimeError("smoke test failed per-parent virtual measure conservation")
    result={"status":"PASS","dataset":ds,"seed":int(seed),"backbone":arm,
      "checkpoint_sha256":checkpoint_sha,"selected_epoch":int(payload["selected_epoch"]),
      "native_E_alpha":native_alpha,"selection_role":"CAL-fit","evaluation_role":"smoke-only",
      "cal_fit_rows":int(len(fit)),"candidate_count":len(rows),
      "candidate_ids":[x["candidate_id"] for x in rows],
      "candidate_metrics":[{k:float(x[k]) for k in ("coordinate_mae_m","transport_w1_m","bf_accuracy",
        "bf_cross_entropy","joint_nll","site_p95_mae_m","site_p99_mae_m","worst_site_mae_m")} for x in rows],
      "geometry":geom,"graph_diagnostics":gdiag,"retained_s2_edges":int(len(s2_edges)),
      "virtual_parent_measure_max_error":float(np.max(np.abs(endpoint_sums[active]-1.0))) if active.any() else 0.0,
      "cal_eval_pack_constructed":False,"cal_eval_rows_scored":0,**TEST_COUNTS}
    _write_json(out/"SMOKE_TEST.json",result)
    print(json.dumps({"stage":"smoke_complete","dataset":ds,"seed":seed,"backbone":arm,
      "candidate_count":len(rows),"cal_eval_rows_scored":0,**TEST_COUNTS},ensure_ascii=False),flush=True)


def run(args):
    out=Path(args.output).resolve(); manifest_path=out/"EXECUTION_MANIFEST.json"
    manifest=_read_json(manifest_path)
    comp._verify_source_manifest(manifest)
    search=_read_json(out/"SPATIAL_SCALE_SEARCH_SPACE.json")["outer_geometry_candidates"]
    inner=_calibration_candidates(); reports=[]
    for ds in s3.DATASETS:
      for seed in s3.SEEDS:
       ctx=s3.reconstruct(ds,seed,Path(args.input_root).resolve(),torch.device(args.device))
       fit=_safe_rows(ctx,ctx["cal_fit"],"CAL-fit"); ev=_safe_rows(ctx,ctx["cal_eval"],"CAL-eval")
       for arm in ARMS:
        rd=out/"runs"/f"{ds}_seed{seed}"/arm; rd.mkdir(parents=True,exist_ok=True)
        sp=rd/"CAL_FIT_SELECTION.json"; lock=rd/"CAL_EVAL.lock"; rp=rd/"CAL_EVAL_REPORT.json"
        if rp.is_file():
            if not sp.is_file() or not lock.is_file():
                raise RuntimeError(f"completed CAL-eval report lacks its selection/lock: {rd}")
            locked_selection=_verify_locked_selection(sp,lock,manifest_path,Path(args.input_root).resolve())
            _verify_selection_context(locked_selection,ctx)
            checkpoint_path=comp._checkpoint_path(Path(args.checkpoint_root).resolve(),ds,seed,arm)
            checkpoint_sha=s3.sha(checkpoint_path)
            if locked_selection.get("checkpoint_sha256")!=checkpoint_sha:
                raise RuntimeError(f"frozen checkpoint changed since selection: {checkpoint_path}")
            saved_report=_read_json(rp)
            if (saved_report.get("selection_sha256")!=s3.sha(sp)
                or saved_report.get("checkpoint_sha256")!=checkpoint_sha
                or saved_report.get("cal_eval_used_for_selection") is not False
                or any(saved_report.get(k)!=0 for k in TEST_COUNTS)):
                raise RuntimeError(f"existing CAL-eval report failed integrity checks: {rp}")
            reports.append(saved_report)
            continue
        if sp.exists() or lock.exists(): raise RuntimeError(f"incomplete locked run exists at {rd}; preserve and inspect")
        ckpt=comp._checkpoint_path(Path(args.checkpoint_root).resolve(),ds,seed,arm); ckpt_sha=s3.sha(ckpt)
        model,payload,native_alpha=comp._load_model(ctx,Path(args.checkpoint_root).resolve(),ds,seed,arm,args.device)
        base_edges=np.asarray(ctx["edges"].detach().cpu(),dtype=np.int64).reshape(-1,2)
        fitpack=s3.make_pack(ctx,model,fit,virtual_gamma=0.0)
        coords=ctx["memory"]["coords"].detach().cpu().numpy(); b=ctx["memory"]["building"].detach().cpu().numpy()
        f=ctx["memory"]["floor"].detach().cpu().numpy(); geoms=search[f"{ds}_seed{seed}"]["geometry_candidates"]
        rows_all=[]; geometry_diagnostics={}
        for geom in geoms:
            graph,_,gdiag=build_cutoff_graph(coords,b,f,base_edges,geom["r_max_s1_m"],geom["tau_graph_m"])
            lens=np.linalg.norm(coords[base_edges[:,0]]-coords[base_edges[:,1]],axis=1) if len(base_edges) else np.empty(0)
            s2=base_edges[lens<=float(geom["r_max_s2_m"])] if len(base_edges) else base_edges
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
            geometry_diagnostics[geom["geometry_id"]]=gdiag
            candidates=[dict(x,geometry_id=geom["geometry_id"],r_max_s1_m=geom["r_max_s1_m"],
              tau_graph_m=geom["tau_graph_m"],r_max_s2_m=geom["r_max_s2_m"],tau_virt_m=geom["tau_virt_m"],
              geometry_role=geom["role"]) for x in inner]
            rows=evaluate_calibration_grid(fitpack,ctx,graph.to(args.device),s2,geom,candidates,endpoint_cache=epdiag)
            for r in rows: r.update(dataset=ds,seed=int(seed),backbone=arm,checkpoint_sha256=ckpt_sha,
              split_hash=s3.key_sha(ctx["keys"][int(i)] for i in fit),
              anchor_order_sha256=_hash_anchor_order(coords,b,f))
            rows_all.extend(rows)
        baseline_config=_legacy_clean_config(ds,seed,arm,fitpack,ctx,ckpt_sha)
        baseline_eval=_eval_config(fitpack,ctx,baseline_config,seed)
        base=_baseline_row(baseline_config,baseline_eval)
        base.update(dataset=ds,seed=int(seed),backbone=arm,checkpoint_sha256=ckpt_sha,
            split_hash=s3.key_sha(ctx["keys"][int(i)] for i in fit),
            anchor_order_sha256=_hash_anchor_order(coords,b,f),geometry_role="clean_baseline_reference")
        rows_all.append(base)
        rows_all=_add_objectives(rows_all,base)
        scored_metric_fields=("coordinate_mae_m","transport_w1_m","bf_accuracy","bf_cross_entropy","joint_nll",
            "site_p95_mae_m","site_p99_mae_m","worst_site_mae_m")
        nonfinite_candidate_count=sum(not all(math.isfinite(float(r.get(k,math.nan))) for k in scored_metric_fields)
            for r in rows_all)
        finite_candidate_count=len(rows_all)-nonfinite_candidate_count
        chosen={}; s1={}; s2sel={}
        for lam in LAMBDAS:
            r=_choose(rows_all,lam)
            if r is None: raise RuntimeError(f"no feasible candidate for lambda={lam}")
            chosen[f"lambda_{lam:.2f}"]=_config(r)
            r1=_choose(rows_all,lam,"single_S1"); s1[f"lambda_{lam:.2f}"]=_config(r1) if r1 else None
            r2=_choose(rows_all,lam,"single_S2"); s2sel[f"lambda_{lam:.2f}"]=_config(r2) if r2 else None
        clean=_attach_virtual_measure(_config(base),geometry_diagnostics)
        for config in chosen.values(): _attach_virtual_measure(config,geometry_diagnostics)
        for config in list(s1.values())+list(s2sel.values()):
            if config is not None: _attach_virtual_measure(config,geometry_diagnostics)
        near_optimal=_near_optimal_sensitivity(rows_all)
        selection={"status":"LOCKED_CAL_FIT","dataset":ds,"seed":int(seed),"backbone":arm,
          "checkpoint_sha256":ckpt_sha,"selected_epoch":int(payload["selected_epoch"]),"native_E_alpha":native_alpha,
          "selection_role":"CAL-fit","cal_eval_not_used_for_selection":True,"gamma_space":0.0,
          "joint_calibrated_base":clean,"baseline_reference_source":baseline_config["baseline_source"],
          "selected_joint":chosen,"single_S1":s1,"single_S2":s2sel,
          "near_optimal_sensitivity":near_optimal,
          "near_optimal_objective_tolerance_absolute":NEAR_OPTIMAL_OBJECTIVE_TOLERANCE,
          "finite_candidate_metric_count":int(finite_candidate_count),
          "nonfinite_candidate_metric_count":int(nonfinite_candidate_count),
          "lambda_2d_nll":list(LAMBDAS),"candidate_count":len(rows_all),"geometry_count":len(geoms),
          "inner_grid_count":len(inner),"geometry_diagnostics":geometry_diagnostics,
          "fit_split_sha256":s3.key_sha(ctx["keys"][int(i)] for i in fit),
          "eval_split_sha256":s3.key_sha(ctx["keys"][int(i)] for i in ev),
          "anchor_order_sha256":_hash_anchor_order(coords,b,f),
          "bf_accuracy_guard":"candidate macro BF accuracy >= clean CAL-fit baseline",
          "bf_cross_entropy_guard":"candidate BF CE <= clean baseline * 1.01 + 1e-8",
          "algorithm_code_sha256":s3.sha(Path(__file__)),
          "source_manifest_sha256":s3.sha(manifest_path),
          "input_run_file_sha256":{p.name:s3.sha(p) for p in sorted(Path(ctx["historical_run"]).iterdir()) if p.is_file()},
          **TEST_COUNTS}
        selection["parameter_config_sha256"]=_stable_hash(selection)
        _write_json(sp,selection,exclusive=True)
        _write_json(lock,{"selection_role":"CAL-fit","evaluation_role":"CAL-eval","selection_frozen_before_cal_eval":True,
          "selection_sha256":s3.sha(sp),"checkpoint_sha256":ckpt_sha,"fit_split_sha256":selection["fit_split_sha256"],
          "eval_split_sha256":selection["eval_split_sha256"],"anchor_order_sha256":selection["anchor_order_sha256"],
          "parameter_config_sha256":selection["parameter_config_sha256"],
          "algorithm_code_sha256":selection["algorithm_code_sha256"],"source_manifest_sha256":selection["source_manifest_sha256"],
          "input_run_file_sha256":selection["input_run_file_sha256"],**TEST_COUNTS},exclusive=True)
        # CAL-eval targets and RSS may only be accessed after the immutable selection lock verifies.
        locked_selection=_verify_locked_selection(sp,lock,manifest_path,Path(args.input_root).resolve())
        _verify_selection_context(locked_selection,ctx)
        evalpack=s3.make_pack(ctx,model,ev,virtual_gamma=0.0)
        configs={"joint_lambda_0.02":chosen["lambda_0.02"],"joint_lambda_0.05":chosen["lambda_0.05"],"joint_calibrated_base":clean}
        for lam_key in ("lambda_0.02","lambda_0.05"):
            if s1[lam_key]: configs[f"single_S1_{lam_key}"]=s1[lam_key]
            if s2sel[lam_key]: configs[f"single_S2_{lam_key}"]=s2sel[lam_key]
        old=_old_combo(ds,seed,arm,ckpt_sha)
        if old: configs["old_combo_1D_2A"]=old
        metrics={}
        for name,config in configs.items(): metrics[name]=_eval_config(evalpack,ctx,config,seed)
        report={"status":"PASS","dataset":ds,"seed":int(seed),"backbone":arm,"checkpoint_sha256":ckpt_sha,
          "selection_sha256":s3.sha(sp),"selection_role":"CAL-fit","evaluation_role":"CAL-eval",
          "cal_eval_used_for_selection":False,"metrics":metrics,"old_combo_available":old is not None,
          "gamma_space":0.0,"test_rows_scored":0,**TEST_COUNTS}
        _write_json(rp,report,exclusive=True); _csv_write(rd/"CANDIDATE_METRICS.csv",rows_all)
        _write_json(rd/"RUN.lock",{"checkpoint_sha256":ckpt_sha,"selection_sha256":s3.sha(sp),
          "fit_split_sha256":selection["fit_split_sha256"],"eval_split_sha256":selection["eval_split_sha256"],
          "anchor_order_sha256":selection["anchor_order_sha256"],"parameter_config_sha256":selection["parameter_config_sha256"],
          "algorithm_code_sha256":selection["algorithm_code_sha256"],"source_manifest_sha256":selection["source_manifest_sha256"],
          "input_run_file_sha256":selection["input_run_file_sha256"],"device":args.device,"gamma_space":0.0,**TEST_COUNTS},exclusive=True)
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
      "Selection is locked on CAL-fit before CAL-eval. CAL-eval is reporting-only. Internal/official TEST scores: 0. Gamma-space is fixed at 0.",
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
    lock_checks=[]
    manifest=_read_json(out/"EXECUTION_MANIFEST.json")
    input_root=Path(manifest["input_root"])
    for report in reports:
        rd=out/"runs"/f"{report['dataset']}_seed{report['seed']}"/report["backbone"]
        sp=rd/"CAL_FIT_SELECTION.json"; lk=rd/"CAL_EVAL.lock"
        try:
            selection=_read_json(sp); lock=_read_json(lk)
            checkpoint_path=comp._checkpoint_path(Path(manifest["checkpoint_root"]),report["dataset"],int(report["seed"]),report["backbone"])
            checkpoint_hash_ok=(checkpoint_path.is_file() and s3.sha(checkpoint_path)==report.get("checkpoint_sha256")
                and report.get("checkpoint_sha256")==selection.get("checkpoint_sha256")==lock.get("checkpoint_sha256"))
            config_hash=selection.get("parameter_config_sha256")
            unhashed=dict(selection); unhashed.pop("parameter_config_sha256",None)
            input_hash_ok=all((input_root/f"{report['dataset']}_seed{report['seed']}"/name).is_file() and
                s3.sha(input_root/f"{report['dataset']}_seed{report['seed']}"/name)==digest
                for name,digest in selection.get("input_run_file_sha256",{}).items())
            lock_checks.append(bool(
                lock.get("selection_sha256")==s3.sha(sp)==report.get("selection_sha256") and
                config_hash==_stable_hash(unhashed) and
                selection.get("algorithm_code_sha256")==s3.sha(Path(__file__)) and
                selection.get("source_manifest_sha256")==s3.sha(out/"EXECUTION_MANIFEST.json") and
                lock.get("parameter_config_sha256")==config_hash and input_hash_ok and checkpoint_hash_ok and
                report.get("cal_eval_used_for_selection") is False and
                report.get("dataset")==selection.get("dataset") and report.get("seed")==selection.get("seed")
                and report.get("backbone")==selection.get("backbone") and
                all(report.get(k)==0 and lock.get(k)==0 and selection.get(k)==0 for k in TEST_COUNTS)))
        except (OSError,KeyError,ValueError):
            lock_checks.append(False)
    expected=len(s3.DATASETS)*len(s3.SEEDS)*len(ARMS)
    status="PASS" if len(reports)==expected and len(selections)==expected and all(lock_checks) and len(files)==expected and len(sensitivity_rows)==expected*len(LAMBDAS) else "INCOMPLETE"
    _write_json(out/"JOINT_FIT_SELECTION.json",{"status":status,"runs":selections,
      "cal_eval_report_count":len(reports),"expected_run_count":expected,
      "selection_role":"CAL-fit","evaluation_role":"CAL-eval","gamma_space":0.0,
      "lambda_2d_nll":list(LAMBDAS),"selection_locks_verified":bool(len(lock_checks)==expected and all(lock_checks)),
      **TEST_COUNTS})
    _write_json(out/"INTEGRITY_REVIEW.json",{"status":status,"phase":"aggregate","completed_runs":len(reports),
      "expected_runs":expected,"selection_lock_checks_passed":int(sum(lock_checks)),
      "selection_locks_verified":bool(len(lock_checks)==expected and all(lock_checks)),
      "selection_locked_before_cal_eval":all(r.get("selection_role")=="CAL-fit" and r.get("cal_eval_used_for_selection") is False for r in reports),
      "gamma_space":0.0,"candidate_files":len(files),"near_optimal_sensitivity_rows":len(sensitivity_rows),
      "near_optimal_sensitivity_complete":len(sensitivity_rows)==expected*len(LAMBDAS),**TEST_COUNTS})
    (out/"RUN_REPORT_V17_DATASET_ADAPTIVE_JOINT_FIT.md").write_text(
      "# V17 Dataset-Adaptive Spatial Joint Fit\n\n"
      "See FIT_NEIGHBOR_DISTANCE_DIAGNOSTICS.md for FIT-only geometry; SPATIAL_SCALE_SEARCH_SPACE.json for candidate design; "
      "JOINT_FIT_SELECTION.json for locked choices and per-run near-optimal sensitivity; "
      "CANDIDATE_METRICS.csv, PARETO_FRONTIER.csv and NEAR_OPTIMAL_SENSITIVITY.csv for candidate diagnostics; "
      "CAL_EVAL_REPORT.md for reporting-only CAL-eval.\n\n"
      f"Completed run reports: {len(reports)} / {len(s3.DATASETS)*len(s3.SEEDS)*len(ARMS)}. Gamma-space=0; TEST scores=0.\n",encoding="utf-8")
    print(json.dumps({"stage":"aggregate_complete","runs":len(reports),**TEST_COUNTS},ensure_ascii=False),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest="command",required=True)
    for name in ("preflight","smoke","run","all","aggregate"):
        q=sub.add_parser(name); q.add_argument("--input-root",default=str(DEFAULT_INPUT))
        q.add_argument("--checkpoint-root",default=str(DEFAULT_CHECKPOINT_ROOT)); q.add_argument("--output",default=str(DEFAULT_OUTPUT))
        q.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    args=p.parse_args(); torch.set_num_threads(4)
    if args.command=="preflight": preflight(args)
    elif args.command=="smoke": smoke(args)
    elif args.command=="run": run(args)
    elif args.command=="aggregate": aggregate(argparse.Namespace(output=args.output))
    else: preflight(args); smoke(args); run(args)
if __name__=="__main__": main()
