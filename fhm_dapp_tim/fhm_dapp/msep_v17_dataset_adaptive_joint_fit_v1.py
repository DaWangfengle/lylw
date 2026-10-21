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
QUANTILES=(0.00,0.01,0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.99,1.00)

def _write_json(path,value,exclusive=False):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x" if exclusive else "w",encoding="utf-8") as f:
        json.dump(s3._json_safe(value),f,ensure_ascii=False,indent=2,allow_nan=False); f.write("\n")

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
        block=xy[members]; delta=block[:,None,:]-block[None,:,:]
        dist=np.sqrt(np.maximum((delta*delta).sum(-1),0.0))
        zeros=int(np.triu(dist==0.0,1).sum()); zero_pairs+=zeros
        np.fill_diagonal(dist,np.inf); sorted_dist=np.sort(dist,axis=1)
        groups[f"{key[0]}::{key[1]}"]={"anchors":len(members),"zero_distance_pairs":zeros,"kth_neighbor":{}}
        for k in (1,2,4,8):
            vals=sorted_dist[:,k-1]; vals=vals[np.isfinite(vals)]; kth[k].extend(vals.tolist())
            groups[f"{key[0]}::{key[1]}"]["kth_neighbor"][str(k)]=_quantiles(vals)
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
        "base_union_knn_edge_distance_m":_quantiles(lengths),
        "base_union_knn_edge_threshold_counts":_threshold_counts(lengths),
        "base_union_knn_edges":int(len(e)),
        "base_union_knn_edge_isolated_anchor_count":int(len(xy)-len(set(e.reshape(-1).tolist()))),
        "groups":groups,"distance_semantics":"FIT anchor k-neighbors are within same building/floor; union-kNN edge lengths are reported separately."}

def build_cutoff_graph(coords,buildings,floors,base_edges,r_max_m,tau_graph_m,*,sinkhorn_iterations=20000,tolerance=1e-10):
    xy=np.asarray(coords,dtype=np.float64); b=np.asarray(buildings,dtype=np.int64); f=np.asarray(floors,dtype=np.int64)
    e=np.asarray(torch.as_tensor(base_edges).detach().cpu(),dtype=np.int64).reshape(-1,2)
    if not (math.isinf(float(r_max_m)) or (math.isfinite(float(r_max_m)) and r_max_m>0)): raise ValueError("R_max must be positive or +inf")
    if not (math.isinf(float(tau_graph_m)) or (math.isfinite(float(tau_graph_m)) and tau_graph_m>0)): raise ValueError("tau_graph must be positive or +inf")
    n=len(xy)
    if len(e):
        e=np.unique(np.sort(e,axis=1),axis=0)
        if np.any(e[:,0]==e[:,1]) or np.any(e<0) or np.any(e>=n): raise ValueError("invalid edge")
        if np.any(b[e[:,0]]!=b[e[:,1]]) or np.any(f[e[:,0]]!=f[e[:,1]]): raise ValueError("cross-BF edge rejected")
        lengths=np.linalg.norm(xy[e[:,0]]-xy[e[:,1]],axis=1); keep=lengths<=float(r_max_m)
        kept=e[keep]; d=lengths[keep]; removed=int((~keep).sum())
    else: kept=np.empty((0,2),dtype=np.int64); d=np.empty(0); removed=0
    a=np.eye(n,dtype=np.float64)
    for (i,j),dist in zip(kept,d):
        exponent=0.0 if math.isinf(float(tau_graph_m)) else float(dist)/float(tau_graph_m)
        w=max(math.exp(-min(exponent,745.0)),np.finfo(np.float64).tiny); a[i,j]=a[j,i]=max(a[i,j],w)
    scale=np.ones(n); converged=False; iterations=0
    for iterations in range(1,sinkhorn_iterations+1):
        mass=scale*(a@scale)
        if not np.isfinite(mass).all() or np.any(mass<=0): raise RuntimeError("invalid Sinkhorn row mass")
        updated=scale/np.sqrt(mass)
        if float(np.max(np.abs(updated-scale)))<=tolerance: scale=updated; converged=True; break
        scale=updated
    w=scale[:,None]*a*scale[None,:]
    re=float(np.max(np.abs(w.sum(1)-1))); ce=float(np.max(np.abs(w.sum(0)-1)))
    if not np.isfinite(w).all() or not np.allclose(w,w.T,atol=2e-10,rtol=2e-10): raise RuntimeError("nonfinite/asymmetric graph")
    if max(re,ce)>2e-6: raise RuntimeError(f"Sinkhorn residual too large: {re},{ce}")
    if np.any((w>0)&((b[:,None]!=b[None,:])|(f[:,None]!=f[None,:]))): raise RuntimeError("cross-BF graph weight")
    rr,cc=np.nonzero(w>0)
    graph=torch.sparse_coo_tensor(torch.as_tensor(np.stack([rr,cc]),dtype=torch.long),
        torch.as_tensor(w[rr,cc],dtype=torch.float32),(n,n)).coalesce()
    deg=np.bincount(kept.reshape(-1),minlength=n) if len(kept) else np.zeros(n,dtype=np.int64)
    return graph,torch.as_tensor(kept.copy(),dtype=torch.long),{
        "r_max_s1_m":float(r_max_m),"tau_graph_m":float(tau_graph_m),"base_edges":int(len(e)),
        "retained_edges":int(len(kept)),"removed_edges":removed,
        "retained_fraction":float(len(kept)/len(e)) if len(e) else 0.0,
        "degree_min":int(deg.min()),"degree_median":float(np.median(deg)),"degree_max":int(deg.max()),
        "zero_neighbor_nodes":int((deg==0).sum()),"sinkhorn_converged":converged,
        "sinkhorn_iterations":iterations,"sinkhorn_row_error":re,"sinkhorn_col_error":ce,
        "retained_edge_distance_m":_quantiles(d),"retained_edge_threshold_counts":_threshold_counts(d)}

def normalized_virtual_endpoint_weights(edges,coords,tau_virt_m):
    e=np.asarray(torch.as_tensor(edges).detach().cpu(),dtype=np.int64).reshape(-1,2); xy=np.asarray(coords,dtype=np.float64); n=len(xy)
    if not (math.isinf(float(tau_virt_m)) or (math.isfinite(float(tau_virt_m)) and tau_virt_m>0)): raise ValueError("tau_virt must be positive or +inf")
    if len(e):
        e=np.sort(e,axis=1)
        if len({tuple(x) for x in e.tolist()})!=len(e) or np.any(e[:,0]==e[:,1]): raise ValueError("virtual edges must be unique undirected non-self edges")
        d=np.linalg.norm(xy[e[:,0]]-xy[e[:,1]],axis=1)
        k=np.ones(len(e)) if math.isinf(float(tau_virt_m)) else np.exp(-np.minimum(d/float(tau_virt_m),745.0))
        k=np.maximum(k,np.finfo(np.float64).tiny); deg=np.bincount(e.reshape(-1),minlength=n); den=np.zeros(n)
        np.add.at(den,e[:,0],k); np.add.at(den,e[:,1],k)
        wi=k/den[e[:,0]]; wj=k/den[e[:,1]]
        for node in range(n):
            ix=np.flatnonzero((e[:,0]==node)|(e[:,1]==node))
            total=sum(wi[z] if e[z,0]==node else wj[z] for z in ix)
            if len(ix) and not np.isclose(total,1.0,atol=1e-10): raise RuntimeError("virtual endpoint weights do not conserve mass")
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
    wi=torch.as_tensor(endpoint_weights["weight_i"],dtype=parent_logits.dtype,device=parent_logits.device)
    wj=torch.as_tensor(endpoint_weights["weight_j"],dtype=parent_logits.dtype,device=parent_logits.device)
    degree=torch.as_tensor(endpoint_weights["degree"],dtype=torch.long,device=parent_logits.device)
    pmass=torch.where(degree>0,parent_logits.new_full((n,),1.0-float(beta)),parent_logits.new_ones((n,)))
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
