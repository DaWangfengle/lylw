import sys
sys.dont_write_bytecode=True
import collections, datetime, hashlib, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from fhm_dapp_tim.fhm_dapp import uji_v17_mother_protocol as original_v17
ROOT=Path.cwd()
EVIDENCE=ROOT/'fhm_dapp_tim/evidence/v17_g2_mechanism'
AUDIT=EVIDENCE/'run_audit'
def load(p): return json.loads(p.read_text(encoding='utf-8'))
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def key_hash(keys): return hashlib.sha256(json.dumps([list(k) for k in sorted(set(keys))],separators=(',',':'),allow_nan=False).encode()).hexdigest()
def row_hash(rows): return hashlib.sha256(np.array(sorted(rows),dtype='<i8').tobytes()).hexdigest()
def write_new(name,obj):
    with (AUDIT/name).open('x',encoding='utf-8',newline='\n') as f: json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False); f.write('\n')
def corr(x,y):
    x=np.asarray(x,dtype=float); y=np.asarray(y,dtype=float)
    if np.linalg.norm(x-x.mean())<1e-10 or np.linalg.norm(y-y.mean())<1e-10: return None
    return float(np.corrcoef(x,y)[0,1])
def main():
    g2=load(EVIDENCE/'G2_MECHANISM_AUDIT.json')
    reg=load(ROOT/'fhm_dapp_tim/evidence/v17_g1_geometry_support/registry_v17_g1.json')
    hist=load(ROOT/'fhm_dapp_tim/evidence/historical_reproduction/v17/registry_v17_reproduction.json')
    gate=load(AUDIT/'PRE_ANALYZE_GATE.json')
    frame=pd.read_csv(ROOT/'data/UJIndoorLoc/trainingData.csv',usecols=['BUILDINGID','FLOOR','LONGITUDE','LATITUDE'])
    raw=[(int(b),int(f),float(x),float(y)) for b,f,x,y in frame[['BUILDINGID','FLOOR','LONGITUDE','LATITUDE']].itertuples(index=False,name=None)]
    roles,_=original_v17.construct_coordinate_roles(set(raw))
    for role,record in hist['roles'].items():
        assert key_hash(roles[role])==record['coordinate_key_sha256'],role
        indices=[i for i,k in enumerate(raw) if k in roles[role]]
        assert row_hash(indices)==record['row_index_sha256'],role
    strata=collections.defaultdict(list)
    for k in roles['development']: strata[k[:2]].append(k)
    cal=set(); conf=set()
    def stable(seed,k): return hashlib.sha256((str(seed)+':'+repr(tuple(k))).encode()).digest()
    for bf,keys in sorted(strata.items()):
        keys=sorted(keys,key=lambda k:(stable(18111,k),k))
        if len(keys)==1: (cal if stable(18112,keys[0])[0]%2==0 else conf).add(keys[0])
        else:
            cut=max(1,min(len(keys)-1,len(keys)//2)); cal.update(keys[:cut]); conf.update(keys[cut:])
    assert len(roles['selection_refit'])==595 and len(cal)==72 and len(conf)==79
    assert not cal&conf and cal|conf==roles['development']
    assert not (cal|conf)&roles['selection_refit'] and not (cal|conf)&roles['outer_test']
    counts=collections.Counter(raw)
    training_by_bf=collections.defaultdict(list)
    for k in sorted(roles['selection_refit']): training_by_bf[k[:2]].append(k)
    role_audits={}; hand=[]; correlation_audits={}; notes=[]
    for role,group in [('calibration',cal),('confirmation',conf)]:
        ordered=sorted(group)
        indices=[i for i,k in enumerate(raw) if k in group]
        expected_rows=1529 if role=='calibration' else 1625
        assert len(indices)==expected_rows
        assert key_hash(group)==reg[role]['coordinate_key_sha256']
        assert row_hash(indices)==reg[role]['row_index_sha256']
        old=load(ROOT/('fhm_dapp_tim/evidence/v17_g1_geometry_support/'+role+'_v17_g1.json'))
        rows=g2['roles'][role]['per_coordinate']
        assert [tuple(x['coordinate']) for x in rows]==ordered
        variants=('gamma_0','gamma_0p7') if role=='calibration' else ('baseline','selected')
        checks=[]
        for variant in old['coordinate_group_vectors']:
            for endpoint,macro in [('exact_mixture_mean_mae_m','mae_2d_m'),('energy_score_2d_m','energy_score_2d_m')]:
                values=old['coordinate_group_vectors'][variant][endpoint]
                assert len(values)==len(ordered)
                restored=math.fsum(values)/len(values)
                assert math.isclose(restored,old['metrics'][variant][macro],abs_tol=1e-12)
                checks.append({'variant':variant,'endpoint':endpoint,'mean_from_vectors':restored,'original_macro':old['metrics'][variant][macro]})
        selected_indices=gate['independent_hand_check_plan'][role+'_sorted_indices_zero_based']
        for i,k in enumerate(ordered):
            anchors=training_by_bf[k[:2]]
            distances=sorted([(math.hypot(a[2]-k[2],a[3]-k[3]),a) for a in anchors],key=lambda x:(x[0],x[1]))
            nearby=[a for d,a in distances if d<=10.0]
            record=rows[i]
            assert record['training_anchors_within_10m']==len(nearby)
            assert record['training_anchor_count_same_stratum']==len(anchors)
            assert record['query_rss_row_count']==counts[k]
            for n in (1,3,5):
                assert math.isclose(record['training_anchor_d%d_m'%n],distances[min(n,len(distances))-1][0],abs_tol=1e-9)
            for endpoint,a,b in [('exact_mixture_mean_mae_m','baseline_mae_m','g1_mae_m'),('energy_score_2d_m','baseline_es_m','g1_es_m')]:
                assert record[a]==old['coordinate_group_vectors'][variants[0]][endpoint][i]
                assert record[b]==old['coordinate_group_vectors'][variants[1]][endpoint][i]
            assert math.isclose(record['mae_gain_m'],record['baseline_mae_m']-record['g1_mae_m'],abs_tol=1e-12)
            if i in selected_indices:
                physical_rows=[j for j,v in enumerate(raw) if v==k]
                hand.append({'role':role,'sorted_vector_index_zero_based':i,'coordinate_key':list(k),'source_CSV_row_indices_zero_based':physical_rows,'source_RSS_row_count':len(physical_rows),'training_anchor_count_same_building_floor':len(anchors),'training_anchors_within_10m_count':len(nearby),'training_anchors_within_10m_keys':[list(a) for a in nearby],'nearest_five_anchors':[{'distance_m':d,'coordinate_key':list(a)} for d,a in distances[:5]],'independently_calculated_d1_d3_d5_m':[distances[n-1][0] for n in (1,3,5)],'original_G1_baseline_MAE_ES':[old['coordinate_group_vectors'][variants[0]][e][i] for e in ('exact_mixture_mean_mae_m','energy_score_2d_m')],'original_G1_gamma0p7_MAE_ES':[old['coordinate_group_vectors'][variants[1]][e][i] for e in ('exact_mixture_mean_mae_m','energy_score_2d_m')],'G2_per_coordinate_matches_original_vectors_and_manual_geometry':True})
        role_audits[role]={'coordinates':len(ordered),'rows':len(indices),'coordinate_key_sha256':key_hash(group),'row_index_sha256':row_hash(indices),'all_coordinates_checked_independently':len(ordered),'all_geometry_and_G1_vector_matches':True,'vector_macro_reconstruction':checks,'first_middle_last_hand_check_indices':selected_indices}
        baseline=np.array([r['baseline_mae_m'] for r in rows])
        gain=np.array([r['mae_gain_m'] for r in rows])
        strata_list=[r['stratum'] for r in rows]
        levels=sorted(set(strata_list))
        design=np.column_stack([np.ones(len(rows)),rankdata(baseline)]+[np.array([float(x==level) for x in strata_list]) for level in levels[1:]])
        q,_=np.linalg.qr(design,mode='reduced')
        gain_rank=rankdata(gain)
        ry=gain_rank-q@(q.T@gain_rank)
        correlations={}
        for name,stored in g2['roles'][role]['associations'].items():
            x=np.array([r[name] for r in rows])
            rank_x=rankdata(x)
            pearson=corr(x,gain); spearman=corr(rank_x,gain_rank)
            assert math.isclose(pearson,stored['pearson_with_gain'],abs_tol=1e-10)
            assert math.isclose(spearman,stored['spearman_with_gain'],abs_tol=1e-10)
            review={'pearson_independently_reconstructed':pearson,'spearman_independently_reconstructed':spearman}
            if 'partial_rank_after_baseline_and_floor_building' in stored:
                rx=rank_x-q@(q.T@rank_x)
                norm=float(np.linalg.norm(rx))
                null=norm<1e-10*max(1.0,float(np.linalg.norm(rank_x)))
                partial=None if null else corr(rx,ry)
                review.update({'rank_predictor_residual_norm':norm,'relative_rank_predictor_residual_norm':norm/max(1.0,float(np.linalg.norm(rank_x))),'raw_G2_partial_rank':stored['partial_rank_after_baseline_and_floor_building'],'independent_QR_partial_rank':partial,'nuisance_design_rank':int(np.linalg.matrix_rank(design)),'residual_zero_to_machine_precision':null})
                if null:
                    assert name=='training_anchor_count_same_stratum'
                    review['interpretation']='N/A: stratum anchor total is exactly determined by building/floor controls; raw tiny nonzero value is numerical residual, not identifiable partial association.'
                    notes.append({'role':role,'proxy':name,'raw_value':stored['partial_rank_after_baseline_and_floor_building'],'interpretable_as_partial_correlation':False,'reason':review['interpretation']})
                else: assert math.isclose(partial,stored['partial_rank_after_baseline_and_floor_building'],abs_tol=1e-9)
            correlations[name]=review
        correlation_audits[role]=correlations
    assert len(hand)==6
    result={'status':'INDEPENDENT_COORDINATE_AND_VECTOR_REVIEW_PASS','recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'method':'Independent pandas coordinate-only CSV read, original V17 role constructor, independently reimplemented frozen hash split, Python math.hypot anchor distances; no G2 geometry/vector/diagnostic function called.','selection_refit':{'coordinates':595,'rows':12717,'coordinate_key_sha256':key_hash(roles['selection_refit']),'row_index_sha256':hist['roles']['selection_refit']['row_index_sha256']},'calibration_confirmation_coordinate_overlap':0,'calibration_confirmation_row_overlap':0,'development_groups':151,'roles':role_audits,'hand_checked_coordinates':hand,'hand_check_count':len(hand),'query_truth_used_for_offline_integrity_review_only':True,'original_G1_and_G2_outputs_not_modified':True,'model_loading_or_prediction_performed':False}
    write_new('INDEPENDENT_COORDINATE_REVIEW.json',result)
    write_new('CORRELATION_RESIDUAL_REVIEW.json',{'status':'DESCRIPTIVE_CORRELATIONS_RECONSTRUCTED_WITH_DISCLOSED_UNIDENTIFIABLE_PROXY','method':'SciPy average ranks; NumPy Pearson and QR projection onto baseline-rank/building/floor nuisance space. No p-values, bootstrap or new confirmation.','roles':correlation_audits,'limitations':notes,'G2_original_values_preserved':True})
    print(json.dumps({'status':result['status'],'hand_check_count':6,'all_physical_coordinates_verified':151,'role_counts':{r:{'coordinates':x['coordinates'],'rows':x['rows']} for r,x in role_audits.items()},'correlation_limitations':notes},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
