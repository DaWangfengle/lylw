"""Independently recompute saved site summaries; never runs a model forward."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
AUDIT = Path(__file__).resolve().parent
METHODS = ["v17_single_raw", "v17_single_temp_t06", "v17_single_temp_fit",
           "v17_single_g13_historical_cal", "v17_single_g13", "v17_single_unified_frozen",
           "unified_joint_no_evidence", "unified_joint"]
FIELDS = {"row_nll":"nll", "correct_neighborhood_mass":"mass", "correct_neighborhood_coverage":"coverage",
          "true_floor_probability":"floor", "true_building_probability":"building",
          "building_floor_joint_accuracy":"joint_correct", "correct_neighborhood_top1_rate":"top1_near",
          "correct_neighborhood_top10_rate":"top10_near", "anchor_entropy_nats":"entropy",
          "nearest_component_center_error_m_oracle":"nearest_center"}


def finite(value):
    if isinstance(value, dict):
        for v in value.values(): finite(v)
    elif isinstance(value, list):
        for v in value: finite(v)
    elif isinstance(value, (float, int)):
        assert math.isfinite(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('smoke','full'))
    args = parser.parse_args()
    evidence = ROOT / 'fhm_dapp_tim/evidence'
    audit, paired = {}, {}
    data_audit = json.loads((AUDIT/'DATA_AUDIT.json').read_text(encoding='utf-8'))
    for name in ('tuji', 'uts', 'uji'):
        output = evidence / (f'msep_single_smoke_{name}' if args.mode == 'smoke' else f'msep_single_{name}_full_v1')
        result = json.loads((output/'RESULT.json').read_text(encoding='utf-8'))
        site_file = json.loads((output/'SITE_DIAGNOSTICS.json').read_text(encoding='utf-8'))
        finite(result); finite(site_file)
        role = 'dev' if args.mode == 'smoke' else 'test'
        metrics_key = 'dev_smoke_metrics' if args.mode == 'smoke' else 'test_metrics'
        assert result['evaluation_role'] == site_file['evaluation_role'] == role
        assert metrics_key in result
        assert ('test_metrics' not in result) if args.mode == 'smoke' else ('dev_smoke_metrics' not in result)
        assert set(result[metrics_key]) == set(site_file['by_model']) == set(METHODS)
        checkpoint_path = output/'CHECKPOINTS.pt'
        checkpoint_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        assert checkpoint_sha == result['checkpoint_sha256']
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        roles = {r: {tuple(k) for k in keys} for r,keys in checkpoint['role_site_keys'].items()}
        assert all(not roles[a] & roles[b] for a in roles for b in roles if a != b)
        for r, keys in roles.items():
            digest = hashlib.sha256(json.dumps(sorted(keys),separators=(',',':')).encode()).hexdigest()
            assert digest == result['role_site_key_sha256'][r] == data_audit[name]['roles'][r]['site_keys_sha256']
            assert len(keys) == data_audit[name]['roles'][r]['sites']
            assert len(checkpoint['role_site_keys'][r]) == data_audit[name]['roles'][r]['rows']
        if args.mode == 'smoke':
            assert 'test' not in checkpoint['role_site_keys']
            assert not (output/'TEST_INVOCATION.lock').exists()
        else:
            lock = json.loads((output/'TEST_INVOCATION.lock').read_text(encoding='utf-8'))
            assert lock['checkpoint_sha256'] == checkpoint_sha and lock['all_learned_choices_frozen']
            for source, digest in lock['source_sha256'].items():
                assert hashlib.sha256((ROOT/'fhm_dapp_tim/fhm_dapp'/source).read_bytes()).hexdigest() == digest
        all_sites, maximum = None, 0.
        descriptions = {}
        for method in METHODS:
            sites = site_file['by_model'][method]
            keys = [tuple(s['site']) for s in sites]
            assert len(set(keys)) == len(keys) == result['role_count'][role]['sites']
            assert set(keys) == roles[role]
            if all_sites is None: all_sites = keys
            assert all_sites == keys
            n = np.asarray([s['scan_count'] for s in sites], dtype=float)
            assert n.sum() == result['role_count'][role]['rows']
            errors = np.asarray([s['err'] for s in sites])
            es = np.asarray([s['es'] for s in sites])
            original = result[metrics_key][method]
            expected = {'site_macro_mae_m':errors.mean(), 'site_macro_es_m':es.mean(),
                        'site_p90_mae_m':np.quantile(errors,.90), 'site_p95_mae_m':np.quantile(errors,.95),
                        'row_mae_m':np.average(errors,weights=n), 'row_es_m':np.average(es,weights=n)}
            expected.update({label:np.average([s[field] for s in sites],weights=n) for label,field in FIELDS.items()})
            for field, value in expected.items():
                difference = abs(float(value)-original[field])
                maximum = max(maximum,difference)
                assert difference <= 2e-5, (name,method,field,difference)
            descriptions[method] = {'site_macro_mae_m':float(errors.mean()), 'site_macro_es_m':float(es.mean()),
                                    'site_p90_mae_m':float(np.quantile(errors,.90)), 'site_p95_mae_m':float(np.quantile(errors,.95)),
                                    'site_p99_mae_m':float(np.quantile(errors,.99)), 'worst_site_mae_m':float(errors.max()),
                                    'site_macro_diagnostics':{field:float(np.mean([s[field] for s in sites])) for field in FIELDS.values()},
                                    'row_building_floor_joint_accuracy':original['building_floor_joint_accuracy']}
        for group in (METHODS[:6], METHODS[6:]):
            reference = site_file['by_model'][group[0]]
            for method in group[1:]:
                assert all(s['nearest_center'] == r['nearest_center'] for s,r in zip(site_file['by_model'][method],reference))
        for method in METHODS[1:3]:
            assert all(s[field] == r[field] for s,r in zip(site_file['by_model'][method],site_file['by_model'][METHODS[0]])
                       for field in ('top1_near','top10_near','coverage'))
        assert .25 <= checkpoint['calibrated_temperature'] <= 3.
        for coefficient_set in ('g13_historical_coefficients','g13_nll_coefficients'):
            assert all(0. <= value <= 1.5 for value in checkpoint[coefficient_set])
        assert torch.equal(checkpoint['frozen_head']['theta'],torch.tensor(checkpoint['g13_nll_coefficients']))
        if checkpoint['g13_nll_coefficients'] == [0.,0.]:
            # A fixed zero global coefficient structurally disables its gate.
            assert result[metrics_key]['v17_single_g13'] == result[metrics_key]['v17_single_temp_t06']
            assert result[metrics_key]['v17_single_unified_frozen'] == result[metrics_key]['v17_single_temp_t06']
        changed = {}
        for prefix in ('encoder.', 'anchor_keys', 'residual_head.'):
            matched = [k for k in checkpoint['base_model'] if k.startswith(prefix)]
            altered = [k for k in matched if not torch.equal(checkpoint['base_model'][k],checkpoint['joint_model'][k])]
            assert altered, (name,prefix)
            changed[prefix] = len(altered)
        for key in ('anchor_coords','anchor_buildings','anchor_floors','prototype_rss','encoder.seen_ap_mask'):
            if key in checkpoint['base_model']:
                assert torch.equal(checkpoint['base_model'][key],checkpoint['joint_model'][key])
        obs = checkpoint['observation_memory']
        assert int(obs['fit_rows_per_anchor'].sum()) == result['role_count']['fit']['rows']
        assert len(obs['fit_rows_per_anchor']) == result['fit_anchor_count']
        audit[name] = {'status':'PASS','evaluation_role':role,'sites':len(all_sites),'scans':int(n.sum()),
                       'maximum_reaggregation_absolute_difference':maximum,'checkpoint_sha256':checkpoint_sha,
                       'all_eight_methods_finite':True,'base_policy_continuous_centers_identical':True,
                       'joint_controls_continuous_centers_identical':True,'joint_changed_parameter_tensors':changed,
                       'summaries':descriptions}
        if args.mode == 'full':
            comparisons = {}
            for left,right in itertools.combinations(METHODS,2):
                lhs,rhs = site_file['by_model'][left],site_file['by_model'][right]
                delta = np.asarray([r['err']-l['err'] for l,r in zip(lhs,rhs)])
                delta_es = np.asarray([r['es']-l['es'] for l,r in zip(lhs,rhs)])
                comparisons[f'{right} minus {left}'] = {
                    'mae_difference_m':float(delta.mean()),'es_difference_m':float(delta_es.mean()),
                    'improved_sites':int((delta<0).sum()),'worsened_sites':int((delta>0).sum()),'unchanged_sites':int((delta==0).sum()),
                    'site_mae_differences_m':delta.tolist(),'site_es_differences_m':delta_es.tolist(),
                    'worst_harms':[{'site':list(all_sites[i]),'reference_mae':lhs[i]['err'],'candidate_mae':rhs[i]['err'],
                                    'delta_mae_m':float(delta[i]),'delta_es_m':float(delta_es[i])}
                                   for i in np.argsort(delta)[-5:][::-1]],
                    'largest_improvements':[{'site':list(all_sites[i]),'delta_mae_m':float(delta[i]),'delta_es_m':float(delta_es[i])}
                                             for i in np.argsort(delta)[:5]]}
            paired[name] = {'ordered_site_keys':[list(k) for k in all_sites],'comparisons':comparisons}
        print(json.dumps({k:v for k,v in audit[name].items() if k!='summaries'}),flush=True)
    with (AUDIT/f'{args.mode.upper()}_RESULT_AUDIT.json').open('x',encoding='utf-8') as handle:
        json.dump(audit,handle,ensure_ascii=False,indent=2,allow_nan=False)
    if paired:
        with (AUDIT/'PAIRED_SITE_ANALYSIS.json').open('x',encoding='utf-8') as handle:
            json.dump(paired,handle,ensure_ascii=False,indent=2,allow_nan=False)


if __name__ == '__main__':
    main()
