"""Post-confirmation development of a full-distribution original-method anchor.

The original nine recipes remain immutable. This follow-up is exploratory;
previously examined confirmation498 cannot be called a fresh independent test.
"""
import argparse
import collections
import numpy as np
from .common import *
from .score_branches import score_readout
from mydata_bench.meter_eval.readout import canonicalize

PHASE_ROOT = RESEARCH / 'post_confirmation_anchor_development_v1'
RHOS = [0., .25, .5, .75, 1.]


def anchored_logits(parent, baseline, original, rho):
    values = [np.asarray(v, dtype=np.float64) for v in [parent, baseline, original]]
    if len({a.shape for a in values}) != 1 or values[0].ndim != 1 or len(values[0]) < 2:
        raise ValueError('Matching full native score vectors required')
    if not all(np.isfinite(a).all() for a in values) or not 0 <= rho <= 1:
        raise ValueError('Finite vectors and convex anchor weight required')
    if rho == 0:
        return list(parent)
    return (values[0] + rho * (values[2] - values[1])).tolist()


def prepare():
    parent_path = RESEARCH / 'frozen_cohort_completion_v1/frozen_manifest_v1.json'
    parent = json.loads(parent_path.read_text())
    confirmation = json.loads(pathlib.Path(parent['confirmation_manifest_path']).read_text())
    cases = []
    for case in parent['cases']:
        source = pathlib.Path(case['folder']) / 'sweep.jsonl'
        if hashlib.sha256(source.read_bytes()).hexdigest() != case['sweep_sha256']:
            raise ValueError('Frozen development source changed')
        cases.append({k: case[k] for k in ['model', 'protocol', 'folder', 'sweep_sha256', 'conditions',
                                           'selected_ks', 'primary', 'best_development_original']})
    create_json(PHASE_ROOT / 'prospective_plan_v1.json', {
        'time': timestamp(), 'phase': 'Post-confirmation exploratory development on development234 only',
        'parent_manifest': str(parent_path), 'parent_manifest_sha256': fingerprint(parent),
        'original_confirmation_manifest_sha256': fingerprint(confirmation),
        'development_ids_path': str(RESEARCH / 'development_v1/development_ids.json'),
        'development_ids_sha256': confirmation['analysis_protocol']['development_ids_sha256'],
        'labels_sha256': confirmation['input_file_sha256']['labels.jsonl'],
        'rhos': RHOS, 'cases': cases,
        'formula': 'z_anchor = z_parent_recipe + rho * (z_original_same_k - z_matched_native); rho in [0,1]. Equivalently (1-rho)z_native + rho*z_original + the unchanged parent attention-evidence term. Every native bin retained.',
        'hypothesis': 'Original strong steering and symmetric region evidence may offer complementary ordinal information. A convex anchor in logit space can change their tradeoff; no guarantee of MAE or endpoint-accuracy improvement follows from the formula.',
        'fixed': 'Keep all nine parent model/input cases, three k values, middle primary, region/scope/radius/gain/confidence/KL settings unchanged. Add only the stated shared scalar anchor weight per case. Do not mutate any original inference file or result.',
        'selection': 'For each case choose one rho shared by its three k values. Require full coverage, both-baseline four-direction/+10pp point gates, same-k original MAE/accuracy improvements at all three k, all-neighbor five-bin MAE improvement, and primary improvement over the development-selected original. Among eligible rhos maximize the smallest same-k original MAE improvement, then the smallest same-k accuracy improvement, then prefer smaller rho. This rule is an explicitly post-confirmation development decision.',
        'boundary': 'The v1 confirmation failure is retained. The current follow-up uses development234 for parameter selection; any later examination on the already-seen confirmation498 is reused-validation exploratory evidence, never a new independent confirmation. Remaining114 overlaps ranking and is also not a fresh independent test. No performance of a new anchored recipe has been computed when this plan is created.',
        'source_sha256': hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()})
    print(PHASE_ROOT / 'prospective_plan_v1.json')


def run():
    plan_path = PHASE_ROOT / 'prospective_plan_v1.json'
    plan = json.loads(plan_path.read_text())
    if hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest() != plan['source_sha256']:
        raise ValueError('Development implementation differs from prospective plan')
    ids = json.loads(pathlib.Path(plan['development_ids_path']).read_text())
    if fingerprint(ids) != plan['development_ids_sha256'] or len(ids) != 234:
        raise ValueError('Development IDs changed')
    neural = {}
    zero_audits = {}
    for case in plan['cases']:
        name = case['model'] + '_' + case['protocol']
        output = PHASE_ROOT / name
        if output.exists():
            raise ValueError('Do not overwrite or silently restart anchored development')
        source = pathlib.Path(case['folder']) / 'sweep.jsonl'
        blob = source.read_bytes()
        if hashlib.sha256(blob).hexdigest() != case['sweep_sha256']:
            raise ValueError('Frozen development source changed')
        indexed = collections.defaultdict(dict)
        for line in blob.decode().splitlines():
            row = canonicalize(json.loads(line))
            indexed[row['condition']][row['example_id']] = row
        output.mkdir()
        new_rows = collections.defaultdict(dict)
        zero_bad = []
        for rho in plan['rhos']:
            for parent_name, k in zip(case['conditions'], case['selected_ks']):
                condition = f'anchor_rho{rho:g}__{parent_name}'
                for eid in ids:
                    required = [indexed[n].get(eid) for n in [parent_name, 'baseline', f'original_all_k{k}']]
                    row = {'time': timestamp(), 'example_id': eid, 'condition': condition,
                           'rho': rho, 'k': k, 'parent_condition': parent_name, 'source': str(source),
                           'source_sha256': case['sweep_sha256'], 'phase': plan['phase']}
                    if any(r is None or r.get('status') != 'ok' for r in required):
                        row.update(status='error', error='One or more fixed parent sources unavailable')
                    else:
                        logits = anchored_logits(*(r['score_logits'] for r in required), rho)
                        row.update(status='ok', score_logits=logits, **score_readout(logits, case['model']))
                        if rho == 0 and any(row.get(field) != required[0].get(field) for field in ['score_logits', 'progress', 'native_prediction']):
                            zero_bad.append({'example_id': eid, 'condition': parent_name})
                    append(output / 'anchored_scores.jsonl', row)
                    new_rows[condition][eid] = row
        create_json(output / 'generation_audit_v1.json', {'time': timestamp(), 'source_sha256': case['sweep_sha256'],
                     'n_expected_new_rows': len(ids) * len(case['conditions']) * len(plan['rhos']),
                     'rho0_parent_exact': not zero_bad, 'rho0_mismatches': zero_bad,
                     'label_access_during_generation': False})
        neural[name] = (case, indexed, new_rows)
        zero_audits[name] = not zero_bad
        print(name, 'label-free generation complete; rho0 exact', not zero_bad, flush=True)
    # Only after every case's new score vectors are saved do we join labels.
    from .metrics import summarize
    from .analyze_development import contrast
    label_blob = (BASE / 'inputs/labels.jsonl').read_bytes()
    if hashlib.sha256(label_blob).hexdigest() != plan['labels_sha256']:
        raise ValueError('Frozen labels changed')
    labels = {r['example_id']: r for r in map(json.loads, label_blob.decode().splitlines())}
    selections = {}
    reports = {}
    for name, (case, indexed, new_rows) in neural.items():
        refs = list(dict.fromkeys(['baseline', 'native_generation_baseline', *case['conditions'],
                                  *[f'original_all_k{k}' for k in case['selected_ks']], case['best_development_original']]))
        summaries = {n: summarize(indexed[n].values(), labels, ids) for n in refs}
        summaries.update({n: summarize(rows.values(), labels, ids) for n, rows in new_rows.items()})
        candidates = []
        for rho in plan['rhos']:
            comparisons = {}
            margins_mae, margins_accuracy = [], []
            for parent_name, k in zip(case['conditions'], case['selected_ks']):
                condition = f'anchor_rho{rho:g}__{parent_name}'
                s = summaries[condition]
                effects = {b: contrast(s, summaries[b], case['model'] in {'meter', 'sole'}) for b in
                           dict.fromkeys(['baseline', 'native_generation_baseline', f'original_all_k{k}', case['best_development_original'], parent_name])}
                original = effects[f'original_all_k{k}']
                gate = s['complete'] and all(effects[b].get('point_estimate_10pp_gate', False)
                                           for b in ['baseline', 'native_generation_baseline'])
                gate = gate and original.get('complete', False) and original.get('mae_delta', 1) < 0 and original.get('accuracy_delta', -1) > 0
                gate = gate and all(s['overall']['mae'] is not None and summaries[b]['overall']['mae'] is not None
                                    and s['overall']['mae'] < summaries[b]['overall']['mae'] for b in ['baseline', 'native_generation_baseline'])
                if parent_name == case['primary']:
                    best = effects[case['best_development_original']]
                    gate = gate and best.get('complete', False) and best.get('mae_delta', 1) < 0 and best.get('accuracy_delta', -1) > 0
                comparisons[condition] = {'effects': effects, 'development_gate': gate}
                margins_mae.append(-original.get('mae_delta', 1))
                margins_accuracy.append(original.get('accuracy_delta', -1))
            candidates.append({'rho': rho, 'eligible': zero_audits[name] and all(c['development_gate'] for c in comparisons.values()),
                               'min_same_k_original_mae_improvement': min(margins_mae),
                               'min_same_k_original_accuracy_improvement': min(margins_accuracy), 'comparisons': comparisons})
        eligible = [c for c in candidates if c['eligible']]
        selected = max(eligible, key=lambda c: (c['min_same_k_original_mae_improvement'], c['min_same_k_original_accuracy_improvement'], -c['rho'])) if eligible else None
        selections[name] = {'rho': selected['rho'] if selected else None, 'eligible': bool(selected),
                            'unchanged_parent_primary': case['primary'], 'unchanged_ks': case['selected_ks']}
        reports[name] = {'case': case, 'summaries': summaries, 'candidates': candidates, 'selection': selections[name]}
        create_json(PHASE_ROOT / name / 'development_analysis_v1.json', reports[name])
        print(name, selections[name], flush=True)
    create_json(PHASE_ROOT / 'development_selection_v1.json', {'time': timestamp(), 'plan_sha256': fingerprint(plan),
                'selections': selections, 'all_nine_development_eligible': all(s['eligible'] for s in selections.values()),
                'nonzero_anchor_cases': [n for n, s in selections.items() if s['rho'] not in [None, 0]],
                'case_analysis_files': {n: str(PHASE_ROOT / n / 'development_analysis_v1.json') for n in reports},
                'boundary': plan['boundary']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    prepare() if args.prepare else run()
