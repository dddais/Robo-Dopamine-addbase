"""Prospective exploratory entropy scaling after the closed anchor evaluation."""
import argparse
import collections
import itertools
import math
import numpy as np
from .common import *
from .score_branches import score_readout
from .anchor_development import anchored_logits
from mydata_bench.meter_eval.readout import canonicalize

PHASE_ROOT = RESEARCH / 'entropy_evidence_development_v1'


def entropy_scaled_logits(parent, baseline, original, scale, beta, rho):
    arrays = [np.asarray(v, dtype=np.float64) for v in [parent, baseline, original]]
    if len({a.shape for a in arrays}) != 1 or arrays[0].ndim != 1 or len(arrays[0]) < 2:
        raise ValueError('Matching full native score vectors required')
    if not all(np.isfinite(a).all() for a in arrays) or scale <= 0 or beta < 0 or not 0 <= rho <= 1:
        raise ValueError('Invalid scores or entropy-scaling parameters')
    base = arrays[1]; shifted = base - base.max()
    logp = shifted - np.log(np.exp(shifted).sum()); prob = np.exp(logp)
    entropy = float(-np.sum(prob * logp) / np.log(len(prob)))
    # Only correct floating-point excursions of the theoretical [0,1] entropy.
    h = min(1., max(0., entropy))
    multiplier = scale * (h ** beta if beta else 1.)
    if multiplier == 1.:
        logits = anchored_logits(parent, baseline, original, rho)
    else:
        logits = (base + multiplier * (arrays[0] - base) + rho * (arrays[2] - base)).tolist()
    return logits, {'normalized_native_entropy': entropy, 'evidence_multiplier': multiplier}


def name_for(parameters, parent):
    return f'entropy_s{parameters["scale"]:g}_b{parameters["beta"]:g}_r{parameters["rho"]:g}__{parent}'


def prepare():
    parent_path = RESEARCH / 'frozen_cohort_completion_v1/frozen_manifest_v1.json'
    anchor_path = RESEARCH / 'post_confirmation_anchor_evaluation_v1/frozen_manifest_v1.json'
    parent = json.loads(parent_path.read_text()); anchor = json.loads(anchor_path.read_text())
    confirmation = json.loads(pathlib.Path(parent['confirmation_manifest_path']).read_text())
    anchor_by_name = {c['model'] + '_' + c['protocol']: c for c in anchor['cases']}
    grid = [dict(scale=s, beta=b, rho=r) for s, b, r in itertools.product([.5, 1., 2., 4.], [0., .5, 1., 2.], [0., .25, .5, 1.])]
    cases = []
    for case in parent['cases']:
        name = case['model'] + '_' + case['protocol']
        source = pathlib.Path(case['folder']) / 'sweep.jsonl'
        if hashlib.sha256(source.read_bytes()).hexdigest() != case['sweep_sha256']:
            raise ValueError('Original development source changed')
        candidates = grid if case['model'] == 'rr' else [dict(scale=1., beta=0., rho=anchor_by_name[name]['rho'])]
        cases.append({**case, 'parameter_grid': candidates})
    create_json(PHASE_ROOT / 'prospective_plan_v1.json', {
        'time': timestamp(), 'phase': 'Exploratory development after original confirmation and anchor/full846 outcomes were seen',
        'parent_manifest_path': str(parent_path), 'parent_manifest_sha256': fingerprint(parent),
        'anchor_manifest_path': str(anchor_path), 'anchor_manifest_sha256': fingerprint(anchor),
        'development_ids_path': str(RESEARCH / 'development_v1/development_ids.json'),
        'development_ids_sha256': confirmation['analysis_protocol']['development_ids_sha256'],
        'labels_sha256': confirmation['input_file_sha256']['labels.jsonl'], 'cases': cases,
        'formula': 'z_new = z0 + scale * [H(softmax(z0))/log(B)]^beta * (z_parent - z0) + rho * (z_original_same_k - z0). The unchanged parent evidence includes its previously frozen gain/confidence/KL mechanism.',
        'hypothesis': 'A completion-expectation gate protects high predicted progress asymmetrically. An additional entropy multiplier is invariant to score-bin permutation and can reduce both low-score and high-score confident changes. Entropy is model uncertainty, not verified correctness; confident errors may be preserved. No performance improvement is guaranteed.',
        'scope': 'Keep all nine model/input cases, original k neighborhoods and primary. Keep the six Meter/Qwen prior development-selected anchor recipes fixed. Search only the three RR cases: 64 scale/beta/rho combinations per case, one common setting across its three k. Include the unchanged parent and prior anchor as nested cases.',
        'selection': 'Require complete development234, both baselines MAE down/suc up/fail up/total +10pp at all three k, same-k original MAE and accuracy improvement at every k, all-neighbor five-bin MAE down, and primary improvement over the original setting selected before confirmation. Among eligible settings maximize the minimum required margin: baseline accuracy-0.1, suc, fail, MAE improvement divided by max(1, reference MAE); same-k original accuracy and normalized MAE margins; plus primary best-original accuracy and normalized MAE margins. Ties prefer fewer modifications, lower beta, scale closer to1, lower rho. Save every candidate, no reused498/full846 performance used to choose these parameters.',
        'boundary': 'This mechanism and selection rule are post-outcome exploratory decisions. The original v1 and anchor v1 failures remain immutable. All 846 examples have now been evaluated with earlier methods; later498 and full846 are reused descriptive validation, not an independent confirmation. No new candidate outcomes were computed before this plan.',
        'mechanism_limits': 'Entropy scaling changes the whole output tilt, not branchwise conservation. Nonzero rho retains an original strong-bias branch. The old parent KL budget does not generally constrain a rescaled or anchored distribution; the retained Qwen KL case has scale1 beta0 rho0 and is exact.',
        'source_sha256': hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        'metric_source_sha256': {n: hashlib.sha256((ROOT / 'mydata_bench/auto_research_addbase' / n).read_bytes()).hexdigest() for n in ['metrics.py', 'analyze_development.py']}})
    print(PHASE_ROOT / 'prospective_plan_v1.json')


def required_margins(effects, originals):
    margins = []
    for base in ['baseline', 'native_generation_baseline']:
        effect, baseline_mae = effects[base], originals[base]
        margins += [effect['accuracy_delta'] - .1, effect['suc_delta'], effect['fail_delta'],
                    -effect['mae_delta'] / max(1., baseline_mae)]
    for name in originals:
        if name in ['baseline', 'native_generation_baseline']:
            continue
        effect = effects[name]
        margins += [effect['accuracy_delta'], -effect['mae_delta'] / max(1., originals[name])]
    return margins


def run():
    plan = json.loads((PHASE_ROOT / 'prospective_plan_v1.json').read_text())
    if hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest() != plan['source_sha256']:
        raise ValueError('Frozen entropy development implementation changed')
    for name, sha in plan['metric_source_sha256'].items():
        if hashlib.sha256((ROOT / 'mydata_bench/auto_research_addbase' / name).read_bytes()).hexdigest() != sha:
            raise ValueError('Frozen metrics changed')
    ids = json.loads(pathlib.Path(plan['development_ids_path']).read_text())
    if len(ids) != 234 or fingerprint(ids) != plan['development_ids_sha256']:
        raise ValueError('Frozen development IDs changed')
    audits = {}
    for case in plan['cases']:
        name = case['model'] + '_' + case['protocol']; out = PHASE_ROOT / name
        if out.exists():
            raise ValueError('Preserve prior outputs; do not silently restart')
        blob = (pathlib.Path(case['folder']) / 'sweep.jsonl').read_bytes()
        if hashlib.sha256(blob).hexdigest() != case['sweep_sha256']:
            raise ValueError('Frozen development source changed')
        indexed = collections.defaultdict(dict)
        for line in blob.decode().splitlines():
            row = canonicalize(json.loads(line)); indexed[row['condition']][row['example_id']] = row
        out.mkdir(); audit = {'n_rows': 0, 'invalid_rows': 0, 'parent_nested_mismatch_ids': [], 'prior_anchor_nested_mismatch_ids': []}
        with (out / 'scores.jsonl').open('x') as handle:
            for parameters in case['parameter_grid']:
                for parent_name, k in zip(case['conditions'], case['selected_ks']):
                    condition = name_for(parameters, parent_name)
                    for eid in ids:
                        required = [indexed[n].get(eid) for n in [parent_name, 'baseline', f'original_all_k{k}']]
                        row = {'example_id': eid, 'condition': condition, 'k': k, 'parameters': parameters, 'parent_condition': parent_name}
                        if any(r is None or r.get('status') != 'ok' for r in required):
                            row.update(status='error', error='One or more frozen source scores invalid'); audit['invalid_rows'] += 1
                        else:
                            logits, diagnostics = entropy_scaled_logits(*(r['score_logits'] for r in required), **parameters)
                            row.update(status='ok', score_logits=logits, **score_readout(logits, case['model']), **diagnostics)
                            if parameters['scale'] == 1 and parameters['beta'] == 0:
                                reference = anchored_logits(*(r['score_logits'] for r in required), parameters['rho'])
                                if reference != logits:
                                    audit['prior_anchor_nested_mismatch_ids'].append([eid, condition])
                                if parameters['rho'] == 0 and any(row.get(f) != required[0].get(f) for f in ['score_logits', 'progress', 'native_prediction']):
                                    audit['parent_nested_mismatch_ids'].append([eid, condition])
                        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n'); audit['n_rows'] += 1
            handle.flush(); os.fsync(handle.fileno())
        audit.update(time=timestamp(), n_expected_rows=234 * 3 * len(case['parameter_grid']),
                     label_access_during_generation=False, source_sha256=case['sweep_sha256'],
                     scores_sha256=hashlib.sha256((out / 'scores.jsonl').read_bytes()).hexdigest())
        create_json(out / 'generation_audit_v1.json', audit); audits[name] = audit
        print(name, 'generated', audit['n_rows'], 'invalid', audit['invalid_rows'], flush=True)
    # Every new score is saved before labels are loaded for parameter selection.
    from .metrics import summarize
    from .analyze_development import contrast
    label_blob = (BASE / 'inputs/labels.jsonl').read_bytes()
    if hashlib.sha256(label_blob).hexdigest() != plan['labels_sha256']:
        raise ValueError('Frozen labels changed')
    labels = {r['example_id']: r for r in map(json.loads, label_blob.decode().splitlines())}
    selections = {}; files = {}
    for case in plan['cases']:
        name = case['model'] + '_' + case['protocol']; out = PHASE_ROOT / name
        refs = list(dict.fromkeys(['baseline', 'native_generation_baseline', *case['conditions'],
                                   *[f'original_all_k{k}' for k in case['selected_ks']], case['best_development_original']]))
        rows = collections.defaultdict(dict)
        for row in read_rows(pathlib.Path(case['folder']) / 'sweep.jsonl'):
            if row['condition'] in refs:
                row = canonicalize(row); rows[row['condition']][row['example_id']] = row
        for row in read_rows(out / 'scores.jsonl'):
            rows[row['condition']][row['example_id']] = row
        summaries = {n: summarize(values.values(), labels, ids) for n, values in rows.items()}
        candidates = []; mae_key = 'continuous_ordinal_mae' if case['model'] in {'meter', 'sole'} else 'mae'
        for parameters in case['parameter_grid']:
            comparisons = {}; margins = []
            for parent_name, k in zip(case['conditions'], case['selected_ks']):
                condition = name_for(parameters, parent_name); summary = summaries[condition]
                comparisons_to = ['baseline', 'native_generation_baseline', f'original_all_k{k}']
                if parent_name == case['primary']:
                    comparisons_to.append(case['best_development_original'])
                comparisons_to = list(dict.fromkeys(comparisons_to))
                effects = {b: contrast(summary, summaries[b], case['model'] in {'meter', 'sole'}) for b in comparisons_to}
                for effect in effects.values():
                    effect.pop('confirmation', None)
                gate = all(effects[b].get('point_estimate_10pp_gate', False) for b in ['baseline', 'native_generation_baseline'])
                gate = gate and all(effects[b].get('complete', False) and effects[b]['mae_delta'] < 0 and effects[b]['accuracy_delta'] > 0 for b in comparisons_to if b not in ['baseline', 'native_generation_baseline'])
                gate = gate and all(summary['overall']['mae'] < summaries[b]['overall']['mae'] for b in ['baseline', 'native_generation_baseline'])
                margins += required_margins(effects, {b: summaries[b]['overall'][mae_key] for b in comparisons_to})
                comparisons[condition] = {'effects': effects, 'development_gate': gate}
            audit = audits[name]
            integrity = audit['n_rows'] == audit['n_expected_rows'] and not audit['invalid_rows'] and not audit['parent_nested_mismatch_ids'] and not audit['prior_anchor_nested_mismatch_ids']
            candidates.append({'parameters': parameters, 'eligible': integrity and all(c['development_gate'] for c in comparisons.values()),
                               'minimum_required_margin': min(margins), 'comparisons': comparisons})
        def order(candidate):
            p = candidate['parameters']
            return (-round(candidate['minimum_required_margin'], 12),
                    sum([p['scale'] != 1, p['beta'] != 0, p['rho'] != 0]), p['beta'], abs(math.log2(p['scale'])), p['rho'])
        eligible = sorted([c for c in candidates if c['eligible']], key=order)
        selected = eligible[0] if eligible else None
        selections[name] = {'parameters': selected['parameters'] if selected else None,
                            'eligible': bool(selected), 'minimum_required_margin': selected['minimum_required_margin'] if selected else None,
                            'unchanged_parent_primary': case['primary'], 'unchanged_ks': case['selected_ks']}
        files[name] = str(out / 'development_analysis_v1.json')
        create_json(files[name], {'time': timestamp(), 'case': case, 'summaries': summaries, 'candidates': candidates, 'selection': selections[name]})
        print(name, selections[name], flush=True)
    create_json(PHASE_ROOT / 'development_selection_v1.json', {
        'time': timestamp(), 'plan_sha256': fingerprint(plan), 'selections': selections,
        'all_nine_development_eligible': all(s['eligible'] for s in selections.values()),
        'case_analysis_files': files, 'boundary': plan['boundary']})
    print(PHASE_ROOT / 'development_selection_v1.json', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true'); args = parser.parse_args()
    prepare() if args.prepare else run()
