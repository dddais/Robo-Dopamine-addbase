"""F8 complete old234 analysis: original scope, both thresholds, fixed F7 comparison."""
import collections
import json
import pathlib
import yaml
from .common import ROOT, BASE, create_json, read_rows, timestamp
from .robust_prepare import DEST, digest
from .metrics import summarize
from .analyze_all_query_reward_gradient_development import original_references
from .analyze_all_query_reward_gradient_full_old import safe_effect as effect
from .f8_reference_comparison import f7_primary_rows


def main():
    policy_path = DEST/'f8_evaluation_policy_v1.json'
    policy = json.loads(policy_path.read_text())
    for path, expected in policy['evaluation_source_sha256'].items():
        if digest(ROOT/path) != expected:
            raise ValueError('Frozen F8 analysis source changed: '+path)
    plan_path = DEST/'ordinal_reward_gradient_development_frozen_plan_v1.json'
    plan = json.loads(plan_path.read_text())
    ids = json.loads((DEST/'development_ids_v1.json').read_text())
    if len(plan['configurations']) != 15:
        raise ValueError('All15 frozen development configurations required')
    loaded = []
    for item in plan['configurations']:
        if digest(item['path']) != item['sha256']:
            raise ValueError('Frozen F8 development configuration changed')
        cfg = yaml.safe_load(pathlib.Path(item['path']).read_text())
        if cfg['evaluation_policy_sha256'] != digest(policy_path):
            raise ValueError('Wrong F8 evaluation policy')
        out = pathlib.Path(cfg['output_dir'])/'development'
        done = json.loads((out/'completion_v1.json').read_text())
        path = out/'predictions.jsonl'
        if digest(path) != done['predictions_sha256'] or json.loads((out/'config.json').read_text()) != cfg:
            raise ValueError('F8 prediction/config provenance mismatch')
        rows = list(read_rows(path))
        keys = [(r['example_id'], r['condition']) for r in rows]
        expected = {(eid, c['name']) for eid in ids for c in cfg['conditions']}
        if len(keys) != len(set(keys)) or set(keys) != expected or len(keys) != done['expected_records']:
            raise ValueError('Incomplete or duplicated F8 old234 predictions')
        loaded.append((item['model']+'_'+item['protocol'], cfg, done, rows))
    labels = {r['example_id']: r for r in read_rows(BASE/'inputs/labels.jsonl') if r['example_id'] in ids}
    references = original_references(labels, ids)
    cases = {}
    for name, cfg, done, rows in loaded:
        grouped = collections.defaultdict(list)
        for row in rows:
            grouped[row['condition']].append(row)
        summaries = {condition: summarize(rr, labels, ids) for condition, rr in grouped.items()}
        base = {r['example_id']: r for r in grouped['baseline']}
        zero = {r['example_id']: r for r in grouped['reward_gradient_zero_k48']}
        zero_exact = all(base[eid].get('status') == zero[eid].get('status') == 'ok' and
            all(base[eid].get(field) == zero[eid].get(field) for field in ['score_logits', 'progress', 'native_prediction']) for eid in ids)
        model = cfg['model']
        thresholds = ['0.125_0.875', '0.2_0.8'] if model == 'meter' else ['native']
        incumbents = {c: s for c, s in summaries.items() if c.startswith('original_all_k') and s['complete']}
        incumbents.update(references['cases'].get(name, {}).get('all_available_summaries', {}))
        loss = 'continuous_ordinal_mae' if model == 'meter' else 'mae'
        best = min(incumbents, key=lambda c: (incumbents[c]['overall'][loss], -incumbents[c]['overall']['accuracy'], int(c.rsplit('k', 1)[1])))
        f7 = summarize(f7_primary_rows(policy, name, 'development', cfg, ids), labels, ids)
        effects = {}
        for k in [32, 48, 64]:
            candidate = summaries[f'reward_gradient_k{k}']
            comparisons = {'baseline': summaries['baseline'], 'native_generation_baseline': summaries['native_generation_baseline'],
                'original_same_k': summaries[f'original_all_k{k}'], 'best_development_original': incumbents[best]}
            if k == 48:
                comparisons.update(raw_all=summaries['raw_all_k48'], same_heads_readout_only=summaries['learned_readout_k48'],
                    learned_positive=summaries['learned_positive_k48'], low_ranked=summaries['reward_gradient_low_k48'],
                    F7_CE_fitted_all_query_heads=f7)
                source = DEST/f'reward_gradient_development_v1/{name}/development'
                f6_done = json.loads((source/'completion_v1.json').read_text())
                if digest(source/'predictions.jsonl') != f6_done['predictions_sha256']:
                    raise ValueError('F6 completed comparison changed')
                f6 = [r for r in read_rows(source/'predictions.jsonl') if r['condition'] == 'reward_gradient_k48']
                comparisons['F6_readout_fitted_heads'] = summarize(f6, labels, ids)
            by_threshold = {threshold: {key: effect(candidate, ref, model, threshold) for key, ref in comparisons.items()}
                            for threshold in thresholds}
            gate = zero_exact and all(e['baseline']['point_gate'] and e['native_generation_baseline']['point_gate'] and
                e['original_same_k']['original_improvement'] and e['best_development_original']['original_improvement']
                for e in by_threshold.values())
            effects[str(k)] = dict(by_threshold=by_threshold, robust_gate=gate)
        cases[name] = dict(model=model, protocol=cfg['protocol'], summaries=summaries, effects=effects,
            case_pass=all(e['robust_gate'] for e in effects.values()), zero_exact=zero_exact, best_original=best,
            predictions_sha256=done['predictions_sha256'], complete_records=len(rows),
            invalid_by_condition={key: s['overall']['n_missing_or_invalid'] for key, s in summaries.items()})
        print(name, 'passed', cases[name]['case_pass'], 'k48', effects['48']['by_threshold'][thresholds[0]]['baseline'], flush=True)
    counts = {model: sum(c['case_pass'] for c in cases.values() if c['model'] == model) for model in ['qwen', 'rr', 'meter']}
    path = DEST/('f8_frame_transport_analysis_'+timestamp().replace(':', '').replace('+', '_')+'.json')
    create_json(path, dict(time=timestamp(), phase='historically exposed old234; descriptive selection only', cases=cases,
        passing_inputs_by_model=counts, all15_complete=len(cases) == 15, development_gate=all(n >= 3 for n in counts.values()),
        frozen_plan_sha256=digest(plan_path), evaluation_policy_sha256=digest(policy_path),
        wrong_region_penalized_means_not_mechanistic_evidence=True,
        final_goal_completed=False, external_confirmation_evaluated=False))
    print('analysis', path, counts, flush=True)


if __name__ == '__main__':
    main()
