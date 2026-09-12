"""F7 old234 descriptive analysis; unchanged scope gate and matched controls."""
import argparse
import collections
import json
import pathlib
from .common import ROOT, BASE, RESEARCH, create_json, read_rows, timestamp
from .robust_prepare import DEST, digest
from .metrics import summarize


def effect(candidate, reference, model, threshold):
    continuous = model == 'meter'
    accuracy = 'endpoint_accuracy_' + threshold if continuous else 'accuracy'
    loss = 'continuous_ordinal_mae' if continuous else 'mae'
    complete = candidate['complete'] and reference['complete']
    delta = dict(complete=complete,
        mae_delta=candidate['overall'][loss]-reference['overall'][loss],
        accuracy_delta=candidate['overall'][accuracy]-reference['overall'][accuracy],
        suc_delta=candidate['by_split']['suc'][accuracy]-reference['by_split']['suc'][accuracy],
        fail_delta=candidate['by_split']['fail'][accuracy]-reference['by_split']['fail'][accuracy])
    delta['point_gate'] = complete and delta['mae_delta'] < 0 and delta['accuracy_delta'] >= .1 and delta['suc_delta'] > 0 and delta['fail_delta'] > 0
    delta['original_improvement'] = complete and delta['mae_delta'] < 0 and delta['accuracy_delta'] > 0
    return delta


def original_references(labels, ids):
    dest = DEST / 'old_original_development_reference_v1.json'
    if dest.exists():
        return json.loads(dest.read_text())
    all_groups = collections.defaultdict(lambda: collections.defaultdict(dict))
    sources = []
    conflicts = []
    for path in sorted(RESEARCH.glob('score*development_full_v1/*/score_branches.jsonl')):
        cfg = json.loads((path.parent / 'config.json').read_text())
        if cfg['model'] not in ['qwen', 'rr', 'meter']:
            continue
        case = cfg['model'] + '_' + cfg['protocol']
        used = False
        for row in read_rows(path):
            if row['example_id'] not in ids or not row['condition'].startswith('original_all_k'):
                continue
            used = True
            group = all_groups[case][row['condition']]
            old = group.get(row['example_id'])
            if old and any(old.get(k) != row.get(k) for k in ['status', 'score_logits', 'progress', 'native_prediction']):
                conflicts.append(dict(case=case, condition=row['condition'], example_id=row['example_id'], path=str(path)))
            else:
                group[row['example_id']] = row
        if used:
            sources.append(dict(path=str(path), sha256=digest(path)))
    if conflicts:
        create_json(DEST / 'old_original_conflicts_v1.json', conflicts)
        raise ValueError('Old original references conflict; inspect before selecting')
    reference = {}
    for case, groups in all_groups.items():
        valid = {}
        for name, by_id in groups.items():
            if set(by_id) == set(ids):
                summary = summarize(list(by_id.values()), labels, ids)
                if summary['complete']:
                    valid[name] = summary
        if valid:
            loss = 'continuous_ordinal_mae' if case.startswith('meter_') else 'mae'
            best = min(valid, key=lambda name: (valid[name]['overall'][loss], -valid[name]['overall']['accuracy'], int(name.rsplit('k', 1)[1])))
            reference[case] = dict(best_condition=best, summary=valid[best], all_available_summaries=valid)
    output = dict(time=timestamp(), cases=reference, sources=sources, conflicts=conflicts,
                  selection='minimum old234 development MAE, then accuracy, then smaller k; no new-method or confirmation outcomes')
    create_json(dest, output)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--references-only', action='store_true')
    parser.add_argument('--family', choices=['f7'], default='f7')
    args = parser.parse_args()
    ids = json.loads((DEST / 'development_ids_v1.json').read_text())
    labels = {r['example_id']: r for r in read_rows(BASE / 'inputs/labels.jsonl') if r['example_id'] in ids}
    references = original_references(labels, ids)
    if args.references_only:
        print('Frozen old original references:', len(references['cases']))
        return
    cases = {}
    experiment_root = DEST / {'f1': 'frame_transport_v1', 'f2': 'frame_transport_vcd_v1', 'f3': 'specificity_v1', 'f4': 'joint_task_v1', 'f5': 'task_interaction_v1', 'f7': 'all_query_reward_gradient_development_v1'}[args.family]
    target_prefix = {'f1': 'frame_transport', 'f2': 'frame_transport', 'f3': 'specificity', 'f4': 'joint_task', 'f5': 'task_interaction', 'f7': 'reward_gradient'}[args.family]
    for folder in sorted(experiment_root.iterdir()):
        path = folder / 'development/predictions.jsonl'
        completion = folder / 'development/completion_v1.json'
        if not completion.exists():
            continue
        cfg = json.loads((folder / 'development/config.json').read_text())
        expected = {(i, c['name']) for i in ids for c in cfg['conditions']}
        if digest(path) != json.loads(completion.read_text())['predictions_sha256']:
            raise ValueError('Frozen F7 development predictions changed')
        rows = list(read_rows(path))
        keys = [(r['example_id'], r['condition']) for r in rows]
        if len(keys) != len(set(keys)) or set(keys) != expected:
            raise ValueError('Duplicate/missing/extra predictions in ' + str(path))
        groups = collections.defaultdict(list)
        for row in rows:
            groups[row['condition']].append(row)
        summaries = {name: summarize(rr, labels, ids) for name, rr in groups.items()}
        base = {r['example_id']: r for r in groups['baseline']}
        zero = {r['example_id']: r for r in groups[target_prefix + '_zero_k48']}
        zero_exact = all(base[i].get('status') == zero[i].get('status') and
                         base[i].get('score_logits') == zero[i].get('score_logits') and
                         base[i].get('progress') == zero[i].get('progress') for i in ids)
        model = cfg['model']
        thresholds = ['0.125_0.875', '0.2_0.8'] if model == 'meter' else ['native']
        # Include the newly rerun original k as possible stronger incumbents.
        incumbents = {name: s for name, s in summaries.items() if name.startswith('original_all_k') and s['complete']}
        incumbents.update(references['cases'].get(folder.name, {}).get('all_available_summaries', {}))
        loss = 'continuous_ordinal_mae' if model == 'meter' else 'mae'
        best_original = min(incumbents, key=lambda name: (incumbents[name]['overall'][loss], -incumbents[name]['overall']['accuracy'], int(name.rsplit('k', 1)[1]))) if incumbents else None
        effects = {}
        for k in cfg['k_neighborhood']:
            candidate = summaries[f'{target_prefix}_k{k}']
            comparisons = {'baseline': summaries['baseline'], 'native_generation_baseline': summaries['native_generation_baseline'],
                           'original_same_k': summaries[f'original_all_k{k}']}
            if best_original:
                comparisons['best_development_original'] = incumbents[best_original]
            if args.family == 'f4' and k == 48:
                comparisons['task_only'] = summaries['task_only_k48']
            if args.family == 'f5' and k == 48:
                comparisons['task_contrast_only'] = summaries['task_contrast_only']
            if args.family == 'f7' and k == 48:
                comparisons['raw_all'] = summaries['raw_all_k48']
                comparisons['same_heads_readout_only'] = summaries['learned_readout_k48']
                comparisons['learned_positive'] = summaries['learned_positive_k48']
                f6_source=DEST/f'reward_gradient_development_v1/{folder.name}/development'
                f6_done=json.loads((f6_source/'completion_v1.json').read_text())
                if digest(f6_source/'predictions.jsonl')!=f6_done['predictions_sha256']:
                    raise ValueError('Frozen F6 scope-comparison predictions changed')
                f6_rows=[r for r in read_rows(f6_source/'predictions.jsonl') if r['condition']==f'reward_gradient_k{k}']
                comparisons['F6_readout_fitted_heads']=summarize(f6_rows,labels,ids)
            if args.family == 'f2':
                comparisons['vcd_only'] = summaries['vcd_only']
                comparisons['vcd_original_same_k'] = summaries[f'vcd_original_k{k}']
                clean_path = DEST / f'frame_transport_v1/{folder.name}/development/predictions.jsonl'
                clean_rows = [r for r in read_rows(clean_path) if r['condition'] == f'frame_transport_k{k}']
                comparisons['F1_without_VCD'] = summarize(clean_rows, labels, ids)
            by_threshold = {threshold: {name: effect(candidate, ref, model, threshold)
                             for name, ref in comparisons.items()} for threshold in thresholds}
            passing = zero_exact and all(c['baseline']['point_gate'] and c['native_generation_baseline']['point_gate'] and
                        c['original_same_k']['original_improvement'] and c.get('best_development_original', {}).get('original_improvement', False)
                        for c in by_threshold.values())
            if args.family == 'f2':
                passing = passing and all(c['vcd_only']['original_improvement'] and c['vcd_original_same_k']['original_improvement']
                                          for c in by_threshold.values())
            effects[str(k)] = dict(by_threshold=by_threshold, robust_gate=passing)
        case_pass = all(x['robust_gate'] for x in effects.values())
        cases[folder.name] = dict(model=model, protocol=cfg['protocol'], summaries=summaries,
            effects=effects, case_pass=case_pass, zero_exact=zero_exact, best_original=best_original,
            predictions_sha256=digest(path), complete_records=len(rows),
            invalid_by_condition={name: s['overall']['n_missing_or_invalid'] for name, s in summaries.items()})
        print(folder.name, 'passed', case_pass, 'zero_exact', zero_exact, 'primary',
              effects['48']['by_threshold'][thresholds[0]]['baseline'], flush=True)
    counts = {model: sum(c['case_pass'] for c in cases.values() if c['model'] == model) for model in ['qwen', 'rr', 'meter']}
    out = DEST / (args.family + '_frame_transport_analysis_' + timestamp().replace(':', '').replace('+', '_') + '.json')
    create_json(out, dict(time=timestamp(), phase='historically exposed development234; no confirmation',
        cases=cases, passing_inputs_by_model=counts, all15_complete=len(cases) == 15,
        development_gate=len(cases) == 15 and all(n >= 3 for n in counts.values()),
        final_goal_completed=False, external_confirmation_evaluated=False))
    print('analysis', out, counts)


if __name__ == '__main__':
    main()
