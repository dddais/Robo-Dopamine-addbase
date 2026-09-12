"""Full-scope original846 F9 description and unchanged strict point gate."""
import collections
import json
import pathlib
import yaml
from .common import ROOT, create_json, read_rows, timestamp
from .metrics import summarize
from .robust_analyze import effect
from .robust_prepare import DEST, digest


def safe_effect(candidate, reference, model, threshold):
    required = ['continuous_ordinal_mae' if model == 'meter' else 'mae']
    if any(s['overall'].get(k) is None for s in [candidate, reference] for k in required):
        return dict(complete=False, point_gate=False, original_improvement=False,
                    reason='Required scoring condition has no valid ordinal outputs')
    return effect(candidate, reference, model, threshold)


def main():
    root = DEST/'row_weighted_reward_gradient_full_old_v1'
    plan_path = root/'frozen_plan_v1.json'
    plan = json.loads(plan_path.read_text())
    for field in ['input', 'labels', 'example_ids']:
        if digest(plan[field+'_file']) != plan[field+'_sha256']:
            raise ValueError('Frozen full846 analysis dependency changed: '+field)
    for path, sha in plan['source_sha256'].items():
        if digest(ROOT/path) != sha:
            raise ValueError('Frozen full846 analysis source changed')
    ids = json.loads(pathlib.Path(plan['example_ids_file']).read_text())
    if len(ids) != 846 or len(set(ids)) != 846:
        raise ValueError('Require original846 complete unique cohort')
    label_rows = [r for r in read_rows(plan['labels_file']) if r['example_id'] in set(ids)]
    labels = {r['example_id']: r for r in label_rows}
    if len(label_rows) != len(labels) or set(labels) != set(ids):
        raise ValueError('Incomplete/duplicate original846 labels')
    cases, rankings, table = {}, {}, []
    for item in plan['configurations']:
        if digest(item['path']) != item['sha256']:
            raise ValueError('Frozen full846 config changed')
        cfg = yaml.safe_load(pathlib.Path(item['path']).read_text())
        name = cfg['model']+'_'+cfg['protocol']
        out = pathlib.Path(cfg['output_dir'])/'full846'
        done = json.loads((out/'completion_v1.json').read_text())
        path = out/'predictions.jsonl'
        if digest(path) != done['predictions_sha256'] or json.loads((out/'config.json').read_text()) != cfg:
            raise ValueError('Full846 prediction/config provenance mismatch')
        rows = list(read_rows(path))
        keys = [(r['example_id'], r['condition']) for r in rows]
        expected = {(eid, c['name']) for eid in ids for c in cfg['conditions']}
        if len(keys) != len(set(keys)) or set(keys) != expected or len(keys) != done['expected_records']:
            raise ValueError('Missing/duplicate/unexpected full846 prediction')
        groups = collections.defaultdict(list)
        for row in rows:
            groups[row['condition']].append(row)
        summaries = {key: summarize(values, labels, ids) for key, values in groups.items()}
        base, zero = ({r['example_id']: r for r in groups[key]} for key in ['baseline', 'reward_gradient_zero_k48'])
        zero_exact = all(base[eid].get('status') == zero[eid].get('status') == 'ok' and
            all(base[eid].get(k) == zero[eid].get(k) for k in ['score_logits', 'progress', 'native_prediction']) for eid in ids)
        thresholds = ['0.125_0.875', '0.2_0.8'] if cfg['model'] == 'meter' else ['native']
        effects = {}
        for k in [32, 48, 64]:
            candidate = summaries[f'reward_gradient_k{k}']
            refs = {'baseline': 'baseline', 'native_generation_baseline': 'native_generation_baseline',
                    'original_same_k': f'original_all_k{k}', 'fixed_best_old_development_original': cfg['best_original_condition']}
            by_threshold = {threshold: {key: safe_effect(candidate, summaries[condition], cfg['model'], threshold)
                for key, condition in refs.items()} for threshold in thresholds}
            gate = zero_exact and all(
                value['baseline']['point_gate'] and value['native_generation_baseline']['point_gate'] and
                value['original_same_k']['original_improvement'] and value['fixed_best_old_development_original']['original_improvement']
                for value in by_threshold.values())
            effects[str(k)] = dict(by_threshold=by_threshold, robust_gate=gate)
        rank_path = pathlib.Path(cfg['ranking_file'])
        if digest(rank_path) != cfg['ranking_sha256']:
            raise ValueError('Learned head ranking changed')
        ranked = json.loads(rank_path.read_text())['ranking']
        rankings[name] = [(r['layer'], r['head']) for r in ranked]
        primary_controls = {key: {threshold: safe_effect(summaries['reward_gradient_k48'], summaries[key], cfg['model'], threshold)
            for threshold in thresholds} for key in ['raw_all_k48', 'learned_positive_k48', 'learned_readout_k48', 'reward_gradient_low_k48']}
        wrong_rows={r['example_id']:r for r in groups['reward_gradient_wrong_k48']}
        feasible_ids=[eid for eid in ids if wrong_rows[eid].get('status')=='ok']
        geometry=sum(r.get('control_geometry_infeasible',False) for r in wrong_rows.values())
        other=sum(r.get('status')!='ok' and not r.get('control_geometry_infeasible',False) for r in wrong_rows.values())
        wrong_feasibility=dict(full_primary_denominator=846,feasible_control_examples=len(feasible_ids),
            geometry_infeasible=geometry,other_control_errors=other,candidate_errors_not_used_for_selection=True,
            interpretation='conditional wrong-control comparison; primary denominator unchanged; unexpected errors prevent geometry-only attribution')
        if feasible_ids:
            candidate_subset=summarize([r for r in groups['reward_gradient_k48'] if r['example_id'] in set(feasible_ids)],labels,feasible_ids)
            control_subset=summarize([wrong_rows[eid] for eid in feasible_ids],labels,feasible_ids)
            wrong_feasibility['candidate_summary']=candidate_subset
            wrong_feasibility['control_summary']=control_subset
            if all(candidate_subset['by_split'][split]['n_expected'] for split in ['suc','fail']):
                wrong_feasibility['effects']={threshold:safe_effect(candidate_subset,control_subset,cfg['model'],threshold) for threshold in thresholds}
        for summary in summaries.values():
            tasks = list(summary['by_task'].values())
            summary['task_macro'] = {metric: sum(t[metric] for t in tasks)/len(tasks)
                for metric in ['accuracy', 'mae', 'continuous_ordinal_mae'] if tasks and all(t.get(metric) is not None for t in tasks)}
        passed = all(e['robust_gate'] for e in effects.values())
        cases[name] = dict(model=cfg['model'], protocol=cfg['protocol'], full846_point_gate=passed,
            zero_exact=zero_exact, effects=effects, summaries=summaries, primary_controls=primary_controls, wrong_control_feasibility=wrong_feasibility,
            predictions_sha256=digest(path), samples=846, errors=done['errors'], cached_old234_records=done['cached_rows'],
            best_original_condition=cfg['best_original_condition'], top8_heads=ranked[:8],
            signed_direction_counts={str(k): dict(collections.Counter(str(r['signed_bias_direction']) for r in ranked[:k])) for k in [8, 32, 48, 64]})
        primary = effects['48']['by_threshold'][thresholds[0]]['baseline']
        table.append((name, passed, primary))
        print(name, 'full846 point gate', passed, 'primary vs baseline', primary, flush=True)
    overlaps = {}
    for a, aa in rankings.items():
        for b, bb in rankings.items():
            if a >= b:
                continue
            overlaps[a+'__'+b] = {str(k): dict(intersection=len(set(aa[:k]) & set(bb[:k])),
                fraction=len(set(aa[:k]) & set(bb[:k]))/k) for k in [8, 32, 64]}
    counts = {model: sum(row['full846_point_gate'] for row in cases.values() if row['model'] == model)
              for model in ['qwen', 'rr', 'meter']}
    path = root/'analysis_v1.json'
    create_json(path, dict(time=timestamp(), cases=cases, passing_inputs_by_model=counts,
        full846_original_scope_gate=all(n >= 3 for n in counts.values()),
        frozen_plan_sha256=digest(plan_path), selected_cases=len(cases), all_selected_cases_reported=True,
        head_index_overlap=overlaps, head_overlap_interpretation='numerical layer/head index overlap only, not functional equivalence across backbones',
        statistics_descriptive_only=True, no_independent_p_values_for_reused846=True,
        native_MAE_and_continuous_ordinal_MAE_reported=True,
        incomplete_condition_MAE_valid_only_never_eligible=True,
        final_reserved_test_confirmation_still_required=True, final_goal_completed=False))
    report = ['# F9完整原始846条描述性结果', '', '**这是已暴露数据上的完整验收，不能当作独立确认。**', '',
              '|模型/输入|完整三k门槛|primary MAE变化|总准确率变化pp|成功变化pp|失败变化pp|',
              '|---|---|---:|---:|---:|---:|']
    for name, passed, p in table:
        if 'mae_delta' in p:
            report.append(f"|{name}|{passed}|{p['mae_delta']:.6f}|{100*p['accuracy_delta']:.4f}|{100*p['suc_delta']:.4f}|{100*p['fail_delta']:.4f}|")
        else:
            report.append(f'|{name}|False|无有效评分|—|—|—|')
    report += ['', '两套Meter阈值、全部k/对照、task/split预测分布、同视频suc−fail配对档位、top8及top8/32/64的head索引重合度见analysis_v1.json。',
               '', '通过输入数：'+json.dumps(counts, ensure_ascii=False)+'。保留外部测试仍待单独冻结和一次性确认，目标未完成。']
    with (root/'REPORT_v1.md').open('x') as f:
        f.write('\n'.join(report)+'\n')
    print('analysis', path, counts, flush=True)


if __name__ == '__main__':
    main()
