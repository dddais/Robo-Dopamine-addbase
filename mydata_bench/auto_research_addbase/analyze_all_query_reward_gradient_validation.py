"""Join only training-validation labels, after all frozen F7 cases finish."""
import collections
import json
import pathlib
import yaml
from .common import ROOT, create_json, read_rows, timestamp
from .robust_prepare import DEST, digest
from .reward_validation_metrics import observations, summarize_observations, effect, paired_cluster_intervals, holm_adjust
from .reward_control_feasibility import feasible_wrong_control


def main():
    plan_path = DEST/'all_query_reward_gradient_validation_frozen_plan_v1.json'
    plan = json.loads(plan_path.read_text())
    for field in ['input', 'example_ids', 'analysis_labels', 'policy']:
        if digest(plan[field+'_file']) != plan[field+'_sha256']:
            raise ValueError('Frozen validation dependency changed: '+field)
    for path, sha in plan['inference_source_sha256'].items():
        if digest(ROOT/path) != sha:
            raise ValueError('Frozen evaluation source changed')
    policy = json.loads(pathlib.Path(plan['policy_file']).read_text())
    ids = json.loads(pathlib.Path(plan['example_ids_file']).read_text())
    configurations = plan['configurations']
    if len(configurations) != 15:
        raise ValueError('All fifteen validation cases required before labels are joined')
    loaded = []
    for item in configurations:
        if digest(item['path']) != item['sha256']:
            raise ValueError('Frozen validation configuration changed')
        cfg = yaml.safe_load(pathlib.Path(item['path']).read_text())
        out = pathlib.Path(cfg['output_dir'])/'validation'
        done = json.loads((out/'completion_v1.json').read_text())
        path = out/'predictions.jsonl'
        if digest(path) != done['predictions_sha256'] or json.loads((out/'config.json').read_text()) != cfg:
            raise ValueError('Prediction/config provenance mismatch')
        rows = list(read_rows(path))
        keys = [(r['example_id'], r['condition']) for r in rows]
        expected = {(eid, c['name']) for eid in ids for c in cfg['conditions']}
        if len(keys) != len(set(keys)) or set(keys) != expected or len(keys) != done['expected_records']:
            raise ValueError('Missing, duplicate or unexpected validation prediction')
        loaded.append((item, cfg, done, rows))
    # F7 joins these reused model-selection labels only after all15 predictions.
    # F6 already analyzed this partition; it is not fresh confirmation.
    # Neither inference nor fitting opens this file; reserved test stays sealed.
    label_rows = list(read_rows(plan['analysis_labels_file']))
    labels = {r['example_id']: r for r in label_rows}
    if len(labels) != len(label_rows) or set(labels) != set(ids):
        raise ValueError('Validation labels are not exactly the frozen cohort')
    if any(r.get('training_partition') != 'validation' for r in label_rows):
        raise ValueError('Expected training-validation labels only')
    samples = [r for r in read_rows(plan['input_file']) if r['example_id'] in set(ids)]
    cases, pvalues = {}, {}
    for item, cfg, done, rows in loaded:
        name = item['model']+'_'+item['protocol']
        grouped = collections.defaultdict(list)
        for row in rows:
            grouped[row['condition']].append(row)
        obs = {key: observations(rr, labels, samples, cfg['model']) for key, rr in grouped.items()}
        summaries = {key: summarize_observations(rr) for key, rr in obs.items()}
        base = {r['example_id']: r for r in grouped['baseline']}
        zero = {r['example_id']: r for r in grouped['reward_gradient_zero_k48']}
        zero_exact = all(base[eid].get('status') == zero[eid].get('status') == 'ok' and
            all(base[eid].get(k) == zero[eid].get(k) for k in ['score_logits', 'progress', 'native_prediction']) for eid in ids)
        effects, uncertainty = {}, {}
        primary_tests = []
        for k in [32, 48, 64]:
            candidate = obs[f'reward_gradient_k{k}']
            refs = {'baseline': 'baseline', 'native_generation_baseline': 'native_generation_baseline',
                    'original_same_k': f'original_all_k{k}', 'fixed_best_old_development_original': cfg['best_original_condition']}
            effects[str(k)] = {key: effect(candidate, obs[condition]) for key, condition in refs.items()}
            uncertainty[str(k)] = {key: paired_cluster_intervals(candidate, obs[condition],
                draws=policy['bootstrap_draws'], seed=policy['bootstrap_seed']) for key, condition in refs.items()}
            if k == 48:
                for result in uncertainty[str(k)].values():
                    primary_tests += [result['metrics'][metric]['centered_bootstrap_one_sided_p_zero_improvement']
                                      for metric in ['mae_delta', 'accuracy_delta']]
                for control in ['raw_all_k48', 'learned_positive_k48', 'learned_readout_k48', 'reward_gradient_low_k48']:
                    effects['primary_control_'+control] = effect(candidate, obs[control])
        point_gate = zero_exact and all(
            all(e['all_grade_improvement'] for e in effects[str(k)].values()) and
            all(effects[str(k)][reference]['endpoint_non_degradation'] for reference in ['baseline', 'native_generation_baseline'])
            for k in [32, 48, 64])
        if not all(e['complete'] for e in effects['48'].values()):
            primary_tests.append(1.)
        pvalues[name] = max(primary_tests)
        wrong_control=feasible_wrong_control(obs['reward_gradient_k48'],obs['reward_gradient_wrong_k48'],grouped['reward_gradient_wrong_k48'])
        f6_source=DEST/f'reward_gradient_validation_v1/{name}/validation'
        f6_done=json.loads((f6_source/'completion_v1.json').read_text())
        if digest(f6_source/'predictions.jsonl')!=f6_done['predictions_sha256']:
            raise ValueError('Frozen F6 validation comparison changed')
        f6_cfg=json.loads((f6_source/'config.json').read_text())
        for field in ['model','protocol','input_sha256','example_ids_sha256']:
            if f6_cfg[field]!=cfg[field]:
                raise ValueError('F6/F7 selection data mismatch')
        f6_rows=[r for r in read_rows(f6_source/'predictions.jsonl') if r['condition']=='reward_gradient_k48']
        effects['primary_F6_readout_fitted_heads']=effect(obs['reward_gradient_k48'],observations(f6_rows,labels,samples,cfg['model']))
        cases[name] = dict(model=cfg['model'], protocol=cfg['protocol'], validation_point_gate=point_gate,
            zero_exact=zero_exact, summaries=summaries, effects=effects, paired_group_uncertainty=uncertainty, wrong_control_feasibility=wrong_control,
            primary_mae_accuracy_IUT_p=pvalues[name], predictions_sha256=done['predictions_sha256'],
            errors=done['errors'], fixed_best_old_development_original=cfg['best_original_condition'])
        print(name, 'validation_gate', point_gate, 'k48 vs baseline', effects['48']['baseline'], flush=True)
    adjusted = holm_adjust(pvalues)
    for name, value in adjusted.items():
        cases[name]['primary_mae_accuracy_IUT_Holm_p'] = value
    counts = {model: sum(c['validation_point_gate'] for c in cases.values() if c['model'] == model)
              for model in ['qwen', 'rr', 'meter']}
    old_paths = sorted(DEST.glob('f7_frame_transport_analysis_*.json'))
    if not old_paths:
        raise ValueError('Full old234 F7 descriptive analysis is required for joint case eligibility')
    old_path = old_paths[-1]
    old = json.loads(old_path.read_text())
    if not old['all15_complete'] or set(old['cases']) != set(cases):
        raise ValueError('Incomplete old234 scope; cannot substitute external validation')
    for name, case in cases.items():
        raw = DEST/f'all_query_reward_gradient_development_v1/{name}/development/predictions.jsonl'
        if digest(raw) != old['cases'][name]['predictions_sha256']:
            raise ValueError('Old234 descriptive prediction provenance mismatch')
        case['old234_and_validation_joint_point_gate'] = case['validation_point_gate'] and old['cases'][name]['case_pass']
    joint_counts = {model: sum(c['old234_and_validation_joint_point_gate'] for c in cases.values() if c['model'] == model)
                    for model in ['qwen', 'rr', 'meter']}
    path = DEST/('f7_training_validation_analysis_'+timestamp().replace(':', '').replace('+', '_')+'.json')
    create_json(path, dict(time=timestamp(), cases=cases, passing_inputs_by_model=counts,
        all_cases_complete=True, validation_gate=all(n >= 3 for n in counts.values()),
        same_input_joint_passing_counts=joint_counts, old234_and_validation_joint_gate=all(n >= 3 for n in joint_counts.values()),
        old234_analysis_file=str(old_path), old234_analysis_sha256=digest(old_path),
        validation_point_gate_does_not_require_10pp_external_mixed_grade_gain=True,
        original_old_dataset_10pp_goal_still_required=True, marginal_CIs_not_simultaneous=True,
        primary_Holm_family='15 fixed model/protocol cases; each IUT uses MAE and all-grade accuracy vs two baselines, same-k and fixed old-best original',
        p_values_approximate_centered_cluster_bootstrap=True, p_values_do_not_test_10pp_lower_bound=True,
        validation_can_select_candidates_and_is_not_final_confirmation=True,validation_previously_used_for_F6_selection=True,
        public_train_may_have_been_used_by_backbone=True, reserved_test_evaluated=False, final_goal_completed=False,
        frozen_plan_sha256=digest(plan_path)))
    print('analysis', path, counts, flush=True)


if __name__ == '__main__':
    main()
