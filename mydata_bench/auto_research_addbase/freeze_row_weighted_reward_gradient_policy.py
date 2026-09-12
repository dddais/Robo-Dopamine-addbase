"""Freeze F9 evaluation before any F9 outcomes; preserve all F8 acceptance gates."""
import copy
import json
from .common import ROOT, create_json, timestamp
from .robust_prepare import DEST, digest


def main():
    previous_path = DEST/'f8_evaluation_policy_v1.json'
    previous = json.loads(previous_path.read_text())
    fit_path = DEST/'row_weighted_reward_gradient_fit_frozen_plan_v1.json'
    fit = json.loads(fit_path.read_text())
    if json.loads((DEST/'f9_fit_completion_v1.json').read_text())['failures']:
        raise ValueError('All fifteen F9 fits must complete successfully')
    sources = dict(previous['evaluation_source_sha256'])
    sources.update(fit['source_sha256'])
    names = ['freeze_row_weighted_reward_gradient_policy.py', 'f9_fit_provenance.py',
        'f9_reference_comparison.py', 'freeze_row_weighted_reward_gradient_development.py',
        'freeze_row_weighted_reward_gradient_validation.py',
        'analyze_row_weighted_reward_gradient_development.py',
        'analyze_row_weighted_reward_gradient_validation.py',
        'row_weighted_reward_gradient_evaluation_queue.py',
        'freeze_row_weighted_reward_gradient_full_old.py',
        'analyze_row_weighted_reward_gradient_full_old.py',
        'row_weighted_reward_gradient_full_old_queue.py']
    for name in names:
        path = ROOT/'mydata_bench/auto_research_addbase'/name
        sources[str(path.relative_to(ROOT))] = digest(path)
    for path, expected in sources.items():
        if digest(ROOT/path) != expected:
            raise ValueError('Frozen dependency changed before F9 evaluation: '+path)
    if (DEST/'external_roboreward_reserved_v1/reward_model_confirmation_claim_v1.json').exists():
        raise ValueError('Reserved test already claimed; cannot claim an unopened-test policy')
    comparisons = {}
    for stage in ['development', 'validation']:
        path = DEST/f'ordinal_reward_gradient_{stage}_frozen_plan_v1.json'
        plan = json.loads(path.read_text())
        if len(plan['configurations']) != 15:
            raise ValueError('All15 prospectively fixed F8 comparisons required')
        comparisons[stage] = {}
        for item in plan['configurations']:
            if digest(item['path']) != item['sha256']:
                raise ValueError('Fixed F8 comparator config changed')
            comparisons[stage][item['model']+'_'+item['protocol']] = item
    snapshots = DEST/'f9_evaluation_source_snapshots_v1'
    snapshots.mkdir(exist_ok=False)
    for name, sha in sources.items():
        path = snapshots/(sha+'.py')
        if not path.exists():
            with path.open('xb') as handle:
                handle.write((ROOT/name).read_bytes())
    policy = copy.deepcopy(previous)
    for key in ['all15_fit_rankings_frozen_together_before_any_F8_evaluation',
                'changes_from_F7', 'F8_reserved_confirmation_requires_new_prospective_policy_and_available_unopened_test',
                'F7_test_opening_or_failure_cannot_be_reused_as_fresh_F8_confirmation']:
        policy.pop(key, None)
    policy.update(time=timestamp(), family='F9 sample-mean native ordinal gradients with episode-cluster variability',
        evaluation_source_sha256=sources, source_snapshots_directory=str(snapshots),
        inherited_F8_policy_file=str(previous_path), inherited_F8_policy_sha256=digest(previous_path),
        fitting_plan_file=str(fit_path), fitting_plan_sha256=digest(fit_path),
        all15_fit_rankings_frozen_together_before_any_F9_evaluation=True,
        changes_from_F8=['full-sample gradient mean', 'episode-cluster sandwich standard error'],
        fitting_estimand='all1942 rows equally weighted; original F8 per-row ordinal loss unchanged',
        gradient_sources='original audited F8 files referenced by source_gradients_file and source_gradients_sha256',
        fixed_F8_comparator_configs=comparisons,
        fixed_F8_comparison_role='descriptive primary-k48 comparison; same per-example loss and inference, different aggregation',
        fixed_F8_results_must_be_complete_before_F9_analysis=True,
        fixed_F7_results_must_be_complete_before_F9_analysis=True,
        old234_and_validation_gates_unchanged_from_F8=True,
        evaluation_and_full846_gpus=[0, 3],
        evaluation_gpu_prerequisite='F7 all15 evaluation inference complete; memory checks before every stage',
        reserved_test_priority='F7 and F8 retain priority; both must finish scientifically ineligible with absent shared claim',
        F9_reserved_confirmation_requires_new_prospective_policy_and_available_unopened_test=True,
        any_earlier_family_test_claim_prevents_fresh_F9_confirmation=True,
        this_candidate_validation_performance_not_observed_before_freeze=True,
        final_goal_completed=False)
    policy['controls'] = previous['controls']+['fixed F8 episode-weighted ordinal primary as descriptive comparator']
    for key in ['validation_point_gate', 'old234_gate', 'joint_gate', 'bootstrap_draws', 'bootstrap_seed',
                'original_best_development_conditions', 'shared_bias_magnitude', 'k_neighborhood', 'primary_k',
                'full846_selection', 'full846_required_original_samples', 'full846_point_gate']:
        if policy[key] != previous[key]:
            raise ValueError('F9 changed acceptance criteria: '+key)
    path = DEST/'f9_evaluation_policy_v1.json'
    create_json(path, policy)
    print('Frozen F9 evaluation/full846 policy:', len(sources), 'sources, SHA', digest(path), flush=True)


if __name__ == '__main__':
    main()
