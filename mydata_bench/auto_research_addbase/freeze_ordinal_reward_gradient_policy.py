"""Prospectively freeze F8 old234/493 and conditional full846; no reserved test."""
import copy
import json
import pathlib
from .common import ROOT, create_json, timestamp
from .robust_prepare import DEST, digest


def main():
    previous_path=DEST/'f7_evaluation_policy_v1.json'
    previous=json.loads(previous_path.read_text())
    policy=copy.deepcopy(previous)
    sources=dict(previous['evaluation_source_sha256'])
    fit_path=DEST/'ordinal_reward_gradient_fit_frozen_plan_v1.json'
    fit=json.loads(fit_path.read_text())
    sources.update(fit['source_sha256'])
    names=['freeze_ordinal_reward_gradient_policy.py', 'f8_reference_comparison.py',
           'freeze_ordinal_reward_gradient_development.py', 'freeze_ordinal_reward_gradient_validation.py',
           'analyze_ordinal_reward_gradient_development.py', 'analyze_ordinal_reward_gradient_validation.py',
           'ordinal_reward_gradient_evaluation_queue.py', 'all_query_reward_gradient_full_old.py',
           'analyze_all_query_reward_gradient_full_old.py',
           'freeze_ordinal_reward_gradient_full_old.py', 'analyze_ordinal_reward_gradient_full_old.py',
           'ordinal_reward_gradient_full_old_queue.py']
    for name in names:
        path=ROOT/'mydata_bench/auto_research_addbase'/name
        sources[str(path.relative_to(ROOT))]=digest(path)
    for path,expected in sources.items():
        if digest(ROOT/path)!=expected:
            raise ValueError('Frozen dependency changed before F8 evaluation policy: '+path)
    if (DEST/'external_roboreward_reserved_v1/reward_model_confirmation_claim_v1.json').exists():
        raise ValueError('Reserved claim opened: revise prospective evidence wording without reusing a test')
    comparisons={}
    for stage in ['development','validation']:
        path=DEST/f'all_query_reward_gradient_{stage}_frozen_plan_v1.json'
        plan=json.loads(path.read_text())
        if len(plan['configurations'])!=15:
            raise ValueError('Fixed F7 comparison requires all15 frozen configurations')
        comparisons[stage]={}
        for item in plan['configurations']:
            if digest(item['path'])!=item['sha256']:
                raise ValueError('Fixed F7 comparison config changed')
            comparisons[stage][item['model']+'_'+item['protocol']]=item
    snapshots=DEST/'f8_evaluation_source_snapshots_v1'
    snapshots.mkdir(exist_ok=False)
    for name,sha in sources.items():
        path=snapshots/(sha+'.py')
        if not path.exists():
            with path.open('xb') as handle:
                handle.write((ROOT/name).read_bytes())
    policy.pop('all15_fit_rankings_frozen_together_before_any_F7_evaluation',None)
    policy.pop('final_old846_and_reserved_test_inference_not_scheduled_here',None)
    policy.update(time=timestamp(),family='F8 cumulative ordinal log-loss fitted signed all-causal-query ROI heads',
        evaluation_source_sha256=sources,source_snapshots_directory=str(snapshots),
        inherited_F7_policy_file=str(previous_path),inherited_F7_policy_sha256=digest(previous_path),
        fitting_plan_file=str(fit_path),fitting_plan_sha256=digest(fit_path),
        all15_fit_rankings_frozen_together_before_any_F8_evaluation=True,
        changes_from_F7=['fitting objective only'],fitting_loss='mean cumulative Bernoulli log loss over every native bin boundary',
        all_native_grade_targets_unchanged=True,evaluation_endpoint_thresholds_used_in_loss=False,
        fixed_F7_comparator_configs=comparisons,
        fixed_F7_comparison_role='descriptive primary-k48 comparison of CE versus cumulative ordinal fitted heads; '
            'same backbone/cohort/kernel/readout/b1, no change to acceptance gates',
        fixed_F7_results_must_be_complete_before_F8_analysis=True,
        old234_and_validation_gates_unchanged_from_F7=True,
        full846_selection='every same-case jointly eligible input, at least3 per model for all3 models',
        full846_required_original_samples=846,
        full846_point_gate='all k32/48/64, both Meter thresholds, vs both baselines MAE down/suc up/fail up/total>=+10pp; '
            'MAE and accuracy improve over same-k and fixed best original',
        final_old846_conditionally_scheduled_but_reserved_test_not_scheduled=True,
        evaluation_and_full846_gpus=[1,2],
        reserved_test_priority='F7 retains its frozen conditional queue; F8 must not race its once-only claim',
        F8_reserved_confirmation_requires_new_prospective_policy_and_available_unopened_test=True,
        F7_test_opening_or_failure_cannot_be_reused_as_fresh_F8_confirmation=True,
        this_candidate_validation_performance_not_observed_before_freeze=True,
        final_goal_completed=False)
    policy['controls']=previous['controls']+['fixed F7 CE-fitted all-query primary as descriptive comparator']
    for key in ['validation_point_gate','old234_gate','joint_gate','bootstrap_draws','bootstrap_seed',
                'original_best_development_conditions','shared_bias_magnitude','k_neighborhood','primary_k']:
        if policy[key]!=previous[key]:
            raise ValueError('F8 silently changed frozen acceptance criteria: '+key)
    path=DEST/'f8_evaluation_policy_v1.json'
    create_json(path,policy)
    print('Frozen F8 evaluation/full846 policy:',len(sources),'sources, SHA',digest(path),flush=True)


if __name__=='__main__':
    main()
