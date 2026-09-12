"""Freeze F9 final-test rules before candidate validation; retain F7/F8 test priority."""
import copy
import json
from .common import ROOT, create_json, timestamp
from .robust_prepare import DEST, digest


def main():
    previous_path=DEST/'f8_confirmation_policy_v1.json'
    previous=json.loads(previous_path.read_text())
    evaluation_path=DEST/'f9_evaluation_policy_v1.json'
    evaluation=json.loads(evaluation_path.read_text())
    claim=DEST/'external_roboreward_reserved_v1/reward_model_confirmation_claim_v1.json'
    if claim.exists():
        raise ValueError('Shared test already claimed; cannot preregister it as unexposed F9 confirmation')
    sources=dict(previous['source_sha256'])
    sources.update(evaluation['evaluation_source_sha256'])
    names=['f9_confirmation_priority.py','freeze_row_weighted_reward_gradient_confirmation_policy.py',
        'freeze_row_weighted_reward_gradient_confirmation.py','analyze_row_weighted_reward_gradient_confirmation.py',
        'row_weighted_reward_gradient_confirmation_queue.py']
    for name in names:
        path=ROOT/'mydata_bench/auto_research_addbase'/name
        sources[str(path.relative_to(ROOT))]=digest(path)
    for path,expected in sources.items():
        if digest(ROOT/path)!=expected:
            raise ValueError('Frozen F9 final dependency changed: '+path)
    for field in ['reserved_input','reserved_groups','reserved_labels']:
        if digest(previous[field+'_file'])!=previous[field+'_sha256']:
            raise ValueError('Frozen reserved dependency changed: '+field)
    snapshots=DEST/'f9_confirmation_source_snapshots_v1'
    snapshots.mkdir(exist_ok=False)
    for name,sha in sources.items():
        path=snapshots/(sha+'.py')
        if not path.exists():
            with path.open('xb') as handle:
                handle.write((ROOT/name).read_bytes())
    policy=copy.deepcopy(previous)
    policy.update(time=timestamp(),family='F9 sample-mean ordinal gradients with episode-cluster variability',
        source_sha256=sources,source_snapshots_directory=str(snapshots),
        earlier_evaluation_policy_file=str(evaluation_path),earlier_evaluation_policy_sha256=digest(evaluation_path),
        inherited_F8_confirmation_policy_file=str(previous_path),inherited_F8_confirmation_policy_sha256=digest(previous_path),
        prerequisite='all15 F9 old234+493 complete, same >=3 inputs for each of3 models jointly eligible; '
            'all eligible inputs complete original846 and >=3/model pass original gate; '
            'F7 and F8 completed scientific ineligibility without claiming shared test',
        head_rankings='each selected case uses exactly its earlier F9 sample-mean ordinal-loss ranking/signs; no refit',
        supervised_fitting_loss='native cumulative ordinal log loss; original native targets and inference unchanged',
        F7_priority='Only completed F7 and F8 scientific ineligibility and absent once-only claim releases this test; '
            'a crash/stopped prerequisite does not release it',
        no_F9_confirmation_if_any_prior_family_claimed_test=True,
        no_test_performance_based_method_replacement=True,
        final_confirmation_gpus=[0,3],
        F9_validation_performance_not_observed_before_final_policy_freeze=True,
        test_reward_performance_unopened_at_policy_freeze=True,
        final_goal_completed=False)
    for key in ['original_dataset_gate','selected_cases','selected_case_count','shared_bias_magnitude',
                'k_neighborhood','primary_k','final_point_gate','final_scientific_evidence_gate',
                'bootstrap_draws','bootstrap_seed','primary_IUT','Holm_family','controls','wrong_control_analysis']:
        if policy[key]!=previous[key]:
            raise ValueError('F9 final gate or controls changed: '+key)
    path=DEST/'f9_confirmation_policy_v1.json'
    create_json(path,policy)
    print('Frozen F9 final policy',len(sources),'sources, SHA',digest(path),flush=True)


if __name__=='__main__':
    main()
