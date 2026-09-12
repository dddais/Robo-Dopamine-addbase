from mydata_bench.auto_research_addbase.reward_validation_metrics import observations
from mydata_bench.auto_research_addbase.reward_control_feasibility import feasible_wrong_control


def test_control_feasibility_does_not_drop_primary_failures_or_change_main_denominator():
    samples = [dict(example_id=str(i), training_episode_group_sha256=str(i), task='place', source_subset='source',
        grounding_status='baseline_fallback' if i == 2 else 'ok') for i in range(4)]
    labels = {str(i): dict(reward=g) for i, g in enumerate([1, 5, 3, 1])}
    def rows(preds):
        return [dict(example_id=str(i), status='error' if p is None else 'ok', native_prediction=p)
                for i, p in enumerate(preds)]
    primary = observations(rows([None, 5, 3, 1]), labels, samples, 'rr')
    raw_control = rows([2, None, 3, 1])
    raw_control[1]['control_geometry_infeasible'] = True
    control = observations(raw_control, labels, samples, 'rr')
    result = feasible_wrong_control(primary, control, raw_control)
    assert result['full_primary_denominator'] == 4 and result['feasible_control_examples'] == 2
    assert result['exclusions'] == dict(no_grounding=1, declared_geometry_infeasible=1, other_control_errors=0)
    assert not result['effect_on_same_feasible_rows']['complete']
    assert result['effect_on_same_feasible_rows']['mae_delta'] == 1.5
    assert len(primary) == len(control) == 4
    del raw_control[1]['control_geometry_infeasible']
    unexplained = feasible_wrong_control(primary, control, raw_control)
    assert unexplained['exclusions']['other_control_errors'] == 1
    assert unexplained['other_control_runtime_errors_prevent_clean_geometry_only_interpretation']
