import pytest
from mydata_bench.auto_research_addbase.reward_validation_metrics import observations, aggregate, paired_cluster_intervals, holm_adjust


def make_case(grades, predictions, model='rr', group_size=1):
    labels = {str(i): {'reward': g} for i, g in enumerate(grades)}
    samples = [dict(example_id=str(i), training_episode_group_sha256=str(i//group_size), task='place',
        source_subset='source', grounding_status='ok') for i in range(len(grades))]
    rows = [dict(example_id=str(i), status='error' if p is None else 'ok',
        **({'progress': p} if model == 'meter' else {'native_prediction': p})) for i, p in enumerate(predictions)]
    return observations(rows, labels, samples, model)


def test_middle_grades_and_failed_outputs_keep_full_denominator():
    obs = make_case([1, 2, 3, 4, 5], [1, 2, 3, None, 5])
    out = aggregate(obs)
    assert out['all_grade_accuracy'] == .8
    assert out['ordinal_mae_full_denominator'] == .8
    assert out['ordinal_mae_valid_only'] == 0
    assert out['n_invalid'] == 1 and not out['complete']
    assert out['endpoint_accuracy_0.125_0.875']['suc']['accuracy'] == 1
    assert out['endpoint_accuracy_0.125_0.875']['fail']['accuracy'] == 1


def test_meter_uses_standard_five_bins_and_separate_endpoint_thresholds():
    out = aggregate(make_case([1, 2, 3, 4, 5], [.125, .25, .5, .75, .8], 'meter'))
    assert out['all_grade_accuracy'] == .6
    assert out['ordinal_mae_full_denominator'] == pytest.approx(.26)
    assert out['five_bin_mae_full_denominator'] == .4
    assert out['endpoint_accuracy_0.125_0.875']['suc']['accuracy'] == 0
    assert out['endpoint_accuracy_0.125_0.875']['fail']['accuracy'] == 0
    assert out['endpoint_accuracy_0.2_0.8']['suc']['accuracy'] == 1
    assert out['endpoint_accuracy_0.2_0.8']['fail']['accuracy'] == 1


def test_whole_group_duplication_does_not_fake_more_independent_evidence():
    a = make_case([1, 5, 1, 5], [1, 5, 2, 4])
    b = make_case([1, 5, 1, 5], [2, 4, 1, 5])
    first = paired_cluster_intervals(a, b, draws=2000)
    def duplicate(rows):
        return [dict(row, example_id=row['example_id']+'_'+str(j)) for row in rows for j in range(5)]
    second = paired_cluster_intervals(duplicate(a), duplicate(b), draws=2000)
    assert first == second
    assert first['groups'] == 4
    assert first['metrics']['accuracy_delta']['percentile95'][0] < 0 < first['metrics']['accuracy_delta']['percentile95'][1]
    assert holm_adjust({'a': .01, 'b': .03, 'c': .2}) == {'a': .03, 'b': .06, 'c': .2}
