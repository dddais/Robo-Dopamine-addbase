import pytest

from mydata_bench.meter_eval.success_metrics import index_rows, metrics, summarize


def labels(rewards):
    return {str(i):dict(reward=y,split='suc' if y==5 else 'fail',subset='task',
                        video_sha256='video',source_suc_id='0') for i,y in enumerate(rewards)}


def test_official_threshold_ties_and_missing_do_not_become_correct_failures():
    truth=labels([5,1,5,1])
    result=metrics({'0':.5,'1':.5,'2':.9},truth,truth)
    assert (result['tp'],result['tn'],result['fp'],result['fn'])==(1,1,0,1)
    assert result['accuracy']==.5
    assert result['suc_accuracy']==result['fail_accuracy']==.5
    assert result['n_invalid_fail']==1
    assert result['n_equal_0_5_suc']==result['n_equal_0_5_fail']==1


def test_endpoint_boundaries_and_auc_ties():
    truth=labels([5,1,5,1])
    result=metrics({'0':.875,'1':.125,'2':.8,'3':.2},truth,truth)
    assert result['endpoint_0.125_0.875']['accuracy']==.25
    assert result['endpoint_0.2_0.8']['accuracy']==.75
    tied=metrics({i:.5 for i in truth},truth,truth)
    assert tied['auroc']==.5
    assert tied['brier']==.25
    assert tied['score_mae']==.5


def test_terminal_head_and_same_video_pair_audit():
    truth=labels([5,1]);rows=[dict(example_id=i,status='ok',success_probability=q,
                                  success_trajectory=[.01,q]) for i,q in [('0',.9),('1',.1)]]
    report=summarize(rows,truth,truth,'success')
    assert report['pairwise']['positive']==1
    assert report['pairwise']['mean_score_gap']==pytest.approx(.8)
    truth['1']['video_sha256']='different_video'
    with pytest.raises(ValueError,match='same-video'):summarize(rows,truth,truth,'success')
    truth['1']['video_sha256']='video';rows[0]['success_trajectory']=[.1]
    with pytest.raises(ValueError,match='terminal'):summarize(rows,truth,truth,'success')


def test_duplicates_and_out_of_range_success_scores_are_rejected():
    row=dict(example_id='0',status='ok',success_probability=1.1,success_trajectory=[1.1])
    with pytest.raises(ValueError,match='Duplicate'):index_rows([row,row])
    truth=labels([5])
    with pytest.raises(ValueError,match='outside'):summarize([row],truth,truth,'success')
