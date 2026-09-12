import numpy as np
import pytest
from mydata_bench.auto_research_addbase.reward_gradient_ranking import rank_reward_gradients


def test_signed_direction_and_group_weighting():
    records=[dict(example_id='a',status='ok',gradient=[[2.,-3.]]),dict(example_id='b',status='ok',gradient=[[2.,-3.]])]
    base=rank_reward_gradients(records,{'a':'episode_a','b':'episode_b'},(1,2),skip_layers=0)
    assert [(r['head'],r['signed_bias_direction']) for r in base['ranking']]==[(1,1),(0,-1)]
    extra=records+[dict(example_id='c',status='ok',gradient=[[2.,-3.]])]
    repeated=rank_reward_gradients(extra,{'a':'episode_a','b':'episode_b','c':'episode_a'},(1,2),skip_layers=0)
    assert repeated['fitting_episode_groups']==2
    assert base['ranking']==repeated['ranking']
    assert all(r['signed_bias_direction']*r['mean_loss_gradient']<0 for r in base['ranking'])


def test_missing_fitting_data_cannot_be_dropped_and_no_roi_is_zero_effect():
    rows=[dict(example_id='a',status='ok',gradient=[[2.,-4.]]),dict(example_id='b',status='no_grounding')]
    result=rank_reward_gradients(rows,{'a':'a','b':'b'},(1,2),skip_layers=0)
    assert result['no_grounding_rows']==1
    assert sorted(r['mean_loss_gradient'] for r in result['ranking'])==[-2.,1.]
    with pytest.raises(ValueError):
        rank_reward_gradients(rows[:1],{'a':'a','b':'b'},(1,2),skip_layers=0)
