import pytest
from mydata_bench.auto_research_addbase.f7_fit_finite_bias_diagnostic import group_effects


def test_episode_mean_retains_zero_fallback_and_does_not_weight_large_groups_more():
    expected={'a':'g1','b':'g1','c':'g2'}
    rows=[dict(example_id=eid,training_episode_group_sha256=expected[eid],status='no_grounding' if eid=='b' else 'ok',
        loss_deltas=dict(all_query=value,same_heads_readout=0.)) for eid,value in [('a',2.),('b',0.),('c',-5.)]]
    result=group_effects(rows,expected)
    assert result['complete'] and result['examples']==3 and result['episode_groups']==2
    assert result['effects']['all_query']['row_mean_delta']==-1.
    assert result['effects']['all_query']['episode_equal_mean_delta']==-2.
    rows[0]['status']='error'
    assert not group_effects(rows,expected)['complete']
    with pytest.raises(ValueError,match='Complete unique'):
        group_effects(rows[:-1],expected)
