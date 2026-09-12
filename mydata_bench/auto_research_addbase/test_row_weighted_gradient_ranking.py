import copy
import numpy as np
import pytest
from mydata_bench.auto_research_addbase.row_weighted_gradient_ranking import rank_row_weighted_gradients
from mydata_bench.auto_research_addbase.reward_gradient_ranking import rank_reward_gradients


def record(eid,value,status='ok'):
    gradient=np.zeros((9,2));gradient[8,0]=value;gradient[8,1]=value/2
    return dict(example_id=eid,status=status,gradient=gradient.tolist() if status=='ok' else None)


def test_equal_cluster_sizes_reduce_to_existing_mean_and_se():
    rows=[record('a',1),record('b',3),record('c',2),record('d',0,'no_grounding')]
    groups=dict(a='g1',b='g1',c='g2',d='g2')
    old=rank_reward_gradients(rows,groups,(9,2));new=rank_row_weighted_gradients(rows,groups,(9,2))
    for a,b in zip(old['ranking'],new['ranking']):
        assert (a['layer'],a['head'],a['signed_bias_direction'])==(b['layer'],b['head'],b['signed_bias_direction'])
        for key in ['mean_loss_gradient','standard_error','score']:
            assert a[key]==pytest.approx(b[key],abs=1e-14)
    assert new['no_grounding_rows']==1 and new['fitting_examples']==4


def test_unequal_cluster_target_and_no_false_precision_from_replication():
    rows=[record(str(i),1.) for i in range(4)]+[record('fallback',0,'no_grounding')]
    groups={str(i):'g1' for i in range(4)};groups['fallback']='g2'
    result=rank_row_weighted_gradients(rows,groups,(9,2));top=result['ranking'][0]
    assert top['mean_loss_gradient']==pytest.approx(.8)
    assert top['standard_error']==pytest.approx(.32)
    assert top['signed_bias_direction']==-1 and result['fitting_episode_groups']==2
    replicated=rows+[dict(row,example_id=row['example_id']+'_rep') for row in rows]
    replicated_groups=dict(groups,**{eid+'_rep':group for eid,group in groups.items()})
    doubled=rank_row_weighted_gradients(replicated,replicated_groups,(9,2))
    for a,b in zip(result['ranking'],doubled['ranking']):
        assert a['mean_loss_gradient']==pytest.approx(b['mean_loss_gradient'])
        assert a['standard_error']==pytest.approx(b['standard_error'])
    with pytest.raises(ValueError,match='unique'):
        rank_row_weighted_gradients(rows+[rows[0]],groups,(9,2))
    failed=copy.deepcopy(rows);failed[0]['status']='error'
    with pytest.raises(ValueError,match='failed'):
        rank_row_weighted_gradients(failed,groups,(9,2))
