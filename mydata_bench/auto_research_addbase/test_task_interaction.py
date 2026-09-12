import pytest
import numpy as np
import torch
from mydata_bench.auto_research_addbase.task_interaction import ablate_task_embeddings, interaction_logits


def test_ablation_is_local_and_restores_after_exception():
    torch.manual_seed(41)
    embedding = torch.nn.Embedding(12, 4)
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    original = embedding(ids)
    audit = {}
    with ablate_task_embeddings(embedding, [1, 3], ids, audit):
        actual = embedding(ids)
    assert torch.equal(actual[:, [0, 2, 4]], original[:, [0, 2, 4]])
    assert torch.count_nonzero(actual[:, [1, 3]]) == 0
    assert torch.equal(embedding(ids), original)
    assert audit['calls'] == 1 and audit['task_embeddings_zero']
    with pytest.raises(ValueError):
        with ablate_task_embeddings(embedding, [1, 3], ids, {}):
            embedding(torch.tensor([[1, 2, 3]]))
    assert torch.equal(embedding(ids), original)


def test_interaction_zero_and_rating_permutation():
    actual = [1.1, 5.2, 4.3, 3.4, 2.5]
    empty = [-2., 0., 1., 2., 7.]
    assert interaction_logits(actual, empty, empty) == actual
    steered = [2., 1., 2., 4., 5.]
    result = np.asarray(interaction_logits(actual, steered, empty))
    perm = np.array([4, 1, 3, 0, 2])
    np.testing.assert_array_equal(interaction_logits(np.array(actual)[perm], np.array(steered)[perm], np.array(empty)[perm]), result[perm])
    # A middle rating remains a legal argmax; all five bins participate.
    assert np.argmax(interaction_logits([0., 0., 9., 0., 0.], empty, empty)) == 2
