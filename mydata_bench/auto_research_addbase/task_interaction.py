"""F5: remove the attention response remaining after task-embedding ablation.

This is a fixed logit interaction contrast, not a calibrated probability or a
claim to identify physical causality. All native rating bins remain available.
"""
from contextlib import contextmanager
import numpy as np
import torch


def interaction_logits(actual_steered, ablated_steered, ablated_baseline):
    arrays = [np.asarray(z, dtype=np.float64) for z in
              [actual_steered, ablated_steered, ablated_baseline]]
    if len({z.shape for z in arrays}) != 1 or arrays[0].ndim != 1 or arrays[0].size < 2:
        raise ValueError('Three matching full rating vectors required')
    if any(not np.isfinite(z).all() for z in arrays):
        raise ValueError('Nonfinite native rating vector')
    if np.array_equal(arrays[1], arrays[2]):
        return list(actual_steered)
    return (arrays[0] - (arrays[1] - arrays[2])).tolist()


@contextmanager
def ablate_task_embeddings(embedding, positions, expected_input_ids, diagnostics):
    if not positions or len(positions) != len(set(positions)):
        raise ValueError('Unique nonempty task positions required')
    diagnostics.update(calls=0, intervention='zero actual task token input embeddings only',
                       token_positions=list(positions), input_ids_grid_timestamps_unchanged=True)

    def hook(module, args, output):
        if len(args) != 1 or not torch.equal(args[0], expected_input_ids):
            raise ValueError('Embedding call does not match full prepared model input')
        if output.ndim != 3 or output.shape[:2] != expected_input_ids.shape:
            raise ValueError('Unexpected input embedding dimensions')
        if min(positions) < 0 or max(positions) >= output.shape[1]:
            raise ValueError('Task position outside prepared input')
        if diagnostics['calls'] != 0:
            raise ValueError('Unexpected repeated embedding call; full prefill only')
        result = output.clone()
        result[:, positions, :] = 0
        diagnostics.update(calls=1, original_task_embedding_max_abs=float(output[:, positions, :].detach().abs().max()),
                           task_embeddings_zero=bool(torch.count_nonzero(result[:, positions, :]) == 0))
        return result

    handle = embedding.register_forward_hook(hook)
    try:
        yield diagnostics
        if diagnostics['calls'] != 1:
            raise ValueError('Task-embedding intervention was not invoked exactly once')
    finally:
        handle.remove()
