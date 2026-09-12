"""F7 differentiable per-head ROI bias throughout all causal prefill queries."""
import copy
from contextlib import contextmanager
import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


@contextmanager
def all_query_reward_gradient_steering(layers, head_biases, target, visual, readout_query_index,
                             active_heads, diagnostics=None):
    if not target or not set(target) <= set(visual):
        raise ValueError('Nonempty grounded target within visual positions required')
    if head_biases.ndim != 2 or head_biases.shape[0] != len(layers):
        raise ValueError('Per-layer/head bias matrix required')
    if not isinstance(readout_query_index, int) or readout_query_index < 0:
        raise ValueError('Actual reward query index required')
    grouped = {}
    for layer, head in active_heads:
        grouped.setdefault(layer, []).append(head)
    if len(active_heads) != sum(len(set(h)) for h in grouped.values()):
        raise ValueError('Duplicate active heads')
    native = ALL_ATTENTION_FUNCTIONS['sdpa']
    diag = diagnostics if diagnostics is not None else {}
    diag.update(calls=0, query_index=readout_query_index, active_heads=len(active_heads),
                only_readout_query_changed=False, all_causal_queries=True, target_count=len(set(target)), visual_count=len(set(visual)))
    base_length = max(visual)+1
    pattern = torch.zeros(base_length, dtype=torch.float32)
    pattern[visual] = -1.
    pattern[target] = 1.
    cache, originals = {}, []

    def attend(module, q, k, v, mask, dropout=0., scaling=None, **kwargs):
        if dropout:
            raise ValueError('Frozen evaluation backbone required')
        output, weights = native(module, q, k, v, mask, dropout=dropout, scaling=scaling, **kwargs)
        batch, heads, qlen, dim = q.shape
        klen = k.shape[-2]
        if batch != 1 or readout_query_index >= qlen or qlen != klen:
            raise ValueError('Full single-example prefill only')
        hh = grouped[module.layer_idx]
        kv = [h//module.num_key_value_groups for h in hh]
        row = slice(0, qlen)
        z = q[:, hh, row, :].float() @ k[:, kv].float().transpose(-1, -2)
        z = z * (scaling if scaling is not None else dim**-.5)
        if mask is None:
            keep = torch.arange(klen, device=q.device)[None, :] <= torch.arange(qlen, device=q.device)[:, None]
            z = z.masked_fill(~keep, float('-inf'))
        else:
            selected = mask[..., :klen].expand(batch, heads, qlen, klen)[:, hh, row, :]
            z = z.masked_fill(~selected, float('-inf')) if selected.dtype == torch.bool else z + selected.float()
        cache_key = (q.device, klen)
        if cache_key not in cache:
            local = pattern[:klen].to(q.device)
            if klen > base_length:
                local = torch.nn.functional.pad(local, (0, klen-base_length))
            cache[cache_key] = local
        delta = head_biases[module.layer_idx, hh].view(1, -1, 1, 1) * cache[cache_key].view(1, 1, 1, -1)
        reference = torch.softmax(z, -1).nan_to_num(0.)
        changed = torch.softmax(z + delta, -1).nan_to_num(0.)
        residual = (changed-reference) @ v[:, kv].float()
        result = output.transpose(1, 2).clone()
        result[:, hh, row, :] = result[:, hh, row, :] + residual.to(result.dtype)
        diag['calls'] += 1
        return result.transpose(1, 2).contiguous(), weights

    name = 'addbase_reward_gradient_all_queries_v1'
    ALL_ATTENTION_FUNCTIONS.register(name, attend)
    try:
        for layer in grouped:
            module = layers[layer].self_attn
            originals.append((module, module.config))
            module.config = copy.copy(module.config)
            module.config._attn_implementation = name
        yield diag
    finally:
        for module, config in originals:
            module.config = config


