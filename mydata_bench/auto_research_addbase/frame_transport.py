"""Frame-local attention transport; no reward, label, or score adjustment.

Within each causal frame, move a fixed fraction of non-target attention mass
to the existing conditional target distribution. Preserve each frame's mass
and every nonvisual key probability under the current Q/K, in real arithmetic.
"""
from contextlib import contextmanager
import copy
import math
import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


def transport_delta(probabilities, frame_targets, fraction):
    """Return probability changes for disjoint (start, end, target_indices)."""
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('fraction must be in [0, 1]')
    delta = torch.zeros_like(probabilities)
    if fraction == 0:
        return delta
    for start, end, selected in frame_targets:
        local = probabilities[..., start:end]
        mask = selected.to(device=local.device, dtype=torch.bool)
        if mask.shape != (end-start,):
            raise ValueError('frame mask shape mismatch')
        target = local * mask
        remainder = local * ~mask
        target_mass = target.sum(-1, keepdim=True)
        other_mass = remainder.sum(-1, keepdim=True)
        conditional = target / target_mass.clamp_min(torch.finfo(local.dtype).tiny)
        # No accessible target means no intervention, rather than creating
        # attention across a causal boundary or dividing by an empty support.
        change = fraction * (other_mass * conditional - remainder)
        delta[..., start:end] = torch.where(target_mass > 0, change, 0.)
    return delta


@contextmanager
def frame_transport(layers, heads, target_positions, spans, fraction=.5,
                    diagnostics=None):
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('fraction must be in [0, 1]')
    targets = set(target_positions)
    visual = {p for span in spans for p in range(span.start, span.end)}
    if not targets or not targets <= visual:
        raise ValueError('nonempty visual target required')
    frames = []
    occupied = set()
    for span in spans:
        indices = set(range(span.start, span.end))
        if occupied & indices:
            raise ValueError('overlapping frame spans')
        occupied.update(indices)
        frames.append((span.start, span.end,
                       torch.tensor([p in targets for p in range(span.start, span.end)])))
    grouped = {}
    for layer, head in heads:
        grouped.setdefault(int(layer), []).append(int(head))
    diag = diagnostics if diagnostics is not None else {}
    diag.update(engine='frame_transport_native_residual_v1', fraction=fraction,
                calls=0, selected_heads=len(heads), frames=len(frames),
                max_frame_mass_error=0., max_nonvisual_change=0.,
                max_causal_change=0., minimum_changed_probability=0.)
    native = ALL_ATTENTION_FUNCTIONS['sdpa']
    saved, index_cache, frame_cache, audited = [], {}, {}, set()

    def attend(module, q, k, v, attention_mask, dropout=0., scaling=None, **kwargs):
        if dropout:
            raise ValueError('inference only')
        output, weights = native(module, q, k, v, attention_mask, dropout=dropout,
                                 scaling=scaling, **kwargs)
        diag['calls'] += 1
        if fraction == 0:
            return output, weights
        batch, nheads, qlen, dimension = q.shape
        klen = k.shape[-2]
        if batch != 1 or max(end for _, end, _ in frames) > klen:
            raise ValueError('one complete prefill example is required')
        cache_key = (module.layer_idx, q.device)
        if cache_key not in index_cache:
            selected = grouped[module.layer_idx]
            index_cache[cache_key] = (
                torch.tensor(selected, device=q.device),
                torch.tensor([h // module.num_key_value_groups for h in selected], device=q.device))
        hi, ki = index_cache[cache_key]
        logits = (q.index_select(1, hi).float() @ k.index_select(1, ki).float().transpose(-1, -2))
        logits *= scaling if scaling is not None else dimension ** -.5
        causal = torch.arange(klen, device=q.device)[None, :] <= (
            torch.arange(qlen, device=q.device)[:, None] + klen-qlen)
        if attention_mask is None:
            logits.masked_fill_(~causal, float('-inf'))
        else:
            mask = attention_mask[..., :klen].expand(batch, nheads, qlen, klen).index_select(1, hi)
            logits = logits.masked_fill(~mask, float('-inf')) if mask.dtype == torch.bool else logits + mask.float()
        probabilities = torch.softmax(logits, -1).nan_to_num(0.)
        if q.device not in frame_cache:
            frame_cache[q.device] = [(start, end, mask.to(q.device)) for start, end, mask in frames]
        delta = transport_delta(probabilities, frame_cache[q.device], fraction)
        if module.layer_idx not in audited:
            audited.add(module.layer_idx)
            errors = [float(delta[..., start:end].sum(-1).abs().max()) for start, end, _ in frames]
            diag['max_frame_mass_error'] = max(diag['max_frame_mass_error'], *errors)
            nonvisual = torch.ones(klen, dtype=torch.bool, device=q.device)
            nonvisual[list(visual)] = False
            if nonvisual.any():
                diag['max_nonvisual_change'] = max(diag['max_nonvisual_change'], float(delta[..., nonvisual].abs().max()))
            diag['max_causal_change'] = max(diag['max_causal_change'], float(delta.masked_fill(causal, 0).abs().max()))
            diag['minimum_changed_probability'] = min(diag['minimum_changed_probability'], float((probabilities + delta).min()))
        residual = delta @ v.index_select(1, ki).float()
        result = output.transpose(1, 2).clone()
        selected_output = result.index_select(1, hi) + residual.to(result.dtype)
        result.index_copy_(1, hi, selected_output)
        return result.transpose(1, 2).contiguous(), weights

    name = 'addbase_frame_transport_v1'
    ALL_ATTENTION_FUNCTIONS.register(name, attend)
    try:
        for layer in grouped:
            module = layers[layer].self_attn
            saved.append((module, module.config))
            module.config = copy.copy(module.config)
            module.config._attn_implementation = name
        yield diag
    finally:
        for module, config in saved:
            module.config = config
