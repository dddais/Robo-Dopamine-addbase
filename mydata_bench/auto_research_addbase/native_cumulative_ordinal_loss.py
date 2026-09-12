"""Proper cumulative log loss on all native ordered score bins.

Only a fitting objective. No inference-time score, bin, or threshold changes.
"""
import torch


def native_grade_target(logits,grade,model):
    if isinstance(grade,bool) or grade not in [1,2,3,4,5]:
        raise ValueError('A true native grade1–5 is required')
    size=5 if model in ['qwen','rr'] else 10 if model=='meter' else None
    if size is None or logits.shape!=(size,):
        raise ValueError('All five native choices or all ten native Meter bins required')
    target=torch.zeros_like(logits,dtype=torch.float32)
    if size==5:
        target[int(grade)-1]=1.
    else:
        position=(grade-1)*9/4
        lower=int(position);upper=min(lower+1,9);fraction=position-lower
        target[lower]+=1-fraction;target[upper]+=fraction
    return target


def cumulative_log_loss(logits,target):
    if logits.ndim!=1 or logits.numel()<2 or target.shape!=logits.shape:
        raise ValueError('One complete ordered probability target and logit vector required')
    values=logits.float();truth=target.to(device=values.device,dtype=values.dtype)
    if not torch.isfinite(values).all() or not torch.isfinite(truth).all() or (truth<0).any() or not torch.isclose(truth.sum(),truth.new_tensor(1.),atol=1e-6,rtol=0.):
        raise ValueError('Finite logits and a normalized nonnegative target required')
    normalizer=torch.logsumexp(values,dim=0)
    # Evaluate both sides directly in log space; do not clamp a rounded CDF.
    lower_log_probability=torch.logcumsumexp(values,dim=0)[:-1]-normalizer
    upper_log_probability=torch.logcumsumexp(values.flip(0),dim=0).flip(0)[1:]-normalizer
    truth_cdf=truth.cumsum(0)[:-1]
    return -(truth_cdf*lower_log_probability+(1-truth_cdf)*upper_log_probability).mean()


def native_cumulative_ordinal_loss(logits,grade,model):
    return cumulative_log_loss(logits,native_grade_target(logits,grade,model))
