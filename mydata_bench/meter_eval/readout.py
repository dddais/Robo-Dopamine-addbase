"""Exact official CPU/float32 expected-bin conversion from saved native logits.

This is model inference postprocessing, independent of labels. Keeping it
separate permits correction/audit of historical derived values without touching
any original prediction file or rerunning a single neural forward pass.
"""
import torch

def official_progress_trajectory(logits):
    values=torch.tensor(logits,dtype=torch.float32,device='cpu')
    if values.ndim!=2 or values.shape[-1]!=10:raise ValueError('Expected frame x 10 native logits')
    probs=values if (values.sum(-1)==1).all() else values.softmax(-1)
    return (probs*torch.linspace(0.,1.,10,dtype=torch.float32)).sum(-1).tolist()

def known_logit_progress_trajectory(logits):
    """Typed counterfactual logits always require softmax, even if sum is 1.

    The official utility also accepts probability arrays and infers that type
    from the sum across bins. Synthesized logit contrasts have a known type;
    an accidental sum of one must not reinterpret them as probabilities.
    """
    values=torch.tensor(logits,dtype=torch.float32,device='cpu')
    if values.ndim!=2 or values.shape[-1]!=10:raise ValueError('Expected frame x 10 logits')
    return (values.softmax(-1)*torch.linspace(0.,1.,10,dtype=torch.float32)).sum(-1).tolist()

def canonicalize(row):
    if row.get('status')!='ok' or 'native_progress_logits' not in row:return row
    typed=row.get('score_readout_type')=='softmax_known_logits'
    trajectory=(known_logit_progress_trajectory if typed else official_progress_trajectory)(row['native_progress_logits'])
    return {**row,'stored_progress_before_precision_audit':row['progress'],'progress':trajectory[-1],'progress_trajectory':trajectory,'readout':'CPU float32 softmax of typed counterfactual logits over native bins' if typed else 'official Robometer eval_server CPU float32 expected-bin conversion'}
