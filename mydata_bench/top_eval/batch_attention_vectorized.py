"""Vectorized per-example region steering for efficient SOLE cohorts."""
import copy
from contextlib import contextmanager
import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3_vl.modeling_qwen3_vl import repeat_kv

@contextmanager
def batch_steering(layers,heads,targets,visuals,bias=6.,kind='bias',query_scope='all',diagnostics=None):
    diag=diagnostics if diagnostics is not None else {};grouped={};old=[];cache={}
    for l,h in heads:grouped.setdefault(l,[]).append(h)
    assert len(targets)==len(visuals)
    for t,v in zip(targets,visuals):assert t and set(t)<=set(v)
    diag.update(kind=kind,bias=bias,query_scope=query_scope,calls=0,target_counts=[len(set(t)) for t in targets],visual_counts=[len(set(v)) for v in visuals],per_example_regions=True,causal_mask_preserved=True,implementation='vectorized_batch_v1')
    def masks(device,length):
        key=(str(device),length)
        if key not in cache:
            t=torch.zeros((len(targets),length),device=device,dtype=torch.bool);v=torch.zeros_like(t)
            for i,(tt,vv) in enumerate(zip(targets,visuals)):
                t[i,[x for x in tt if x<length]]=True;v[i,[x for x in vv if x<length]]=True
            delta=t.float()*bias
            if kind!='target_only':delta-=(v&~t).float()*bias
            cache[key]=(v[:,None,None,:],delta[:,None,None,:])
        return cache[key]
    name='addbase_sole_batch_vectorized_v1'
    def attend(module,q,k,v,mask,dropout=0.,scaling=None,**kwargs):
        k=repeat_kv(k,module.num_key_value_groups);v=repeat_kv(v,module.num_key_value_groups)
        b,h,ql,d=q.shape;kl=k.shape[-2];scale=scaling if scaling is not None else d**-.5
        if b!=len(targets):raise ValueError('Batch/region cardinality mismatch')
        if mask is None:
            keep=torch.arange(kl,device=q.device)[None,:]<=torch.arange(ql,device=q.device)[:,None]+kl-ql
            mask=torch.zeros((1,1,ql,kl),dtype=q.dtype,device=q.device).masked_fill(~keep,float('-inf'))
        else:mask=mask[...,:kl]
        if mask.dtype==torch.bool:mask=torch.zeros_like(mask,dtype=q.dtype).masked_fill(~mask,float('-inf'))
        active=query_scope=='all' or (query_scope=='prefill' and ql>1) or (query_scope=='decode' and ql==1) or (query_scope=='last_prompt' and ql>1)
        if active:
            hh=grouped[module.layer_idx];rows=slice(-1,None) if query_scope=='last_prompt' else slice(None)
            base=mask.expand(b,h,ql,kl);new=base.clone().float() if kind=='visual_preserve' else base.clone()
            visual,delta=masks(q.device,kl)
            if kind=='visual_preserve':
                z=(q[:,hh,rows,:].float()@k[:,hh,:,:].float().transpose(-1,-2))*scale+base[:,hh,rows,:].float()
                z0=torch.logsumexp(z.masked_fill(~visual,float('-inf')),-1,keepdim=True)
                z1=torch.logsumexp((z+delta).masked_fill(~visual,float('-inf')),-1,keepdim=True)
                correction=torch.where(torch.isfinite(z0)&torch.isfinite(z1),z1-z0,torch.zeros_like(z0))
                change=delta-correction*visual
            else:change=delta
            new[:,hh,rows,:]+=change.to(new.dtype);mask=new;diag['calls']+=1
        # Explicitly match the neutral baseline's kernel for long reasoning.
        from torch.nn.attention import sdpa_kernel,SDPBackend
        with sdpa_kernel(SDPBackend.MATH):
            out=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.,scale=scale)
        return out.transpose(1,2).contiguous(),None
    ALL_ATTENTION_FUNCTIONS.register(name,attend)
    try:
        for l in grouped:
            m=layers[l].self_attn;old.append((m,m.config));m.config=copy.copy(m.config);m.config._attn_implementation=name
        yield diag
    finally:
        for m,c in old:m.config=c
