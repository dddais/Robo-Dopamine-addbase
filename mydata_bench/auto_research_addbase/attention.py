"""Causal attention interventions, including visual-mass-preserving steering.

Uses Transformers' public attention dispatch on selected modules. All other
modules keep their original backend. No model output or label enters this code.
"""
from __future__ import annotations
import copy,math
from contextlib import contextmanager
from types import SimpleNamespace
import numpy as np
import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3_vl.modeling_qwen3_vl import repeat_kv

@contextmanager
def steering(layers,heads,target,visual,bias=6.,kind='bias',query_scope='all',diagnostics=None,engine='legacy'):
    if engine=='native_residual':
        from mydata_bench.top_eval.batch_attention_residual import batch_steering
        with batch_steering(layers,heads,[target],[visual],bias,kind,query_scope,diagnostics) as diag:yield diag
        return
    if engine!='legacy':raise ValueError(engine)
    if kind not in {'bias','visual_preserve','target_only'}:raise ValueError(kind)
    diagnostics=diagnostics if diagnostics is not None else {}
    target=sorted(set(target));visual=sorted(set(visual));other=sorted(set(visual)-set(target))
    assert set(target)<=set(visual) and not set(target)&set(other)
    if not target:raise ValueError('No target image cells')
    grouped={}
    for layer,head in heads:grouped.setdefault(int(layer),[]).append(int(head))
    diagnostics.update({'kind':kind,'query_scope':query_scope,'target_tokens':len(target),'other_visual_tokens':len(other),'calls':0,'query_rows':0,'selected_head_count':len(heads),'causal_mask_preserved':True,'max_log_visual_normalizer_error':0.})
    old=[]
    name='addbase_attention_v1'
    def interface(module,q,k,v,mask,dropout=0.,scaling=None,**kwargs):
        k=repeat_kv(k,module.num_key_value_groups);v=repeat_kv(v,module.num_key_value_groups)
        scale=scaling if scaling is not None else 1/math.sqrt(q.shape[-1])
        b,h,ql,d=q.shape;kl=k.shape[-2];dtype=q.dtype;dev=q.device
        if mask is None:
            causal=torch.arange(kl,device=dev)[None,:] <= (torch.arange(ql,device=dev)[:,None]+kl-ql)
            base=torch.zeros((1,1,ql,kl),device=dev,dtype=dtype).masked_fill(~causal,float('-inf'))
        elif mask.dtype==torch.bool:
            base=torch.zeros_like(mask,dtype=dtype).masked_fill(~mask,float('-inf'))
        else:base=mask[...,:kl].to(dtype)
        scoped=query_scope=='all' or (query_scope=='prefill' and ql>1) or (query_scope=='decode' and ql==1) or (query_scope=='last_prompt' and ql>1)
        if not scoped:
            out=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=base,dropout_p=0.,scale=scale)
            return out.transpose(1,2).contiguous(),None
        selected_heads=grouped[module.layer_idx]
        additive=base.expand(b,h,ql,kl).clone()
        if kind=='visual_preserve':additive=additive.float()
        row_idx=torch.arange(ql,device=dev) if query_scope!='last_prompt' else torch.tensor([ql-1],device=dev)
        t=[i for i in target if i<kl];vv=[i for i in visual if i<kl];nn=[i for i in other if i<kl]
        for head in selected_heads:
            delta=torch.zeros((b,len(row_idx),kl),device=dev,dtype=torch.float32)
            if t:delta[:,:,t]=float(bias)
            if nn and kind!='target_only':delta[:,:,nn]=-float(bias)
            if kind=='visual_preserve':
                logits=(q[:,head,row_idx,:].float() @ k[:,head,:,:].float().transpose(-1,-2))*scale
                logits=logits+base.expand(b,h,ql,kl)[:,head,row_idx,:].float()
                z0=torch.logsumexp(logits[:,:,vv],dim=-1,keepdim=True)
                z1=torch.logsumexp((logits+delta)[:,:,vv],dim=-1,keepdim=True)
                valid=torch.isfinite(z0)&torch.isfinite(z1)
                correction=torch.where(valid,z1-z0,torch.zeros_like(z0))
                delta[:,:,vv]-=correction
                # The analytic equality is checked on first call only; avoid GPU sync per decode.
                if diagnostics['calls']==0:
                    check=torch.logsumexp((logits+delta)[:,:,vv],dim=-1,keepdim=True)-z0
                    err=float(check[valid].abs().max().item()) if valid.any() else 0.
                    diagnostics['max_log_visual_normalizer_error']=max(diagnostics['max_log_visual_normalizer_error'],err)
            additive[:,head,row_idx,:]+=delta.to(additive.dtype)
        diagnostics['calls']+=1;diagnostics['query_rows']+=len(row_idx)
        if kind=='visual_preserve':
            # cuDNN on this A100/torch build rejects float32 bias with bf16 Q.
            # PyTorch math SDPA supports this documented mixed-mask contract.
            from torch.nn.attention import sdpa_kernel,SDPBackend
            with sdpa_kernel(SDPBackend.MATH):
                out=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=additive,dropout_p=0.,scale=scale)
        else:
            out=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=additive,dropout_p=0.,scale=scale)
        return out.transpose(1,2).contiguous(),None
    ALL_ATTENTION_FUNCTIONS.register(name,interface)
    try:
        for layer in grouped:
            module=layers[layer].self_attn;old.append((module,module.config))
            module.config=copy.copy(module.config);module.config._attn_implementation=name
        yield diagnostics
    finally:
        for module,config in old:module.config=config

def collect_ranking(layers,forward,query_index,target,visual):
    raw=[];vis=[];handles=[];old=[]
    shared_config=layers[0].self_attn.config
    shared_implementation=shared_config._attn_implementation
    # Force the language model's causal-mask builder to materialize its mask.
    # Switching only a child module to eager while its parent remains SDPA
    # would let SDPA elide the mask, leaking future tokens during ranking.
    shared_config._attn_implementation='eager'
    n=len(layers);raw=np.zeros((n,layers[0].self_attn.config.num_attention_heads));vis=np.zeros_like(raw)
    def capture(i):
        def hook(module,args,out):
            if out[1] is None:raise RuntimeError('Missing eager weights')
            a=out[1][0,:,query_index,:]
            raw[i]=a[:,target].float().sum(-1).detach().cpu().numpy()
            vis[i]=a[:,visual].float().sum(-1).detach().cpu().numpy()
            return out[0],None
        return hook
    try:
        for i,layer in enumerate(layers):
            m=layer.self_attn;old.append((m,m.config));m.config=copy.copy(m.config);m.config._attn_implementation='eager'
            handles.append(m.register_forward_hook(capture(i)))
        with torch.inference_mode():forward()
    finally:
        for h in handles:h.remove()
        for m,c in old:m.config=c
        shared_config._attn_implementation=shared_implementation
    frac=len(target)/len(visual)
    return {'raw_mass':raw.tolist(),'image_mass':vis.tolist(),'excess_mass':(raw-frac*vis).tolist(),'visual_enrichment':(np.divide(raw,vis,out=np.zeros_like(raw),where=vis>0)-frac).tolist(),'query_index':query_index,'target_count':len(target),'visual_count':len(visual)}
