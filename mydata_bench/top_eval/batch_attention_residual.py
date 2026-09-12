"""Sparse attention residual on top of the unchanged native SDPA output.

In real arithmetic O' = O + (softmax(Z') - softmax(Z))V. Computing
only the selected heads avoids moving every layer to a slow math kernel.
The subtraction occurs before the value product, so a zero intervention
produces an exactly zero residual, including with low precision model weights.
This is an explicitly versioned numerical implementation, not bitwise parity
with a full math-SDPA replacement at nonzero bias.
"""
import copy
from contextlib import contextmanager
import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


@contextmanager
def batch_steering(layers,heads,targets,visuals,bias=6.,kind='bias',query_scope='all',diagnostics=None,capture_safe=False,readout_query_index=None):
    if kind not in {'bias','target_only','visual_preserve'}:raise ValueError(kind)
    if query_scope=='readout' and (not isinstance(readout_query_index,int) or readout_query_index<0):raise ValueError('Explicit native readout query index required')
    diag=diagnostics if diagnostics is not None else {};grouped={};old=[];cache={};index_cache={}
    for layer,head in heads:grouped.setdefault(layer,[]).append(head)
    for target,visual in zip(targets,visuals):assert target and set(target)<=set(visual)
    native=ALL_ATTENTION_FUNCTIONS['sdpa']
    diag.update(kind=kind,bias=bias,query_scope=query_scope,calls=0,
                engine='native_sdpa_plus_float32_selected_head_residual',
                target_counts=[len(set(t)) for t in targets],
                visual_counts=[len(set(v)) for v in visuals],
                causal_mask_preserved=True,per_example_regions=True,capture_safe_indexing=capture_safe,readout_query_index=readout_query_index)
    # Regions are static during autoregressive generation. Build them once on
    # CPU and transfer once, then pad zeros as the KV cache grows. Rebuilding
    # per-example CUDA index assignments at every decode length was expensive.
    base_length=max(max(vv) for vv in visuals)+1
    target_cpu=torch.zeros((len(targets),base_length),dtype=torch.bool)
    visual_cpu=torch.zeros_like(target_cpu)
    for i,(tt,vv) in enumerate(zip(targets,visuals)):
        target_cpu[i,tt]=True;visual_cpu[i,vv]=True
    delta_cpu=target_cpu.float()*bias
    if kind!='target_only':delta_cpu-=(visual_cpu&~target_cpu).float()*bias
    def masks(device,length):
        key=(device,length)
        if key not in cache:
            base_key=(device,'base')
            if base_key not in cache:cache[base_key]=(visual_cpu.to(device)[:,None,None,:],delta_cpu.to(device)[:,None,None,:])
            vis,delta=cache[base_key]
            if length>base_length:
                vis=torch.nn.functional.pad(vis,(0,length-base_length),value=False)
                delta=torch.nn.functional.pad(delta,(0,length-base_length),value=0.)
            elif length<base_length:vis=vis[...,:length];delta=delta[...,:length]
            cache[key]=(vis,delta)
        return cache[key]
    def attend(module,q,k,v,mask,dropout=0.,scaling=None,**kwargs):
        if dropout:raise ValueError('Inference-only implementation requires zero dropout')
        output,weights=native(module,q,k,v,mask,dropout=dropout,scaling=scaling,**kwargs)
        b,h,ql,d=q.shape;kl=k.shape[-2]
        active=query_scope in {'all','readout'} or (query_scope=='prefill' and ql>1) or (query_scope=='decode' and ql==1) or (query_scope=='last_prompt' and ql>1)
        if not active:return output,weights
        if b!=len(targets):raise ValueError('Batch/region count mismatch')
        hh=grouped[module.layer_idx];kv=[head//module.num_key_value_groups for head in hh]
        if capture_safe:
            index_key=(q.device,module.layer_idx)
            if index_key not in index_cache:
                index_cache[index_key]=(torch.tensor(hh,device=q.device),torch.tensor(kv,device=q.device))
            hi,ki=index_cache[index_key]
        rows=slice(-1,None) if query_scope=='last_prompt' else slice(None)
        if query_scope=='readout':
            if readout_query_index>=ql:raise ValueError('Readout query must be in the current full prefill; autoregressive use is unsupported')
            rows=slice(readout_query_index,readout_query_index+1)
        scale=scaling if scaling is not None else d**-.5
        qs=q.index_select(1,hi)[:,:,rows,:] if capture_safe else q[:,hh,rows,:]
        ks=k.index_select(1,ki) if capture_safe else k[:,kv,:,:]
        vs=v.index_select(1,ki) if capture_safe else v[:,kv,:,:]
        z=qs.float()@ks.float().transpose(-1,-2)*scale
        if mask is None:
            query_ids=torch.arange(ql,device=q.device)[rows]+kl-ql
            keep=torch.arange(kl,device=q.device)[None,:]<=query_ids[:,None]
            z=z.masked_fill(~keep,float('-inf'))
        else:
            expanded=mask[...,:kl].expand(b,h,ql,kl)
            base=expanded.index_select(1,hi)[:,:,rows,:] if capture_safe else expanded[:,hh,rows,:]
            z=z.masked_fill(~base,float('-inf')) if base.dtype==torch.bool else z+base.float()
        visual,delta=masks(q.device,kl)
        if kind=='visual_preserve':
            z0=torch.logsumexp(z.masked_fill(~visual,float('-inf')),-1,keepdim=True)
            z1=torch.logsumexp((z+delta).masked_fill(~visual,float('-inf')),-1,keepdim=True)
            correction=torch.where(torch.isfinite(z0)&torch.isfinite(z1),z1-z0,torch.zeros_like(z0))
            delta=delta-correction*visual
        original=torch.softmax(z,dim=-1).nan_to_num(0.)
        changed=torch.softmax(z+delta,dim=-1).nan_to_num(0.)
        residual=(changed-original)@vs.float()
        result=output.transpose(1,2).clone()
        if capture_safe:
            selected=result.index_select(1,hi)
            selected[:,:,rows,:]+=residual.to(result.dtype)
            result.index_copy_(1,hi,selected)
        else:result[:,hh,rows,:]+=residual.to(result.dtype)
        diag['calls']+=1
        return result.transpose(1,2).contiguous(),weights
    name='addbase_sole_batch_residual_v1'
    ALL_ATTENTION_FUNCTIONS.register(name,attend)
    try:
        for layer in grouped:
            m=layers[layer].self_attn;old.append((m,m.config))
            m.config=copy.copy(m.config);m.config._attn_implementation=name
        yield diag
    finally:
        for m,config in old:m.config=config
