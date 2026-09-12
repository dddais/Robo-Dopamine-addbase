"""Batched SOLE attention with separate seven-step recurrence per condition."""
from __future__ import annotations
import argparse,copy,json,pathlib,time,traceback
from contextlib import contextmanager,nullcontext
import torch,yaml
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3_vl.modeling_qwen3_vl import repeat_kv
from mydata_bench.auto_research_addbase.common import *
from .runtime import SoleRuntime,parse_answer
from .batch_baseline import pad_batch

@contextmanager
def batch_steering_reference(layers,heads,targets,visuals,bias=6.,kind='bias',query_scope='all',diagnostics=None):
    diag=diagnostics if diagnostics is not None else {};grouped={};old=[]
    for l,h in heads:grouped.setdefault(l,[]).append(h)
    assert len(targets)==len(visuals)
    for t,v in zip(targets,visuals):assert t and set(t)<=set(v)
    diag.update(kind=kind,bias=bias,query_scope=query_scope,calls=0,target_counts=[len(set(t)) for t in targets],visual_counts=[len(set(v)) for v in visuals],per_example_regions=True,causal_mask_preserved=True)
    name='addbase_sole_batch_attention_v1'
    def attend(module,q,k,v,mask,dropout=0.,scaling=None,**kwargs):
        k=repeat_kv(k,module.num_key_value_groups);v=repeat_kv(v,module.num_key_value_groups)
        b,h,ql,d=q.shape;kl=k.shape[-2];scale=scaling if scaling is not None else d**-.5
        if b!=len(targets):raise ValueError('Batch/region cardinality mismatch')
        if mask is None:
            keep=torch.arange(kl,device=q.device)[None,:] <= torch.arange(ql,device=q.device)[:,None]+kl-ql
            mask=torch.zeros((1,1,ql,kl),dtype=q.dtype,device=q.device).masked_fill(~keep,float('-inf'))
        else:mask=mask[...,:kl]
        if mask.dtype==torch.bool:mask=torch.zeros_like(mask,dtype=q.dtype).masked_fill(~mask,float('-inf'))
        active=query_scope=='all' or (query_scope=='prefill' and ql>1) or (query_scope=='decode' and ql==1) or (query_scope=='last_prompt' and ql>1)
        if active:
            base=mask.expand(b,h,ql,kl);new=base.clone().float() if kind=='visual_preserve' else base.clone()
            rows=torch.arange(ql,device=q.device) if query_scope!='last_prompt' else torch.tensor([ql-1],device=q.device)
            for head in grouped[module.layer_idx]:
                for i,(target,visual) in enumerate(zip(targets,visuals)):
                    target=[x for x in target if x<kl];visual=[x for x in visual if x<kl];other=list(set(visual)-set(target))
                    delta=torch.zeros((len(rows),kl),device=q.device,dtype=torch.float32);delta[:,target]=bias
                    if kind!='target_only':delta[:,other]=-bias
                    if kind=='visual_preserve':
                        z=q[i,head,rows,:].float()@k[i,head].float().T*scale+base[i,head,rows,:].float()
                        z0=torch.logsumexp(z[:,visual],-1,keepdim=True);z1=torch.logsumexp((z+delta)[:,visual],-1,keepdim=True)
                        c=torch.where(torch.isfinite(z0)&torch.isfinite(z1),z1-z0,torch.zeros_like(z0));delta[:,visual]-=c
                    new[i,head,rows,:]+=delta.to(new.dtype)
            mask=new;diag['calls']+=1
        if kind=='visual_preserve':
            from torch.nn.attention import sdpa_kernel,SDPBackend
            with sdpa_kernel(SDPBackend.MATH):
                out=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.,scale=scale)
        else:
            out=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.,scale=scale)
        return out.transpose(1,2).contiguous(),None
    ALL_ATTENTION_FUNCTIONS.register(name,attend)
    try:
        for l in grouped:
            m=layers[l].self_attn;old.append((m,m.config));m.config=copy.copy(m.config);m.config._attn_implementation=name
        yield diag
    finally:
        for m,c in old:m.config=c

from .batch_attention_vectorized import batch_steering

def prepare_regions(runtime,sample,inp,condition):
    target,alignment=runtime.positions(sample,inp,condition.get('scope','all_frames'))
    visual=[p for span in inp['spans'] for p in range(span.start,span.end)]
    if condition.get('region')=='wrong':
        from .controls import wrong_control
        target=wrong_control(inp,target,alignment)
    return target,visual,alignment

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--batch-size',type=int,default=16);ap.add_argument('--pilot-batches',type=int);ap.add_argument('--retry-errors',action='store_true');ap.add_argument('--preload-head-indices',action='store_true');ap.add_argument('--allow-missing-context',action='store_true');args=ap.parse_args()
    cfg=yaml.safe_load(pathlib.Path(args.config).read_text());out=pathlib.Path(cfg['output_dir']);out.mkdir(parents=True,exist_ok=True)
    if args.pilot_batches and not cfg.get('pilot'):raise ValueError('Separate pilot config required')
    if (out/'config.json').exists():
        if json.loads((out/'config.json').read_text())!=cfg:raise ValueError('Config mismatch')
    else:create_json(out/'config.json',cfg)
    ranking=json.loads(pathlib.Path(cfg.get('ranking_path',out/'ranking.json')).read_text())['ranking']
    conditions=cfg.get('conditions') or [{'name':'baseline','k':0}]+[{'name':f'{r}_k{k}','k':k,'bias':6.,'kind':'bias','scope':cfg.get('scope','all_frames'),'region':'wrong' if r=='wrong' else 'target','head_group':'low' if r=='low_rank' else 'high'} for k in [8,32,64] for r in ['target','wrong','low_rank']]
    samples=list(read_rows(BASE/'inputs/cohort.jsonl'))
    if cfg.get('example_ids_file'):
        allow=set(json.loads(pathlib.Path(cfg['example_ids_file']).read_text()));samples=[s for s in samples if s['example_id'] in allow]
    if args.pilot_batches:samples=samples[:args.batch_size*args.pilot_batches]
    schedule=cfg.get('intervention_schedule','all_steps')
    if schedule not in {'all_steps','terminal_step'}:raise ValueError(schedule)
    priors={};missing=[]
    if schedule=='terminal_step':
        source=pathlib.Path(cfg['baseline_context_path'])
        priors={r['example_id']:r for r in read_rows(source) if r.get('condition','baseline')=='baseline' and r.get('status')=='ok'}
        missing=[s['example_id'] for s in samples if s['example_id'] not in priors or len(priors[s['example_id']].get('progress_trajectory',[]))!=8]
        if missing and not args.allow_missing_context:raise ValueError(f'Terminal-step comparison requires completed native seven-step baseline context: {len(missing)} missing')
    result_path=out/'attention.jsonl';known=set()
    if result_path.exists():
        latest={(r['example_id'],r['condition']):r for r in read_rows(result_path)}
        known={key for key,row in latest.items() if not args.retry_errors or row['status']=='ok'}
    append(out/'launches.jsonl',{'stage':'attention','batch_size':args.batch_size,'preload_head_indices':args.preload_head_indices,'allow_missing_context':args.allow_missing_context,**provenance(cfg)})
    for eid in missing:
        for condition in conditions:
            key=(eid,condition['name'])
            if key in known:continue
            append(result_path,{'time':timestamp(),'example_id':eid,'condition':condition['name'],'condition_config':condition,
                               'status':'error','phase':'native_baseline_context','error':'Native baseline did not produce a valid complete seven-step history; terminal comparison undefined',
                               'context_baseline_source':cfg['baseline_context_path'],'intervention_schedule':schedule,'newly_generated_steps':0})
            known.add(key)
    runtime=SoleRuntime(cfg)
    # The unmodified and intervened conditions must share explicit causal
    # masks and the same SDPA kernel. Otherwise long CoT amplifies backend
    # rounding differences even when the intervention bias is zero.
    engine=cfg.get('attention_engine','matched_math')
    apply_steering=batch_steering
    language=runtime.model.model.language_model
    if engine=='matched_math':
        language.config=copy.copy(language.config);language.config._attn_implementation='eager'
    elif engine=='native_residual':
        from .batch_attention_residual import batch_steering as apply_steering
        if args.preload_head_indices:
            from functools import partial
            apply_steering=partial(apply_steering,capture_safe=True)
    else:raise ValueError(engine)
    from torch.nn.attention import sdpa_kernel,SDPBackend
    started=time.time()
    for offset in range(0,len(samples),args.batch_size):
        original_batch=samples[offset:offset+args.batch_size]
        for condition in conditions:
            batch=[s for s in original_batch if (s['example_id'],condition['name']) not in known]
            if not batch:continue
            ordered=ranking if condition.get('head_group','high')=='high' else list(reversed(ranking));heads=[(r['layer'],r['head']) for r in ordered[:condition['k']]]
            prepared=[runtime.prepare(s,cfg['protocol']) for s in batch];errors={}
            trajectories=[list(priors[s['example_id']]['progress_trajectory'][:-1]) for s in batch] if schedule=='terminal_step' else [[0.] for s in batch]
            traces=[list(priors[s['example_id']]['steps'][:-1]) for s in batch] if schedule=='terminal_step' else [[] for s in batch]
            for step in (range(7,8) if schedule=='terminal_step' else range(1,8)):
                active=[i for i in range(len(batch)) if i not in errors]
                if not active:break
                try:
                    ready=[];ins=[];regions=[]
                    for i in active:
                        try:
                            inp=runtime.step_input(batch[i],prepared[i],step,trajectories[i][-1])
                            region=prepare_regions(runtime,batch[i],inp,condition) if heads else ([],[],[])
                            ready.append(i);ins.append(inp);regions.append(region)
                        except Exception as exc:
                            errors[i]=repr(exc)
                            append(out/'attention_step_traces.jsonl',{'time':timestamp(),'example_id':batch[i]['example_id'],'condition':condition['name'],'step':step,'status':'error','phase':'prepare_region','error':repr(exc)})
                    active=ready
                    if not active:break
                    padded=pad_batch([x['inputs'] for x in ins],runtime.processor.tokenizer.pad_token_id)
                    diag={};context=nullcontext();alignments=[[] for i in active]
                    if heads:
                        targets=[];visuals=[]
                        for j,i in enumerate(active):
                            t,v,align=regions[j];shift=padded['input_ids'].shape[1]-ins[j]['inputs']['input_ids'].shape[1]
                            targets.append([x+shift for x in t]);visuals.append([x+shift for x in v]);alignments[j]=align
                        context=apply_steering(runtime.layers,heads,targets,visuals,condition.get('bias',6.),condition.get('kind','bias'),condition.get('query_scope','all'),diag)
                    with context,torch.inference_mode(),(sdpa_kernel(SDPBackend.MATH) if engine=='matched_math' else nullcontext()):
                        output=runtime.model.generate(**padded,max_new_tokens=cfg.get('max_new_tokens',512),do_sample=False,temperature=None,top_p=None,top_k=None,use_cache=True,pad_token_id=runtime.processor.tokenizer.pad_token_id)
                    for j,i in enumerate(active):
                        raw=runtime.processor.tokenizer.decode(output[j,padded['input_ids'].shape[1]:],skip_special_tokens=True).strip()
                        row={'time':timestamp(),'example_id':batch[i]['example_id'],'condition':condition['name'],'step':step,'previous_progress':trajectories[i][-1],'raw_output':raw,'hook_diagnostics':diag,'tracking_alignment':alignments[j]}
                        try:
                            progress=parse_answer(raw);row.update(status='ok',progress=progress);trajectories[i].append(progress);traces[i].append(row)
                        except Exception as exc:row.update(status='error',error=repr(exc));errors[i]=repr(exc)
                        append(out/'attention_step_traces.jsonl',row)
                    del output,padded,ins
                except Exception as exc:
                    traceback.print_exc()
                    for i in active:errors[i]=repr(exc)
                    torch.cuda.empty_cache();break
            for i,s in enumerate(batch):
                row={'time':timestamp(),'example_id':s['example_id'],'condition':condition['name'],'condition_config':condition,'progress_trajectory':trajectories[i],'steps':traces[i],'generation_batch_size':len(batch),'attention_backend':engine,'preload_head_indices':args.preload_head_indices,'intervention_schedule':schedule,'context_baseline_source':cfg.get('baseline_context_path') if schedule=='terminal_step' else None,'newly_generated_steps':1 if schedule=='terminal_step' else 7,'video_sha256':s['video_sha256']}
                if i in errors:row.update(status='error',error=errors[i])
                else:row.update(status='ok',progress=trajectories[i][-1])
                append(result_path,row)
            print(json.dumps({'offset':offset,'condition':condition['name'],'batch':len(batch),'errors':len(errors),'total_seconds':time.time()-started}),flush=True)
    append(out/'completion_events.jsonl',{'stage':'attention','time':timestamp(),'batch_size':args.batch_size,'expected_samples':len(samples),'expected_conditions':len(conditions)})
if __name__=='__main__':main()
