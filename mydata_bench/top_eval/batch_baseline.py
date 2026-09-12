"""Batched official SOLE recurrence; complete seven-step histories, no label access.

Attention comparisons also regenerate singleton baseline in their own runner so
baseline/steering use the same arithmetic batch shape. This full-data baseline
reports batch size explicitly and keeps every intermediate raw generation.
"""
from __future__ import annotations
import argparse,json,pathlib,time,traceback
import torch,yaml
from mydata_bench.auto_research_addbase.common import *
from .runtime import SoleRuntime,parse_answer

def pad_batch(inputs,pad_token):
    max_len=max(x['input_ids'].shape[1] for x in inputs);result={}
    for key in ['input_ids','attention_mask']:
        val=pad_token if key=='input_ids' else 0
        result[key]=torch.cat([torch.nn.functional.pad(x[key],(max_len-x[key].shape[1],0),value=val) for x in inputs],dim=0)
    for key in ['pixel_values','image_grid_thw','pixel_values_videos','video_grid_thw']:
        if key in inputs[0]:result[key]=torch.cat([x[key] for x in inputs],dim=0)
    unknown=set.union(*(set(x) for x in inputs))-set(result)
    if unknown:raise ValueError(f'Unaccounted processor keys: {unknown}')
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--batch-size',type=int,default=16);ap.add_argument('--pilot-batches',type=int);args=ap.parse_args()
    cfg=yaml.safe_load(pathlib.Path(args.config).read_text());out=pathlib.Path(cfg['output_dir']);out.mkdir(parents=True,exist_ok=True)
    if args.pilot_batches and not cfg.get('pilot'):raise ValueError('Use a separate pilot config')
    if (out/'config.json').exists():
        if json.loads((out/'config.json').read_text())!=cfg:raise ValueError('Config mismatch')
    else:create_json(out/'config.json',cfg)
    append(out/'launches.jsonl',{'stage':'baseline','batch_size':args.batch_size,**provenance(cfg)})
    result_path=out/'baseline.jsonl';known=set()
    if result_path.exists():known={r['example_id'] for r in read_rows(result_path)}
    samples=[r for r in read_rows(BASE/'inputs/full.jsonl') if r['example_id'] not in known]
    if args.pilot_batches:samples=samples[:args.batch_size*args.pilot_batches]
    runtime=SoleRuntime(cfg);started=time.time()
    for offset in range(0,len(samples),args.batch_size):
        batch=samples[offset:offset+args.batch_size];prepared=[runtime.prepare(s,cfg['protocol']) for s in batch]
        trajectories=[[0.] for _ in batch];traces=[[] for _ in batch];errors={}
        for step in range(1,8):
            active=[i for i in range(len(batch)) if i not in errors]
            if not active:break
            tick=time.time()
            try:
                ins=[runtime.step_input(batch[i],prepared[i],step,trajectories[i][-1]) for i in active]
                padded=pad_batch([x['inputs'] for x in ins],runtime.processor.tokenizer.pad_token_id)
                with torch.inference_mode():
                    output=runtime.model.generate(**padded,max_new_tokens=cfg.get('max_new_tokens',512),do_sample=False,temperature=None,top_p=None,top_k=None,use_cache=True,pad_token_id=runtime.processor.tokenizer.pad_token_id)
                for j,i in enumerate(active):
                    raw=runtime.processor.tokenizer.decode(output[j,padded['input_ids'].shape[1]:],skip_special_tokens=True).strip()
                    record={'example_id':batch[i]['example_id'],'step':step,'previous_progress':trajectories[i][-1],'raw_output':raw,'batch_size':len(active),'time':timestamp()}
                    try:
                        p=parse_answer(raw);record.update(status='ok',progress=p);trajectories[i].append(p);traces[i].append(record)
                    except Exception as e:record.update(status='error',error=repr(e));errors[i]=repr(e)
                    append(out/'step_traces.jsonl',record)
                print(json.dumps({'offset':offset,'step':step,'batch':len(active),'seconds':time.time()-tick,'errors':len(errors)}),flush=True)
                del output,padded,ins
            except Exception as exc:
                traceback.print_exc()
                for i in active:errors[i]=repr(exc)
                torch.cuda.empty_cache();break
        for i,s in enumerate(batch):
            record={'example_id':s['example_id'],'condition':'baseline','time':timestamp(),'video_sha256':s['video_sha256'],'progress_trajectory':trajectories[i],'steps':traces[i],'generation_batch_size':len(batch),'native_readout':'recursive absolute progress; raw unrounded terminal percentage','initial_progress':0.}
            if i in errors:record.update(status='error',error=errors[i])
            else:record.update(status='ok',progress=trajectories[i][-1])
            append(result_path,record)
        print(json.dumps({'finished':offset+len(batch),'total':len(samples),'seconds':time.time()-started}),flush=True)
    append(out/'completion_events.jsonl',{'stage':'baseline','time':timestamp(),'new_records':len(samples),'batch_size':args.batch_size});print('COMPLETED baseline',flush=True)
if __name__=='__main__':main()
