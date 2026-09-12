"""Resumable, condition-isolated baseline/ranking/attention evaluation."""
from __future__ import annotations
import argparse,json,pathlib,time,traceback
import numpy as np
import torch,yaml
from .common import *

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--stage',choices=['baseline','rank','attention','sweep'],required=True);ap.add_argument('--limit',type=int);ap.add_argument('--retry-errors',action='store_true');args=ap.parse_args()
    cfg=yaml.safe_load(pathlib.Path(args.config).read_text());out=pathlib.Path(cfg['output_dir']);out.mkdir(parents=True,exist_ok=True)
    if args.limit and not cfg.get('pilot',False):raise ValueError('Limited runs require separately named pilot config')
    path=out/f'{args.stage}.jsonl';known={}
    if path.exists():
        for r in read_rows(path):known[(r['example_id'],r.get('condition','baseline'))]=r
    manifest=out/'config.json'
    if manifest.exists():
        if json.loads(manifest.read_text())!=cfg:raise ValueError('Output directory belongs to another config')
    else:create_json(manifest,cfg)
    append(out/'launches.jsonl',{'stage':args.stage,**provenance(cfg)})
    if cfg['model']=='meter':
        from mydata_bench.meter_eval.runtime import MeterRuntime
        runtime=MeterRuntime(cfg)
    elif cfg['model']=='sole':
        from mydata_bench.top_eval.runtime import SoleRuntime
        runtime=SoleRuntime(cfg)
    else:
        from .discrete_runtime import DiscreteRuntime
        runtime=DiscreteRuntime(cfg)
    if args.stage=='baseline':which='full'
    elif args.stage=='rank':which='ranking'
    else:which='cohort'
    samples=list(read_rows(BASE/f'inputs/{which}.jsonl'))
    if cfg.get('example_ids_file') and args.stage!='rank':
        allow=set(json.loads(pathlib.Path(cfg['example_ids_file']).read_text()));samples=[s for s in samples if s['example_id'] in allow]
    if args.limit:samples=samples[:args.limit]
    if args.stage in {'attention','sweep'}:
        ranking=json.loads((out/'ranking.json').read_text())['ranking']
        conditions=cfg.get('conditions')
        if conditions is None:
            conditions=[{'name':'baseline','k':0}]
            for k in [8,32,64]:
                for group in ['target','wrong','low_rank']:
                    conditions.append({'name':f'{group}_k{k}','k':k,'kind':'bias','bias':6.,'scope':cfg.get('scope','all_frames'),'region':'wrong' if group=='wrong' else 'target','head_group':'low' if group=='low_rank' else 'high','query_scope':'all'})
    else:conditions=[{'name':'ranking' if args.stage=='rank' else 'baseline','k':0}]
    started=time.time();done=0
    for i,sample in enumerate(samples):
        pending=[c for c in conditions if (sample['example_id'],c['name']) not in known or (args.retry_errors and known[(sample['example_id'],c['name'])].get('status')!='ok')]
        if not pending:continue
        try:prepared=runtime.prepare(sample,cfg['protocol'])
        except Exception as exc:
            traceback.print_exc()
            for c in pending:append(path,{'example_id':sample['example_id'],'condition':c['name'],'status':'error','phase':'prepare','error':repr(exc),'time':timestamp()})
            continue
        for c in pending:
            record={'example_id':sample['example_id'],'condition':c['name'],'condition_config':c,'video_sha256':sample['video_sha256'],'time':timestamp()}
            tick=time.time()
            try:
                if args.stage=='rank':res=runtime.rank(sample,prepared,cfg.get('scope','all_frames'))
                else:
                    heads=[]
                    if c['k']:
                        ordered=ranking if c.get('head_group','high')=='high' else list(reversed(ranking))
                        heads=[(r['layer'],r['head']) for r in ordered[:c['k']]]
                    res=runtime.predict(sample,prepared,c,heads)
                    if not np.isfinite(res['progress']):raise ValueError('Non-finite prediction')
                record.update(status='ok',**res)
            except Exception as exc:
                traceback.print_exc();record.update(status='error',phase='inference',error=repr(exc));torch.cuda.empty_cache()
            record['elapsed_seconds']=time.time()-tick;append(path,record);done+=1
        if i%10==0:print(json.dumps({'stage':args.stage,'sample':i+1,'total':len(samples),'new_conditions':done,'seconds':time.time()-started}),flush=True)
    if args.stage=='rank':
        rows={r['example_id']:r for r in read_rows(path)};ok=[r for r in rows.values() if r['status']=='ok']
        if len(ok)!=len(samples):raise RuntimeError(f'Incomplete ranking {len(ok)}/{len(samples)}')
        arr=np.mean([r[cfg.get('ranking_metric','raw_mass')] for r in ok],axis=0)
        ranked=sorted([{'layer':l,'head':h,'score':float(arr[l,h])} for l in range(cfg.get('skip_early_layers',8),arr.shape[0]) for h in range(arr.shape[1])],key=lambda r:(-r['score'],r['layer'],r['head']))
        target=out/'ranking.json'
        result={'metric':cfg.get('ranking_metric','raw_mass'),'sample_count':len(ok),'skip_early_layers':cfg.get('skip_early_layers',8),'query':'last trained progress token for meter; final prompt token for generative models','scope':cfg.get('scope','all_frames'),'ranking':ranked}
        if not target.exists():create_json(target,result)
    append(out/'completion_events.jsonl',{'stage':args.stage,'time':timestamp(),'expected_samples':len(samples),'conditions':len(conditions),'new_records':done})
    print('COMPLETED',args.stage,flush=True)
if __name__=='__main__':main()
