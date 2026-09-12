"""Offline-only metric joins. None of this module is imported by inference."""
from __future__ import annotations
import argparse,collections,datetime,json,math,pathlib
import numpy as np
from .common import *
from mydata_bench.protocol import progress_to_reward
from mydata_bench.meter_eval.readout import canonicalize

def summarize(rows,labels,expected_ids):
    by={r['example_id']:canonicalize(r) for r in rows};expected=set(expected_ids)
    valid={i:by[i] for i in expected if i in by and by[i].get('status')=='ok'}
    def group(ids):
        ids=set(ids);ok=sorted(ids&set(valid));n=len(ids)
        if not n:return {'n_expected':0,'n_valid':0}
        pp=[float(valid[i]['progress']) for i in ok];ys=[labels[i]['reward'] for i in ok]
        pred=[int(valid[i].get('native_prediction',progress_to_reward(p))) for i,p in zip(ok,pp)]
        dist=collections.Counter(pred)
        result={'n_expected':n,'n_valid':len(ok),'n_missing_or_invalid':n-len(ok),'continuous_ordinal_mae':float(np.mean([abs(1+4*p-y) for p,y in zip(pp,ys)])) if ok else None,'mae':float(np.mean([abs(p-y) for p,y in zip(pred,ys)])) if ok else None,'accuracy':sum(p==y for p,y in zip(pred,ys))/n,'prediction_distribution':{str(p):dist[p] for p in range(1,6)},'mean_progress':float(np.mean(pp)) if ok else None}
        for low,high in [(.125,.875),(.2,.8)]:
            # Lower boundary follows repository mapping: p < low => reward 1;
            # upper is inclusive, p >= high => reward 5. Intermediate != endpoint.
            correct=sum((p<low and y==1) or (p>=high and y==5) for p,y in zip(pp,ys))
            result[f'endpoint_accuracy_{low}_{high}']=correct/n
        return result
    tasks=sorted({labels[i]['subset'] for i in expected});splits=['suc','fail']
    result={'overall':group(expected),'by_split':{s:group([i for i in expected if labels[i]['split']==s]) for s in splits},'by_task':{t:group([i for i in expected if labels[i]['subset']==t]) for t in tasks},'by_task_split':{t:{s:group([i for i in expected if labels[i]['subset']==t and labels[i]['split']==s]) for s in splits} for t in tasks},'complete':len(valid)==len(expected)}
    differences=[];continuous=[];pairs_expected=0
    for i in sorted(expected):
        lab=labels[i];source=lab['source_suc_id']
        if lab['split']!='fail' or source not in expected:continue
        if labels[source]['video_sha256']!=lab['video_sha256']:raise AssertionError('Pair video mismatch')
        pairs_expected+=1
        if i not in valid or source not in valid:continue
        a=valid[source];b=valid[i]
        x=int(a.get('native_prediction',progress_to_reward(a['progress'])))-int(b.get('native_prediction',progress_to_reward(b['progress'])))
        differences.append('negative' if x<0 else str(x));continuous.append(a['progress']-b['progress'])
    dist=collections.Counter(differences)
    result['pairwise']={'expected_pairs_with_suc_in_cohort':pairs_expected,'valid_pairs':len(differences),'discrete_suc_minus_fail_counts':{k:dist[k] for k in ['negative','0','1','2','3','4']},'mean_progress_gap':float(np.mean(continuous)) if continuous else None}
    return result

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output-root',default=str(BASE));args=ap.parse_args();root=pathlib.Path(args.output_root)
 labels={r['example_id']:r for r in read_rows(BASE/'inputs/labels.jsonl')};cohort=[r['example_id'] for r in read_rows(BASE/'inputs/cohort.jsonl')];splits=json.loads((BASE/'inputs/splits.json').read_text());report={};heads={}
 for folder in sorted(root.iterdir()):
  if not folder.is_dir() or not (folder/'config.json').exists():continue
  cfg=json.loads((folder/'config.json').read_text());data={};baseline={}
  if cfg.get('pilot'):continue
  if (folder/'baseline.jsonl').exists():
   rows=list(read_rows(folder/'baseline.jsonl'));baseline={r['example_id']:r for r in rows}
   data['baseline_full']=summarize(rows,labels,labels);data['baseline_cohort']=summarize(rows,labels,cohort)
  for stage in ['attention','sweep']:
   if not (folder/f'{stage}.jsonl').exists():continue
   grouped=collections.defaultdict(list)
   for r in read_rows(folder/f'{stage}.jsonl'):grouped[r['condition']].append(r)
   data[stage]={name:summarize(rows,labels,cohort) for name,rows in grouped.items()}
  if (folder/'ranking.json').exists():
   ranked=json.loads((folder/'ranking.json').read_text())['ranking'];heads[folder.name]=[(r['layer'],r['head']) for r in ranked];data['top8_heads']=[f'L{l}H{h}' for l,h in heads[folder.name][:8]]
  report[folder.name]={'config':cfg,**data}
 overlaps={}
 for a,aa in heads.items():
  for b,bb in heads.items():
   if a>=b:continue
   overlaps[a+'__'+b]={str(k):{'intersection_count':len(set(aa[:k])&set(bb[:k])),'overlap_fraction':len(set(aa[:k])&set(bb[:k]))/k,'interpretation':'index overlap only; fine-tuning does not guarantee functional equivalence across models'} for k in [8,32,64]}
 stamp=datetime.datetime.now().strftime('%Y%m%dT%H%M%S');dest=root/f'metrics_snapshot_{stamp}.json';create_json(dest,{'time':timestamp(),'experiments':report,'ranking_overlaps':overlaps,'mae_definition':'mean absolute error on native 1–5 output; continuous models additionally report |1+4p-y| and the repository five-bin adapter MAE','accuracy_denominator':'all expected samples; invalid/missing count as inaccurate','selection':'descriptive snapshot, not confirmation acceptance'})
 print(dest)
 for name,r in report.items():
  if 'baseline_full' not in r:continue
  for group in ['baseline_full','baseline_cohort']:
   s=r[group];print(name,group,s['overall'],'suc',s['by_split']['suc'].get('accuracy'),'fail',s['by_split']['fail'].get('accuracy'))
if __name__=='__main__':main()
