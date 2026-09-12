"""Offline paired effects with video-cluster uncertainty; no inference imports.

Resample complete video-content groups to retain suc/fail instruction pairs.
Missing or invalid predictions prevent a full-cohort acceptance claim. Controls
can be inspected on explicitly enumerated feasible subsets, never silently.
"""
import argparse,collections,json,pathlib
import numpy as np
from .common import BASE,create_json,read_rows,timestamp
from mydata_bench.meter_eval.readout import canonicalize
from mydata_bench.protocol import progress_to_reward


def latest(path,condition):
    return {r['example_id']:canonicalize(r) for r in read_rows(path) if r.get('condition')==condition}


def paired(candidate,baseline,labels,expected_ids,continuous=False,draws=20000,seed=20260908):
    expected=sorted(set(expected_ids))
    valid=[i for i in expected if candidate.get(i,{}).get('status')=='ok' and baseline.get(i,{}).get('status')=='ok']
    excluded=[{'example_id':i,'candidate':candidate.get(i,{}).get('error','missing' if i not in candidate else candidate[i].get('status')),'baseline':baseline.get(i,{}).get('error','missing' if i not in baseline else baseline[i].get('status'))} for i in expected if i not in set(valid)]
    if not valid:return {'complete':False,'n_expected':len(expected),'n_paired':0,'excluded':excluded}
    group_ids=sorted({labels[i]['video_sha256'] for i in valid});group_index={g:j for j,g in enumerate(group_ids)}
    groups=np.array([group_index[labels[i]['video_sha256']] for i in valid])
    y=np.array([labels[i]['reward'] for i in valid]);suc=y==5;fail=y==1
    def predictions(rows):
        pp=np.array([rows[i]['progress'] for i in valid],dtype=float)
        discrete=np.array([rows[i].get('native_prediction',progress_to_reward(float(p))) for i,p in zip(valid,pp)])
        return pp,discrete
    ca,cd=predictions(candidate);ba,bd=predictions(baseline)
    loss_c=np.abs((1+4*ca if continuous else cd)-y);loss_b=np.abs((1+4*ba if continuous else bd)-y)
    measures={'mae':(loss_c-loss_b,np.ones(len(valid),dtype=bool),-1,float(loss_b.mean())),
              'accuracy':((cd==y).astype(float)-(bd==y),np.ones(len(valid),dtype=bool),1,float((bd==y).mean())),
              'suc_accuracy':((cd==y).astype(float)-(bd==y),suc,1,float((bd[suc]==y[suc]).mean()) if suc.any() else None),
              'fail_accuracy':((cd==y).astype(float)-(bd==y),fail,1,float((bd[fail]==y[fail]).mean()) if fail.any() else None)}
    for low,high in [(.125,.875),(.2,.8)]:
        cc=((ca<low)&fail)|((ca>=high)&suc);bc=((ba<low)&fail)|((ba>=high)&suc)
        for split,mask in [('all',np.ones(len(valid),dtype=bool)),('suc',suc),('fail',fail)]:
            measures[f'endpoint_{low}_{high}_{split}']=(cc.astype(float)-bc,mask,1,float(bc[mask].mean()) if mask.any() else None)
    rng=np.random.default_rng(seed);g=len(group_ids)
    weights=rng.multinomial(g,np.full(g,1/g),size=draws)
    signs=rng.choice([-1,1],size=(draws,g))
    effects={}
    for name,(difference,mask,direction,base) in measures.items():
        if not mask.any():effects[name]={'n':0};continue
        sums=np.bincount(groups,weights=difference*mask,minlength=g)
        counts=np.bincount(groups,weights=mask.astype(float),minlength=g)
        denominators=weights@counts;good=denominators>0
        bootstrap=(weights[good]@sums)/denominators[good]
        delta=float(difference[mask].mean());observed=float(direction*sums.sum())
        random_null=direction*(signs@sums)
        effects[name]={'n':int(mask.sum()),'baseline':base,'candidate':base+delta,'delta_candidate_minus_baseline':delta,
                       'relative_improvement':direction*delta/base if base not in (None,0) else None,
                       'ci95_video_cluster_percentile':np.quantile(bootstrap,[.025,.975]).tolist(),
                       'one_sided_cluster_sign_flip_p':float((1+(random_null>=observed-1e-12).sum())/(draws+1)),
                       'bootstrap_draws_with_nonzero_denominator':int(good.sum())}
    core=['mae','accuracy','suc_accuracy','fail_accuracy']
    complete=len(valid)==len(expected)
    directions=effects['mae']['delta_candidate_minus_baseline']<0 and all(effects[k].get('delta_candidate_minus_baseline',0)>0 for k in core[1:])
    gate=complete and directions and effects['accuracy']['delta_candidate_minus_baseline']>=.1-1e-12
    per_task={}
    for task in sorted({labels[i]['subset'] for i in valid}):
        mask=np.array([labels[i]['subset']==task for i in valid]);per_task[task]={}
        for name in core:
            difference,scope,_,_=measures[name];m=mask&scope
            per_task[task][name]={'n':int(m.sum()),'delta':float(difference[m].mean()) if m.any() else None}
    return {'complete':complete,'n_expected':len(expected),'n_paired':len(valid),'n_video_clusters':g,
            'excluded':excluded,'effects':effects,'per_task':per_task,'point_estimate_10pp_gate':gate,
            'joint_direction_intersection_union_p':max(effects[k].get('one_sided_cluster_sign_flip_p',1) for k in core),
            'inference_notes':'Paired video-content-cluster percentile bootstrap and one-sided cluster label-swap/sign-flip test. Core p tests directional improvement, not a 10pp lower bound. Exploratory selection is not removed by these intervals.','seed':seed,'draws':draws}


def holm(pvalues):
    ordered=sorted(pvalues,key=pvalues.get);result={};previous=0.
    for rank,key in enumerate(ordered):
        previous=max(previous,min(1.,pvalues[key]*(len(ordered)-rank)));result[key]=previous
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);p.add_argument('--file',default='sweep.jsonl');p.add_argument('--baseline',default='baseline');p.add_argument('--conditions',required=True);p.add_argument('--draws',type=int,default=20000);p.add_argument('--output',required=True);a=p.parse_args()
    folder=pathlib.Path(a.folder);cfg=json.loads((folder/'config.json').read_text())
    ids=json.loads(pathlib.Path(cfg['example_ids_file']).read_text()) if cfg.get('example_ids_file') else [r['example_id'] for r in read_rows(BASE/'inputs/cohort.jsonl')]
    labels={r['example_id']:r for r in read_rows(BASE/'inputs/labels.jsonl')};baseline=latest(folder/a.file,a.baseline)
    contrasts={c:paired(latest(folder/a.file,c),baseline,labels,ids,cfg['model'] in {'meter','sole'},a.draws) for c in a.conditions.split(',')}
    ps={name:r['joint_direction_intersection_union_p'] for name,r in contrasts.items() if 'joint_direction_intersection_union_p' in r}
    create_json(a.output,{'time':timestamp(),'phase':cfg.get('stage','descriptive; assess split provenance'),'folder':str(folder),'baseline':a.baseline,'contrasts':contrasts,'holm_within_requested_contrasts':holm(ps)})
    for c,r in contrasts.items():print(c,'paired',r['n_paired'],'gate',r.get('point_estimate_10pp_gate'),{k:(round(v['delta_candidate_minus_baseline'],5),v['ci95_video_cluster_percentile']) for k,v in r.get('effects',{}).items() if k in ['mae','accuracy','suc_accuracy','fail_accuracy']})
if __name__=='__main__':main()
