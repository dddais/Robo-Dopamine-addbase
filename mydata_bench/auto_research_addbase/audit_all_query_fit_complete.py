"""Training-record diagnostics after all15 frozen F7 fits; no evaluation labels.

This audit does not select a new method, modify a ranking, or trigger inference.
"""
import hashlib
import itertools
import json
import pathlib
import time
import numpy as np
from .common import create_json, read_rows, timestamp
from .robust_prepare import DEST, digest


def gradient_digest(row):
    if row.get('gradient') is None:
        return None
    values=np.asarray(row['gradient'],dtype=np.float64)
    if values.ndim!=2 or not np.isfinite(values).all():
        raise ValueError('Invalid preserved fitting gradient')
    return hashlib.sha256(values.tobytes()).hexdigest()


def compare_case(name):
    directories={key:DEST/folder/name/'fit' for key,folder in [('F6','reward_gradient_v1'),('F7','all_query_reward_gradient_v1')]}
    done={key:json.loads((path/'completion_v1.json').read_text()) for key,path in directories.items()}
    for key,path in directories.items():
        if done[key]['examples']!=1942 or done[key]['fitting_episode_groups']!=446:
            raise ValueError('Unexpected fitting cohort size')
        for filename,field in [('gradient_records_v1.jsonl','gradients_sha256'),('ranking_v1.json','ranking_sha256')]:
            if digest(path/filename)!=done[key][field]:
                raise ValueError('Frozen '+key+' fit artifact changed')
    old={}
    for row in read_rows(directories['F6']/'gradient_records_v1.jsonl'):
        eid=row['example_id']
        if eid in old:
            raise ValueError('Duplicate F6 fit ID')
        old[eid]={field:row.get(field) for field in ['status','training_episode_group_sha256','native_logits','supervised_loss']}
        old[eid]['gradient_digest']=gradient_digest(row)
    seen=set();counts=dict(measured=0,no_grounding=0,native_logits_exact=0,loss_exact=0,gradient_changed=0);mismatches=[]
    for row in read_rows(directories['F7']/'gradient_records_v1.jsonl'):
        eid=row['example_id']
        if eid in seen or eid not in old:
            raise ValueError('Unexpected/duplicate F7 fit ID')
        seen.add(eid);reference=old[eid]
        if any(row.get(field)!=reference[field] for field in ['status','training_episode_group_sha256']):
            mismatches.append(dict(example_id=eid,reason='status/group mismatch'))
        if row['status']=='no_grounding':
            counts['no_grounding']+=1
        elif row['status']=='ok':
            counts['measured']+=1
            counts['native_logits_exact']+=row['native_logits']==reference['native_logits']
            counts['loss_exact']+=row['supervised_loss']==reference['supervised_loss']
            counts['gradient_changed']+=gradient_digest(row)!=reference['gradient_digest']
            if row['native_logits']!=reference['native_logits'] or row['supervised_loss']!=reference['supervised_loss']:
                mismatches.append(dict(example_id=eid,reason='zero-bias native logits/loss mismatch'))
        else:
            raise ValueError('Failed gradient in completed fit')
    if seen!=set(old) or len(seen)!=1942:
        raise ValueError('Incomplete/changed fit cohort')
    return dict(examples=len(seen),counts=counts,zero_forward_exact=not mismatches,mismatches=mismatches,
        f6_gradient_records_sha256=done['F6']['gradients_sha256'],f7_gradient_records_sha256=done['F7']['gradients_sha256'])


def cross_protocol_diagnostic(model):
    protocols=['image_text','text_image','text_video','video_text','interleaved']
    rankings={};hashes={}
    for protocol in protocols:
        path=DEST/f'all_query_reward_gradient_v1/{model}_{protocol}/fit/ranking_v1.json'
        done=json.loads((path.parent/'completion_v1.json').read_text())
        if digest(path)!=done['ranking_sha256']:
            raise ValueError('F7 learned ranking changed')
        rankings[protocol]={(r['layer'],r['head']):r for r in json.loads(path.read_text())['ranking']}
        hashes[protocol]=done['ranking_sha256']
    keys=sorted(rankings[protocols[0]])
    if any(set(rows)!=set(keys) for rows in rankings.values()):
        raise ValueError('Head architecture differs across same-model input protocols')
    mu=np.array([[rankings[p][h]['mean_loss_gradient'] for h in keys] for p in protocols])
    se=np.array([[rankings[p][h]['standard_error'] for h in keys] for p in protocols])
    paired={}
    for i,j in itertools.combinations(range(5),2):
        a,b=mu[i],mu[j]
        heads_a=sorted(rankings[protocols[i]],key=lambda h:(-rankings[protocols[i]][h]['score'],-abs(rankings[protocols[i]][h]['mean_loss_gradient']),*h))[:48]
        heads_b=sorted(rankings[protocols[j]],key=lambda h:(-rankings[protocols[j]][h]['score'],-abs(rankings[protocols[j]][h]['mean_loss_gradient']),*h))[:48]
        paired[protocols[i]+'__'+protocols[j]]=dict(mean_gradient_cosine=float(a@b/(np.linalg.norm(a)*np.linalg.norm(b))),
            sign_agreement=float(np.mean(np.sign(a)==np.sign(b))),top48_overlap=len(set(heads_a)&set(heads_b)))
    positive_direction_margin=(-mu-1.96*se).min(axis=0)
    negative_direction_margin=(mu-1.96*se).min(axis=0)
    best_margin=np.maximum(positive_direction_margin,negative_direction_margin)
    return dict(protocols=protocols,heads=len(keys),ranking_sha256=hashes,pairwise=paired,
        all5_same_sign_heads=int(np.all(np.sign(mu)==np.sign(mu[0:1]),axis=0).sum()),
        positive_worst_protocol_penalized_heads=int((best_margin>0).sum()),
        shared_direction_diagnostic='for each head compare sign +/-1 and take max_s min_protocol(-s*mean_gradient-1.96SE)',
        sufficient_positive_heads_for_three_fixed_k=bool((best_margin>0).sum()>=64),
        no_new_inference_ranking_or_configuration_selected=True)


def main():
    launch=json.loads((DEST/'f7_fit_queue_launch_v1.json').read_text())
    complete=DEST/'f7_all_fit_queues_completion_v1.json'
    while not complete.exists():
        process=pathlib.Path('/proc')/str(launch['pid'])/'cmdline'
        if not process.exists() or b'all_query_reward_gradient_fit_queue' not in process.read_bytes():
            raise RuntimeError('F7 fitting queue terminal without completion; do not restart it')
        time.sleep(20)
    done=json.loads(complete.read_text())
    if done['failures']:
        create_json(DEST/'f7_complete_fit_mechanism_audit_v1.json',dict(time=timestamp(),eligible=False,
            reason='Fitting failures preserved; no complete fit diagnostics',fitting_completion_sha256=digest(complete)))
        return
    plan=json.loads((DEST/'all_query_reward_gradient_fit_frozen_plan_v1.json').read_text())
    names=sorted(item['model']+'_'+item['protocol'] for item in plan['configurations'])
    if len(names)!=len(set(names)) or len(names)!=15:
        raise ValueError('Require all15 unique frozen fitting cases')
    cases={}
    for name in names:
        cases[name]=compare_case(name)
        print(name,cases[name]['counts'],'zero_forward_exact',cases[name]['zero_forward_exact'],flush=True)
    cross={model:cross_protocol_diagnostic(model) for model in ['qwen','rr','meter']}
    create_json(DEST/'f7_complete_fit_mechanism_audit_v1.json',dict(time=timestamp(),cases=cases,
        all15_complete=True,all_zero_forward_exact=all(r['zero_forward_exact'] for r in cases.values()),
        cross_protocol=cross,fitting_completion_sha256=digest(complete),
        source='frozen fit records only; no validation/test outputs or labels',
        interpretation='zero-forward provenance and training-gradient diagnostics only; no guarantee of finite-bias loss/accuracy, no simultaneous confidence claim, no new method selected',
        final_goal_completed=False))
    print('Completed full F7 fitting diagnostics',flush=True)


if __name__=='__main__':
    main()
