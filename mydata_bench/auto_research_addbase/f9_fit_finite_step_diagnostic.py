"""Fixed Qwen training-only F9 finite intervention; all earlier native forwards retained."""
import argparse
import collections
import fcntl
import json
import os
import pathlib
import time
import traceback
import numpy as np
import torch
import yaml
from .common import ROOT, append, create_json, output_path, read_rows, timestamp
from .robust_prepare import digest
from .all_query_reward_gradient_runtime import AllQueryRewardGradientRuntime
from .all_query_reward_gradient_validation import inference_value
from .native_cumulative_ordinal_loss import native_cumulative_ordinal_loss
from .reward_gradient_attention import native_grade_loss
from .score_branches import score_readout


def measured(logits, grade):
    tensor=torch.tensor(logits,dtype=torch.float32,device='cpu')
    prediction=score_readout(logits,'qwen')['native_prediction']
    return dict(cumulative_ordinal_loss=float(native_cumulative_ordinal_loss(tensor,grade,'qwen')),
        categorical_CE=float(native_grade_loss(tensor,grade,'qwen')),native_prediction=prediction,
        native_accuracy=float(prediction==grade),native_MAE=float(abs(prediction-grade)))


def aggregate(records, expected_ids):
    by_id={r['example_id']:r for r in records}
    if len(records)!=len(by_id) or set(by_id)!=set(expected_ids) or any(r['status']=='error' for r in records):
        return dict(complete=False,reason='Missing/duplicate/failed required training outputs; no incomplete mean claimed')
    keys=['cumulative_ordinal_loss','categorical_CE','native_accuracy','native_MAE']
    result={}
    for branch in ['baseline','F7','F8','F9']:
        groups=collections.defaultdict(list)
        for row in records:
            groups[row['training_episode_group_sha256']].append(row['metrics'][branch])
        result[branch]=dict(row_mean={k:float(np.mean([r['metrics'][branch][k] for r in records])) for k in keys},
            episode_equal_mean={k:float(np.mean([np.mean([r[k] for r in values]) for values in groups.values()])) for k in keys},
            by_grade={str(g):dict(n=sum(r['grade']==g for r in records),
                **{k:(float(np.mean([r['metrics'][branch][k] for r in records if r['grade']==g])) if any(r['grade']==g for r in records) else None) for k in keys}) for g in [1,2,3,4,5]})
    return dict(complete=True,examples=len(records),episode_groups=len(groups),branches=result)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    cfg=yaml.safe_load(pathlib.Path(args.config).read_text())
    if cfg['model']!='qwen' or cfg['protocol']!='text_video':
        raise ValueError('This prospectively selected diagnostic is only Qwen text_video')
    if cfg['primary_k']!=48 or cfg['inference_bias_magnitude']!=1. or cfg['k_neighborhood']!=[32,48,64]:
        raise ValueError('Only the unchanged F9 b1/k48 primary diagnostic is authorized')
    for path,expected in cfg['source_sha256'].items():
        if digest(ROOT/path)!=expected:
            raise ValueError('Frozen diagnostic source changed: '+path)
    for field in ['training_input','fit_ids','fit_labels','ranking','f8_reference_records','f7_ranking','f8_ranking','f8_fit_records']:
        if digest(cfg[field+'_file'])!=cfg[field+'_sha256']:
            raise ValueError('Frozen diagnostic input changed: '+field)
    ids=json.loads(pathlib.Path(cfg['fit_ids_file']).read_text())
    samples=[r for r in read_rows(cfg['training_input_file']) if r['example_id'] in set(ids)]
    labels={r['example_id']:r['reward'] for r in read_rows(cfg['fit_labels_file'])}
    if len(samples)!=1942 or len(set(ids))!=1942 or {r['example_id'] for r in samples}!=set(ids) or set(labels)!=set(ids):
        raise ValueError('Complete fitting-only1942 required')
    if any(r['training_partition']!='fit' for r in samples):
        raise ValueError('Only fitting data allowed')
    cached_rows=list(read_rows(cfg['f8_reference_records_file']));cache={r['example_id']:r for r in cached_rows}
    fitted={r['example_id']:r for r in read_rows(cfg['f8_fit_records_file'])}
    if len(cached_rows)!=len(cache) or set(cache)!=set(ids) or set(fitted)!=set(ids):
        raise ValueError('Reference records must cover entire fitting cohort')
    ranking=json.loads(pathlib.Path(cfg['ranking_file']).read_text())['ranking']
    f7_ranking=json.loads(pathlib.Path(cfg['f7_ranking_file']).read_text())['ranking']
    f8_ranking=json.loads(pathlib.Path(cfg['f8_ranking_file']).read_text())['ranking']
    gradients={(r['layer'],r['head']):r['mean_loss_gradient'] for r in ranking}
    slopes={branch:sum(gradients[(r['layer'],r['head'])]*r['signed_bias_direction'] for r in order[:48])
            for branch,order in [('F7',f7_ranking),('F8',f8_ranking),('F9',ranking)]}
    out=output_path(cfg['output_dir']);out.mkdir(parents=True,exist_ok=True)
    lock=(out/'writer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    create_json(out/'launch_v1.json',dict(time=timestamp(),pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        config_sha256=digest(args.config),fit_only=True,fixed_b1_k48=True))
    runtime=AllQueryRewardGradientRuntime(cfg);runtime.runtime.model.requires_grad_(False)
    heads=[(r['layer'],r['head']) for r in ranking[:48]]
    condition=dict(name='reward_gradient_k48',kind='reward_gradient',bias=1.,k=48,scope='all_frames',signed=True)
    rows=[];errors=0;begin=time.monotonic()
    for index,sample in enumerate(samples,1):
        eid=sample['example_id'];grade=labels[eid]
        row=dict(time=timestamp(),example_id=eid,grade=grade,training_episode_group_sha256=sample['training_episode_group_sha256'])
        try:
            prepared=runtime.prepare(sample)
            baseline=inference_value(runtime,sample,prepared,dict(name='baseline',k=0),[],None)
            candidate=inference_value(runtime,sample,prepared,condition,heads,baseline)
            if cache[eid]['status'] not in ['ok','no_grounding'] or baseline['score_logits']!=cache[eid]['score_logits']['baseline']:
                raise ValueError('Actual baseline does not exactly reproduce frozen F8 finite-step baseline')
            no_roi=sample['grounding_status']!='ok'
            if no_roi:
                if not candidate.get('baseline_fallback') or candidate['score_logits']!=baseline['score_logits']:
                    raise ValueError('Actual frozen no-ROI fallback changed')
            elif not candidate['hook_diagnostics'].get('all_causal_queries') or candidate['tracking_alignment']!=fitted[eid]['tracking_alignment']:
                raise ValueError('All-query intervention/localization differs from F8 fitting')
            logits=dict(baseline=baseline['score_logits'],F7=cache[eid]['score_logits']['F7'],F8=cache[eid]['score_logits']['F8'],F9=candidate['score_logits'])
            row.update(status='no_grounding' if no_roi else 'ok',score_logits=logits,
                metrics={key:measured(value,grade) for key,value in logits.items()},
                baseline_exact_to_f8_diagnostic=True,actual_baseline_measured=True,actual_noROI_fallback_applied=no_roi,
                hook_diagnostics=candidate.get('hook_diagnostics'),tracking_alignment=candidate.get('tracking_alignment'))
        except Exception as exc:
            traceback.print_exc();row.update(status='error',error=repr(exc));errors+=1;torch.cuda.empty_cache()
        rows.append(row);append(out/'records_v1.jsonl',row)
        if index%25==0 or index==len(samples):
            print(json.dumps(dict(time=timestamp(),processed=index,expected=len(samples),errors=errors,elapsed_seconds=time.monotonic()-begin)),flush=True)
    create_json(out/'completion_v1.json',dict(time=timestamp(),summary=aggregate(rows,ids),errors=errors,
        records_sha256=digest(out/'records_v1.jsonl'),ordinal_objective_first_order_row_slopes=slopes,
        all1942_actual_baseline_measured=all(r.get('actual_baseline_measured') for r in rows),
        interpretation='pre-frozen fit-only comparison of F9, F8 and F7 at same b1/k48; common sample-mean ordinal gradient; no efficacy/generalization gate',
        no_hyperparameters_or_rankings_changed=True,validation_and_test_not_read=True,final_goal_completed=False))
    if errors:
        raise ValueError('Diagnostic errors preserved; no silent retry or incomplete fit result')


if __name__=='__main__':
    main()
