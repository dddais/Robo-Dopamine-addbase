"""Fixed primary-k finite-bias loss diagnostic on fit data only.

No parameter search, validation labels, old234 scores, or reserved-test reads.
"""
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
from .common import ROOT,append,create_json,read_rows,timestamp
from .robust_prepare import DEST,digest
from .all_query_reward_gradient_runtime import AllQueryRewardGradientRuntime
from .reward_gradient_attention import native_grade_loss
from .all_query_reward_gradient_validation import inference_value


def group_effects(records,expected):
    by_id={r['example_id']:r for r in records}
    if len(by_id)!=len(records) or set(by_id)!=set(expected):
        raise ValueError('Complete unique expected fitting records required')
    if any(r['status'] not in ['ok','no_grounding'] for r in records):
        return dict(complete=False,reason='Required fit diagnostic errors retained; no complete-cohort mean reported')
    grouped=collections.defaultdict(list)
    for eid,group in expected.items():
        row=by_id[eid]
        if row['training_episode_group_sha256']!=group:
            raise ValueError('Frozen fitting group changed')
        grouped[group].append(row)
    metrics={}
    for key in ['all_query','same_heads_readout']:
        values=[r['loss_deltas'][key] for r in records]
        means=[np.mean([r['loss_deltas'][key] for r in rr]) for rr in grouped.values()]
        metrics[key]=dict(row_mean_delta=float(np.mean(values)),episode_equal_mean_delta=float(np.mean(means)),
            rows_negative=sum(v<0 for v in values),rows_positive=sum(v>0 for v in values),rows_zero=sum(v==0 for v in values))
    contrast=[r['loss_deltas']['all_query']-r['loss_deltas']['same_heads_readout'] for r in records]
    grouped_contrast=[np.mean([r['loss_deltas']['all_query']-r['loss_deltas']['same_heads_readout'] for r in rr]) for rr in grouped.values()]
    return dict(complete=True,examples=len(records),episode_groups=len(grouped),effects=metrics,
        all_query_minus_matched_readout=dict(row_mean_delta=float(np.mean(contrast)),episode_equal_mean_delta=float(np.mean(grouped_contrast))))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    cfg=yaml.safe_load(pathlib.Path(args.config).read_text())
    for p,sha in cfg['source_sha256'].items():
        if digest(ROOT/p)!=sha:
            raise ValueError('Frozen diagnostic source changed: '+p)
    for field in ['training_input','fit_ids','fit_labels','ranking','f6_ranking','f7_fit_records']:
        if digest(cfg[field+'_file'])!=cfg[field+'_sha256']:
            raise ValueError('Frozen fit diagnostic dependency changed: '+field)
    ids=json.loads(pathlib.Path(cfg['fit_ids_file']).read_text())
    samples=[r for r in read_rows(cfg['training_input_file']) if r['example_id'] in set(ids)]
    labels={r['example_id']:r['reward'] for r in read_rows(cfg['fit_labels_file'])}
    if len(samples)!=len(ids) or len(set(ids))!=1942 or set(labels)!=set(ids) or any(r['training_partition']!='fit' for r in samples):
        raise ValueError('Require complete fitting-only1942; no validation/test substitution')
    cached={r['example_id']:{key:r.get(key) for key in ['native_logits','tracking_alignment']} for r in read_rows(cfg['f7_fit_records_file'])}
    ranking=json.loads(pathlib.Path(cfg['ranking_file']).read_text())['ranking'];selected=ranking[:48]
    heads=[(r['layer'],r['head']) for r in selected]
    previous={(r['layer'],r['head']):r for r in json.loads(pathlib.Path(cfg['f6_ranking_file']).read_text())['ranking']}
    slopes=dict(all_query=sum(r['mean_loss_gradient']*r['signed_bias_direction'] for r in selected),
        same_heads_readout=sum(previous[(r['layer'],r['head'])]['mean_loss_gradient']*r['signed_bias_direction'] for r in selected))
    out=pathlib.Path(cfg['output_dir']);out.mkdir(parents=True,exist_ok=True)
    lock=(out/'writer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    create_json(out/'config.json',cfg)
    create_json(out/'launch_v1.json',dict(time=timestamp(),pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),config_sha256=digest(args.config),
        fit_labels_used=True,validation_and_test_labels_used=False,parameter_search=False))
    runtime=AllQueryRewardGradientRuntime(cfg);runtime.runtime.model.requires_grad_(False)
    conditions=dict(baseline=dict(name='baseline',k=0),
        all_query=dict(name='all_query',k=48,kind='reward_gradient',bias=1.,signed=True,scope='all_frames',query_scope='all'),
        same_heads_readout=dict(name='same_heads_readout',k=48,kind='reward_gradient',bias=1.,signed=True,scope='all_frames',query_scope='readout'))
    start=time.monotonic();records=[];errors=0
    for index,sample in enumerate(samples,1):
        row=dict(time=timestamp(),example_id=sample['example_id'],training_episode_group_sha256=sample['training_episode_group_sha256'])
        fallback=sample['grounding_status']!='ok'
        try:
            prepared=runtime.prepare(sample);outputs={};losses={};baseline=None
            for key,condition in conditions.items():
                value=inference_value(runtime,sample,prepared,condition,[] if key=='baseline' else heads,baseline)
                if key=='baseline':
                    baseline=value
                outputs[key]=value
                losses[key]=float(native_grade_loss(torch.tensor(value['score_logits'],device=prepared['inputs']['input_ids'].device),labels[sample['example_id']],cfg['model']))
            if fallback:
                if not all(outputs[key].get('baseline_fallback') and outputs[key]['score_logits']==outputs['baseline']['score_logits'] for key in ['all_query','same_heads_readout']):
                    raise ValueError('Actual noROI baseline fallback mismatch')
            else:
                if not outputs['all_query']['hook_diagnostics'].get('all_causal_queries') or not outputs['same_heads_readout']['hook_diagnostics'].get('only_readout_query_changed'):
                    raise ValueError('Actual query-scope diagnostics mismatch')
                if outputs['all_query']['tracking_alignment']!=cached[sample['example_id']]['tracking_alignment']:
                    raise ValueError('Actual ROI alignment differs from frozen fitting record')
            if not all(np.isfinite(v) for v in losses.values()):
                raise ValueError('Nonfinite measured fitting loss')
            row.update(status='no_grounding' if fallback else 'ok',losses=losses,
                loss_deltas={key:losses[key]-losses['baseline'] for key in ['all_query','same_heads_readout']},
                actual_baseline_measured=True,actual_frozen_noROI_fallback_applied=fallback,
                baseline_logits_match_frozen_fit=None if fallback else outputs['baseline']['score_logits']==cached[sample['example_id']]['native_logits'],
                score_logits={key:value['score_logits'] for key,value in outputs.items()},
                hook_diagnostics={key:value.get('hook_diagnostics') for key,value in outputs.items()})
        except Exception as exc:
            traceback.print_exc();row.update(status='error',error=repr(exc));errors+=1;torch.cuda.empty_cache()
        append(out/'records_v1.jsonl',row);records.append(row)
        if index%25==0 or index==len(samples):
            print(json.dumps(dict(time=timestamp(),processed=index,expected=len(samples),errors=errors,elapsed_seconds=time.monotonic()-start)),flush=True)
    expected={r['example_id']:r['training_episode_group_sha256'] for r in samples}
    summary=group_effects(records,expected)
    create_json(out/'completion_v1.json',dict(time=timestamp(),summary=summary,errors=errors,records_sha256=digest(out/'records_v1.jsonl'),
        measured_rows=sum(r['status']=='ok' for r in records),noROI_actual_baseline_fallback_rows=sum(r['status']=='no_grounding' for r in records),
        actual_baseline_logits_match_cached_fit=sum(r.get('baseline_logits_match_frozen_fit') is True for r in records),
        matched_heads_first_order_slopes=slopes,
        interpretation='fit-only finite b1 CE changes at fixed k48. Matched heads/directions isolate query scope on fit data; no claim about held-out reward metrics or original+10pp goal.',
        no_hyperparameters_or_rankings_changed=True,validation_and_test_performance_not_opened=True,final_goal_completed=False))
    if errors:
        raise RuntimeError('Fitting diagnostic errors preserved; no silent retry')


if __name__=='__main__':
    main()
