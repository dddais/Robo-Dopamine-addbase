"""F8 fit: signed cumulative ordinal-loss gradients at all causal queries."""
import argparse
import fcntl
import json
import os
import pathlib
import time
import traceback
import torch
import yaml
from .common import ROOT, append, create_json, output_path, read_rows, timestamp
from .robust_prepare import digest
from .score_branches import ScoreRuntime
from .native_cumulative_ordinal_loss import native_cumulative_ordinal_loss as native_grade_loss
from .all_query_reward_gradient_attention import all_query_reward_gradient_steering as reward_gradient_steering
from .reward_gradient_engineering import raw_logits
from .reward_gradient_ranking import rank_reward_gradients


def sample_gradient(runtime, sample, grade, shape):
    prepared = runtime.prepare(sample)
    target, alignment = runtime.runtime.positions(sample, prepared, 'all_frames')
    visual = [i for span in prepared['spans'] for i in range(span.start, span.end)]
    query = prepared.get('score_query_index', prepared['query_index'])
    active = [(l,h) for l in range(8,shape[0]) for h in range(shape[1])]
    biases = torch.zeros(shape, device=prepared['inputs']['input_ids'].device, requires_grad=True)
    diag = {}
    with reward_gradient_steering(runtime.runtime.layers,biases,target,visual,query,active,diag):
        logits = raw_logits(runtime,prepared)
        loss = native_grade_loss(logits,grade,runtime.config['model'])
        gradient, = torch.autograd.grad(loss,biases)
    if not torch.isfinite(gradient).all() or torch.count_nonzero(gradient[:8]) or diag['calls']!=shape[0]-8:
        raise ValueError('Invalid measured training gradient')
    return dict(status='ok',gradient=gradient.detach().cpu().tolist(),native_logits=logits.detach().cpu().tolist(),
                supervised_loss=float(loss.detach()),hook_diagnostics=diag,tracking_alignment=alignment,
                input_tokens=prepared['inputs']['input_ids'].shape[1],native_query=query)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    args=p.parse_args()
    cfg=yaml.safe_load(pathlib.Path(args.config).read_text())
    for name, expected in cfg['source_sha256'].items():
        if digest(ROOT/name)!=expected:
            raise ValueError('Frozen F8 source changed: '+name)
    for field in ['training_input','fit_labels','fit_ids']:
        if digest(cfg[field+'_file'])!=cfg[field+'_sha256']:
            raise ValueError('Frozen fitting dependency changed: '+field)
    out=output_path(cfg['output_dir'])/'fit'
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'writer.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    create_json(out/'launch_v1.json',dict(time=timestamp(),pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        config_sha256=digest(args.config),source_sha256=cfg['source_sha256'],external_fit_supervised=True,
        old_development_and_reserved_test_labels_read=False))
    ids=set(json.loads(pathlib.Path(cfg['fit_ids_file']).read_text()))
    samples=[r for r in read_rows(cfg['training_input_file']) if r['example_id'] in ids]
    labels={r['example_id']:r['reward'] for r in read_rows(cfg['fit_labels_file'])}
    if len(samples)!=len(ids) or set(labels)!=ids or any(r['training_partition']!='fit' or not r['training_eligible'] for r in samples):
        raise ValueError('Fitting data do not match frozen fitting-only cohort')
    runtime=ScoreRuntime(cfg)
    runtime.runtime.model.requires_grad_(False)
    shape=(len(runtime.runtime.layers),runtime.runtime.model.config.text_config.num_attention_heads)
    records=[]
    errors=0
    begin=time.monotonic()
    for index,sample in enumerate(samples,1):
        row=dict(time=timestamp(),example_id=sample['example_id'],training_episode_group_sha256=sample['training_episode_group_sha256'])
        if sample['grounding_status']!='ok':
            row.update(status='no_grounding',gradient=None,reason=sample.get('grounding_fallback_reason'),
                interpretation='method falls back to baseline; derivative with respect to ROI steering is zero')
        else:
            try:
                row.update(sample_gradient(runtime,sample,labels[sample['example_id']],shape))
            except Exception as exc:
                traceback.print_exc()
                row.update(status='error',error=repr(exc))
                errors+=1
                torch.cuda.empty_cache()
        append(out/'gradient_records_v1.jsonl',row)
        records.append(row)
        if index%25==0 or index==len(samples):
            print(json.dumps(dict(time=timestamp(),processed=index,expected=len(samples),errors=errors,elapsed_seconds=time.monotonic()-begin)),flush=True)
    if errors:
        create_json(out/'fit_failure_v1.json',dict(time=timestamp(),errors=errors,all_expected_records_saved=True))
        raise ValueError('Required fitting gradients failed; no head ranking selected')
    expected_groups={r['example_id']:r['training_episode_group_sha256'] for r in samples}
    ranking=rank_reward_gradients(records,expected_groups,shape)
    create_json(out/'ranking_v1.json',ranking)
    create_json(out/'completion_v1.json',dict(time=timestamp(),examples=len(samples),fitting_episode_groups=ranking['fitting_episode_groups'],
        gradients_sha256=digest(out/'gradient_records_v1.jsonl'),ranking_sha256=digest(out/'ranking_v1.json'),
        validation_and_test_performance_not_evaluated=True,backbone_weights_frozen=True))


if __name__=='__main__':
    main()
