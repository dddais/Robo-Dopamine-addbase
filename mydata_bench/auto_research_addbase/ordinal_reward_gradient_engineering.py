"""F8 ordinal-loss all-query engineering; synthetic grade3 only, no fitting."""
import argparse
import json
import os
import pathlib
import time
import torch
import yaml
from .common import ROOT, create_json, read_rows, timestamp
from .robust_prepare import DEST, digest
from .score_branches import ScoreRuntime
from .native_cumulative_ordinal_loss import native_cumulative_ordinal_loss as native_grade_loss
from .all_query_reward_gradient_attention import all_query_reward_gradient_steering as reward_gradient_steering


def raw_logits(runtime, prepared):
    output = runtime.runtime.model(**prepared['inputs'], **({} if runtime.config['model']=='meter' else {'use_cache':False}))
    if runtime.config['model']=='meter':
        return output['progress_logits'][-1].float()
    return output.logits[0, -1, runtime.choice_ids].float()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['rr','qwen','meter'], required=True)
    p.add_argument('--protocol', required=True)
    args = p.parse_args()
    name = args.model+'_'+args.protocol
    cfg_path = ROOT / f'mydata_bench/configs/v2_crossmodel/addbase_robust_frame_transport_v1_{name}.yaml'
    cfg = yaml.safe_load(cfg_path.read_text())
    out = DEST/'ordinal_reward_gradient_engineering_v1'/name
    out.mkdir(parents=True, exist_ok=True)
    create_json(out/'launch_v1.json', dict(time=timestamp(), pid=os.getpid(), gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        source_sha256=digest(__file__), kernel_sha256=digest(ROOT/'mydata_bench/auto_research_addbase/all_query_reward_gradient_attention.py'),
        loss_source_sha256=digest(ROOT/'mydata_bench/auto_research_addbase/native_cumulative_ordinal_loss.py'),
        config_sha256=digest(cfg_path), labels_read=False, synthetic_probe_grade=3,
        purpose='gradient/zero/readout plumbing only; grade3 is synthetic, not a supplied annotation; no fitted head ranking'))
    ids = set(json.loads(pathlib.Path(cfg['engineering_ids_file']).read_text()))
    samples = [r for r in read_rows(cfg['input_file']) if r['example_id'] in ids]
    baseline_path = DEST/f'frame_transport_v1/{name}/development/predictions.jsonl'
    baselines = {r['example_id']:r for r in read_rows(baseline_path) if r['example_id'] in ids and r['condition']=='baseline'}
    runtime = ScoreRuntime(cfg)
    runtime.runtime.model.requires_grad_(False)
    layers = runtime.runtime.layers
    n_heads = runtime.runtime.model.config.text_config.num_attention_heads
    active = [(layer, head) for layer in range(8,len(layers)) for head in range(n_heads)]
    checks = []
    for sample in samples:
        begin = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        prepared = runtime.prepare(sample)
        target, _ = runtime.runtime.positions(sample, prepared, 'all_frames')
        visual = [i for span in prepared['spans'] for i in range(span.start,span.end)]
        query = prepared.get('score_query_index', prepared['query_index'])
        with torch.no_grad():
            baseline = raw_logits(runtime, prepared)
            zero_bias = torch.zeros(len(layers), n_heads, device=baseline.device)
            with reward_gradient_steering(layers, zero_bias, target, visual, query, active):
                zero = raw_logits(runtime, prepared)
        biases = torch.zeros(len(layers), n_heads, device=baseline.device, requires_grad=True)
        diag = {}
        with reward_gradient_steering(layers, biases, target, visual, query, active, diag):
            grad_logits = raw_logits(runtime, prepared)
            loss = native_grade_loss(grad_logits, 3, args.model)
            gradient, = torch.autograd.grad(loss, biases)
        result = dict(example_id=sample['example_id'], input_tokens=prepared['inputs']['input_ids'].shape[1],
            query=query, zero_inference_exact=torch.equal(baseline, zero),
            original_cached_logits_exact=baseline.cpu().tolist()==baselines[sample['example_id']].get('score_logits'),
            gradient_forward_zero_max_difference=float((grad_logits.detach()-baseline).abs().max()),
            finite_gradient=bool(torch.isfinite(gradient).all()), nonzero_gradient_count=int(torch.count_nonzero(gradient)),
            first8_gradients_exact_zero=bool(torch.count_nonzero(gradient[:8])==0),
            backbone_parameters_frozen=all(not v.requires_grad and v.grad is None for v in runtime.runtime.model.parameters()),
            hook_calls=diag['calls'], synthetic_grade3_loss=float(loss.detach()),
            maximum_absolute_gradient=float(gradient.abs().max()), peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            elapsed_seconds=time.monotonic()-begin, gradient=gradient.detach().cpu().tolist())
        checks.append(result)
        print(json.dumps({k:v for k,v in result.items() if k!='gradient'}), flush=True)
        del grad_logits, loss, gradient, biases
    passed = len(checks)==len(ids) and all(c['zero_inference_exact'] and c['original_cached_logits_exact']
        and c['gradient_forward_zero_max_difference']==0 and c['finite_gradient'] and c['nonzero_gradient_count']>0 and c['first8_gradients_exact_zero']
        and c['backbone_parameters_frozen'] and c['hook_calls']==len(layers)-8 for c in checks)
    create_json(out/'engineering_audit_v1.json', dict(time=timestamp(), passed=passed, checks=checks,
        synthetic_probe_not_training=True, labels_read=False, external_test_performance_not_opened=True))
    if not passed:
        raise ValueError('Gradient engineering failed; preserve and inspect')


if __name__ == '__main__':
    main()
