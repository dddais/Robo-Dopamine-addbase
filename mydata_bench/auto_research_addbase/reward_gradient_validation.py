"""Label-free F6 inference on the separately frozen public train-validation rows."""
import argparse
import fcntl
import json
import os
import pathlib
import time
import traceback
import yaml
from .common import ROOT, append, create_json, output_path, read_rows, timestamp
from .robust_prepare import digest
from .reward_gradient_runtime import RewardGradientRuntime


def inference_value(runtime, sample, prepared, condition, heads, baseline):
    # Grounding absence is a baseline fallback, never a score or exclusion rule.
    if condition.get('k', 0) and sample['grounding_status'] != 'ok':
        if baseline is None:
            raise ValueError('No valid baseline available for grounding fallback')
        return dict(baseline, baseline_fallback=True, fallback_reason=sample.get('grounding_fallback_reason'),
                    hook_diagnostics={}, tracking_alignment=[])
    return dict(runtime.predict(sample, prepared, condition, heads), baseline_fallback=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--stage', choices=['engineering', 'validation'], required=True)
    args = p.parse_args()
    cfg = yaml.safe_load(pathlib.Path(args.config).read_text())
    for path, expected in cfg['source_sha256'].items():
        if digest(ROOT/path) != expected:
            raise ValueError('Frozen evaluation source changed: '+path)
    for field in ['input', 'ranking', 'original_ranking', 'example_ids', 'engineering_ids', 'tracking_manifest']:
        if digest(cfg[field+'_file']) != cfg[field+'_sha256']:
            raise ValueError('Frozen inference dependency changed: '+field)
    ids = json.loads(pathlib.Path(cfg['engineering_ids_file'] if args.stage == 'engineering' else cfg['example_ids_file']).read_text())
    samples = [s for s in read_rows(cfg['input_file']) if s['example_id'] in set(ids)]
    if len(ids) != len(set(ids)) or {s['example_id'] for s in samples} != set(ids) or len(samples) != len(ids):
        raise ValueError('Unknown/duplicate frozen validation IDs')
    if any(s['training_partition'] != 'validation' or not s['training_eligible'] or 'reward' in s for s in samples):
        raise ValueError('Validation input scope/label boundary violation')
    tracks = json.loads(pathlib.Path(cfg['tracking_manifest_file']).read_text())
    for sample in samples:
        if sample['grounding_status'] == 'ok':
            path = sample['tracking_path']
            if digest(path) != tracks[path]:
                raise ValueError('Frozen localization changed')
            mapped = json.loads(pathlib.Path(path).read_text())['frames']
            if {int(r['frame_index']) for r in mapped if r.get('bbox') is not None} != set(sample['image_source_indices']):
                raise ValueError('Grounding does not cover exactly the displayed source frames')
    out = output_path(cfg['output_dir'])/args.stage
    out.mkdir(parents=True, exist_ok=True)
    lock = (out/'writer.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.stage == 'validation':
        if not json.loads((out.parent/'engineering/engineering_audit_v1.json').read_text())['passed']:
            raise ValueError('Actual validation-media engineering gate failed')
    create_json(out/'config.json', cfg)
    create_json(out/'launch_v1.json', dict(time=timestamp(), pid=os.getpid(), gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        config_sha256=digest(args.config), stage=args.stage, labels_read=False, reserved_test_evaluated=False))
    orders = {'reward_gradient': json.loads(pathlib.Path(cfg['ranking_file']).read_text())['ranking'],
              'original': json.loads(pathlib.Path(cfg['original_ranking_file']).read_text())['ranking']}
    conditions = cfg['conditions']
    if args.stage == 'engineering':
        names = ['baseline', 'native_generation_baseline', 'original_all_k48', 'reward_gradient_k48', 'reward_gradient_zero_k48']
        conditions = [c for c in conditions if c['name'] in names]
    if conditions[0]['name'] != 'baseline':
        raise ValueError('Baseline must precede fallback conditions')
    runtime = RewardGradientRuntime(cfg)
    runtime.runtime.model.requires_grad_(False)
    begin, errors, audits = time.monotonic(), 0, []
    for index, sample in enumerate(samples, 1):
        current, baseline = {}, None
        try:
            prepared = runtime.prepare(sample)
        except Exception as exc:
            traceback.print_exc()
            for condition in conditions:
                append(out/'predictions.jsonl', dict(time=timestamp(), example_id=sample['example_id'],
                    condition=condition['name'], status='error', phase='prepare', error=repr(exc)))
                errors += 1
            continue
        for condition in conditions:
            tick = time.monotonic()
            row = dict(time=timestamp(), example_id=sample['example_id'], video_sha256=sample['video_sha256'],
                condition=condition['name'], condition_config=condition, grounding_status=sample['grounding_status'])
            order = orders[condition.get('ranking', 'reward_gradient')]
            if condition.get('head_group') == 'low':
                order = list(reversed(order))
            heads = [(r['layer'], r['head']) for r in order[:condition.get('k', 0)]]
            try:
                value = inference_value(runtime, sample, prepared, condition, heads, baseline)
                if condition['name'] == 'baseline':
                    baseline = value
                row.update(status='ok', **value)
            except Exception as exc:
                traceback.print_exc()
                row.update(status='error', error=repr(exc))
                errors += 1
            row['elapsed_seconds'] = time.monotonic()-tick
            append(out/'predictions.jsonl', row)
            current[condition['name']] = row
        if args.stage == 'engineering':
            base, zero = current['baseline'], current['reward_gradient_zero_k48']
            target = current['reward_gradient_k48']
            fallback = sample['grounding_status'] != 'ok'
            audits.append(dict(example_id=sample['example_id'], grounding_fallback=fallback,
                checks=dict(all_conditions_ok=all(r['status'] == 'ok' for r in current.values()),
                    zero_logits_exact=base.get('score_logits') == zero.get('score_logits'),
                    zero_progress_exact=base.get('progress') == zero.get('progress'),
                    target_hook_or_fallback=(target.get('baseline_fallback', False) and
                        target.get('score_logits') == base.get('score_logits') and target.get('progress') == base.get('progress'))
                        if fallback else target.get('hook_diagnostics', {}).get('calls', 0) > 0)))
        if index % 10 == 0 or index == len(samples):
            print(json.dumps(dict(time=timestamp(), processed=index, expected=len(samples), errors=errors,
                elapsed_seconds=time.monotonic()-begin)), flush=True)
    if args.stage == 'engineering':
        passed = len(audits) == len(ids) and all(all(a['checks'].values()) for a in audits)
        create_json(out/'engineering_audit_v1.json', dict(time=timestamp(), passed=passed, rows=audits, labels_read=False))
        if not passed:
            raise ValueError('Validation-media engineering failed; records retained')
    create_json(out/'completion_v1.json', dict(time=timestamp(), samples=len(samples),
        expected_records=len(samples)*len(conditions), errors=errors, labels_read=False,
        predictions_sha256=digest(out/'predictions.jsonl'), elapsed_seconds=time.monotonic()-begin))


if __name__ == '__main__':
    main()
