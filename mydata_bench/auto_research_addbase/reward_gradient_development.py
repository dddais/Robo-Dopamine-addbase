"""F6 old-development inference: externally fitted signed heads, native score readout."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--stage', choices=['engineering', 'development'], required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(pathlib.Path(args.config).read_text())
    for path, expected in cfg['source_sha256'].items():
        if digest(ROOT / path) != expected:
            raise ValueError('Frozen source changed: ' + path)
    for field in ['input', 'ranking', 'original_ranking']:
        if digest(cfg[field + '_file']) != cfg[field + '_sha256']:
            raise ValueError('Frozen dependency changed: ' + field)
    source = pathlib.Path(cfg['clean_source_dir']) / 'development'
    source_config = json.loads((source / 'config.json').read_text())
    for field in ['model', 'model_path', 'processor_path', 'protocol', 'input_sha256', 'example_ids_file']:
        if cfg.get(field) != source_config.get(field):
            raise ValueError('Incompatible cached F1 baseline: ' + field)
    if source_config['ranking_sha256'] != cfg['original_ranking_sha256']:
        raise ValueError('Original head ranking mismatch')
    source_done = json.loads((source / 'completion_v1.json').read_text())
    if digest(source / 'predictions.jsonl') != source_done['predictions_sha256']:
        raise ValueError('Cached clean predictions changed')
    old_rows = list(read_rows(source / 'predictions.jsonl'))
    cached = {(r['example_id'], r['condition']): r for r in old_rows}
    if len(cached) != len(old_rows):
        raise ValueError('Duplicate cached records')
    out = output_path(cfg['output_dir']) / args.stage
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'writer.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.stage == 'development':
        if not json.loads((out.parent / 'engineering/engineering_audit_v1.json').read_text())['passed']:
            raise ValueError('Engineering gate failed')
    create_json(out / 'config.json', cfg)
    create_json(out / 'launch_v1.json', dict(time=timestamp(), pid=os.getpid(),
        gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), config_sha256=digest(args.config),
        source_sha256=cfg['source_sha256'], cached_predictions_sha256=source_done['predictions_sha256']))
    ids = set(json.loads(pathlib.Path(cfg['engineering_ids_file'] if args.stage == 'engineering' else cfg['example_ids_file']).read_text()))
    samples = [r for r in read_rows(cfg['input_file']) if r['example_id'] in ids]
    if len(samples) != len(ids):
        raise ValueError('Unknown expected IDs')
    rankings = {'reward_gradient': json.loads(pathlib.Path(cfg['ranking_file']).read_text())['ranking'],
                'original': json.loads(pathlib.Path(cfg['original_ranking_file']).read_text())['ranking']}
    conditions = cfg['conditions']
    if args.stage == 'engineering':
        conditions = [c for c in conditions if c['name'] in ['baseline', 'original_all_k48', 'reward_gradient_k48', 'reward_gradient_zero_k48']]
    runtime = RewardGradientRuntime(cfg)
    begin = time.monotonic()
    errors, engineering = 0, {}
    for index, sample in enumerate(samples, 1):
        try:
            prepared = runtime.prepare(sample)
        except Exception as exc:
            for c in conditions:
                append(out / 'predictions.jsonl', dict(time=timestamp(), example_id=sample['example_id'],
                    condition=c['name'], condition_config=c, status='error', error=repr(exc), phase='prepare'))
                errors += 1
            continue
        for condition in conditions:
            tick = time.monotonic()
            row = dict(time=timestamp(), example_id=sample['example_id'], video_sha256=sample['video_sha256'],
                       condition=condition['name'], condition_config=condition)
            order = rankings[condition.get('ranking', 'reward_gradient')]
            if condition.get('head_group') == 'low':
                order = list(reversed(order))
            heads = [(r['layer'], r['head']) for r in order[:condition.get('k', 0)]]
            try:
                is_reference = condition.get('cached_reference', False)
                if is_reference and args.stage == 'development':
                    old = cached[(sample['example_id'], condition['name'])]
                    if old['status'] != 'ok':
                        raise ValueError('Preserved cached failure: ' + old.get('error', 'unknown'))
                    value = {k: v for k, v in old.items() if k not in ['time', 'example_id', 'condition', 'condition_config', 'status', 'elapsed_seconds']}
                    value['cached_source'] = str(source / 'predictions.jsonl')
                else:
                    value = runtime.predict(sample, prepared, condition, heads)
                    if is_reference:
                        old = cached[(sample['example_id'], condition['name'])]
                        if old['status'] != 'ok' or any(old.get(k) != value.get(k) for k in ['score_logits', 'progress', 'native_prediction']):
                            raise ValueError('Actual-vs-cached reference mismatch')
                        value['cached_reference_recomputed_exact'] = True
                row.update(status='ok', **value)
            except Exception as exc:
                traceback.print_exc()
                row.update(status='error', error=repr(exc))
                errors += 1
            row['elapsed_seconds'] = time.monotonic()-tick
            append(out / 'predictions.jsonl', row)
            if args.stage == 'engineering':
                engineering[(sample['example_id'], condition['name'])] = row
        if index % 10 == 0 or index == len(samples):
            print(json.dumps(dict(time=timestamp(), processed=index, expected=len(samples), errors=errors,
                                  elapsed_seconds=time.monotonic()-begin)), flush=True)
    if args.stage == 'engineering':
        checks = []
        for sample in samples:
            rr = {name: engineering.get((sample['example_id'], name), {}) for name in
                  ['baseline', 'original_all_k48', 'reward_gradient_k48', 'reward_gradient_zero_k48']}
            checks.append(dict(example_id=sample['example_id'], complete=all(r.get('status') == 'ok' for r in rr.values()),
                zero_logits_exact=rr['baseline'].get('score_logits') == rr['reward_gradient_zero_k48'].get('score_logits'),
                zero_progress_exact=rr['baseline'].get('progress') == rr['reward_gradient_zero_k48'].get('progress'),
                baseline_cache_exact=rr['baseline'].get('cached_reference_recomputed_exact', False),
                original_cache_exact=rr['original_all_k48'].get('cached_reference_recomputed_exact', False),
                target_hook_called=rr['reward_gradient_k48'].get('hook_diagnostics', {}).get('calls', 0) > 0))
        passed = len(checks) == len(ids) and all(all(v for k, v in row.items() if k != 'example_id') for row in checks)
        create_json(out / 'engineering_audit_v1.json', dict(time=timestamp(), passed=passed, checks=checks,
            externally_supervised_head_ranking=True, no_frame_mass_conservation_claim=True, labels_not_used=True))
        if not passed:
            raise ValueError('Engineering audit failed; original failure preserved')
    create_json(out / 'completion_v1.json', dict(time=timestamp(), samples=len(samples), expected_records=len(samples)*len(conditions),
        errors=errors, predictions_sha256=digest(out / 'predictions.jsonl'), elapsed_seconds=time.monotonic()-begin))


if __name__ == '__main__':
    main()
