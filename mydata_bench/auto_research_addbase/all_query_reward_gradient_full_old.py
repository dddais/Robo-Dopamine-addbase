"""Full846 descriptive F7 evaluation, preserving exact already-frozen old234 rows."""
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
from .all_query_reward_gradient_runtime import AllQueryRewardGradientRuntime as RewardGradientRuntime, WrongRegionUnavailable


def normalized_condition(condition):
    return {k: v for k, v in condition.items() if k != 'cached_reference'}


def verified_cache(cfg):
    source = pathlib.Path(cfg['old234_output_dir'])/'development'
    if digest(source/'config.json') != cfg['old234_config_sha256']:
        raise ValueError('Frozen old234 source config changed')
    done = json.loads((source/'completion_v1.json').read_text())
    path = source/'predictions.jsonl'
    if digest(path) != cfg['old234_predictions_sha256'] or digest(path) != done['predictions_sha256']:
        raise ValueError('Old234 cached prediction provenance mismatch')
    old = json.loads((source/'config.json').read_text())
    for field in ['model', 'model_path', 'processor_path', 'protocol', 'input_sha256', 'ranking_sha256', 'original_ranking_sha256']:
        if cfg.get(field) != old.get(field):
            raise ValueError('Incompatible old234 cached inference: '+field)
    current = {c['name']: normalized_condition(c) for c in cfg['conditions']}
    for condition in old['conditions']:
        if current.get(condition['name']) != normalized_condition(condition):
            raise ValueError('Cached condition changed')
    rows = list(read_rows(path))
    cache = {(r['example_id'], r['condition']): r for r in rows}
    expected_ids = json.loads(pathlib.Path(old['example_ids_file']).read_text())
    expected = {(eid, c['name']) for eid in expected_ids for c in old['conditions']}
    if len(rows) != len(cache) or set(cache) != expected:
        raise ValueError('Incomplete/duplicate old234 cache')
    return cache, path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--stage', choices=['engineering', 'full846'], required=True)
    args = p.parse_args()
    cfg = yaml.safe_load(pathlib.Path(args.config).read_text())
    for path, sha in cfg['source_sha256'].items():
        if digest(ROOT/path) != sha:
            raise ValueError('Frozen full846 source changed: '+path)
    for field in ['input', 'ranking', 'original_ranking', 'example_ids', 'engineering_ids']:
        if digest(cfg[field+'_file']) != cfg[field+'_sha256']:
            raise ValueError('Frozen full846 dependency changed: '+field)
    cache, cache_path = verified_cache(cfg)
    ids = json.loads(pathlib.Path(cfg['engineering_ids_file'] if args.stage == 'engineering' else cfg['example_ids_file']).read_text())
    samples = [r for r in read_rows(cfg['input_file']) if r['example_id'] in set(ids)]
    if len(samples) != len(ids) or len({r['example_id'] for r in samples}) != len(ids):
        raise ValueError('Require complete unique full846/engineering IDs')
    out = output_path(cfg['output_dir'])/args.stage
    out.mkdir(parents=True, exist_ok=True)
    lock = (out/'writer.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.stage == 'full846':
        if len(ids) != 846 or not json.loads((out.parent/'engineering/engineering_audit_v1.json').read_text())['passed']:
            raise ValueError('Require all846 and passed actual cache parity engineering')
    create_json(out/'config.json', cfg)
    create_json(out/'launch_v1.json', dict(time=timestamp(), pid=os.getpid(), gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        config_sha256=digest(args.config), old234_predictions_sha256=digest(cache_path), labels_read=False,
        all846_historically_exposed_descriptive_only=True))
    rankings = {'reward_gradient': json.loads(pathlib.Path(cfg['ranking_file']).read_text())['ranking'],
                'original': json.loads(pathlib.Path(cfg['original_ranking_file']).read_text())['ranking']}
    conditions = cfg['conditions']
    if args.stage == 'engineering':
        conditions = [c for c in conditions if c['name'] in ['baseline', 'original_all_k48', 'reward_gradient_k48', 'reward_gradient_zero_k48', 'learned_readout_k48']]
    runtime = RewardGradientRuntime(cfg)
    begin, errors, cached_rows, checks = time.monotonic(), 0, 0, []
    for index, sample in enumerate(samples, 1):
        need_forward = args.stage == 'engineering' or any((sample['example_id'], c['name']) not in cache for c in conditions)
        prepare_error, prepared = None, None
        if need_forward:
            try:
                prepared = runtime.prepare(sample)
            except Exception as exc:
                traceback.print_exc()
                prepare_error = repr(exc)
        current = {}
        for condition in conditions:
            key = (sample['example_id'], condition['name'])
            tick = time.monotonic()
            row = dict(time=timestamp(), example_id=sample['example_id'], video_sha256=sample['video_sha256'],
                       condition=condition['name'], condition_config=condition)
            if args.stage == 'full846' and key in cache:
                old = cache[key]
                row.update({k: v for k, v in old.items() if k not in ['time', 'condition_config', 'elapsed_seconds']})
                row['reused_old234_prediction_source'] = str(cache_path)
                cached_rows += 1
            else:
                order = rankings[condition.get('ranking', 'reward_gradient')]
                if condition.get('head_group') == 'low':
                    order = list(reversed(order))
                heads = [(r['layer'], r['head']) for r in order[:condition.get('k', 0)]]
                try:
                    if prepare_error:
                        raise ValueError('Preserved preparation error: '+prepare_error)
                    row.update(status='ok', **runtime.predict(sample, prepared, condition, heads))
                    if args.stage == 'engineering':
                        old = cache[key]
                        if old['status'] != 'ok' or any(old.get(f) != row.get(f) for f in ['score_logits', 'progress', 'native_prediction']):
                            raise ValueError('Actual full846 runtime differs from frozen old234 result')
                        row['actual_old234_cache_exact'] = True
                except Exception as exc:
                    traceback.print_exc()
                    row.update(status='error', error=repr(exc), control_geometry_infeasible=isinstance(exc, WrongRegionUnavailable))
            errors += row['status'] != 'ok'
            row['elapsed_seconds'] = time.monotonic()-tick
            append(out/'predictions.jsonl', row)
            current[condition['name']] = row
        if args.stage == 'engineering':
            base, zero = current['baseline'], current['reward_gradient_zero_k48']
            checks.append(dict(example_id=sample['example_id'], complete=all(r['status'] == 'ok' for r in current.values()),
                all_conditions_actual_cache_exact=all(r.get('actual_old234_cache_exact', False) for r in current.values()),
                zero_logits_exact=base.get('score_logits') == zero.get('score_logits'),
                zero_progress_exact=base.get('progress') == zero.get('progress'),
                target_all_queries=current['reward_gradient_k48'].get('hook_diagnostics',{}).get('all_causal_queries',False),
                control_readout_only=current['learned_readout_k48'].get('hook_diagnostics',{}).get('only_readout_query_changed',False)))
        if index % 10 == 0 or index == len(samples):
            print(json.dumps(dict(time=timestamp(), processed=index, expected=len(samples), errors=errors,
                cached_rows=cached_rows, elapsed_seconds=time.monotonic()-begin)), flush=True)
    if args.stage == 'engineering':
        passed = len(checks) == len(ids) and all(all(v for k, v in r.items() if k != 'example_id') for r in checks)
        create_json(out/'engineering_audit_v1.json', dict(time=timestamp(), passed=passed, checks=checks, labels_read=False))
        if not passed:
            raise ValueError('Actual full846 engineering failed; evidence retained')
    create_json(out/'completion_v1.json', dict(time=timestamp(), samples=len(samples),
        expected_records=len(samples)*len(conditions), errors=errors, cached_rows=cached_rows,
        predictions_sha256=digest(out/'predictions.jsonl'), elapsed_seconds=time.monotonic()-begin,
        all846_descriptive_only=True, reserved_test_evaluated=False))


if __name__ == '__main__':
    main()
