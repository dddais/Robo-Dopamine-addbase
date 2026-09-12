"""Label-free inference for the frozen frame-local transport experiment."""
import argparse
import fcntl
import hashlib
import json
import os
import pathlib
import time
import traceback
import torch
import yaml
from .common import ROOT, create_json, append, read_rows, timestamp, output_path
from .score_branches import ScoreRuntime
from .frame_transport import frame_transport
from mydata_bench.attention_eval.masking import matched_wrong_position_set


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


class FrameTransportRuntime(ScoreRuntime):
    def predict(self, sample, prepared, condition, heads):
        if condition.get('kind') != 'frame_transport':
            return super().predict(sample, prepared, condition, heads)
        target, alignment = self.runtime.positions(sample, prepared, 'all_frames')
        if condition.get('region') == 'wrong':
            wrong = []
            for span in prepared['spans']:
                local = [p for p in target if span.start <= p < span.end]
                if not local:
                    continue
                positions = matched_wrong_position_set(span, local, spatial_merge_size=2)
                if positions is None:
                    raise ValueError('No equal-area disjoint same-frame wrong control')
                wrong.extend(positions)
            target = wrong
        diagnostics = {}
        with frame_transport(self.runtime.layers, heads, target, prepared['spans'],
                             condition['fraction'], diagnostics):
            result = super().predict(sample, prepared, {}, [])
        result.update(hook_diagnostics=diagnostics, tracking_alignment=alignment,
                      engine='frame_transport_native_residual_v1')
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--stage', choices=['engineering', 'development'], required=True)
    args = parser.parse_args()
    config = yaml.safe_load(pathlib.Path(args.config).read_text())
    for path, expected in config['source_sha256'].items():
        if digest(ROOT / path) != expected:
            raise ValueError('Frozen source changed: ' + path)
    for field in ['input', 'ranking']:
        if digest(config[field + '_file']) != config[field + '_sha256']:
            raise ValueError('Frozen input changed: ' + field)
    out = output_path(config['output_dir']) / args.stage
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'writer.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / 'completion_v1.json').exists():
        raise RuntimeError('Stage already complete; refusing to repeat it')
    if args.stage == 'development':
        audit = json.loads((out.parent / 'engineering/engineering_audit_v1.json').read_text())
        if not audit['passed']:
            raise RuntimeError('Engineering gate did not pass')
    create_json(out / 'config.json', config)
    ids_path = config['engineering_ids_file'] if args.stage == 'engineering' else config['example_ids_file']
    ids = set(json.loads(pathlib.Path(ids_path).read_text()))
    samples = [r for r in read_rows(config['input_file']) if r['example_id'] in ids]
    if len(samples) != len(ids):
        raise ValueError('Missing expected samples')
    ranking = json.loads(pathlib.Path(config['ranking_file']).read_text())['ranking']
    conditions = config['conditions']
    if args.stage == 'engineering':
        conditions = [c for c in conditions if c['name'] in ['baseline', 'frame_transport_k48', 'frame_transport_zero_k48']]
    create_json(out / 'launch_v1.json', dict(time=timestamp(), pid=os.getpid(),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        config_sha256=digest(args.config), source_sha256=config['source_sha256'],
        stage=args.stage, samples=len(samples), conditions=len(conditions)))
    runtime = FrameTransportRuntime(config)
    start = time.monotonic()
    audit_rows = {}
    errors = 0
    for index, sample in enumerate(samples):
        try:
            prepared = runtime.prepare(sample)
        except Exception as exc:
            traceback.print_exc()
            for condition in conditions:
                append(out / 'predictions.jsonl', dict(time=timestamp(), example_id=sample['example_id'],
                    condition=condition['name'], status='error', error=repr(exc), phase='prepare'))
                errors += 1
            continue
        for condition in conditions:
            ordered = list(reversed(ranking)) if condition.get('head_group') == 'low' else ranking
            heads = [(r['layer'], r['head']) for r in ordered[:condition.get('k', 0)]]
            row = dict(time=timestamp(), example_id=sample['example_id'], video_sha256=sample['video_sha256'],
                       condition=condition['name'], condition_config=condition)
            tick = time.monotonic()
            try:
                row.update(status='ok', **runtime.predict(sample, prepared, condition, heads))
            except Exception as exc:
                traceback.print_exc()
                row.update(status='error', error=repr(exc))
                errors += 1
                torch.cuda.empty_cache()
            row['elapsed_seconds'] = time.monotonic()-tick
            append(out / 'predictions.jsonl', row)
            if args.stage == 'engineering':
                audit_rows[(sample['example_id'], condition['name'])] = row
        if (index+1) % 10 == 0 or index == len(samples)-1:
            print(json.dumps(dict(time=timestamp(), processed=index+1, expected=len(samples),
                                  elapsed_seconds=time.monotonic()-start, errors=errors)), flush=True)
    if args.stage == 'engineering':
        checks = []
        for sample in samples:
            baseline = audit_rows.get((sample['example_id'], 'baseline'), {})
            zero = audit_rows.get((sample['example_id'], 'frame_transport_zero_k48'), {})
            target = audit_rows.get((sample['example_id'], 'frame_transport_k48'), {})
            diag = target.get('hook_diagnostics', {})
            checks.append(dict(example_id=sample['example_id'],
                complete=all(r.get('status') == 'ok' for r in [baseline, zero, target]),
                zero_logits_exact=baseline.get('score_logits') == zero.get('score_logits'),
                zero_progress_exact=baseline.get('progress') == zero.get('progress'),
                hook_calls_positive=diag.get('calls', 0) > 0,
                frame_mass=diag.get('max_frame_mass_error', 1.) <= 1e-6,
                nonvisual=diag.get('max_nonvisual_change', 1.) == 0,
                causal=diag.get('max_causal_change', 1.) == 0,
                nonnegative=diag.get('minimum_changed_probability', -1.) >= -1e-7))
        passed = len(checks) == len(ids) and all(all(v for k, v in c.items() if k != 'example_id') for c in checks)
        create_json(out / 'engineering_audit_v1.json', dict(time=timestamp(), passed=passed, checks=checks,
                    no_labels_used=True, no_performance_selection=True))
        if not passed:
            raise RuntimeError('Engineering checks failed; raw failures preserved')
    create_json(out / 'completion_v1.json', dict(time=timestamp(), samples=len(samples),
        expected_records=len(samples)*len(conditions), errors=errors, elapsed_seconds=time.monotonic()-start,
        predictions_sha256=digest(out / 'predictions.jsonl')))


if __name__ == '__main__':
    main()
