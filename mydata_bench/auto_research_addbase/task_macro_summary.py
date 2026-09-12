"""Supplement micro metrics with equal-task descriptive summaries.

Custom mydata tasks are not the 23 official RoboRewardBench subsets. This
supplement never changes frozen selection or confirmation point acceptance.
"""
import argparse
from .common import *


def collect(value, path=()):
    result = {}
    if isinstance(value, dict):
        if {'overall', 'by_task', 'by_split'} <= value.keys():
            tasks = {k: v for k, v in value['by_task'].items() if v.get('n_expected', 0)}
            metrics = {}
            for metric in ['mae', 'continuous_ordinal_mae', 'accuracy', 'endpoint_accuracy_0.125_0.875', 'endpoint_accuracy_0.2_0.8']:
                defined = {k: v[metric] for k, v in tasks.items() if v.get(metric) is not None}
                metrics[metric] = {'n_tasks_expected': len(tasks), 'n_tasks_defined': len(defined),
                                   'micro': value['overall'].get(metric),
                                   'macro_task': sum(defined.values()) / len(tasks) if tasks and len(defined) == len(tasks) else None,
                                   'undefined_tasks': sorted(set(tasks) - set(defined))}
            result['/'.join(map(str, path))] = {'metrics': metrics,
                'complete_sample_coverage': value.get('complete'),
                'task_sample_sizes': {k: v['n_expected'] for k, v in tasks.items()},
                'task_valid_sizes': {k: v['n_valid'] for k, v in tasks.items()}}
            return result
        for key, item in value.items():
            result.update(collect(item, path + (key,)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            result.update(collect(item, path + (index,)))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    reports = []
    for source in args.sources:
        source = pathlib.Path(source).resolve()
        blob = source.read_bytes()
        report = collect(json.loads(blob))
        if not report:
            raise ValueError(f'No supported summaries in {source}')
        reports.append({'source': str(source), 'sha256': hashlib.sha256(blob).hexdigest(), 'summaries': report})
        print(source.name, len(report), 'equal-task summaries')
    create_json(args.output, {'time': timestamp(), 'reports': reports,
        'definition': 'Arithmetic mean of per-task metrics over every task represented in the specified cohort. Per-task accuracy retains all expected examples in its denominator. MAE is marked undefined if an entire represented task lacks a valid score; per-task valid counts and overall completeness are explicit.',
        'paper_comparable': False,
        'boundary': 'The paper MAE formula is mean absolute error on 1..5 labels; its leaderboard Overall is a group-wise average over 23 RoboRewardBench subsets. These custom tasks are a different dataset. Task-macro values are a descriptive supplement, not a changed frozen confirmation endpoint.',
        'paper_source': 'https://arxiv.org/html/2601.00675v2#S3'})


if __name__ == '__main__':
    main()
