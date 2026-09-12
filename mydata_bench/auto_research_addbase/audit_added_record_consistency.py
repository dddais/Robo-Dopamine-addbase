"""Read-only duplicate and scientific-record audit of completed added baselines."""
import argparse
import collections
from .common import *

FIELDS = ['status', 'error', 'phase', 'progress', 'progress_trajectory',
          'success_probability', 'success_trajectory', 'native_progress_logits',
          'previous_progress', 'raw_output', 'video_sha256', 'condition_config']


def scientific(row):
    value = {key: row[key] for key in FIELDS if key in row}
    if 'steps' in row:
        value['steps'] = [{**scientific(step), 'step': step.get('step')}
                          for step in row['steps']]
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    for protocol in ['video_text', 'interleaved']:
        if not (BASE / f'sole_{protocol}_attention_residual_v1/completion_events.jsonl').exists():
            raise ValueError('Wait for both remaining mandatory SOLE all-step runs')
    reports = {}
    for config_path in sorted(BASE.glob('*/config.json')):
        config = json.loads(config_path.read_text())
        if config.get('model') not in {'meter', 'sole'} or config.get('pilot') or config.get('example_ids_file'):
            continue
        folder = config_path.parent
        for name in ['baseline.jsonl', 'attention.jsonl', 'step_traces.jsonl', 'attention_step_traces.jsonl']:
            source = folder / name
            if not source.exists():
                continue
            hashes = {}; counts = collections.Counter(); conflicts = []; statuses = collections.Counter()
            digest = hashlib.sha256(); size_before = source.stat().st_size
            with source.open('rb') as handle:
                for line_number, line in enumerate(handle, 1):
                    digest.update(line)
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    key = (row['example_id'], row.get('condition', 'baseline'), row.get('step'))
                    value = fingerprint(scientific(row))
                    if key in hashes and hashes[key] != value:
                        conflicts.append({'key': key, 'line': line_number, 'previous_sha256': hashes[key], 'current_sha256': value})
                    hashes[key] = value; counts[key] += 1; statuses[row.get('status', 'missing_status')] += 1
            if source.stat().st_size != size_before:
                raise ValueError(f'Input changed during final audit: {source}')
            reports[str(source.relative_to(BASE))] = {
                'source_sha256': digest.hexdigest(), 'source_bytes': size_before,
                'n_raw_records': sum(counts.values()), 'n_unique_keys': len(counts),
                'n_repeated_keys': sum(n > 1 for n in counts.values()),
                'n_extra_duplicate_records': sum(n - 1 for n in counts.values()),
                'raw_status_counts': dict(statuses), 'scientific_conflicts': conflicts}
            print(source.relative_to(BASE), 'unique', len(counts), 'duplicates', sum(n - 1 for n in counts.values()), 'conflicts', len(conflicts), flush=True)
    create_json(args.output, {
        'time': timestamp(), 'files': reports,
        'all_scientific_duplicate_records_consistent': all(not r['scientific_conflicts'] for r in reports.values()),
        'scope': 'Completed non-pilot Meter/SOLE native baselines, full attention and terminal-step supplements. Errors and duplicates remain in place. This audits saved scientific fields, not an independent neural rerun; labels are not read.',
        'compared_fields': FIELDS + ['steps recursively with step index'],
        'analysis_source_sha256': hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()})


if __name__ == '__main__':
    main()
