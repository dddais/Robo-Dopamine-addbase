"""Freeze F1 configurations and data exposure audit before model outcomes."""
import hashlib
import json
import pathlib
import yaml
from .common import ROOT, BASE, RESEARCH, create_json, read_rows, timestamp

DEST = ROOT / 'results/mydata_bench/experiments_v2_corssmodel/auto_research/session_20260909_robust_v1'
PROTOCOLS = ['text_video', 'video_text', 'text_image', 'image_text', 'interleaved']


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def main():
    source = ROOT / 'mydata_bench/auto_research_addbase'
    code = {str(p.relative_to(ROOT)): digest(p) for p in [
        source / 'frame_transport.py', source / 'robust_runner.py', source / 'robust_prepare.py',
        source / 'score_branches.py', source / 'discrete_runtime.py',
        ROOT / 'mydata_bench/meter_eval/runtime.py', ROOT / 'mydata_bench/top_eval/batch_attention_residual.py']}
    full = list(read_rows(BASE / 'inputs/full.jsonl'))
    cohort = list(read_rows(BASE / 'inputs/cohort.jsonl'))
    ranking = list(read_rows(BASE / 'inputs/ranking.jsonl'))
    seen = {row['video_sha256'] for row in cohort + ranking}
    remainder = [row for row in full if row['video_sha256'] not in seen]
    create_json(DEST / 'data_exposure_audit_v1.json', {
        'time': timestamp(), 'full_samples': len(full), 'cohort_samples': len(cohort),
        'ranking_samples': len(ranking), 'remainder_samples': len(remainder),
        'remainder_video_clusters': len({r['video_sha256'] for r in remainder}),
        'remainder_ids': [r['example_id'] for r in remainder],
        'remainder_has_never_seen_baseline': False,
        'old_846_is_development_only': True,
        'source_sha256': {str(BASE / ('inputs/' + n)): digest(BASE / ('inputs/' + n))
                          for n in ['full.jsonl', 'cohort.jsonl', 'ranking.jsonl', 'splits.json']}})
    splits = json.loads((BASE / 'inputs/splits.json').read_text())
    ids = splits['development']
    samples = sorted((r for r in cohort if r['example_id'] in ids), key=lambda r: (r['video_sha256'], r['example_id']))
    engineering = []
    for row in samples:
        if not engineering or row['video_sha256'] != engineering[0]['video_sha256']:
            engineering.append(row)
        if len(engineering) == 2:
            break
    create_json(DEST / 'development_ids_v1.json', ids)
    create_json(DEST / 'engineering_ids_v1.json', [r['example_id'] for r in engineering])
    conditions = [dict(name='baseline', k=0), dict(name='native_generation_baseline', k=0, kind='native_generation')]
    for k in [32, 48, 64]:
        conditions.extend([
            dict(name=f'original_all_k{k}', k=k, kind='bias', bias=6., scope='all_frames'),
            dict(name=f'frame_transport_k{k}', k=k, kind='frame_transport', fraction=.5, scope='all_frames')])
    conditions.extend([
        dict(name='frame_transport_zero_k48', k=48, kind='frame_transport', fraction=0., scope='all_frames'),
        dict(name='frame_transport_wrong_k48', k=48, kind='frame_transport', fraction=.5, scope='all_frames', region='wrong'),
        dict(name='frame_transport_low_k48', k=48, kind='frame_transport', fraction=.5, scope='all_frames', head_group='low')])
    configs = []
    for model in ['qwen', 'rr', 'meter']:
        for protocol in PROTOCOLS:
            old = ROOT / f'mydata_bench/configs/v2_crossmodel/addbase_development_v1_{model}_{protocol}.yaml'
            cfg = yaml.safe_load(old.read_text())
            ranking_path = RESEARCH / f'development_v1/{model}_{protocol}/ranking.json'
            if not ranking_path.exists():
                ranking_path = BASE / f'{model}_{protocol}_v1/ranking.json'
            if not ranking_path.exists():
                raise FileNotFoundError(ranking_path)
            cfg = {k: v for k, v in cfg.items() if k in ['model', 'model_path', 'processor_path', 'protocol', 'max_new_tokens']}
            cfg.update(stage='exploratory_development234_frozen_F1',
                       input_file=str(BASE / 'inputs/cohort.jsonl'),
                       input_sha256=digest(BASE / 'inputs/cohort.jsonl'),
                       example_ids_file=str(DEST / 'development_ids_v1.json'),
                       engineering_ids_file=str(DEST / 'engineering_ids_v1.json'),
                       output_dir=str(DEST / f'frame_transport_v1/{model}_{protocol}'),
                       ranking_file=str(ranking_path), ranking_sha256=digest(ranking_path),
                       conditions=conditions, primary_k=48, k_neighborhood=[32, 48, 64],
                       source_sha256=code, output_score_adjustment='none',
                       attention_engine='native_residual', final_confirmation_opened=False)
            path = ROOT / f'mydata_bench/configs/v2_crossmodel/addbase_robust_frame_transport_v1_{model}_{protocol}.yaml'
            with path.open('x') as handle:
                yaml.safe_dump(cfg, handle, sort_keys=False)
            configs.append(dict(model=model, protocol=protocol, path=str(path), sha256=digest(path)))
    create_json(DEST / 'frame_transport_frozen_plan_v1.json', dict(
        time=timestamp(), configurations=configs, source_sha256=code,
        method='fixed half of each frame non-target mass transported to causal target',
        development_samples=len(ids), engineering_samples=len(engineering),
        shared_fraction=.5, shared_primary_k=48, shared_k_neighborhood=[32, 48, 64],
        no_new_method_outcomes_seen_at_freeze=True, inference_has_no_label_file=True,
        interpretation='prospective candidate in historically exposed development data; not independent confirmation'))
    print(DEST, 'frozen', len(configs), 'cases', len(ids), 'samples', len(conditions), 'conditions')


if __name__ == '__main__':
    main()
