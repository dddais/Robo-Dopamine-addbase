"""Freeze validation configs after all fits, before any F6 evaluation scores."""
import json
import pathlib
import yaml
from .common import ROOT, create_json, read_rows, timestamp
from .prepare_external_head_training import OUT
from .robust_prepare import DEST, digest


def main():
    policy_path = DEST/'f6_evaluation_policy_v1.json'
    policy = json.loads(policy_path.read_text())
    for path, sha in policy['evaluation_source_sha256'].items():
        if digest(ROOT/path) != sha:
            raise ValueError('Prospective evaluation source changed: '+path)
    queue_done = json.loads((DEST/'f6_all_fit_queues_completion_v1.json').read_text())
    if queue_done['failures']:
        raise ValueError('At least one required F6 fitting case failed')
    dev_plan = json.loads((DEST/'reward_gradient_development_frozen_plan_v1.json').read_text())
    if len(dev_plan['configurations']) != 15:
        raise ValueError('Require all fifteen F6 configurations')
    inputs = OUT/'grounding_v1/grounded_inputs_v1.jsonl'
    done = json.loads((OUT/'grounding_v1/grounding_completion_v1.json').read_text())
    if digest(inputs) != done['output_sha256']:
        raise ValueError('Grounded training input changed')
    all_rows = list(read_rows(inputs))
    samples = [r for r in all_rows if r['training_partition'] == 'validation']
    ids_path = OUT/'training_validation_ids_v1.json'
    ids = json.loads(ids_path.read_text())
    if len(samples) != len(ids) or {r['example_id'] for r in samples} != set(ids) or set(ids) != set(policy['validation_ids']):
        raise ValueError('Frozen validation cohort changed')
    fit = [r for r in all_rows if r['training_partition'] == 'fit']
    if {r['training_episode_group_sha256'] for r in fit} & {r['training_episode_group_sha256'] for r in samples}:
        raise ValueError('Fit/validation episode overlap')
    grounded = sorted((r['example_id'] for r in samples if r['grounding_status'] == 'ok'))
    fallback = sorted((r['example_id'] for r in samples if r['grounding_status'] != 'ok'))
    if len(grounded) < 2:
        raise ValueError('Insufficient actual grounded validation examples for engineering')
    engineering_path = OUT/'f6_validation_engineering_ids_v1.json'
    create_json(engineering_path, grounded[:2]+fallback[:1])
    tracks_path = OUT/'f6_validation_tracking_sha256_v1.json'
    create_json(tracks_path, {r['tracking_path']: digest(r['tracking_path']) for r in samples if r['grounding_status'] == 'ok'})
    configs = []
    for item in dev_plan['configurations']:
        if digest(item['path']) != item['sha256']:
            raise ValueError('F6 development config changed')
        cfg = yaml.safe_load(pathlib.Path(item['path']).read_text())
        name = item['model']+'_'+item['protocol']
        for stage in ['engineering', 'development']:
            if (pathlib.Path(cfg['output_dir'])/stage/'launch_v1.json').exists():
                raise ValueError('F6 evaluation already started before validation freeze')
        conditions = [{k: v for k, v in c.items() if k != 'cached_reference'} for c in cfg['conditions']]
        best = policy['original_best_development_conditions'][name]
        if best not in {c['name'] for c in conditions}:
            conditions.append(dict(name=best, k=int(best.rsplit('k', 1)[1]), kind='bias', bias=6.,
                                   scope='all_frames', ranking='original'))
        cfg.pop('clean_source_dir', None)
        cfg.update(output_dir=str(DEST/f'reward_gradient_validation_v1/{name}'),
            input_file=str(inputs), input_sha256=digest(inputs),
            example_ids_file=str(ids_path), example_ids_sha256=digest(ids_path),
            engineering_ids_file=str(engineering_path), engineering_ids_sha256=digest(engineering_path),
            tracking_manifest_file=str(tracks_path), tracking_manifest_sha256=digest(tracks_path),
            conditions=conditions, best_original_condition=best,
            stage='F6_separate_public_train_validation', inference_reads_labels=False,
            grounding_fallback='typed score baseline for every ROI method and original ROI control; keep all rows',
            evaluation_policy_sha256=digest(policy_path))
        cfg['source_sha256'].update(policy['evaluation_source_sha256'])
        path = ROOT/f'mydata_bench/configs/v2_crossmodel/addbase_robust_reward_gradient_validation_v1_{name}.yaml'
        with path.open('x') as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        configs.append(dict(model=item['model'], protocol=item['protocol'], path=str(path), sha256=digest(path)))
    label_file = OUT/'training_validation_labels_only_v1.jsonl'
    create_json(DEST/'reward_gradient_validation_frozen_plan_v1.json', dict(time=timestamp(), configurations=configs,
        input_file=str(inputs), input_sha256=digest(inputs), example_ids_file=str(ids_path), example_ids_sha256=digest(ids_path),
        analysis_labels_file=str(label_file), analysis_labels_sha256=digest(label_file),
        policy_file=str(policy_path), policy_sha256=digest(policy_path),
        inference_source_sha256=policy['evaluation_source_sha256'],
        examples=len(samples), episode_groups=len({r['training_episode_group_sha256'] for r in samples}),
        fit_examples=len(fit), expected_cases=15, validation_performance_not_observed_before_freeze=True,
        analyze_labels_only_after_all_cases_finish=True, reserved_test_performance_not_opened=True))
    print('Frozen independent F6 training validation:', len(samples), 'rows,', len(configs), 'cases', flush=True)


if __name__ == '__main__':
    main()
