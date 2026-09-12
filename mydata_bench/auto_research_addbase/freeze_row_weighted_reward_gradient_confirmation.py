"""Freeze all final reserved-test F9 settings after full original-scope success."""
import json
import pathlib
import yaml
from .common import ROOT, create_json, read_rows, timestamp
from .robust_prepare import DEST, digest
from .f9_confirmation_priority import confirmation_priority


def selected_confirmation_cases(analysis):
    cases = sorted(name for name, row in analysis['cases'].items() if row['full846_point_gate'])
    counts = {model: sum(analysis['cases'][name]['model'] == model for name in cases) for model in ['qwen', 'rr', 'meter']}
    if not analysis['full846_original_scope_gate'] or any(n < 3 for n in counts.values()):
        raise ValueError('Original full846 scope is not met; reserved test must remain unopened')
    return cases


def main():
    priority=confirmation_priority()
    if priority['state']!='released':
        raise ValueError('F9 shared test not available: '+str(priority))
    old_root = DEST/'row_weighted_reward_gradient_full_old_v1'
    analysis_path = old_root/'analysis_v1.json'
    analysis = json.loads(analysis_path.read_text())
    names = selected_confirmation_cases(analysis)
    old_plan_path = old_root/'frozen_plan_v1.json'
    if digest(old_plan_path) != analysis['frozen_plan_sha256']:
        raise ValueError('Wrong full846 plan provenance')
    old_plan = json.loads(old_plan_path.read_text())
    configurations = {r['model']+'_'+r['protocol']: r for r in old_plan['configurations']}
    policy_path = DEST/'f9_confirmation_policy_v1.json'
    policy = json.loads(policy_path.read_text())
    for path, sha in policy['source_sha256'].items():
        if digest(ROOT/path) != sha:
            raise ValueError('Frozen final evaluation source changed: '+path)
    external = DEST/'external_roboreward_reserved_v1'
    inputs = external/'grounding_v1/grounded_inputs_v1.jsonl'
    input_done = json.loads((external/'grounding_v1/grounding_completion_v1.json').read_text())
    if digest(inputs) != input_done['output_sha256'] or digest(inputs) != policy['reserved_input_sha256']:
        raise ValueError('Reserved grounded inputs changed')
    samples = list(read_rows(inputs))
    if len(samples) != 2831 or len({r['example_id'] for r in samples}) != 2831 or any(r['preparation_status'] != 'ready' for r in samples):
        raise ValueError('Require all2831 ready reserved examples')
    group_path = external/'episode_grouping_records_v1.jsonl'
    if digest(group_path) != policy['reserved_groups_sha256']:
        raise ValueError('Reserved episode grouping changed')
    groups = list(read_rows(group_path))
    if len(groups) != 2831 or {r['example_id'] for r in groups} != {r['example_id'] for r in samples}:
        raise ValueError('Incomplete/duplicate reserved episode membership')
    label_path = external/'labels_sealed_v1.jsonl'
    if digest(label_path) != policy['reserved_labels_sha256']:
        raise ValueError('Original sealed test labels changed')
    selected = []
    for name in names:
        item = configurations[name]
        if digest(item['path']) != item['sha256']:
            raise ValueError('Full846 frozen config changed')
        cfg = yaml.safe_load(pathlib.Path(item['path']).read_text())
        predictions = pathlib.Path(cfg['output_dir'])/'full846/predictions.jsonl'
        if digest(predictions) != analysis['cases'][name]['predictions_sha256']:
            raise ValueError('Full846 predictions changed')
        for field in ['ranking', 'original_ranking']:
            if digest(cfg[field+'_file']) != cfg[field+'_sha256']:
                raise ValueError('Frozen ranking changed')
        summaries = analysis['cases'][name]['summaries']
        originals = {key: value for key, value in summaries.items() if key.startswith('original_all_k') and value['complete']}
        loss = 'continuous_ordinal_mae' if cfg['model'] == 'meter' else 'mae'
        best = min(originals, key=lambda key: (originals[key]['overall'][loss], -originals[key]['overall']['accuracy'], int(key.rsplit('k', 1)[1])))
        if best not in {c['name'] for c in cfg['conditions']}:
            raise ValueError('Strong original reference lacks an actual tested condition')
        selected.append((name, cfg, best))
    # Exclusive one-time claim is made before any reserved reward-model forward.
    # This is a local workflow convention, not third-party enforced blinding.
    claim_path = external/'reward_model_confirmation_claim_v1.json'
    priority=confirmation_priority()
    if priority['state']!='released':
        raise ValueError('Shared test priority changed before exclusive claim')
    create_json(claim_path, dict(time=timestamp(), F7_F8_release_evidence=priority, family='F9_sample_mean_ordinal_signed_all_query_reward_gradient',
        policy_sha256=digest(policy_path), selection_full846_analysis_sha256=digest(analysis_path), cases=names,
        once_only=True, all_conditions_and_rankings_frozen_before_performance=True,
        prohibition='No method/parameter/case replacement based on reserved performance; report all selected cases including failures'))
    out = DEST/'row_weighted_reward_gradient_confirmation_v1'
    create_json(out/'example_ids_v1.json', [r['example_id'] for r in samples])
    grounded = sorted(r['example_id'] for r in samples if r['grounding_status'] == 'ok')
    fallback = sorted(r['example_id'] for r in samples if r['grounding_status'] != 'ok')
    create_json(out/'engineering_ids_v1.json', grounded[:2]+fallback[:1])
    create_json(out/'tracking_sha256_v1.json', {r['tracking_path']: digest(r['tracking_path']) for r in samples if r['grounding_status'] == 'ok'})
    final_configs = []
    for name, cfg, best in selected:
        old_best = cfg['best_original_condition']
        for field in ['old234_output_dir', 'old234_predictions_sha256', 'old234_config_sha256']:
            cfg.pop(field, None)
        cfg.update(output_dir=str(out/name), input_file=str(inputs), input_sha256=digest(inputs),
            example_ids_file=str(out/'example_ids_v1.json'), example_ids_sha256=digest(out/'example_ids_v1.json'),
            engineering_ids_file=str(out/'engineering_ids_v1.json'), engineering_ids_sha256=digest(out/'engineering_ids_v1.json'),
            tracking_manifest_file=str(out/'tracking_sha256_v1.json'), tracking_manifest_sha256=digest(out/'tracking_sha256_v1.json'),
            confirmation_claim_file=str(claim_path), confirmation_claim_sha256=digest(claim_path),
            best_original_condition=best, earlier_old234_best_original_condition=old_best,
            stage='F9_once_only_reserved2831_confirmation', all846_previously_exposed=True,
            independent_confirmation=True, reserved_test_evaluated=False,
            selection='all full846 passing cases among previously joint-old234+validation eligible cases',
            strong_original_selection='minimum actual old846 native ordinal MAE; tie accuracy then smaller k; no test outcomes',
            final_policy_sha256=digest(policy_path))
        cfg['source_sha256'].update(policy['source_sha256'])
        path = ROOT/f'mydata_bench/configs/v2_crossmodel/addbase_robust_row_weighted_reward_gradient_confirmation_v1_{name}.yaml'
        with path.open('x') as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        final_configs.append(dict(model=cfg['model'], protocol=cfg['protocol'], path=str(path), sha256=digest(path)))
    create_json(out/'frozen_plan_v1.json', dict(time=timestamp(), configurations=final_configs,
        input_file=str(inputs), input_sha256=digest(inputs), example_ids_file=str(out/'example_ids_v1.json'),
        example_ids_sha256=digest(out/'example_ids_v1.json'), groups_file=str(group_path), groups_sha256=digest(group_path),
        analysis_labels_file=str(label_path), analysis_labels_sha256=digest(label_path),
        policy_file=str(policy_path), policy_sha256=digest(policy_path), source_sha256=policy['source_sha256'],
        confirmation_claim_file=str(claim_path), confirmation_claim_sha256=digest(claim_path),
        selection_analysis_file=str(analysis_path), selection_analysis_sha256=digest(analysis_path),
        examples=2831, episode_groups=1172, expected_cases=len(final_configs),
        test_performance_not_joined_before_freeze=True, all_selected_cases_must_finish_before_label_join=True,
        final_test_failure_cannot_be_repaired_by_reusing_this_test_for_new_confirmation=True))
    print('Once-only reserved confirmation frozen:', len(final_configs), 'cases, all2831 examples', flush=True)


if __name__ == '__main__':
    main()
