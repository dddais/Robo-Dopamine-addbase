"""Freeze original complete846 follow-up only after same-case joint eligibility."""
import argparse
import json
import pathlib
import yaml
from .common import ROOT, BASE, create_json, read_rows, timestamp
from .robust_prepare import DEST, digest


def eligible_cases(analysis):
    cases = [name for name, row in analysis['cases'].items() if row['old234_and_validation_joint_point_gate']]
    counts = {model: sum(analysis['cases'][name]['model'] == model for name in cases) for model in ['qwen', 'rr', 'meter']}
    if len(analysis['cases']) != 15 or not analysis['all_cases_complete'] or not analysis['old234_and_validation_joint_gate'] or any(n < 3 for n in counts.values()):
        raise ValueError('F7 has not met joint old234 + separate-training-validation eligibility')
    return sorted(cases)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis', required=True)
    args = parser.parse_args()
    analysis_path = pathlib.Path(args.analysis)
    analysis = json.loads(analysis_path.read_text())
    names = eligible_cases(analysis)
    validation_plan_path = DEST/'all_query_reward_gradient_validation_frozen_plan_v1.json'
    if digest(validation_plan_path) != analysis['frozen_plan_sha256']:
        raise ValueError('Validation analysis has the wrong frozen plan')
    val_plan = json.loads(validation_plan_path.read_text())
    dev_plan = json.loads((DEST/'all_query_reward_gradient_development_frozen_plan_v1.json').read_text())
    vals = {r['model']+'_'+r['protocol']: r for r in val_plan['configurations']}
    devs = {r['model']+'_'+r['protocol']: r for r in dev_plan['configurations']}
    inputs = BASE/'inputs/cohort.jsonl'
    rows = list(read_rows(inputs))
    if len(rows) != 846 or len({r['example_id'] for r in rows}) != 846:
        raise ValueError('Require original complete846 cohort, no substitution')
    dest = DEST/'all_query_reward_gradient_full_old_v1'
    create_json(dest/'full846_ids_v1.json', [r['example_id'] for r in rows])
    source_files = ['all_query_reward_gradient_full_old.py', 'freeze_all_query_reward_gradient_full_old.py',
                    'analyze_all_query_reward_gradient_full_old.py', 'all_query_reward_gradient_full_old_queue.py']
    sources = {str((ROOT/'mydata_bench/auto_research_addbase'/p).relative_to(ROOT)):
               digest(ROOT/'mydata_bench/auto_research_addbase'/p) for p in source_files}
    configs = []
    for name in names:
        if any(digest(r['path']) != r['sha256'] for r in [vals[name], devs[name]]):
            raise ValueError('Earlier frozen F7 config changed')
        val_cfg = yaml.safe_load(pathlib.Path(vals[name]['path']).read_text())
        cfg = yaml.safe_load(pathlib.Path(devs[name]['path']).read_text())
        old_out = pathlib.Path(cfg['output_dir'])
        old_path = old_out/'development/predictions.jsonl'
        old_done = json.loads((old_out/'development/completion_v1.json').read_text())
        if digest(old_path) != old_done['predictions_sha256']:
            raise ValueError('Old234 prediction provenance changed')
        validation_path = pathlib.Path(val_cfg['output_dir'])/'validation/predictions.jsonl'
        if digest(validation_path) != analysis['cases'][name]['predictions_sha256']:
            raise ValueError('Validation prediction provenance changed')
        if digest(inputs) != cfg['input_sha256'] or digest(cfg['ranking_file']) != cfg['ranking_sha256']:
            raise ValueError('Frozen full846 media or learned ranking changed')
        cfg.update(output_dir=str(dest/name), old234_output_dir=str(old_out),
            old234_predictions_sha256=digest(old_path), old234_config_sha256=digest(old_out/'development/config.json'),
            input_file=str(inputs), example_ids_file=str(dest/'full846_ids_v1.json'),
            example_ids_sha256=digest(dest/'full846_ids_v1.json'),
            engineering_ids_sha256=digest(cfg['engineering_ids_file']),
            conditions=val_cfg['conditions'], best_original_condition=val_cfg['best_original_condition'],
            stage='F7_complete_original846_descriptive',
            selection_analysis_sha256=digest(analysis_path), all846_previously_exposed=True,
            independent_confirmation=False, reserved_test_evaluated=False)
        cfg.pop('clean_source_dir', None)
        cfg['source_sha256'].update(sources)
        path = ROOT/f'mydata_bench/configs/v2_crossmodel/addbase_robust_all_query_reward_gradient_full_old_v1_{name}.yaml'
        with path.open('x') as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        configs.append(dict(model=cfg['model'], protocol=cfg['protocol'], path=str(path), sha256=digest(path)))
    create_json(dest/'frozen_plan_v1.json', dict(time=timestamp(), configurations=configs, examples=846,
        source_sha256=sources, selection_analysis_file=str(analysis_path), selection_analysis_sha256=digest(analysis_path),
        selection='every jointly eligible case; no selection by full846 F7 or reserved test outcomes',
        input_file=str(inputs), input_sha256=digest(inputs), labels_file=str(BASE/'inputs/labels.jsonl'),
        labels_sha256=digest(BASE/'inputs/labels.jsonl'), example_ids_file=str(dest/'full846_ids_v1.json'),
        example_ids_sha256=digest(dest/'full846_ids_v1.json'),
        scientific_role='previously exposed original846 full-scope descriptive gate; not an independent confirmation',
        shared_bias_magnitude=1., primary_k=48, k_neighborhood=[32, 48, 64],
        both_meter_thresholds_required=True, lower_mae_higher_suc_fail_and_total_10pp_vs_two_baselines_required=True,
        same_k_and_frozen_best_old_original_improvement_required=True,
        reports_all_conditions_tasks_splits_pairs_and_head_index_overlap=True,
        final_reserved_test_freeze_and_confirmation_still_required=True))
    print('Frozen full846 F7 follow-up:', len(configs), 'cases; all original846 retained', flush=True)


if __name__ == '__main__':
    main()
