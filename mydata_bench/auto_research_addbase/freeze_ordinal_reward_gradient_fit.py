"""Freeze F8 fitting from the F7 cohort after three ordinal engineering audits."""
import copy
import json
import pathlib
import yaml
from .common import ROOT, create_json, timestamp
from .robust_prepare import DEST, digest


def ordinal_config(previous, sources):
    cfg = copy.deepcopy(previous)
    name = cfg['model']+'_'+cfg['protocol']
    cfg.update(output_dir=str(DEST/f'ordinal_reward_gradient_v1/{name}'),
        loss='mean cumulative Bernoulli log loss over every native ordered-bin boundary; '
             'same native grade targets as F7, including Meter adjacent-bin interpolation',
        parameter='per-head signed ROI log-bias at every causal prefill query; unchanged native score readout',
        candidate_family='F8_cumulative_ordinal_all_query_supervised_heads',
        stage='F8_independent_fit_only',
        changes_from_F7=['fitting objective only'],
        hypothesis_origin='posthoc F7 finite-step fit diagnostics; no F8 validation performance seen',
        validation_has_been_used_for_model_selection=True,
        validation_not_independent_confirmation=True,
        evaluation_endpoint_thresholds_used_in_loss=False,
        inference_score_adjustment='none')
    cfg['source_sha256'].update(sources)
    return cfg


def main():
    queue_path = DEST/'f8_fit_queue_plan_v1.json'
    queue = json.loads(queue_path.read_text())
    for path, expected in queue['dependencies_sha256'].items():
        if digest(ROOT/path) != expected:
            raise ValueError('Frozen prospective F8 fitting source changed: '+path)
    previous_path = DEST/'all_query_reward_gradient_fit_frozen_plan_v1.json'
    if digest(previous_path) != queue['previous_fit_plan_sha256']:
        raise ValueError('F7 fitting cohort plan changed')
    previous = json.loads(previous_path.read_text())
    engineering = []
    for case in ['rr_image_text', 'qwen_text_video', 'meter_text_image']:
        path = DEST/'ordinal_reward_gradient_engineering_v1'/case/'engineering_audit_v1.json'
        audit = json.loads(path.read_text())
        if not audit['passed'] or not audit['synthetic_probe_not_training'] or audit['labels_read']:
            raise ValueError('Required F8 synthetic engineering failed')
        launch = json.loads((path.parent/'launch_v1.json').read_text())
        for field, source in [('source_sha256', 'ordinal_reward_gradient_engineering.py'),
                              ('loss_source_sha256', 'native_cumulative_ordinal_loss.py')]:
            if launch[field] != queue['dependencies_sha256']['mydata_bench/auto_research_addbase/'+source]:
                raise ValueError('Engineering used an unregistered F8 source')
        engineering.append(dict(path=str(path), sha256=digest(path)))
    configs = []
    for item in previous['configurations']:
        if digest(item['path']) != item['sha256']:
            raise ValueError('Previous fixed fitting config changed')
        old = yaml.safe_load(pathlib.Path(item['path']).read_text())
        for field in ['training_input', 'fit_ids', 'fit_labels']:
            if digest(old[field+'_file']) != old[field+'_sha256']:
                raise ValueError('Independent fitting cohort dependency changed: '+field)
        cfg = ordinal_config(old, queue['dependencies_sha256'])
        name = cfg['model']+'_'+cfg['protocol']
        path = ROOT/f'mydata_bench/configs/v2_crossmodel/addbase_robust_ordinal_reward_gradient_fit_v1_{name}.yaml'
        with path.open('x') as handle:
            yaml.safe_dump(cfg, handle, sort_keys=False)
        configs.append(dict(model=cfg['model'], protocol=cfg['protocol'], path=str(path), sha256=digest(path)))
    if len(configs) != 15:
        raise ValueError('All three models and five protocols required')
    create_json(DEST/'ordinal_reward_gradient_fit_frozen_plan_v1.json', dict(time=timestamp(),
        configurations=configs, source_sha256=queue['dependencies_sha256'], engineering=engineering,
        prospective_queue_sha256=digest(queue_path), previous_fit_plan_sha256=digest(previous_path),
        fit_examples=previous['fit_examples'], validation_examples=previous['validation_examples'],
        fit_episode_groups=previous['fit_episode_groups'], validation_episode_groups=previous['validation_episode_groups'],
        unchanged_fit_cohort=True, unchanged_native_grade_targets=True, unchanged_native_loss=False,
        backbone_frozen=True, rank_score='abs(mean_episode_gradient)-1.96*SE_episode',
        direction='negative sign(mean_episode_gradient)', shared_bias_magnitude=1.,
        primary_k=48, k_neighborhood=[32, 48, 64], first8_layers_excluded=True,
        fitting_objective='native cumulative ordinal log loss',
        hypothesis='CE rewards correct-bin probability without distinguishing error distance; '
                   'ordered cumulative loss tests ordinal alignment without changing inference',
        output_score_adjustment='none', all_native_rating_bins_retained=True,
        validation_is_reusable_selection_data_not_new_independent_confirmation=True,
        reserved_test_not_scheduled=True, final_goal_completed=False))
    print('Frozen F8:', len(configs), 'fits; cumulative ordinal loss, unchanged cohort and inference', flush=True)


if __name__ == '__main__':
    main()
