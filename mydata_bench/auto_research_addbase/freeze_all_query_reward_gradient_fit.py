"""Freeze F7 fitting after three real all-query engineering audits pass."""
import json
import pathlib
import yaml
from .common import ROOT, create_json, timestamp
from .robust_prepare import DEST, digest


def main():
    previous = json.loads((DEST/'reward_gradient_fit_frozen_plan_v1.json').read_text())
    engineering = []
    for case in ['rr_image_text', 'qwen_text_video', 'meter_text_image']:
        path = DEST/'all_query_reward_gradient_engineering_v1'/case/'engineering_audit_v1.json'
        audit = json.loads(path.read_text())
        if not audit['passed'] or any(row['gradient_forward_zero_max_difference'] != 0 for row in audit['checks']):
            raise ValueError('Required F7 real-model zero/gradient engineering failed')
        engineering.append(dict(path=str(path), sha256=digest(path)))
    new_sources = ['all_query_reward_gradient_attention.py', 'all_query_reward_gradient_engineering.py',
                   'all_query_reward_gradient_fit.py', 'freeze_all_query_reward_gradient_fit.py']
    sources = {str((ROOT/'mydata_bench/auto_research_addbase'/name).relative_to(ROOT)):
               digest(ROOT/'mydata_bench/auto_research_addbase'/name) for name in new_sources}
    configs = []
    for item in previous['configurations']:
        if digest(item['path']) != item['sha256']:
            raise ValueError('Previous fixed fitting data config changed')
        cfg = yaml.safe_load(pathlib.Path(item['path']).read_text())
        name = cfg['model']+'_'+cfg['protocol']
        for field in ['training_input', 'fit_ids', 'fit_labels']:
            if digest(cfg[field+'_file']) != cfg[field+'_sha256']:
                raise ValueError('Independent fitting input/labels changed')
        cfg.update(output_dir=str(DEST/f'all_query_reward_gradient_v1/{name}'),
            parameter='per-head signed ROI log-bias at every causal prefill query; score loss at unchanged native readout',
            attention_query_scope='all', candidate_family='F7_all_causal_query_supervised_heads',
            stage='F7_independent_fit_only', old_development_labels_used_for_fit=False,
            validation_has_been_used_for_F6_model_selection=True, validation_not_independent_confirmation=True,
            shared_inference_bias_magnitude=1., shared_k_neighborhood=[32, 48, 64], primary_k=48)
        cfg['source_sha256'].update(sources)
        path = ROOT/f'mydata_bench/configs/v2_crossmodel/addbase_robust_all_query_reward_gradient_fit_v1_{name}.yaml'
        with path.open('x') as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        configs.append(dict(model=cfg['model'], protocol=cfg['protocol'], path=str(path), sha256=digest(path)))
    create_json(DEST/'all_query_reward_gradient_fit_frozen_plan_v1.json', dict(time=timestamp(),
        configurations=configs, source_sha256=sources, engineering=engineering,
        fit_examples=previous['fit_examples'], validation_examples=previous['validation_examples'],
        fit_episode_groups=previous['fit_episode_groups'], validation_episode_groups=previous['validation_episode_groups'],
        unchanged_fit_cohort_and_native_grade_loss=True, backbone_frozen=True,
        rank_score='abs(mean_episode_gradient)-1.96*SE_episode', direction='negative sign(mean_episode_gradient)',
        shared_bias_magnitude=1., primary_k=48, k_neighborhood=[32, 48, 64],
        hypothesis='readout-only intervention was weak; earlier causal attention may alter task/visual representations',
        first8_layers_excluded=True, output_score_adjustment='none',
        validation_is_reusable_selection_data_not_new_independent_confirmation=True,
        reserved_test_reward_performance_not_opened=True, final_goal_completed=False))
    print('Frozen F7 fits:', len(configs), 'cases; same1942 fit examples, all-causal-query gradients', flush=True)


if __name__ == '__main__':
    main()
