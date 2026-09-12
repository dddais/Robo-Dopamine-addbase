"""Freeze F7 all-query learned heads, controls and old-development inference after all fits."""
import json
import pathlib
import yaml
from .common import ROOT,create_json,timestamp
from .robust_prepare import DEST,digest


def main():
    policy_path=DEST/'f7_evaluation_policy_v1.json'
    policy=json.loads(policy_path.read_text())
    for path,sha in policy['evaluation_source_sha256'].items():
        if digest(ROOT/path)!=sha:
            raise ValueError('Prospective F7 evaluation source changed: '+path)
    plan=json.loads((DEST/'frame_transport_frozen_plan_v1.json').read_text())
    configs=[]
    for item in plan['configurations']:
        name=item['model']+'_'+item['protocol']
        source=DEST/'all_query_reward_gradient_v1'/name/'fit'
        done=json.loads((source/'completion_v1.json').read_text())
        ranking=source/'ranking_v1.json'
        if digest(ranking)!=done['ranking_sha256']:
            raise ValueError('Learned fitting ranking changed')
        cfg=yaml.safe_load(pathlib.Path(item['path']).read_text())
        conditions=[dict(name='baseline',k=0,cached_reference=True),
                    dict(name='native_generation_baseline',k=0,kind='native_generation',cached_reference=True)]
        for k in [32,48,64]:
            conditions.extend([
                dict(name=f'original_all_k{k}',k=k,kind='bias',bias=6.,scope='all_frames',ranking='original',cached_reference=True),
                dict(name=f'reward_gradient_k{k}',k=k,kind='reward_gradient',bias=1.,scope='all_frames',signed=True)])
        conditions.extend([
            dict(name='reward_gradient_zero_k48',k=48,kind='reward_gradient',bias=0.,scope='all_frames',signed=True),
            dict(name='reward_gradient_wrong_k48',k=48,kind='reward_gradient',bias=1.,scope='all_frames',signed=True,region='wrong'),
            dict(name='reward_gradient_low_k48',k=48,kind='reward_gradient',bias=1.,scope='all_frames',signed=True,head_group='low'),
            dict(name='raw_all_k48',k=48,kind='reward_gradient',bias=1.,scope='all_frames',query_scope='all',ranking='original',signed=False),
            dict(name='learned_positive_k48',k=48,kind='reward_gradient',bias=1.,scope='all_frames',signed=False),
            dict(name='learned_readout_k48',k=48,kind='reward_gradient',bias=1.,scope='all_frames',signed=True,query_scope='readout')])
        cfg.update(clean_source_dir=cfg['output_dir'],output_dir=str(DEST/f'all_query_reward_gradient_development_v1/{name}'),
            original_ranking_file=cfg['ranking_file'],original_ranking_sha256=cfg['ranking_sha256'],
            ranking_file=str(ranking),ranking_sha256=digest(ranking),conditions=conditions,
            primary_k=48,k_neighborhood=[32,48,64],stage='F7_all_query_externally_fitted_heads_old234_development',
            output_score_adjustment='none',ranking_supervised_source='separate official train fitting partition only')
        for filename in ['reward_gradient_attention.py','all_query_reward_gradient_attention.py','all_query_reward_gradient_runtime.py','all_query_reward_gradient_development.py','freeze_all_query_reward_gradient_development.py']:
            path=ROOT/'mydata_bench/auto_research_addbase'/filename
            cfg['source_sha256'][str(path.relative_to(ROOT))]=digest(path)
        cfg['source_sha256'].update(policy['evaluation_source_sha256'])
        cfg['evaluation_policy_sha256']=digest(policy_path)
        path=ROOT/f'mydata_bench/configs/v2_crossmodel/addbase_robust_all_query_reward_gradient_development_v1_{name}.yaml'
        with path.open('x') as f:yaml.safe_dump(cfg,f,sort_keys=False)
        configs.append(dict(model=item['model'],protocol=item['protocol'],path=str(path),sha256=digest(path)))
    create_json(DEST/'all_query_reward_gradient_development_frozen_plan_v1.json',dict(time=timestamp(),configurations=configs,
        learned_rankings_frozen_after_external_fit_before_old_development_evaluation=True,
        shared_bias_magnitude=1.,shared_k_neighborhood=[32,48,64],primary_k=48,
        output_score_adjustment='none',external_test_performance_not_opened=True,
        training_validation_selection_still_required=True,validation_reused_after_F6_not_new_confirmation=True))


if __name__=='__main__':
    main()
