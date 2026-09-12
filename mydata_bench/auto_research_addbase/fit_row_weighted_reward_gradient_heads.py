"""F9 CPU fitting from complete F8 per-example gradients; no reward evaluation."""
import copy
import json
import pathlib
import time
import numpy as np
import yaml
from .common import ROOT, create_json, read_rows, timestamp
from .robust_prepare import DEST, digest
from .row_weighted_gradient_ranking import rank_row_weighted_gradients


def wait_for(path, launch_path, marker):
    launch=json.loads(pathlib.Path(launch_path).read_text())
    while not path.exists():
        process=pathlib.Path('/proc')/str(launch['pid'])/'cmdline'
        if not process.exists() or marker not in process.read_bytes():
            raise RuntimeError('Required F8 process stopped before completion: '+str(path))
        time.sleep(20)


def main():
    policy_path=DEST/'f9_fit_policy_v1.json'
    policy=json.loads(policy_path.read_text())
    for name,expected in policy['source_sha256'].items():
        if digest(ROOT/name)!=expected:
            raise ValueError('Frozen F9 fitting dependency changed: '+name)
    if digest(policy['previous_fit_plan_file'])!=policy['previous_fit_plan_sha256']:
        raise ValueError('F8 frozen fitting plan changed')
    done_path=DEST/'f8_all_fit_queues_completion_v1.json'
    wait_for(done_path,DEST/'f8_fit_queue_launch_v1.json',b'ordinal_reward_gradient_fit_queue')
    if json.loads(done_path.read_text())['failures']:
        create_json(DEST/'f9_fit_prerequisite_failure_v1.json',dict(time=timestamp(),reason='F8 fit failures retained'))
        return
    audit_path=DEST/'f8_complete_fit_mechanism_audit_v1.json'
    wait_for(audit_path,DEST/'f8_complete_fit_mechanism_audit_launch_v1.json',b'audit_ordinal_fit_complete')
    audit=json.loads(audit_path.read_text())
    if not audit['all15_complete']:
        raise ValueError('All15 F8 loss/zero provenance audits required')
    previous=json.loads(pathlib.Path(policy['previous_fit_plan_file']).read_text())
    if len(previous['configurations'])!=15:
        raise ValueError('All15 model/protocol fitting cases required')
    loaded=[]
    for item in previous['configurations']:
        if digest(item['path'])!=item['sha256']:
            raise ValueError('Frozen F8 fitting config changed')
        cfg=yaml.safe_load(pathlib.Path(item['path']).read_text())
        if cfg['candidate_family']!='F8_cumulative_ordinal_all_query_supervised_heads':
            raise ValueError('F9 requires exactly the F8 cumulative ordinal gradient field')
        for field in ['training_input','fit_ids','fit_labels']:
            if digest(cfg[field+'_file'])!=cfg[field+'_sha256']:
                raise ValueError('Frozen training-only dependency changed')
        source=pathlib.Path(cfg['output_dir'])/'fit'
        done=json.loads((source/'completion_v1.json').read_text())
        gradients=source/'gradient_records_v1.jsonl'
        name=cfg['model']+'_'+cfg['protocol']
        if digest(gradients)!=done['gradients_sha256'] or audit['cases'][name]['f8_records_sha256']!=done['gradients_sha256']:
            raise ValueError('F8 gradient records do not match completed implementation audit')
        if done['examples']!=1942 or done['fitting_episode_groups']!=446:
            raise ValueError('All1942 fit rows and446 groups required')
        loaded.append((name,cfg,gradients,done))
    configurations=[]
    for name,old_cfg,gradients,previous_done in loaded:
        cfg=copy.deepcopy(old_cfg)
        ids=json.loads(pathlib.Path(cfg['fit_ids_file']).read_text())
        samples=[r for r in read_rows(cfg['training_input_file']) if r['example_id'] in set(ids)]
        if len(samples)!=1942 or {r['example_id'] for r in samples}!=set(ids) or any(r['training_partition']!='fit' for r in samples):
            raise ValueError('Full training-only sample membership required')
        expected={r['example_id']:r['training_episode_group_sha256'] for r in samples}
        records=list(read_rows(gradients))
        shape=np.asarray(next(r['gradient'] for r in records if r['status']=='ok')).shape
        ranked=rank_row_weighted_gradients(records,expected,shape)
        if ranked['fitting_examples']!=1942 or ranked['fitting_episode_groups']!=446:
            raise ValueError('Incomplete F9 reaggregation')
        out=DEST/'row_weighted_reward_gradient_v1'/name/'fit'
        create_json(out/'ranking_v1.json',dict(ranked,source_F8_gradients_file=str(gradients),
            source_F8_gradients_sha256=digest(gradients),fitting_policy_sha256=digest(policy_path)))
        cfg.pop('changes_from_F7',None)
        cfg.update(output_dir=str(out.parent),source_sha256=dict(cfg['source_sha256'],**policy['source_sha256']),
            candidate_family='F9_sample_weighted_ordinal_gradient_with_cluster_variability',
            stage='F9_independent_fit_only_CPU_reaggregation',
            ranking='abs(row_mean_gradient)-1.96*episode_cluster_SE; direction=-sign(row_mean)',
            fitting_estimand='full1942 sample mean of unchanged F8 native cumulative ordinal log loss',
            changes_from_F8=['sample-mean gradient aggregation','episode-cluster sandwich variability estimator'],
            hypothesis_origin='fit-only weighting diagnostic; no F9 evaluation outcomes observed',
            source_gradients_file=str(gradients),source_gradients_sha256=digest(gradients),
            inference_score_adjustment='none',fitting_policy_sha256=digest(policy_path))
        cfg_path=ROOT/f'mydata_bench/configs/v2_crossmodel/addbase_robust_row_weighted_reward_gradient_fit_v1_{name}.yaml'
        with cfg_path.open('x') as handle:
            yaml.safe_dump(cfg,handle,sort_keys=False)
        create_json(out/'completion_v1.json',dict(time=timestamp(),examples=1942,fitting_episode_groups=446,
            source_gradients_file=str(gradients),source_gradients_sha256=digest(gradients),
            ranking_sha256=digest(out/'ranking_v1.json'),config_sha256=digest(cfg_path),
            no_new_backbone_forward_required=True,validation_and_test_not_evaluated=True))
        configurations.append(dict(model=cfg['model'],protocol=cfg['protocol'],path=str(cfg_path),sha256=digest(cfg_path)))
        print(name,'complete F9 row-weighted fitted ranking; no evaluation',flush=True)
    create_json(DEST/'row_weighted_reward_gradient_fit_frozen_plan_v1.json',dict(time=timestamp(),
        configurations=configurations,source_sha256=policy['source_sha256'],fitting_policy_sha256=digest(policy_path),
        fit_examples=1942,fit_episode_groups=446,validation_examples=493,validation_episode_groups=108,
        F8_complete_fit_audit_sha256=digest(audit_path),unchanged_native_ordinal_loss=True,
        all_native_grade_targets_retained=True,backbone_frozen=True,
        shared_bias_magnitude=1.,primary_k=48,k_neighborhood=[32,48,64],first8_layers_excluded=True,
        output_score_adjustment='none',validation_and_test_evaluation_not_scheduled=True,final_goal_completed=False))
    create_json(DEST/'f9_fit_completion_v1.json',dict(time=timestamp(),cases=15,failures=[],
        evaluation_and_once_only_final_confirmation_still_required=True,final_goal_completed=False))


if __name__=='__main__':
    main()
