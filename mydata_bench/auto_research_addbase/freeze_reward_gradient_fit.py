"""Freeze F6 fit cohorts and sources after label-free media preparation."""
import json
import pathlib
import yaml
from .common import ROOT, create_json, read_rows, timestamp
from .prepare_external_head_training import OUT
from .robust_prepare import DEST, PROTOCOLS, digest


def main():
    done=json.loads((OUT/'grounding_v1/grounding_completion_v1.json').read_text())
    inputs=OUT/'grounding_v1/grounded_inputs_v1.jsonl'
    if digest(inputs)!=done['output_sha256']:
        raise ValueError('Frozen grounded training inputs changed')
    rows=list(read_rows(inputs))
    if len({r['example_id'] for r in rows})!=len(rows) or any(not r['training_eligible'] for r in rows):
        raise ValueError('Invalid eligible training rows')
    fit=[r for r in rows if r['training_partition']=='fit']
    val=[r for r in rows if r['training_partition']=='validation']
    if len(fit)+len(val)!=len(rows) or {r['training_episode_group_sha256'] for r in fit} & {r['training_episode_group_sha256'] for r in val}:
        raise ValueError('Fitting and validation are not complete disjoint episode groups')
    selected=json.loads((OUT/'training_selection_completion_v1.json').read_text())
    source_labels=OUT/'training_labels_v1.jsonl'
    if digest(source_labels)!=selected['training_labels_sha256']:
        raise ValueError('Official training labels changed')
    labels={r['example_id']:r for r in read_rows(source_labels)}
    for part,samples in [('fit',fit),('training_validation',val)]:
        ids=[r['example_id'] for r in samples]
        create_json(OUT/(part+'_ids_v1.json'),ids)
        with (OUT/(part+'_labels_only_v1.jsonl')).open('x') as handle:
            for eid in ids:
                handle.write(json.dumps(labels[eid],ensure_ascii=False)+'\n')
    engineering=[]
    for case in ['rr_image_text','qwen_text_video','meter_text_image']:
        path=DEST/'reward_gradient_engineering_v1'/case/'engineering_audit_v1.json'
        audit=json.loads(path.read_text())
        if not audit['passed']:
            raise ValueError('Required actual-model gradient engineering failed')
        engineering.append(dict(path=str(path),sha256=digest(path)))
    names=['reward_gradient_attention.py','reward_gradient_engineering.py','reward_gradient_ranking.py',
           'reward_gradient_fit.py','freeze_reward_gradient_fit.py','score_branches.py','discrete_runtime.py']
    source_files=[ROOT/'mydata_bench/auto_research_addbase'/name for name in names]
    source_files += [ROOT/'mydata_bench/meter_eval'/name for name in ['runtime.py','model.py','readout.py']]
    sources={str(path.relative_to(ROOT)):digest(path) for path in source_files}
    configs=[]
    for model in ['qwen','rr','meter']:
        for protocol in PROTOCOLS:
            original=ROOT/f'mydata_bench/configs/v2_crossmodel/addbase_robust_frame_transport_v1_{model}_{protocol}.yaml'
            old=yaml.safe_load(original.read_text())
            cfg={k:old[k] for k in ['model','protocol','model_path','processor_path','max_new_tokens'] if k in old}
            cfg.update(output_dir=str(DEST/f'reward_gradient_v1/{model}_{protocol}'),
                training_input_file=str(inputs),training_input_sha256=digest(inputs),
                fit_ids_file=str(OUT/'fit_ids_v1.json'),fit_ids_sha256=digest(OUT/'fit_ids_v1.json'),
                fit_labels_file=str(OUT/'fit_labels_only_v1.jsonl'),fit_labels_sha256=digest(OUT/'fit_labels_only_v1.jsonl'),
                source_sha256=sources,loss='native1–5 CE; Meter linear adjacent-bin target for each native grade',
                parameter='per-head signed ROI log-bias at actual score query only',
                gradient_at_bias=0.,ranking='abs(mean_episode_gradient)-1.96*SE_episode; direction negative sign(mean)',
                inference_bias_magnitude=1.,primary_k=48,k_neighborhood=[32,48,64],
                backbone_frozen=True,old_development_labels_used_for_fit=False,held_out_labels_used_for_fit=False)
            path=ROOT/f'mydata_bench/configs/v2_crossmodel/addbase_robust_reward_gradient_fit_v1_{model}_{protocol}.yaml'
            with path.open('x') as f:yaml.safe_dump(cfg,f,sort_keys=False)
            configs.append(dict(model=model,protocol=protocol,path=str(path),sha256=digest(path)))
    create_json(DEST/'reward_gradient_fit_frozen_plan_v1.json',dict(time=timestamp(),configurations=configs,source_sha256=sources,
        fit_examples=len(fit),validation_examples=len(val),fit_episode_groups=len({r['training_episode_group_sha256'] for r in fit}),
        validation_episode_groups=len({r['training_episode_group_sha256'] for r in val}),engineering=engineering,
        shared_inference_parameters=dict(bias_magnitude=1.,primary_k=48,k_neighborhood=[32,48,64]),
        candidate_fit_and_validation_performance_not_seen=True,reserved_test_performance_not_opened=True,
        fitting_changes_only_head_order_and_direction=True,all_native_rating_bins_retained=True,
        final_evaluation_config_and_learned_ranking_hashes_must_be_frozen_before_evaluation=True))
    print('Frozen F6 fitting:',len(configs),'cases;',len(fit),'fit examples;',len(val),'validation examples')


if __name__=='__main__':
    main()
