import json
import pytest
import yaml
from mydata_bench.auto_research_addbase import analyze_ordinal_reward_gradient_validation as validation
from mydata_bench.auto_research_addbase.f8_reference_comparison import f7_primary_rows
from mydata_bench.auto_research_addbase.robust_prepare import digest
from mydata_bench.auto_research_addbase.freeze_ordinal_reward_gradient_full_old import eligible_cases


def test_f8_incomplete_validation_blocks_label_join(tmp_path, monkeypatch):
    dummy=tmp_path/'dummy.json'
    dummy.write_text('{}')
    ids=tmp_path/'ids.json'
    ids.write_text(json.dumps(['example-a']))
    configs=[]
    for model in ['qwen','rr','meter']:
        for protocol in ['text_video','video_text','text_image','image_text','interleaved']:
            path=tmp_path/(model+'_'+protocol+'.yaml')
            path.write_text(yaml.safe_dump(dict(model=model,protocol=protocol,output_dir=str(tmp_path/model/protocol))))
            configs.append(dict(model=model,protocol=protocol,path=str(path),sha256=digest(path)))
    plan=dict(configurations=configs,inference_source_sha256={})
    for field in ['input','example_ids','analysis_labels','policy']:
        path=ids if field=='example_ids' else dummy
        plan[field+'_file']=str(path)
        plan[field+'_sha256']=digest(path)
    (tmp_path/'ordinal_reward_gradient_validation_frozen_plan_v1.json').write_text(json.dumps(plan))
    monkeypatch.setattr(validation,'DEST',tmp_path)
    def forbidden_label_parse(path):
        raise AssertionError('Incomplete F8 cases must block label parsing')
    monkeypatch.setattr(validation,'read_rows',forbidden_label_parse)
    with pytest.raises(FileNotFoundError,match='completion_v1.json'):
        validation.main()


def test_fixed_ce_comparator_preserves_errors_rejects_protocol_and_missing_ids(tmp_path):
    cfg=dict(model='rr',protocol='image_text',model_path='frozen',input_sha256='input',
        example_ids_file='fixed-ids',output_dir=str(tmp_path/'ce'),
        conditions=[dict(name='reward_gradient_k48',k=48)])
    path=tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(cfg))
    policy=dict(fixed_F7_comparator_configs={'validation':{'rr_image_text':dict(path=str(path),sha256=digest(path))}})
    stage=tmp_path/'ce/validation'
    stage.mkdir(parents=True)
    (stage/'config.json').write_text(json.dumps(cfg))
    records=stage/'predictions.jsonl'
    records.write_text(json.dumps(dict(example_id='a',condition='reward_gradient_k48',status='ok'))+'\n'+
        json.dumps(dict(example_id='b',condition='reward_gradient_k48',status='error',error='preserved failure'))+'\n')
    (stage/'completion_v1.json').write_text(json.dumps(dict(predictions_sha256=digest(records))))
    result=f7_primary_rows(policy,'rr_image_text','validation',cfg,['a','b'])
    assert len(result)==2 and result[1]['status']=='error'
    with pytest.raises(ValueError,match='mismatch: protocol'):
        f7_primary_rows(policy,'rr_image_text','validation',dict(cfg,protocol='text_image'),['a','b'])
    with pytest.raises(ValueError,match='incomplete or duplicated'):
        f7_primary_rows(policy,'rr_image_text','validation',cfg,['a','b','c'])


def test_f8_scope_cannot_substitute_extra_successes_from_another_model():
    cases={m+'_'+str(i):dict(model=m,old234_and_validation_joint_point_gate=i<3)
           for m in ['qwen','rr','meter'] for i in range(5)}
    analysis=dict(cases=cases,all_cases_complete=True,old234_and_validation_joint_gate=True)
    assert len(eligible_cases(analysis))==9
    cases['rr_2']['old234_and_validation_joint_point_gate']=False
    cases['meter_3']['old234_and_validation_joint_point_gate']=True
    with pytest.raises(ValueError,match='joint old234'):
        eligible_cases(analysis)
