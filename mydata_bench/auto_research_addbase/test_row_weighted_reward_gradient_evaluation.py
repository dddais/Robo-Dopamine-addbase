import json
import pytest
import yaml
from mydata_bench.auto_research_addbase import analyze_row_weighted_reward_gradient_validation as validation
from mydata_bench.auto_research_addbase.f9_reference_comparison import f8_primary_rows
from mydata_bench.auto_research_addbase.robust_prepare import digest
from mydata_bench.auto_research_addbase.freeze_row_weighted_reward_gradient_full_old import eligible_cases


def test_f9_incomplete_validation_blocks_label_join(tmp_path, monkeypatch):
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
    (tmp_path/'row_weighted_reward_gradient_validation_frozen_plan_v1.json').write_text(json.dumps(plan))
    monkeypatch.setattr(validation,'DEST',tmp_path)
    def forbidden_label_parse(path):
        raise AssertionError('Incomplete F9 cases must block label parsing')
    monkeypatch.setattr(validation,'read_rows',forbidden_label_parse)
    with pytest.raises(FileNotFoundError,match='completion_v1.json'):
        validation.main()


def test_fixed_ordinal_comparator_preserves_errors_rejects_protocol_and_missing_ids(tmp_path):
    cfg=dict(model='rr',protocol='image_text',model_path='frozen',input_sha256='input',
        example_ids_file='fixed-ids',output_dir=str(tmp_path/'ce'),
        conditions=[dict(name='reward_gradient_k48',k=48)])
    path=tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(cfg))
    policy=dict(fixed_F8_comparator_configs={'validation':{'rr_image_text':dict(path=str(path),sha256=digest(path))}})
    stage=tmp_path/'ce/validation'
    stage.mkdir(parents=True)
    (stage/'config.json').write_text(json.dumps(cfg))
    records=stage/'predictions.jsonl'
    records.write_text(json.dumps(dict(example_id='a',condition='reward_gradient_k48',status='ok'))+'\n'+
        json.dumps(dict(example_id='b',condition='reward_gradient_k48',status='error',error='preserved failure'))+'\n')
    (stage/'completion_v1.json').write_text(json.dumps(dict(predictions_sha256=digest(records))))
    result=f8_primary_rows(policy,'rr_image_text','validation',cfg,['a','b'])
    assert len(result)==2 and result[1]['status']=='error'
    with pytest.raises(ValueError,match='mismatch: protocol'):
        f8_primary_rows(policy,'rr_image_text','validation',dict(cfg,protocol='text_image'),['a','b'])
    with pytest.raises(ValueError,match='incomplete or duplicated'):
        f8_primary_rows(policy,'rr_image_text','validation',cfg,['a','b','c'])


def test_f9_scope_cannot_substitute_extra_successes_from_another_model():
    cases={m+'_'+str(i):dict(model=m,old234_and_validation_joint_point_gate=i<3)
           for m in ['qwen','rr','meter'] for i in range(5)}
    analysis=dict(cases=cases,all_cases_complete=True,old234_and_validation_joint_gate=True)
    assert len(eligible_cases(analysis))==9
    cases['rr_2']['old234_and_validation_joint_point_gate']=False
    cases['meter_3']['old234_and_validation_joint_point_gate']=True
    with pytest.raises(ValueError,match='joint old234'):
        eligible_cases(analysis)


def test_f9_verifies_referenced_f8_gradients_without_local_copy(tmp_path):
    from mydata_bench.auto_research_addbase.f9_fit_provenance import verify_fit_artifacts
    source=tmp_path/'original_f8/gradient_records_v1.jsonl'
    source.parent.mkdir()
    source.write_text('{"example_id":"preserved"}\n')
    out=tmp_path/'f9/fit'
    out.mkdir(parents=True)
    ranking=out/'ranking_v1.json'
    ranking.write_text(json.dumps(dict(source_F8_gradients_file=str(source),source_F8_gradients_sha256=digest(source))))
    cfg=dict(source_gradients_file=str(source),source_gradients_sha256=digest(source))
    for field in ['training_input','fit_ids','fit_labels']:
        p=tmp_path/(field+'.json')
        p.write_text('{}')
        cfg[field+'_file']=str(p)
        cfg[field+'_sha256']=digest(p)
    done=out/'completion_v1.json'
    done.write_text(json.dumps(dict(examples=1942,fitting_episode_groups=446,config_sha256='frozen-config',
        ranking_sha256=digest(ranking),source_gradients_file=str(source),source_gradients_sha256=digest(source))))
    expected=digest(source)
    assert not (out/'gradient_records_v1.jsonl').exists()
    assert verify_fit_artifacts(done,cfg,expected,'frozen-config')['examples']==1942
    with pytest.raises(ValueError,match='configuration'):
        verify_fit_artifacts(done,cfg,expected,'wrong-config')
    source.write_text('{"example_id":"tampered"}\n')
    with pytest.raises(ValueError,match='completed F8 audit'):
        verify_fit_artifacts(done,cfg,expected,'frozen-config')
