import json
import pytest
import yaml
from mydata_bench.auto_research_addbase.freeze_all_query_reward_gradient_confirmation import selected_confirmation_cases
from mydata_bench.auto_research_addbase import analyze_all_query_reward_gradient_confirmation as analyzer
from mydata_bench.auto_research_addbase.robust_prepare import digest


def test_reserved_confirmation_cannot_replace_failed_original_model_scope():
    cases = {f'{model}_{i}': dict(model=model, full846_point_gate=i < 3)
             for model in ['qwen', 'rr', 'meter'] for i in range(5)}
    analysis = dict(cases=cases, full846_original_scope_gate=True)
    assert len(selected_confirmation_cases(analysis)) == 9
    cases['qwen_2']['full846_point_gate'] = False
    cases['meter_3']['full846_point_gate'] = True
    with pytest.raises(ValueError, match='scope is not met'):
        selected_confirmation_cases(analysis)


def test_incomplete_reserved_outputs_block_test_label_join(tmp_path, monkeypatch):
    root = tmp_path/'all_query_reward_gradient_confirmation_v1'
    root.mkdir()
    dummy = tmp_path/'dummy.json'
    dummy.write_text('{}')
    ids = tmp_path/'ids.json'
    ids.write_text(json.dumps(['reserved-'+str(i) for i in range(2831)]))
    configurations=[]
    selection=dict(full846_original_scope_gate=True,cases={})
    for model in ['rr','qwen','meter']:
        for protocol in ['image_text','text_image','text_video']:
            name=model+'_'+protocol
            cfg=tmp_path/(name+'.yaml')
            cfg.write_text(yaml.safe_dump(dict(output_dir=str(root/name),model=model,protocol=protocol)))
            configurations.append(dict(model=model,protocol=protocol,path=str(cfg),sha256=digest(cfg)))
            selection['cases'][name]=dict(model=model,full846_point_gate=True)
    selection_path=tmp_path/'selection.json'
    selection_path.write_text(json.dumps(selection))
    plan = dict(source_sha256={}, expected_cases=9, configurations=configurations)
    for field in ['input', 'example_ids', 'groups', 'policy', 'confirmation_claim', 'selection_analysis']:
        path = ids if field == 'example_ids' else selection_path if field == 'selection_analysis' else dummy
        plan[field+'_file'] = str(path)
        plan[field+'_sha256'] = digest(path)
    (root/'frozen_plan_v1.json').write_text(json.dumps(plan))
    monkeypatch.setattr(analyzer, 'DEST', tmp_path)
    def forbidden_read_rows(path):
        raise AssertionError('No label/input rows may be parsed before all frozen cases finish')
    monkeypatch.setattr(analyzer, 'read_rows', forbidden_read_rows)
    with pytest.raises(FileNotFoundError, match='completion_v1.json'):
        analyzer.main()
    assert not (root/'performance_opening_v1.json').exists()


def test_duplicate_or_omitted_reserved_cases_are_rejected_before_join():
    configurations=[dict(model=m,protocol=str(i),path=m+'_'+str(i)) for m in ['rr','qwen','meter'] for i in range(3)]
    selection=dict(full846_original_scope_gate=True,cases={r['model']+'_'+r['protocol']:dict(model=r['model'],full846_point_gate=True) for r in configurations})
    assert len(analyzer.validate_configuration_scope(configurations,selection))==9
    duplicate=configurations[:-1]+[configurations[0]]
    with pytest.raises(ValueError,match='Duplicate'):
        analyzer.validate_configuration_scope(duplicate,selection)
    with pytest.raises(ValueError,match='every eligible'):
        analyzer.validate_configuration_scope(configurations[:-1],selection)
