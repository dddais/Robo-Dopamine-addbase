import json
import pytest
from mydata_bench.auto_research_addbase.all_query_reward_gradient_full_old import verified_cache
from mydata_bench.auto_research_addbase.freeze_all_query_reward_gradient_full_old import eligible_cases
from mydata_bench.auto_research_addbase.robust_prepare import digest


def test_full846_requires_same_three_joint_inputs_in_every_model():
    cases = {f'{model}_{i}': dict(model=model, old234_and_validation_joint_point_gate=i < 3)
             for model in ['qwen', 'rr', 'meter'] for i in range(5)}
    analysis = dict(cases=cases, all_cases_complete=True, old234_and_validation_joint_gate=True)
    assert len(eligible_cases(analysis)) == 9
    # A high overall count or a stale success boolean cannot rescue one model.
    cases['rr_2']['old234_and_validation_joint_point_gate'] = False
    cases['meter_3']['old234_and_validation_joint_point_gate'] = True
    with pytest.raises(ValueError, match='joint old234'):
        eligible_cases(analysis)


def test_cached_failures_are_preserved_and_changed_conditions_rejected(tmp_path):
    stage = tmp_path/'old234/development'
    stage.mkdir(parents=True)
    ids = tmp_path/'ids.json'
    ids.write_text(json.dumps(['a', 'b']))
    old = dict(model='rr', model_path='frozen-model', protocol='image_text', input_sha256='input-hash',
        ranking_sha256='ranking-hash', original_ranking_sha256='original-hash', example_ids_file=str(ids),
        conditions=[dict(name='baseline', k=0, cached_reference=True)])
    (stage/'config.json').write_text(json.dumps(old))
    predictions = stage/'predictions.jsonl'
    predictions.write_text(json.dumps(dict(example_id='a', condition='baseline', status='ok', native_prediction=3))+'\n'+
        json.dumps(dict(example_id='b', condition='baseline', status='error', error='original error retained'))+'\n')
    (stage/'completion_v1.json').write_text(json.dumps(dict(predictions_sha256=digest(predictions))))
    cfg = dict(old, old234_output_dir=str(stage.parent), old234_predictions_sha256=digest(predictions),
        old234_config_sha256=digest(stage/'config.json'), conditions=[dict(name='baseline', k=0)])
    cache, _ = verified_cache(cfg)
    assert len(cache) == 2 and cache[('b', 'baseline')]['status'] == 'error'
    cfg['conditions'][0]['k'] = 1
    with pytest.raises(ValueError, match='condition changed'):
        verified_cache(cfg)
