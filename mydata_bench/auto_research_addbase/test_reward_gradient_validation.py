import pytest
from mydata_bench.auto_research_addbase.reward_gradient_validation import inference_value


def test_no_grounding_uses_baseline_without_calling_region_or_model():
    class Runtime:
        def predict(self, *args):
            raise AssertionError('No region/model call is needed for fallback')
    # The fallback deliberately retains a middle-bin result.
    baseline = dict(score_logits=[0., 1., 3., 1., 0.], native_prediction=3, progress=.5)
    result = inference_value(Runtime(), {'grounding_status': 'baseline_fallback'}, {},
        {'kind': 'reward_gradient', 'k': 48}, [(8, 1)], baseline)
    assert result['native_prediction'] == 3 and result['score_logits'] == baseline['score_logits']
    assert result['baseline_fallback'] and result['progress'] == .5
    assert 'baseline_fallback' not in baseline
    with pytest.raises(ValueError, match='No valid baseline'):
        inference_value(Runtime(), {'grounding_status': 'baseline_fallback'}, {}, {'k': 48}, [(8, 1)], None)
