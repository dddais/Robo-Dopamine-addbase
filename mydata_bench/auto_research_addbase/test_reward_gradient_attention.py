from types import SimpleNamespace
import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from mydata_bench.auto_research_addbase.reward_gradient_attention import reward_gradient_steering, native_grade_loss


def test_bias_gradient_matches_finite_difference_and_zero_is_exact():
    torch.manual_seed(10650)
    q, k, v = [torch.randn(1, 4, 7, 4) for _ in range(3)]
    module = SimpleNamespace(layer_idx=0, num_key_value_groups=1,
        config=SimpleNamespace(_attn_implementation='sdpa'))
    layers = [SimpleNamespace(self_attn=module)]
    native = ALL_ATTENTION_FUNCTIONS['sdpa']
    base, _ = native(module, q, k, v, None, scaling=.5)
    probe = torch.randn_like(base)

    def run(biases):
        with reward_gradient_steering(layers, biases, [1], [0, 1, 2], 5, [(0, 0), (0, 2)]):
            return ALL_ATTENTION_FUNCTIONS['addbase_reward_gradient_readout_v1'](module, q, k, v, None, scaling=.5)[0]

    biases = torch.zeros(1, 4, requires_grad=True)
    actual = run(biases)
    assert torch.equal(actual, base)
    grad, = torch.autograd.grad((actual*probe).sum(), biases)
    assert grad[0, 1] == grad[0, 3] == 0
    delta = torch.zeros_like(biases)
    delta[0, 0] = .001
    finite = (((run(delta)-run(-delta))*probe).sum()/.002)
    torch.testing.assert_close(grad[0, 0], finite, atol=2e-4, rtol=2e-3)
    moved = run(delta*1000)
    assert torch.equal(moved[:, [0, 1, 2, 3, 4, 6]], base[:, [0, 1, 2, 3, 4, 6]])
    assert torch.equal(moved[:, :, [1, 3]], base[:, :, [1, 3]])
    assert module.config._attn_implementation == 'sdpa'


def test_all_training_grades_have_native_targets():
    for model, bins in [('qwen', 5), ('rr', 5), ('meter', 10)]:
        for grade in range(1, 6):
            logits = torch.zeros(bins, requires_grad=True)
            loss = native_grade_loss(logits, grade, model)
            loss.backward()
            assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
            assert abs(float(logits.grad.sum())) < 1e-6
            if grade == 3:
                assert logits.grad[0] > 0 and logits.grad[-1] > 0
