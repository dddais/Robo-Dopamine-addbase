from types import SimpleNamespace
import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from mydata_bench.auto_research_addbase.all_query_reward_gradient_attention import all_query_reward_gradient_steering


def test_all_query_zero_causality_unselected_heads_and_gradient():
    torch.manual_seed(20260910)
    q, k, v = [torch.randn(1, 4, 7, 4) for _ in range(3)]
    module = SimpleNamespace(layer_idx=0, num_key_value_groups=1,
        config=SimpleNamespace(_attn_implementation='sdpa'))
    layers = [SimpleNamespace(self_attn=module)]
    native = ALL_ATTENTION_FUNCTIONS['sdpa']
    base, _ = native(module, q, k, v, None, scaling=.5)
    probe = torch.randn_like(base)
    def run(biases, mask=None):
        with all_query_reward_gradient_steering(layers, biases, [3], [2, 3, 4], 6, [(0, 0), (0, 2)]):
            return ALL_ATTENTION_FUNCTIONS['addbase_reward_gradient_all_queries_v1'](module, q, k, v, mask, scaling=.5)[0]
    biases = torch.zeros(1, 4, requires_grad=True)
    zero = run(biases)
    assert torch.equal(zero, base)
    grad, = torch.autograd.grad((zero*probe).sum(), biases)
    assert grad[0, 1] == grad[0, 3] == 0
    delta = torch.zeros_like(biases)
    delta[0, 0] = .001
    finite = (((run(delta)-run(-delta))*probe).sum()/.002)
    torch.testing.assert_close(grad[0, 0], finite, atol=3e-4, rtol=2e-3)
    changed = run(delta*1000)
    assert torch.equal(changed[:, :2], base[:, :2])  # No visual key is causally visible yet.
    assert torch.equal(changed[:, :, [1, 3]], base[:, :, [1, 3]])
    assert not torch.equal(changed[:, 3:6, 0], base[:, 3:6, 0])  # Genuine pre-readout intervention.
    mask = torch.ones(1, 1, 7, 7, dtype=torch.bool).tril()
    torch.testing.assert_close(run(delta*1000, mask), changed, atol=1e-6, rtol=1e-6)
    assert module.config._attn_implementation == 'sdpa'
