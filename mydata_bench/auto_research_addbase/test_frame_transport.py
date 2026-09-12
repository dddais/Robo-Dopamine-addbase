import unittest
from types import SimpleNamespace
import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from .frame_transport import transport_delta, frame_transport


class FrameTransportTests(unittest.TestCase):
    def test_known_transport_and_no_accessible_target(self):
        p = torch.tensor([[.1, .2, .3, .4], [0., .2, .3, .5]], dtype=torch.float64)
        frame = [(0, 3, torch.tensor([True, False, False]))]
        d = transport_delta(p, frame, .5)
        torch.testing.assert_close(p[0] + d[0], torch.tensor([.35, .1, .15, .4], dtype=torch.float64))
        self.assertTrue(torch.equal(d[1], torch.zeros(4, dtype=torch.float64)))

    def test_native_hook_causality_frames_and_untouched_heads(self):
        torch.manual_seed(31)
        q = torch.randn(1, 4, 9, 8)
        k = torch.randn(1, 2, 9, 8)
        v = torch.eye(9).expand(1, 2, 9, 9).clone()
        module = SimpleNamespace(layer_idx=0, num_key_value_groups=2,
                                 config=SimpleNamespace(_attn_implementation='sdpa'))
        layers = [SimpleNamespace(self_attn=module)]
        spans = [SimpleNamespace(start=1, end=4), SimpleNamespace(start=5, end=8)]
        native = ALL_ATTENTION_FUNCTIONS['sdpa']
        base, _ = native(module, q, k, v, None, scaling=8**-.5)
        with frame_transport(layers, [(0, 1), (0, 3)], [2, 7], spans, .5) as diag:
            result, _ = ALL_ATTENTION_FUNCTIONS['addbase_frame_transport_v1'](module, q, k, v, None, scaling=8**-.5)
        self.assertTrue(torch.equal(base[:, :, [0, 2]], result[:, :, [0, 2]]))
        for span in spans:
            torch.testing.assert_close(base[..., span.start:span.end].sum(-1), result[..., span.start:span.end].sum(-1), atol=2e-7, rtol=1e-6)
        torch.testing.assert_close(base[..., [0, 4, 8]], result[..., [0, 4, 8]], atol=0, rtol=0)
        self.assertEqual(diag['max_causal_change'], 0)
        self.assertGreaterEqual(diag['minimum_changed_probability'], -1e-7)
        self.assertEqual(module.config._attn_implementation, 'sdpa')
        with frame_transport(layers, [(0, 1)], [2, 7], spans, 0):
            zero, _ = ALL_ATTENTION_FUNCTIONS['addbase_frame_transport_v1'](module, q, k, v, None, scaling=8**-.5)
        self.assertTrue(torch.equal(base, zero))

    def test_full_target_and_zero_are_exact(self):
        p = torch.softmax(torch.randn(2, 3, 7), -1)
        frames = [(0, 7, torch.ones(7, dtype=torch.bool))]
        self.assertTrue(torch.equal(transport_delta(p, frames, .5), torch.zeros_like(p)))
        self.assertTrue(torch.equal(transport_delta(p, frames, 0), torch.zeros_like(p)))


if __name__ == '__main__':
    unittest.main()
