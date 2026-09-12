"""Numerical invariants catch silent no-op, leakage, and score-shift bugs."""
import copy,unittest
from types import SimpleNamespace
import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from .attention import steering
class AttentionTests(unittest.TestCase):
 def setUp(self):
  torch.manual_seed(21)
  self.q=torch.randn(1,2,6,8);self.k=torch.randn(1,2,6,8)
  self.v=torch.eye(6).expand(1,2,6,6).clone()
  self.module=SimpleNamespace(layer_idx=0,num_key_value_groups=1,config=SimpleNamespace(_attn_implementation='sdpa'))
  self.layers=[SimpleNamespace(self_attn=self.module)]
 def eval(self,kind,bias,scope='all'):
  with steering(self.layers,[(0,0)],[1],[0,1,2],bias,kind,scope) as diag:
   out,_=ALL_ATTENTION_FUNCTIONS['addbase_attention_v1'](self.module,self.q,self.k,self.v,None,scaling=8**-.5)
  return out.transpose(1,2),diag
 def base(self):return torch.nn.functional.scaled_dot_product_attention(self.q,self.k,self.v,is_causal=True)
 def test_zero_bias(self):
  a,_=self.eval('bias',0);torch.testing.assert_close(a,self.base())
 def test_mass_preservation_and_nonvisual_unchanged(self):
  a,diag=self.eval('visual_preserve',1.5);b=self.base()
  torch.testing.assert_close(a[...,:3].sum(-1),b[...,:3].sum(-1),atol=1e-6,rtol=1e-6)
  torch.testing.assert_close(a[...,3:],b[...,3:],atol=1e-6,rtol=1e-6)
  self.assertGreater(a[0,0,-1,1].item(),b[0,0,-1,1].item())
  self.assertLess(diag['max_log_visual_normalizer_error'],1e-5)
 def test_future_and_other_head(self):
  a,_=self.eval('bias',6);b=self.base()
  self.assertEqual(torch.triu(a[0,0],diagonal=1).abs().sum(),0)
  torch.testing.assert_close(a[:,1],b[:,1]);self.assertEqual(self.module.config._attn_implementation,'sdpa')
 def test_all_queries_and_last_prompt(self):
  a,d=self.eval('bias',2);b=self.base();self.assertFalse(torch.allclose(a[0,0,2],b[0,0,2]));self.assertEqual(d['query_rows'],6)
  c,d=self.eval('bias',2,'last_prompt');torch.testing.assert_close(c[:,:,:-1],b[:,:,:-1]);self.assertEqual(d['query_rows'],1)
if __name__=='__main__':unittest.main()
