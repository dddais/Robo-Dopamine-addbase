import unittest
from types import SimpleNamespace
import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from mydata_bench.top_eval.batch_attention import batch_steering
from .attention import steering
class BatchAttentionTests(unittest.TestCase):
 def test_per_example_regions_and_mass(self):
  torch.manual_seed(9);q=torch.randn(2,2,6,8);k=torch.randn_like(q);v=torch.eye(6).expand(2,2,6,6).clone()
  m=SimpleNamespace(layer_idx=0,num_key_value_groups=1,config=SimpleNamespace(_attn_implementation='sdpa'));layers=[SimpleNamespace(self_attn=m)]
  targets=[[1],[2]];visuals=[[0,1,2],[1,2,3]]
  with batch_steering(layers,[(0,0)],targets,visuals,1.5,'visual_preserve'):
   out,_=ALL_ATTENTION_FUNCTIONS['addbase_sole_batch_vectorized_v1'](m,q,k,v,None,scaling=8**-.5)
  for i in range(2):
   with steering(layers,[(0,0)],targets[i],visuals[i],1.5,'visual_preserve'):
    one,_=ALL_ATTENTION_FUNCTIONS['addbase_attention_v1'](m,q[i:i+1],k[i:i+1],v[i:i+1],None,scaling=8**-.5)
   torch.testing.assert_close(out[i:i+1],one,atol=1e-6,rtol=1e-6)
  base=torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2)
  for i in range(2):
   torch.testing.assert_close(out[i,:,:,visuals[i]].sum(-1),base[i,:,:,visuals[i]].sum(-1),atol=1e-6,rtol=1e-6)
  self.assertEqual(m.config._attn_implementation,'sdpa')
if __name__=='__main__':unittest.main()
