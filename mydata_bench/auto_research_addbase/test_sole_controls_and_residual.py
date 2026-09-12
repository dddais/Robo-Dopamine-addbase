import unittest
from types import SimpleNamespace
import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from mydata_bench.attention_eval.masking import ImageSpan,bbox_to_token_positions
from mydata_bench.top_eval.controls import montage_wrong_control
from mydata_bench.top_eval.batch_attention_residual import batch_steering

class SoleControlsTests(unittest.TestCase):
 def test_same_panel_matching(self):
  span=ImageSpan('montage','unused',10,10+12*36,(1,24,72));inp={'spans':[span]}
  alignment=[{'history_slot':j,'mapped_bbox':[j*389+80,80,j*389+170,170],'image_size':[1162,384]} for j in range(3)]
  target=sorted({p for a in alignment for p in bbox_to_token_positions(span,a['mapped_bbox'],a['image_size'],2)})
  wrong=montage_wrong_control(inp,target,alignment)
  self.assertEqual(len(wrong),len(target));self.assertFalse(set(wrong)&set(target))
  for j in range(3):
   def panel_count(values):return sum(j*389/1162<=((p-span.start)%36+.5)/36<(j*389+384)/1162 for p in values)
   self.assertEqual(panel_count(wrong),panel_count(target))
 def test_infeasible_is_explicit(self):
  span=ImageSpan('montage','unused',0,12*36,(1,24,72))
  alignment=[{'history_slot':0,'mapped_bbox':[0,0,384,384],'image_size':[1162,384]}]
  target=bbox_to_token_positions(span,alignment[0]['mapped_bbox'],alignment[0]['image_size'],2)
  with self.assertRaises(ValueError):montage_wrong_control({'spans':[span]},target,alignment)

class ResidualTests(unittest.TestCase):
 def test_zero_and_nonzero_against_dense_reference(self):
  torch.manual_seed(13)
  q=torch.randn(2,4,6,8);k=torch.randn(2,2,6,8);v=torch.randn_like(k)
  m=SimpleNamespace(layer_idx=0,num_key_value_groups=2,config=SimpleNamespace(_attn_implementation='sdpa'),is_causal=True)
  layers=[SimpleNamespace(self_attn=m)];target=[[1],[2]];visual=[[0,1,2],[1,2,3]]
  base,_=ALL_ATTENTION_FUNCTIONS['sdpa'](m,q,k,v,None,scaling=8**-.5)
  for kind in ['bias','target_only','visual_preserve']:
   for bias in [0.,1.5]:
    with batch_steering(layers,[(0,1)],target,visual,bias,kind):
     out,_=ALL_ATTENTION_FUNCTIONS['addbase_sole_batch_residual_v1'](m,q,k,v,None,scaling=8**-.5)
    if not bias:torch.testing.assert_close(base,out,atol=0,rtol=0)
    from mydata_bench.top_eval.batch_attention import batch_steering_reference
    with batch_steering_reference(layers,[(0,1)],target,visual,bias,kind):
     dense,_=ALL_ATTENTION_FUNCTIONS['addbase_sole_batch_attention_v1'](m,q,k,v,None,scaling=8**-.5)
    torch.testing.assert_close(out,dense,atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(out[:,:,[0,2,3],:],base[:,:,[0,2,3],:],atol=0,rtol=0)
  self.assertEqual(m.config._attn_implementation,'sdpa')
if __name__=='__main__':unittest.main()
