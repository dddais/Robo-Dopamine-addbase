import unittest
import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
from .attention import collect_ranking
class RankingCausality(unittest.TestCase):
 def test_collect_ranking_preserves_causal_forward(self):
  torch.manual_seed(19)
  c=Qwen3VLTextConfig(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8,rope_scaling={'rope_type':'default','mrope_section':[1,1,2],'mrope_interleaved':True})
  c._attn_implementation='sdpa';m=Qwen3VLTextModel(c).eval();ids=torch.tensor([[1,2,3,4,5,6]])
  with torch.inference_mode():baseline=m(input_ids=ids,use_cache=False).last_hidden_state
  outputs=[]
  def forward():outputs.append(m(input_ids=ids,use_cache=False).last_hidden_state)
  r=collect_ranking(m.layers,forward,0,[1],[1,2])
  self.assertEqual(sum(sum(x) for x in r['raw_mass']),0.)
  torch.testing.assert_close(outputs[0],baseline,atol=2e-6,rtol=2e-6)
  self.assertEqual(m.config._attn_implementation,'sdpa')
if __name__=='__main__':unittest.main()
