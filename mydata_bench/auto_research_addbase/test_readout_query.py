"""The score-query intervention must leave every other query/head unchanged."""
import unittest
from types import SimpleNamespace
import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from mydata_bench.top_eval.batch_attention_residual import batch_steering


class ReadoutQueryTests(unittest.TestCase):
    def test_native_query_only_and_index_engine_parity(self):
        torch.manual_seed(20260909)
        q=torch.randn(2,4,8,6);k=torch.randn(2,2,8,6);v=torch.randn_like(k)
        m=SimpleNamespace(layer_idx=0,num_key_value_groups=2,config=SimpleNamespace(_attn_implementation='sdpa'),is_causal=True)
        layers=[SimpleNamespace(self_attn=m)];heads=[(0,0),(0,3)]
        target=[[1],[2]];visual=[[0,1,2],[1,2,3]]
        native,_=ALL_ATTENTION_FUNCTIONS['sdpa'](m,q,k,v,None,scaling=6**-.5)
        for kind in ['bias','visual_preserve']:
            for bias in [0.,1.5,-1.5]:
                results=[]
                for safe in [False,True]:
                    with batch_steering(layers,heads,target,visual,bias,kind,'readout',capture_safe=safe,readout_query_index=6):
                        out,_=ALL_ATTENTION_FUNCTIONS['addbase_sole_batch_residual_v1'](m,q,k,v,None,scaling=6**-.5)
                    results.append(out)
                    torch.testing.assert_close(out[:,[0,1,2,3,4,5,7]],native[:,[0,1,2,3,4,5,7]],atol=0,rtol=0)
                    torch.testing.assert_close(out[:,:,[1,2]],native[:,:,[1,2]],atol=0,rtol=0)
                    if bias==0:torch.testing.assert_close(out,native,atol=0,rtol=0)
                    else:self.assertFalse(torch.equal(out[:,6,[0,3]],native[:,6,[0,3]]))
                torch.testing.assert_close(results[0],results[1],atol=0,rtol=0)
        self.assertEqual(m.config._attn_implementation,'sdpa')


if __name__=='__main__':unittest.main()
