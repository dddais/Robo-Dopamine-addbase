import math,unittest
import numpy as np
import torch
from mydata_bench.top_eval.answer_distribution import sequence_scores,token_trie,repeat_prefix_cache
from .score_branches import score_readout


class CharacterTokenizer:
    def __init__(self,fuse=False):self.fuse=fuse

    def __call__(self,texts,**kwargs):
        encoded=[];offsets=[]
        for text in texts:
            ids=[];spans=[];i=0
            while i<len(text):
                size=3 if self.fuse and text[i:i+3]=='%</' else 1
                ids.append(1000 if size==3 else ord(text[i]));spans.append((i,i+size));i+=size
            encoded.append(ids);offsets.append(spans)
        return {'input_ids':encoded,'offset_mapping':offsets}


class NumericDistributionTests(unittest.TestCase):
    def test_digit_tokenizer_includes_termination(self):
        common,suffixes,nodes=token_trie(CharacterTokenizer(),'<answer>')
        for number,suffix in zip(range(-100,101),suffixes):
            self.assertEqual(''.join(map(chr,common+list(suffix))),f'<answer>{number}%')
        self.assertGreater(len(nodes),201)

    def test_fused_native_delimiter_retains_complete_token(self):
        common,suffixes,nodes=token_trie(CharacterTokenizer(fuse=True),'<answer>')
        for number,suffix in zip(range(-100,101),suffixes):
            self.assertEqual(suffix[-1],1000)
            self.assertNotIn(ord('%'),suffix)
            self.assertEqual(''.join(map(chr,common+list(suffix[:-1]))),f'<answer>{number}')
        self.assertEqual(len(set(suffixes)),201)
        self.assertFalse(any(a!=b and b[:len(a)]==a for a in suffixes for b in suffixes))
    def test_signed_number_joint_probability(self):
        # A negative answer must include the probability of its sign branch.
        # Ignoring that factor would incorrectly favor the conditional 0.9.
        suffixes=[(0,),(1,2)]
        scores=sequence_scores(suffixes,{():np.log([.4,.1,.5]),(1,):np.log([.05,.05,.9])})
        self.assertAlmostEqual(math.exp(scores[0]),.4)
        self.assertAlmostEqual(math.exp(scores[1]),.09)
        self.assertGreater(scores[0],scores[1])
    def test_every_native_integer_is_retained_without_clipping(self):
        for percent in range(-100,101):
            logits=[-20.]*201;logits[percent+100]=0.
            r=score_readout(logits,'sole')
            self.assertEqual(r['native_percent_prediction'],percent)
            self.assertEqual(r['progress'],percent/100.)
            self.assertEqual(len(r['native_score_values']),201)

    def test_branch_cache_matches_full_causal_forward(self):
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
        torch.manual_seed(93)
        c=Qwen3VLTextConfig(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8,rope_scaling={'rope_type':'default','mrope_section':[1,1,2],'mrope_interleaved':True})
        c._attn_implementation='sdpa';model=Qwen3VLTextModel(c).eval()
        prefix=torch.tensor([[1,2,3,4]]);branches=torch.tensor([[5,6],[7,8]])
        with torch.inference_mode():
            expected=model(input_ids=torch.cat([prefix.repeat(2,1),branches],-1),use_cache=False).last_hidden_state[:,-2:]
            cache=model(input_ids=prefix,use_cache=True).past_key_values
            copied=repeat_prefix_cache(cache,2,c);position=torch.arange(4,6)
            actual=model(input_ids=branches,attention_mask=torch.ones(2,6),past_key_values=copied,
                cache_position=position,position_ids=position[None,None,:].expand(3,2,-1),use_cache=True).last_hidden_state
        torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-6)
        self.assertEqual(int(cache.get_seq_length()),4)
        self.assertEqual(int(copied.get_seq_length()),6)


if __name__=='__main__':unittest.main()
