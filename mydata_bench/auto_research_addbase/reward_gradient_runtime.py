"""Inference using frozen externally learned signed heads; native score readout."""
import json
import pathlib
import torch
from .score_branches import ScoreRuntime
from .reward_gradient_attention import reward_gradient_steering
from mydata_bench.attention_eval.masking import matched_wrong_position_set


class RewardGradientRuntime(ScoreRuntime):
    def __init__(self,config):
        super().__init__(config)
        ranking=json.loads(pathlib.Path(config['ranking_file']).read_text())['ranking']
        self.directions={(row['layer'],row['head']):row['signed_bias_direction'] for row in ranking}

    def predict(self,sample,prepared,condition,heads):
        if condition.get('kind')!='reward_gradient':
            return super().predict(sample,prepared,condition,heads)
        target,alignment=self.runtime.positions(sample,prepared,'all_frames')
        if condition.get('region')=='wrong':
            wrong=[]
            for span in prepared['spans']:
                local=[p for p in target if span.start<=p<span.end]
                if not local:
                    continue
                chosen=matched_wrong_position_set(span,local,spatial_merge_size=2)
                if chosen is None:
                    raise ValueError('No equal-area disjoint same-frame wrong control')
                wrong.extend(chosen)
            target=wrong
        visual=[p for span in prepared['spans'] for p in range(span.start,span.end)]
        n_heads=self.runtime.model.config.text_config.num_attention_heads
        biases=torch.zeros(len(self.runtime.layers),n_heads,device=prepared['inputs']['input_ids'].device)
        for layer,head in heads:
            direction=self.directions[(layer,head)] if condition.get('signed',True) else 1
            biases[layer,head]=direction*condition['bias']
        query=prepared.get('score_query_index',prepared['query_index'])
        diag={}
        with reward_gradient_steering(self.runtime.layers,biases,target,visual,query,heads,diag):
            result=super().predict(sample,prepared,{},[])
        result.update(hook_diagnostics=diag,tracking_alignment=alignment,
            active_biases=[dict(layer=l,head=h,bias=float(biases[l,h])) for l,h in heads],
            engine='externally_supervised_signed_reward_query_heads_v1',output_score_adjustment='none')
        return result
