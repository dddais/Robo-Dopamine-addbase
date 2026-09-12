"""Score all 201 native signed integer percentages after SOLE's own reasoning.

The first six recursive states and terminal reasoning text come from that
example's native baseline. No label, endpoint target or external reasoning is
used. Token-trie likelihoods account for different positive/negative lengths.
"""
import json,pathlib,re
from contextlib import nullcontext
from collections import defaultdict
import numpy as np
import torch
from transformers.cache_utils import DynamicCache
from mydata_bench.auto_research_addbase.common import read_rows
from .runtime import SoleRuntime
from .batch_attention import prepare_regions
from .batch_attention_residual import batch_steering


PERCENTAGES=list(range(-100,101))


def token_trie(tokenizer,prefix):
    # The delimiter is required: P('1') alone is always >= P('10'), which
    # would spuriously favor short numbers when digits are separate tokens.
    # Tokenize the complete native closing tag before truncating at the first
    # token that covers the percentage delimiter. In this checkpoint "%</"
    # is fused; encoding a standalone "%" asks for an unnatural token.
    texts=[prefix+str(value)+'%</answer>' for value in PERCENTAGES]
    encoded=tokenizer(texts,add_special_tokens=False,return_offsets_mapping=True,padding=False,truncation=False)
    sequences=[]
    for value,ids,offsets in zip(PERCENTAGES,encoded['input_ids'],encoded['offset_mapping']):
        marker=len(prefix+str(value))
        boundaries=[i+1 for i,(left,right) in enumerate(offsets) if left<=marker<right]
        if len(boundaries)!=1:raise ValueError('Native percent marker must map to exactly one canonical token')
        sequences.append(ids[:boundaries[0]])
    common=0
    while common<min(map(len,sequences)) and len({seq[common] for seq in sequences})==1:common+=1
    suffixes=[tuple(seq[common:]) for seq in sequences]
    if any(not s for s in suffixes) or len(set(suffixes))!=201:raise ValueError('Native numeric candidates must have distinct nonempty suffixes')
    if any(a!=b and b[:len(a)]==a for a in suffixes for b in suffixes):raise ValueError('Numeric likelihood candidates must be prefix-free')
    nodes=sorted({s[:i] for s in suffixes for i in range(len(s))},key=lambda s:(len(s),s))
    if len(nodes)>512:raise ValueError(f'Unexpected numeric tokenization complexity: {len(nodes)} states')
    return sequences[0][:common],suffixes,nodes


def sequence_scores(suffixes,log_probabilities):
    return [sum(float(log_probabilities[s[:i]][token]) for i,token in enumerate(s)) for s in suffixes]


def repeat_prefix_cache(cache,batch,config):
    return DynamicCache(ddp_cache_data=[(k.repeat(batch,1,1,1),v.repeat(batch,1,1,1)) for k,v in cache.to_legacy_cache()],config=config)


class SoleAnswerDistribution:
    def __init__(self,config):
        self.config=config;self.runtime=SoleRuntime(config)
        self.priors={r['example_id']:r for r in read_rows(config['baseline_context_path']) if r.get('status')=='ok'}

    def prepare(self,sample,media_condition='observed'):
        if media_condition not in {'observed','static_first'}:raise ValueError(media_condition)
        prior=self.priors.get(sample['example_id'])
        if prior is None or len(prior.get('progress_trajectory',[]))!=8:raise ValueError('Native complete seven-step baseline required')
        raw=prior['steps'][-1]['raw_output']
        matches=list(re.finditer(r'<answer>\s*(?=[+-]?\d)',raw))
        if len(matches)!=1:raise ValueError('Exactly one native terminal numeric answer prefix required')
        prefix=raw[:matches[0].end()]
        common,suffixes,nodes=token_trie(self.runtime.processor.tokenizer,prefix)
        source=sample if media_condition=='observed' else {**sample,'image_paths':[sample['image_paths'][0]]*len(sample['image_paths'])}
        prepared=self.runtime.prepare(source,self.config['protocol'])
        inp=self.runtime.step_input(source,prepared,7,prior['progress_trajectory'][-2]);inputs=dict(inp['inputs'])
        tail=torch.tensor([common],device=inputs['input_ids'].device,dtype=inputs['input_ids'].dtype)
        inputs['input_ids']=torch.cat([inputs['input_ids'],tail],-1);inputs['attention_mask']=torch.cat([inputs['attention_mask'],torch.ones_like(tail)],-1)
        inp.update(inputs=inputs,score_query_index=inputs['input_ids'].shape[1]-1,native_reasoning_prefix=prefix,
                   numeric_suffixes=suffixes,numeric_trie_nodes=nodes,native_prior=prior)
        if media_condition=='static_first':
            first=sample['image_source_indices'][0];inp['sources']=[first]*3
            inp['media_counterfactual']={'type':'first_frame_repeated','source_path':sample['image_paths'][0],
                'pixel_source_frame_index':first,'presentation_timestamps':'unchanged from observed input',
                'grounding_sources':'first source frame in all three history panels',
                'previous_progress_and_reasoning':'native observed baseline, held identical across branches'}
        return inp

    def predict(self,sample,prepared,condition,heads):
        prior=prepared['native_prior']
        if condition.get('kind')=='native_generation':
            return {'progress':prior['progress'],'progress_trajectory':prior['progress_trajectory'],
                    'native_generation_source':self.config['baseline_context_path'],'raw_output':prior['steps'][-1]['raw_output'],
                    'readout':'unchanged native full seven-step baseline result'}
        target=visual=alignment=[]
        if heads:target,visual,alignment=prepare_regions(self.runtime,sample,prepared,condition)
        probabilities={};diags=[];needed=defaultdict(set);groups=defaultdict(list)
        for suffix in prepared['numeric_suffixes']:
            for i,token in enumerate(suffix):needed[suffix[:i]].add(token)
        for node in prepared['numeric_trie_nodes']:
            if node:groups[len(node)].append(node)
        if condition.get('query_scope','all')!='all':raise ValueError('Cached SOLE numeric trie currently supports all-query steering only')
        def context(batch,diag):
            return batch_steering(self.runtime.layers,heads,[target]*batch,[visual]*batch,condition.get('bias',3.),condition.get('kind','visual_preserve'),'all',diag,capture_safe=True) if heads else nullcontext()
        def collect(nodes,logits):
            logp=logits.float().log_softmax(-1)
            for j,node in enumerate(nodes):
                tokens=sorted(needed[node]);probabilities[node]=dict(zip(tokens,logp[j,tokens].cpu().tolist()))
        diag={};inputs=prepared['inputs'];length=inputs['input_ids'].shape[1];device=inputs['input_ids'].device
        with context(1,diag),torch.inference_mode():
            output=self.runtime.model(**inputs,use_cache=True,logits_to_keep=1)
            cache=output.past_key_values;delta=self.runtime.model.model.rope_deltas.clone();collect([()],output.logits[:,-1]);del output
        diags.append(diag);chunk_size=int(self.config.get('numeric_trie_batch_size',16))
        if chunk_size<1:raise ValueError('Positive numeric trie batch size required')
        for depth,nodes in sorted(groups.items()):
            for offset in range(0,len(nodes),chunk_size):
                group=nodes[offset:offset+chunk_size];batch=len(group);diag={}
                tokens=torch.tensor(group,device=device,dtype=inputs['input_ids'].dtype)
                attention=torch.cat([inputs['attention_mask'].repeat(batch,1),torch.ones_like(tokens)],-1)
                cache_position=torch.arange(length,length+depth,device=device)
                positions=(cache_position[None,:]+delta.repeat(batch,1)).unsqueeze(0).expand(3,-1,-1)
                with context(batch,diag),torch.inference_mode():
                    branch_cache=repeat_prefix_cache(cache,batch,self.runtime.model.config)
                    output=self.runtime.model(input_ids=tokens,attention_mask=attention,position_ids=positions,
                        past_key_values=branch_cache,cache_position=cache_position,use_cache=True,logits_to_keep=1)
                    collect(group,output.logits[:,-1]);del output,branch_cache
                diags.append(diag)
        del cache
        logits=sequence_scores(prepared['numeric_suffixes'],probabilities)
        if not np.isfinite(logits).all():raise ValueError('Nonfinite native percentage likelihood')
        percent=PERCENTAGES[int(np.argmax(logits))]
        return {'score_logits':logits,'native_percent_prediction':percent,'progress':percent/100.,
                'native_score_values':PERCENTAGES,'native_numeric_token_trie_states':len(probabilities),
                'native_percentage_format_probability':float(np.exp(np.asarray(logits,dtype=np.float64)).sum()),
                'native_numeric_suffixes':[list(s) for s in prepared['numeric_suffixes']],'numeric_trie_batch_size':chunk_size,
                'native_reasoning_prefix':prepared['native_reasoning_prefix'],
                'native_previous_progress':prior['progress_trajectory'][-2],
                'media_counterfactual':prepared.get('media_counterfactual'),
                'tracking_alignment':alignment,'hook_diagnostics':{'calls':sum(d.get('calls',0) for d in diags),'trie_node_diagnostics':diags},
                'readout':'MAP over all201 signed integer percentages; percent delimiter uses canonical full-closing-tag tokenization; cached prefix-free token-trie joint probabilities; no clipping or endpoint remapping',
                'intervention_schedule':'terminal answer distribution, own native first-six-step history and reasoning held fixed',
                'engine':'native_residual'}
