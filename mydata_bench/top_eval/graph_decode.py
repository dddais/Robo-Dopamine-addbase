"""Greedy Qwen3-VL decoding with an explicit static cache and CUDA graph.

The same eager decode function is available for arithmetic/caching audits.
No token constraints or score postprocessing are applied. Every sequence stops
at the checkpoint's native EOS, and completed rows receive the native pad token.
"""
import math
import torch
from transformers.cache_utils import StaticCache


def greedy_static(model,inputs,max_new_tokens=512,pad_token_id=None,use_graph=True,diagnostics=None):
    diag=diagnostics if diagnostics is not None else {}
    generation=model.generation_config
    for key,allowed in {'repetition_penalty':(1.,),'no_repeat_ngram_size':(0,),'min_length':(0,),
                        'min_new_tokens':(None,0),'bad_words_ids':(None,),'suppress_tokens':(None,),
                        'begin_suppress_tokens':(None,),'forced_bos_token_id':(None,),
                        'forced_eos_token_id':(None,),'num_beams':(1,)}.items():
        if getattr(generation,key,None) not in allowed:raise ValueError(f'Unsupported generation setting {key}')
    ids=inputs['input_ids'];device=ids.device;b,length=ids.shape
    if max_new_tokens<1:raise ValueError('Positive generation length required')
    pad_token_id=generation.pad_token_id if pad_token_id is None else pad_token_id
    eos=generation.eos_token_id;eos=[eos] if isinstance(eos,int) else sorted(set(eos))
    eos=torch.tensor(eos,device=device,dtype=ids.dtype)
    cache_length=math.ceil((length+max_new_tokens)/16)*16
    cache=StaticCache(config=model.config,max_cache_len=cache_length)
    attention=inputs.get('attention_mask',torch.ones_like(ids))
    positions,rope_deltas=model.model.get_rope_index(ids,inputs.get('image_grid_thw'),inputs.get('video_grid_thw'),attention_mask=attention)
    model.model.rope_deltas=rope_deltas
    with torch.inference_mode():
        prefill=model(**inputs,position_ids=positions,past_key_values=cache,
                      cache_position=torch.arange(length,device=device),use_cache=True,logits_to_keep=1)
        first=prefill.logits[:,-1,:].argmax(-1);del prefill
        generated=torch.full((b,max_new_tokens),pad_token_id,device=device,dtype=ids.dtype)
        generated[:,0]=first;finished=torch.isin(first,eos)
        diag.update(decoding_engine='static_cuda_graph' if use_graph else 'static_eager',
                    cache_length=cache_length,prompt_length=length,graph_replays=0,
                    explicit_decode_mask=True,position_rule='cache_position + native multimodal rope_delta')
        if max_new_tokens==1 or bool(finished.all()):return torch.cat([ids,generated[:,:1]],-1)
        token_input=first[:,None].clone();cache_position=torch.tensor([length],device=device,dtype=torch.long)
        position_input=(cache_position+rope_deltas).unsqueeze(0).expand(3,-1,-1).contiguous()
        minimum=torch.finfo(model.dtype).min
        mask=torch.full((b,1,1,cache_length),minimum,device=device,dtype=model.dtype)
        mask[:,:,:, :length]=torch.where(attention[:,None,None,:].bool(),0.,minimum)
        mask[:,:,:,length]=0.
        def forward():
            return model(input_ids=token_input,attention_mask=mask,position_ids=position_input,
                         past_key_values=cache,cache_position=cache_position,use_cache=True,
                         logits_to_keep=1).logits[:,-1,:]
        graph=None;captured_logits=None
        if use_graph:
            # Only the current decode slot is overwritten by warmup/capture;
            # the prefilled prefix and all earlier KV slots remain unchanged.
            warmup=torch.cuda.Stream(device=device)
            warmup.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(warmup):
                for _ in range(2):temporary=forward()
            torch.cuda.current_stream(device).wait_stream(warmup)
            del temporary
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):captured_logits=forward()
        generated_count=1
        for step in range(1,max_new_tokens):
            if graph is None:logits=forward()
            else:graph.replay();logits=captured_logits;diag['graph_replays']+=1
            next_token=logits.argmax(-1)
            next_token=torch.where(finished,torch.full_like(next_token,pad_token_id),next_token)
            generated[:,step]=next_token;generated_count=step+1
            finished|=torch.isin(next_token,eos)
            if bool(finished.all()) or step==max_new_tokens-1:break
            token_input.copy_(next_token[:,None]);cache_position.add_(1)
            position_input.copy_((cache_position+rope_deltas).unsqueeze(0).expand(3,-1,-1))
            mask[:,:,:,length+step]=0.
        diag['generated_steps']=generated_count
        result=torch.cat([ids,generated[:,:generated_count]],-1)
        del graph,captured_logits,cache
        return result
