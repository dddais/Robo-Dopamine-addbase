"""Qwen3-VL / RoboReward runtime sharing verified causal intervention code."""
import numpy as np,torch
from PIL import Image
from contextlib import nullcontext
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
from mydata_bench.meter_eval.runtime import MeterRuntime
from mydata_bench.roboreward_eval.runner import ROBOREWARD_PROMPT,parse_native_score
from mydata_bench.qwen_eval.protocols import INTERLEAVED_REWARD_PROMPT
from mydata_bench.attention_eval.masking import ImageSpan,matched_wrong_position_set
from mydata_bench.attention_eval.runtime import find_contiguous_spans
from .attention import steering,collect_ranking

class DiscreteRuntime(MeterRuntime):
 def __init__(self,config):
  self.config=config;self.processor=AutoProcessor.from_pretrained(config['model_path'],local_files_only=True)
  self.processor.image_processor.size={'shortest_edge':1024,'longest_edge':50176}
  self.model,self.loading_info=Qwen3VLForConditionalGeneration.from_pretrained(config['model_path'],torch_dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa',local_files_only=True,output_loading_info=True)
  if any(self.loading_info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']):raise RuntimeError(self.loading_info)
  self.model.eval();self.layers=self.model.model.language_model.layers
 def prepare(self,sample,protocol):
  imgs=[Image.open(p).convert('RGB') for p in sample['image_paths']];prompt=ROBOREWARD_PROMPT.format(task=sample['task']);text={'type':'text','text':prompt};video=protocol in {'text_video','video_text'}
  if video:
   # Native video token representation, eight deterministic endpoint-preserving
   # source frames, with original-video indices/timestamps explicitly supplied.
   frames=np.stack([np.array(im) for im in imgs]);item={'type':'video','video':frames}
   content=[text,item] if protocol=='text_video' else [item,text]
  elif protocol=='interleaved':
   fragments=INTERLEAVED_REWARD_PROMPT.format(task=sample['task']).split('<image>');content=[]
   for i,part in enumerate(fragments):
    content.append({'type':'text','text':part})
    if i<len(imgs):content.append({'type':'image','image':imgs[i]})
  else:
   items=[{'type':'image','image':im} for im in imgs];content=[text]+items if protocol=='text_image' else items+[text]
  kwargs={}
  if video:
   sr=sample['image_sampling_record'];kwargs={'do_sample_frames':False,'video_metadata':{'fps':sr['source_fps'],'total_num_frames':sr['decoded_frame_count'],'frames_indices':sample['image_source_indices']},'size':{'shortest_edge':8*1024,'longest_edge':8*50176}}
  inputs=self.processor.apply_chat_template([{'role':'user','content':content}],tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt',**kwargs)
  ids=inputs['input_ids'][0].tolist();grids=inputs['video_grid_thw' if video else 'image_grid_thw'].tolist()
  if video:
   grid=grids[0];positions=[i for i,x in enumerate(ids) if x==151656];cells=grid[1]*grid[2]//4
   assert len(positions)==grid[0]*cells
   ranges=[(positions[i*cells],positions[(i+1)*cells-1]+1) for i in range(grid[0])];grids=[[1,grid[1],grid[2]] for _ in ranges];sources=[sample['image_source_indices'][i*2:i*2+2] for i in range(len(ranges))]
  else:ranges=find_contiguous_spans(ids,151655);sources=[[i] for i in sample['image_source_indices']]
  assert len(ranges)==len(grids)==len(sources)
  spans=[ImageSpan(f'frame_{i}',sample['image_paths'][-1],a,b,tuple(g)) for i,((a,b),g) in enumerate(zip(ranges,grids))]
  for span in spans:assert span.token_count==np.prod(span.grid_thw)//4
  inputs={k:v.to('cuda:0',dtype=torch.bfloat16) if v.is_floating_point() else v.to('cuda:0') for k,v in inputs.items() if isinstance(v,torch.Tensor)}
  return {'inputs':inputs,'spans':spans,'sources':sources,'image_size':imgs[-1].size,'query_index':len(ids)-1,'protocol':protocol}
 def rank(self,sample,prepared,scope):
  target,_=self.positions(sample,prepared,scope);visual=[p for s in prepared['spans'] for p in range(s.start,s.end)]
  return collect_ranking(self.layers,lambda:self.model(**prepared['inputs'],use_cache=False),prepared['query_index'],target,visual)
 def predict(self,sample,prepared,condition,heads=()):
  diag={};alignment=[];context=nullcontext()
  if heads:
   target,alignment=self.positions(sample,prepared,condition.get('scope','all_frames'));visual=[p for s in prepared['spans'] for p in range(s.start,s.end)]
   if condition.get('region')=='wrong':
    wrong=[]
    for span in prepared['spans']:
     local=[p for p in target if span.start<=p<span.end]
     if not local:continue
     chosen=matched_wrong_position_set(span,local,spatial_merge_size=2)
     if chosen is None:raise ValueError('No equal-area disjoint wrong region')
     wrong+=chosen
    target=wrong
   context=steering(self.layers,heads,target,visual,condition.get('bias',6),condition.get('kind','bias'),condition.get('query_scope','all'),diag,engine=self.config.get('attention_engine','legacy'))
  with context,torch.inference_mode():
   output=self.model.generate(**prepared['inputs'],max_new_tokens=self.config.get('max_new_tokens',128),do_sample=False,temperature=None,top_p=None,use_cache=True,pad_token_id=self.processor.tokenizer.pad_token_id)
  raw=self.processor.tokenizer.decode(output[0,prepared['inputs']['input_ids'].shape[1]:],skip_special_tokens=True).strip();pred=parse_native_score(raw)
  return {'native_prediction':pred,'progress':(pred-1)/4,'raw_output':raw,'hook_diagnostics':diag,'tracking_alignment':alignment,'input_token_count':prepared['inputs']['input_ids'].shape[1]}
