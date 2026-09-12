from __future__ import annotations
import json
from contextlib import nullcontext
import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor
from mydata_bench.attention_eval.masking import ImageSpan,bbox_to_token_positions,matched_wrong_position_set
from mydata_bench.attention_eval.runtime import find_contiguous_spans
from mydata_bench.auto_research_addbase.attention import steering,collect_ranking
from .model import RobometerModel

PROMPT="The task for the robot is '{task}'. Given the trajectory video, predict the task progress at each frame, how far along the robot is towards completing the task, a float between 0 and 1, where 0 is the starting state and 1 is when the task is completed. If the robot is not performing the same task, predict 0 progress."
PROG='<|prog_token|>'

class MeterRuntime:
    def __init__(self,config):
        self.config=config
        self.processor=AutoProcessor.from_pretrained(config['processor_path'],local_files_only=True)
        for token in ['<|split_token|>','<|reward_token|>','<|pref_token|>','<|sim_token|>',PROG]:
            self.processor.tokenizer.add_special_tokens({'additional_special_tokens':[token]})
        assert self.processor.tokenizer.convert_tokens_to_ids(PROG)==151673 and len(self.processor.tokenizer)==151674
        self.model,self.loading_info=RobometerModel.from_pretrained(config['model_path'],torch_dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa',local_files_only=True,output_loading_info=True)
        if self.loading_info['missing_keys'] or self.loading_info['unexpected_keys'] or self.loading_info.get('mismatched_keys'):
            raise RuntimeError(f'Checkpoint mismatch: {self.loading_info}')
        self.model.eval();self.layers=self.model.model.language_model.layers
    def prepare(self,sample,protocol):
        imgs=[Image.open(p).convert('RGB') for p in sample['image_paths']]
        # Match official collator: downsize longest side to 480 (bicubic).
        media=[]
        for im in imgs:
            s=min(1.,480/max(im.size));media.append(im.resize((int(im.width*s),int(im.height*s)),Image.Resampling.BICUBIC) if s<1 else im)
        text={'type':'text','text':PROMPT.format(task=sample['task'])}
        if protocol in {'text_video','video_text'}:
            item={'type':'video','video':np.stack([np.array(im) for im in media])}
            content=[text,item] if protocol=='text_video' else [item,text]
            content+=[{'type':'text','text':PROG}]
        elif protocol=='image_text':
            content=[{'type':'image','image':im} for im in media]+[text,{'type':'text','text':PROG}]
        else:
            content=[text]
            for i,im in enumerate(media):
                if protocol=='interleaved':content.append({'type':'text','text':f'Frame {i+1}: '})
                content.extend([{'type':'image','image':im},{'type':'text','text':PROG}])
        messages=[{'role':'user','content':content}]
        kwargs={}
        if protocol in {'text_video','video_text'}:
            sr=sample['image_sampling_record'];kwargs={'do_sample_frames':False,'video_metadata':{'fps':sr['source_fps'],'total_num_frames':sr['decoded_frame_count'],'frames_indices':sample['image_source_indices']}}
        inputs=self.processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=False,add_vision_id=True,enable_thinking=False,return_dict=True,return_tensors='pt',**kwargs)
        ids=inputs['input_ids'][0].tolist();isvideo=protocol in {'text_video','video_text'}
        spans0=find_contiguous_spans(ids,151656 if isvideo else 151655)
        grids=inputs['video_grid_thw' if isvideo else 'image_grid_thw'].tolist()
        if isvideo:
            grid=grids[0];spatial=grid[1]*grid[2]//4
            positions=[i for i,x in enumerate(ids) if x==151656]
            if len(positions)!=grid[0]*spatial:raise ValueError('Video token/grid mismatch')
            spans0=[(positions[i*spatial],positions[(i+1)*spatial-1]+1) for i in range(grid[0])]
            grids=[[1,grid[1],grid[2]] for _ in spans0]
            sources=[sample['image_source_indices'][2*i:2*i+2] for i in range(len(spans0))]
        else:sources=[[i] for i in sample['image_source_indices']]
        assert len(spans0)==len(grids)==len(sources)
        spans=[ImageSpan(f'frame_{i}',sample['image_paths'][-1],a,b,tuple(grid)) for i,((a,b),grid) in enumerate(zip(spans0,grids))]
        for span in spans:assert span.token_count==np.prod(span.grid_thw)//4
        inputs={k:v.to('cuda:0',dtype=torch.bfloat16) if v.is_floating_point() else v.to('cuda:0') for k,v in inputs.items() if isinstance(v,torch.Tensor)}
        return {'inputs':inputs,'spans':spans,'sources':sources,'image_size':imgs[-1].size,'query_index':max(i for i,x in enumerate(ids) if x==151673),'prompt':self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=False,add_vision_id=True,enable_thinking=False),'protocol':protocol}
    def positions(self,sample,prepared,scope='all_frames'):
        track=json.load(open(sample['tracking_path']))
        by={int(r['frame_index']):r['bbox'] for r in track['frames'] if r.get('bbox') is not None}
        result=[];alignment=[]
        for i,(span,sources) in enumerate(zip(prepared['spans'],prepared['sources'])):
            if scope=='last_frame' and i!=len(prepared['spans'])-1:continue
            matched=[min(by,key=lambda x:abs(x-s)) for s in sources]
            bboxes=[by[x] for x in matched]
            # Union matches every source frame contributing to a temporal tubelet.
            bbox=[min(b[0] for b in bboxes),min(b[1] for b in bboxes),max(b[2] for b in bboxes),max(b[3] for b in bboxes)]
            result+=bbox_to_token_positions(span,bbox,prepared['image_size'],2)
            alignment.append({'span':i,'sources':sources,'track_indices':matched,'bbox':bbox})
        if not result:raise ValueError('Empty target')
        return sorted(set(result)),alignment
    def rank(self,sample,prepared,scope):
        target,_=self.positions(sample,prepared,scope);visual=[p for s in prepared['spans'] for p in range(s.start,s.end)]
        return collect_ranking(self.layers,lambda:self.model(**prepared['inputs']),prepared['query_index'],target,visual)
    def predict(self,sample,prepared,condition,heads=()):
        diag={};context=nullcontext();alignment=[]
        if heads:
            target,alignment=self.positions(sample,prepared,condition.get('scope','all_frames'))
            visual=[p for s in prepared['spans'] for p in range(s.start,s.end)]
            if condition.get('region')=='wrong':
                wrong=[]
                for span in prepared['spans']:
                    local=[p for p in target if span.start<=p<span.end]
                    if not local:continue
                    chosen=matched_wrong_position_set(span,local,spatial_merge_size=2)
                    if chosen is None:raise ValueError('No exact size-matched wrong-region control')
                    wrong+=chosen
                target=wrong
            context=steering(self.layers,heads,target,visual,bias=condition.get('bias',6),kind=condition.get('kind','bias'),query_scope=condition.get('query_scope','all'),diagnostics=diag,engine=self.config.get('attention_engine','legacy'))
        with context,torch.inference_mode():result=self.model(**prepared['inputs'])
        return {'progress':float(result['progress'][-1]),'progress_trajectory':result['progress'].float().cpu().tolist(),'success_probability':float(result['success'][-1]),'success_trajectory':result['success'].float().cpu().tolist(),'native_progress_logits':result['progress_logits'].float().cpu().tolist(),'hook_diagnostics':diag,'tracking_alignment':alignment,'num_input_tokens':prepared['inputs']['input_ids'].shape[1]}
