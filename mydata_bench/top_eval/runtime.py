from __future__ import annotations
import ast,hashlib,json,pathlib,re
from contextlib import nullcontext
import cv2,numpy as np,torch
from PIL import Image
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
from mydata_bench.auto_research_addbase.common import RESEARCH
from mydata_bench.auto_research_addbase.attention import steering,collect_ranking
from mydata_bench.attention_eval.masking import ImageSpan,bbox_to_token_positions,matched_wrong_position_set
from mydata_bench.attention_eval.runtime import find_contiguous_spans

# Execute only the previously inspected official prompt constants and two pure
# image-layout functions; importing the full module would initialize vLLM.
def official_contract():
    path=RESEARCH/'references/rewardgen_official/rewardgen/sole.py'
    tree=ast.parse(path.read_text())
    names={'system_prompt_template','user_question_template_external_view','resize_with_padding','create_composite_frame'}
    body=[n for n in tree.body if (isinstance(n,ast.FunctionDef) and n.name in names) or (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in n.targets))]
    ns={'np':np,'cv2':cv2};exec(compile(ast.Module(body=body,type_ignores=[]),str(path),'exec'),ns)
    return ns

def parse_answer(raw):
    matches=re.findall(r'<answer>\s*([+-]?\d+(?:\.\d+)?)\s*%?\s*</answer>',raw)
    if len(matches)!=1:raise ValueError('Expected exactly one complete numeric <answer>')
    return float(matches[0])/100.

class SoleRuntime:
    def __init__(self,config):
        self.config=config;self.contract=official_contract()
        self.processor=AutoProcessor.from_pretrained(config['model_path'],local_files_only=True)
        self.model,self.loading_info=Qwen3VLForConditionalGeneration.from_pretrained(config['model_path'],torch_dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa',local_files_only=True,output_loading_info=True)
        if any(self.loading_info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']):raise RuntimeError(self.loading_info)
        self.model.eval();self.layers=self.model.model.language_model.layers
        self.priors={}
        baseline=pathlib.Path(config['output_dir'])/'baseline.jsonl'
        if baseline.exists():
            from mydata_bench.auto_research_addbase.common import read_rows
            self.priors={r['example_id']:r for r in read_rows(baseline) if r['status']=='ok'}
    def prepare(self,sample,protocol):
        return {'images':[np.array(Image.open(p).convert('RGB')) for p in sample['image_paths']],'protocol':protocol}
    def step_input(self,sample,prepared,step,previous):
        ims=prepared['images'];protocol=prepared['protocol'];idx=[0,step-1,step]
        history=[ims[i] for i in idx]
        sources=[sample['image_source_indices'][i] for i in idx]
        question=self.contract['user_question_template_external_view'].format(task_description=sample['task'],prev_progress=f'{previous*100:g}')
        system=self.contract['system_prompt_template']
        if protocol in {'image_text','text_image'}:
            montage=self.contract['create_composite_frame'](None,history[0],None,history[1],None,history[2],view_type='external')
            image=Image.fromarray(montage)
            # Official inference resizes the entire montage with factor 28,
            # then the Qwen3 processor applies its native factor 32.
            from qwen_vl_utils import smart_resize
            hh,ww=smart_resize(image.height,image.width,factor=28,min_pixels=3136,max_pixels=12845056)
            image=image.resize((ww,hh))
            media={'type':'image','image':image};text={'type':'text','text':question}
            content=[media,text] if protocol=='image_text' else [text,media]
            input_images=[image];video=False
        elif protocol=='interleaved':
            question=(f'The task description is: {sample["task"]}. The first timestep progress is 0%. '
                      f'The previous timestep progress is {previous*100:g}%. Predict the current timestep task progress. '
                      'The three external-camera observations below are the first, previous, and current timesteps respectively.')
            content=[{'type':'text','text':question}]
            input_images=[Image.fromarray(self.contract['resize_with_padding'](im)) for im in history]
            for label,im in zip(['FIRST','PREVIOUS','CURRENT'],input_images):content.extend([{'type':'text','text':label+': '},{'type':'image','image':im}])
            video=False
        else:
            question=(f'The task description is: {sample["task"]}. The video shows the first, previous, and current timesteps, '
                      'each repeated twice to preserve a separate temporal patch for each observation. '
                      f'The first timestep progress is 0%. The previous timestep progress is {previous*100:g}%. Predict the current timestep task progress.')
            frames=np.stack([self.contract['resize_with_padding'](im) for im in history for _ in range(2)])
            media={'type':'video','video':frames};text={'type':'text','text':question}
            content=[text,media] if protocol=='text_video' else [media,text];input_images=[];video=True
        messages=[{'role':'system','content':[{'type':'text','text':system}]},{'role':'user','content':content}]
        # Keep full official prompt; never truncate image-token spans.
        kwargs={'do_sample_frames':False,'video_metadata':{'fps':2.,'total_num_frames':6,'frames_indices':list(range(6))}} if video else {}
        inputs=self.processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt',**kwargs)
        ids=inputs['input_ids'][0].tolist();grids=inputs['video_grid_thw' if video else 'image_grid_thw'].tolist()
        if video:
            grid=grids[0];positions=[i for i,t in enumerate(ids) if t==151656];cells=grid[1]*grid[2]//4
            assert grid[0]==3 and len(positions)==3*cells
            spans=[ImageSpan(f'history_{i}',sample['image_paths'][idx[i]],positions[i*cells],positions[(i+1)*cells-1]+1,(1,grid[1],grid[2])) for i in range(3)]
        else:
            ranges=find_contiguous_spans(ids,151655)
            assert len(ranges)==len(grids)==len(input_images)
            spans=[ImageSpan(f'history_{i}',sample['image_paths'][idx[min(i,2)]],a,b,tuple(g)) for i,((a,b),g) in enumerate(zip(ranges,grids))]
        for span in spans:assert span.token_count==np.prod(span.grid_thw)//4
        return {'inputs':{k:v.to('cuda:0',dtype=torch.bfloat16) if v.is_floating_point() else v.to('cuda:0') for k,v in inputs.items() if isinstance(v,torch.Tensor)},'spans':spans,'sources':sources,'idx':idx,'original_sizes':[(im.shape[1],im.shape[0]) for im in history],'input_images':input_images,'query_index':len(ids)-1,'protocol':protocol,'step':step,'previous_progress':previous,'prompt':self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)}
    def positions(self,sample,inp,scope='all_frames'):
        track=json.load(open(sample['tracking_path']));by={int(r['frame_index']):r['bbox'] for r in track['frames'] if isinstance(r.get('bbox'),list)}
        result=[];alignment=[]
        for j,(source,(w,h)) in enumerate(zip(inp['sources'],inp['original_sizes'])):
            if scope=='last_frame' and j!=2:continue
            matched=min(by,key=lambda x:abs(source-x));bbox=by[matched]
            # Exact official padding: int(new width/height), integer centered offsets.
            scale=384/max(w,h);nw,nh=int(w*scale),int(h*scale);ox=(384-nw)//2;oy=(384-nh)//2
            b=[bbox[0]*nw/w+ox,bbox[1]*nh/h+oy,bbox[2]*nw/w+ox,bbox[3]*nh/h+oy]
            if inp['protocol'] in {'image_text','text_image'}:
                span=inp['spans'][0];im=inp['input_images'][0];sx=im.width/(3*384+2*5);sy=im.height/384
                b=[(b[0]+j*389)*sx,b[1]*sy,(b[2]+j*389)*sx,b[3]*sy];size=im.size
            else:span=inp['spans'][j];size=(384,384)
            result+=bbox_to_token_positions(span,b,size,2);alignment.append({'history_slot':j,'source_frame':source,'tracking_frame':matched,'mapped_bbox':b,'image_size':size})
        if not result:raise ValueError('No aligned target cells')
        return sorted(set(result)),alignment
    def decode(self,sample,inp,condition,heads=()):
        diag={};alignment=[];context=nullcontext()
        if heads:
            target,alignment=self.positions(sample,inp,condition.get('scope','all_frames'));visual=[p for s in inp['spans'] for p in range(s.start,s.end)]
            if condition.get('region')=='wrong':
                # Montage target may contain several rectangles. Select an equal
                # cardinality nonoverlapping control within EACH temporal panel.
                wrong=[]
                for span in inp['spans']:
                    local=[p for p in target if span.start<=p<span.end]
                    if not local:continue
                    other=[p for p in range(span.start,span.end) if p not in set(target)]
                    if len(other)<len(local):raise ValueError('Insufficient wrong-region cells')
                    wrong.extend(other[-len(local):])
                target=wrong
            context=steering(self.layers,heads,target,visual,condition.get('bias',6),condition.get('kind','bias'),condition.get('query_scope','all'),diag)
        with context,torch.inference_mode():
            output=self.model.generate(**inp['inputs'],max_new_tokens=self.config.get('max_new_tokens',512),do_sample=False,temperature=None,top_p=None,top_k=None,use_cache=True,pad_token_id=self.processor.tokenizer.pad_token_id)
        raw=self.processor.tokenizer.decode(output[0,inp['inputs']['input_ids'].shape[1]:],skip_special_tokens=True).strip()
        try:p=parse_answer(raw)
        except Exception as exc:
            # Preserve malformed/truncated model output rather than silently
            # coercing it to a previous value or endpoint.
            raise ValueError(f'{exc}; raw_output={raw!r}') from exc
        return {'progress':p,'raw_output':raw,'generated_tokens':output.shape[1]-inp['inputs']['input_ids'].shape[1],'hook_diagnostics':diag,'tracking_alignment':alignment}
    def predict(self,sample,prepared,condition,heads=()):
        p=0.;trajectory=[0.];steps=[]
        for step in range(1,len(prepared['images'])):
            inp=self.step_input(sample,prepared,step,p);r=self.decode(sample,inp,condition,heads);p=r['progress'];trajectory.append(p);steps.append({'step':step,'previous_progress':inp['previous_progress'],**r})
        return {'progress':p,'progress_trajectory':trajectory,'steps':steps,'official_initial_progress':0.,'initial_progress_is_not_terminal_prediction':True}
    def rank(self,sample,prepared,scope):
        prior=self.priors.get(sample['example_id'])
        if prior is None:prior=self.predict(sample,prepared,{},())
        previous=prior['progress_trajectory'][-2];inp=self.step_input(sample,prepared,len(prepared['images'])-1,previous)
        target,_=self.positions(sample,inp,scope);visual=[p for s in inp['spans'] for p in range(s.start,s.end)]
        result=collect_ranking(self.layers,lambda:self.model(**inp['inputs'],use_cache=False),inp['query_index'],target,visual)
        return {**result,'previous_progress':previous,'ranking_context':'own baseline predicted previous timestep'}
