"""Label-free cache/decoding engineering audit; never a scientific metric run."""
import argparse, math, time
from contextlib import nullcontext
import torch, yaml
from transformers.cache_utils import StaticCache
from mydata_bench.auto_research_addbase.common import *
from .runtime import SoleRuntime
from .batch_baseline import pad_batch
from .graph_decode import greedy_static
from .batch_attention_residual import batch_steering
from .batch_attention import prepare_regions


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--batch-size',type=int,default=2);p.add_argument('--max-new-tokens',type=int,default=96)
    p.add_argument('--protocols',default='image_text')
    p.add_argument('--condition',choices=['baseline','zero_bias','zero_preserve','bias','visual_preserve'],default='baseline')
    p.add_argument('--modes',default='hf_dynamic,hf_static,static_eager,static_graph')
    a=p.parse_args()
    out=output_path(a.output);out.mkdir(parents=True,exist_ok=True)
    cfg=yaml.safe_load((ROOT/'mydata_bench/configs/v2_crossmodel_addbase/sole_image_text_v1.yaml').read_text())
    append(out/'launches.jsonl',{'audit_only':True,**provenance(cfg)})
    ids=set(json.loads((RESEARCH/'development_v1/development_ids.json').read_text()))
    samples=[s for s in read_rows(BASE/'inputs/cohort.jsonl') if s['example_id'] in ids][:a.batch_size]
    runtime=SoleRuntime(cfg);reports=[]
    for protocol in a.protocols.split(','):
        prepared=[runtime.step_input(s,runtime.prepare(s,protocol),1,0.) for s in samples]
        inputs=pad_batch([x['inputs'] for x in prepared],runtime.processor.tokenizer.pad_token_id);length=inputs['input_ids'].shape[1]
        targets=[];visuals=[];heads=[]
        kind={'zero_bias':'bias','zero_preserve':'visual_preserve'}.get(a.condition,a.condition)
        bias=0. if a.condition.startswith('zero') else (3. if kind=='visual_preserve' else 6.)
        if a.condition!='baseline':
            ranking=json.loads((BASE/'sole_image_text_v1/ranking.json').read_text())['ranking']
            heads=[(r['layer'],r['head']) for r in ranking[:64]]
            for sample,inp in zip(samples,prepared):
                target,visual,_=prepare_regions(runtime,sample,inp,{'scope':'all_frames'})
                shift=length-inp['inputs']['input_ids'].shape[1]
                targets.append([x+shift for x in target]);visuals.append([x+shift for x in visual])
        outputs={};diagnostics={}
        for mode in a.modes.split(','):
            torch.cuda.synchronize();tick=time.time();diag={}
            context=batch_steering(runtime.layers,heads,targets,visuals,bias,kind,diagnostics=diag,capture_safe=mode=='hf_dynamic_safe' or not mode.startswith('hf_')) if heads else nullcontext()
            with context,torch.inference_mode():
                if mode.startswith('hf_'):
                    kwargs={}
                    if mode=='hf_static':kwargs['past_key_values']=StaticCache(config=runtime.model.config,max_cache_len=math.ceil((length+a.max_new_tokens)/16)*16)
                    result=runtime.model.generate(**inputs,max_new_tokens=a.max_new_tokens,do_sample=False,temperature=None,top_p=None,top_k=None,use_cache=True,pad_token_id=runtime.processor.tokenizer.pad_token_id,disable_compile=True,**kwargs)
                    del kwargs
                else:
                    result=greedy_static(runtime.model,inputs,a.max_new_tokens,runtime.processor.tokenizer.pad_token_id,mode=='static_graph',diag)
            torch.cuda.synchronize();diag['seconds']=time.time()-tick
            outputs[mode]=result[:,length:].cpu().tolist();diagnostics[mode]=diag
            append(out/'raw_outputs.jsonl',{'time':timestamp(),'protocol':protocol,'condition':a.condition,'mode':mode,'example_ids':[s['example_id'] for s in samples],'tokens':outputs[mode],'diagnostics':diag})
            print(protocol,mode,diag,flush=True);del result
            torch.cuda.empty_cache()
        report={'protocol':protocol,'condition':a.condition,'prompt_length':length,'batch_size':len(samples),'diagnostics':diagnostics}
        for left,right,key in [('hf_dynamic','hf_static','dynamic_vs_static_equal'),('hf_static','static_eager','hf_static_vs_manual_eager_equal'),('static_eager','static_graph','eager_vs_graph_equal'),('hf_dynamic','hf_dynamic_safe','dynamic_index_implementation_equal')]:
            if left in outputs and right in outputs:report[key]=outputs[left]==outputs[right]
        reports.append(report);print(report,flush=True)
    create_json(out/'audit.json',{'time':timestamp(),'reports':reports,'labels_used':False,
        'passed_static_implementation':all(r.get('hf_static_vs_manual_eager_equal',False) and r.get('eager_vs_graph_equal',False) for r in reports),
        'max_new_tokens':a.max_new_tokens,'scope':'Engineering audit; truncated token probes are not reward predictions.'})


if __name__=='__main__':main()
