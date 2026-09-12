"""Native score-distribution branches for prospective contrastive experiments.

All five answer choices or all ten trained progress bins remain available.
The inference module does not read labels, split names, or endpoint targets.
"""
import argparse,fcntl,json,pathlib,time,traceback
from contextlib import nullcontext
import numpy as np
import torch,yaml
from .common import *
from mydata_bench.attention_eval.masking import matched_wrong_position_set
from mydata_bench.top_eval.batch_attention_residual import batch_steering
from mydata_bench.meter_eval.readout import official_progress_trajectory,known_logit_progress_trajectory


def score_readout(logits,model):
    values=torch.tensor(logits,dtype=torch.float32)
    if model=='meter':return {'progress':known_logit_progress_trajectory([logits])[0],'native_progress_logits':[logits],'score_readout_type':'softmax_known_logits'}
    if model=='sole':
        if values.shape!=(201,):raise ValueError('All 201 native signed integer percentages are required')
        percent=int(np.asarray(logits,dtype=np.float64).argmax())-100
        return {'progress':percent/100.,'native_percent_prediction':percent,'native_score_values':list(range(-100,101))}
    if values.shape!=(5,):raise ValueError('All five native reward choices are required')
    prediction=int(values.argmax())+1
    return {'native_prediction':prediction,'progress':(prediction-1)/4}


class ScoreRuntime:
    def __init__(self,config):
        self.config=config
        self.sole_adapter=None
        if config['model']=='sole':
            from mydata_bench.top_eval.answer_distribution import SoleAnswerDistribution
            self.sole_adapter=SoleAnswerDistribution(config);self.runtime=self.sole_adapter.runtime
        elif config['model']=='meter':
            from mydata_bench.meter_eval.runtime import MeterRuntime
            self.runtime=MeterRuntime(config)
        elif config['model'] in {'qwen','rr'}:
            from .discrete_runtime import DiscreteRuntime
            self.runtime=DiscreteRuntime(config)
            tokenizer=self.runtime.processor.tokenizer
            choices=[tokenizer.encode(f'ANSWER: {i}',add_special_tokens=False) for i in range(1,6)]
            self.prefix=choices[0][:-1];self.choice_ids=[tokens[-1] for tokens in choices]
            assert len(set(self.choice_ids))==5 and all(tokens[:-1]==self.prefix for tokens in choices)
        else:raise ValueError(f'Unsupported native score model: {config["model"]}')
    def prepare(self,sample,media_condition='observed'):
        if self.sole_adapter is not None:return self.sole_adapter.prepare(sample,media_condition)
        if media_condition not in {'observed','static_first'}:raise ValueError(media_condition)
        source=sample if media_condition=='observed' else {**sample,'image_paths':[sample['image_paths'][0]]*len(sample['image_paths'])}
        prepared=self.runtime.prepare(source,self.config['protocol'])
        if media_condition=='static_first':
            # Preserve native token/timestamp structure, while each displayed
            # observation contains the first frame. Grounding follows displayed
            # pixels, not the unchanged presentation timestamps.
            first_index=sample['image_source_indices'][0]
            prepared['sources']=[[first_index]*len(indices) for indices in prepared['sources']]
            prepared['media_counterfactual']={'type':'first_frame_repeated','source_path':sample['image_paths'][0],
                'pixel_source_frame_index':first_index,'presentation_timestamps':'unchanged from observed input',
                'grounding_sources':'first source frame for every displayed temporal observation'}
        if self.config['model']!='meter':
            inputs=dict(prepared['inputs']);prefix=torch.tensor([self.prefix],device=inputs['input_ids'].device)
            inputs['input_ids']=torch.cat([inputs['input_ids'],prefix],dim=-1)
            inputs['attention_mask']=torch.cat([inputs['attention_mask'],torch.ones_like(prefix)],dim=-1)
            prepared={**prepared,'native_inputs':prepared['inputs'],'inputs':inputs,'score_query_index':inputs['input_ids'].shape[1]-1}
        return prepared
    def predict(self,sample,prepared,condition,heads):
        if self.sole_adapter is not None:return self.sole_adapter.predict(sample,prepared,condition,heads)
        runtime=self.runtime;context=nullcontext();diag={};alignment=[]
        if condition.get('kind')=='native_generation':
            native={**prepared,'inputs':prepared.get('native_inputs',prepared['inputs'])}
            return runtime.predict(sample,native,{},())
        if heads:
            target,alignment=runtime.positions(sample,prepared,condition.get('scope','last_frame'))
            visual=[p for span in prepared['spans'] for p in range(span.start,span.end)]
            if condition.get('region')=='wrong':
                wrong=[]
                for span in prepared['spans']:
                    local=[p for p in target if span.start<=p<span.end]
                    if not local:continue
                    selected=matched_wrong_position_set(span,local,spatial_merge_size=2)
                    if selected is None:raise ValueError('No equal-area disjoint same-frame wrong control')
                    wrong+=selected
                target=wrong
            query_scope=condition.get('query_scope','all')
            context=batch_steering(runtime.layers,heads,[target],[visual],condition.get('bias',3.),condition.get('kind','visual_preserve'),query_scope,diag,
                                   readout_query_index=prepared.get('score_query_index',prepared['query_index']) if query_scope=='readout' else None)
        with context,torch.inference_mode():
            output=runtime.model(**prepared['inputs'],**({} if self.config['model']=='meter' else {'use_cache':False}))
        extra={'media_counterfactual':prepared.get('media_counterfactual')}
        if self.config['model']=='meter':
            trajectory_logits=output['progress_logits'].float().cpu().tolist();logits=trajectory_logits[-1]
            # Keep both readouts explicitly. The separate native-generation
            # baseline retains the official whole-sequence heuristic. Every
            # typed score branch consistently softmaxes its known logits.
            extra.update(native_progress_logits=trajectory_logits,official_whole_sequence_progress=official_progress_trajectory(trajectory_logits)[-1])
        else:
            all_logits=output.logits[0,-1].float()
            logits=all_logits[self.choice_ids].cpu().tolist()
            extra.update({'common_answer_prefix_token_ids':self.prefix,'native_choice_token_ids':self.choice_ids,
                   'unrestricted_next_token_id':int(all_logits.argmax()),
                   'total_probability_of_five_native_choices':float((torch.logsumexp(all_logits[self.choice_ids],0)-torch.logsumexp(all_logits,0)).exp())})
        if not np.isfinite(logits).all():raise ValueError('Nonfinite score logits')
        return {'score_logits':logits,**score_readout(logits,self.config['model']),**extra,
                'hook_diagnostics':diag,'tracking_alignment':alignment,'engine':'native_residual',
                'readout':'native_ten_bin_softmax_expected_progress_typed' if self.config['model']=='meter' else 'five_native_answer_logits_after_common_format_prefix'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--retry-errors',action='store_true');p.add_argument('--supplement');a=p.parse_args()
    cfg=yaml.safe_load(pathlib.Path(a.config).read_text());out=output_path(cfg['output_dir']);out.mkdir(parents=True,exist_ok=True)
    # Keep a live advisory lock for this process's entire inference run. Older
    # detached schedulers must not introduce a second writer to the same file.
    writer_lock=(out/'score_branches_writer_v1.lock').open('a')
    try:fcntl.flock(writer_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError as exc:raise RuntimeError('Another score-branch writer already owns this output directory') from exc
    if (out/'config.json').exists():
        if json.loads((out/'config.json').read_text())!=cfg:raise ValueError('Config mismatch')
    else:create_json(out/'config.json',cfg)
    ranking=json.loads((out/'ranking.json').read_text())['ranking']
    ids=set(json.loads(pathlib.Path(cfg['example_ids_file']).read_text()))
    samples=[s for s in read_rows(BASE/'inputs/cohort.jsonl') if s['example_id'] in ids]
    if len(samples)!=len(ids):raise ValueError('Unknown sample ids')
    path=out/'score_branches.jsonl';known={}
    if path.exists():known={(r['example_id'],r['condition']):r for r in read_rows(path)}
    supplement=json.loads(pathlib.Path(a.supplement).read_text()) if a.supplement else None
    branches=cfg['branch_conditions']+(supplement.get('branch_conditions',[]) if supplement else [])
    if len({c['name'] for c in branches})!=len(branches):raise ValueError('Supplement must add new branch names')
    append(out/'launches.jsonl',{'stage':'score_branches','supplement':supplement,'supplement_sha256':fingerprint(supplement) if supplement else None,**provenance(cfg)})
    runtime=ScoreRuntime(cfg);started=time.time()
    for i,sample in enumerate(samples):
        pending=[c for c in branches if (sample['example_id'],c['name']) not in known or (a.retry_errors and known[(sample['example_id'],c['name'])]['status']!='ok')]
        if not pending:continue
        try:prepared=runtime.prepare(sample)
        except Exception as exc:
            traceback.print_exc()
            for condition in pending:append(path,{'time':timestamp(),'example_id':sample['example_id'],'condition':condition['name'],'status':'error','phase':'prepare','error':repr(exc)})
            continue
        prepared_cache={'observed':prepared}
        for condition in pending:
            row={'time':timestamp(),'example_id':sample['example_id'],'video_sha256':sample['video_sha256'],'condition':condition['name'],'condition_config':condition}
            ordered=ranking if condition.get('head_group','high')=='high' else list(reversed(ranking))
            heads=[(r['layer'],r['head']) for r in ordered[:condition.get('k',0)]]
            tick=time.time()
            try:
                media_condition=condition.get('media_condition','observed')
                if media_condition not in prepared_cache:
                    alternate=runtime.prepare(sample,media_condition)
                    equality={key:torch.equal(prepared['inputs'][key],alternate['inputs'][key]) for key in ['input_ids','attention_mask','image_grid_thw','video_grid_thw'] if key in prepared['inputs']}
                    if not all(equality.values()):raise ValueError(f'Static reference changed native token/timestamp structure: {equality}')
                    alternate['media_counterfactual']['nonpixel_input_equality']=equality
                    prepared_cache[media_condition]=alternate
                row.update(status='ok',**runtime.predict(sample,prepared_cache[media_condition],condition,heads))
            except Exception as exc:traceback.print_exc();row.update(status='error',error=repr(exc));torch.cuda.empty_cache()
            row['elapsed_seconds']=time.time()-tick;append(path,row)
        if i%10==0:print(json.dumps({'sample':i+1,'total':len(samples),'seconds':time.time()-started}),flush=True)
    append(out/'completion_events.jsonl',{'stage':'score_branches','time':timestamp(),'expected_samples':len(samples),'expected_conditions':len(branches),'supplement_sha256':fingerprint(supplement) if supplement else None})
if __name__=='__main__':main()
