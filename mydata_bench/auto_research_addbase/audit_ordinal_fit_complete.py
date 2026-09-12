"""CPU-only complete-fit F8/F7 provenance audit; never reads validation performance."""
import collections
import json
import pathlib
import time
import numpy as np
import torch
import yaml
from .common import ROOT, create_json, read_rows, timestamp
from .robust_prepare import DEST, digest
from .native_cumulative_ordinal_loss import native_cumulative_ordinal_loss


def compare_case(name):
    configs=[]
    paths=[]
    done=[]
    for family in ['all_query', 'ordinal']:
        cfg_path=ROOT/f'mydata_bench/configs/v2_crossmodel/addbase_robust_{family}_reward_gradient_fit_v1_{name}.yaml'
        plan=json.loads((DEST/f'{family}_reward_gradient_fit_frozen_plan_v1.json').read_text())
        item=next(item for item in plan['configurations'] if item['model']+'_'+item['protocol']==name)
        if str(cfg_path)!=item['path'] or digest(cfg_path)!=item['sha256']:
            raise ValueError('Frozen F7/F8 fitting configuration changed')
        cfg=yaml.safe_load(cfg_path.read_text())
        out=pathlib.Path(cfg['output_dir'])/'fit'
        completion=json.loads((out/'completion_v1.json').read_text())
        records=out/'gradient_records_v1.jsonl'
        if digest(records)!=completion['gradients_sha256'] or digest(out/'ranking_v1.json')!=completion['ranking_sha256']:
            raise ValueError('Frozen complete fitting records changed')
        for field in ['training_input', 'fit_ids', 'fit_labels']:
            if digest(cfg[field+'_file'])!=cfg[field+'_sha256']:
                raise ValueError('Frozen fit data changed')
        configs.append(cfg);paths.append(records);done.append(completion)
    for field in ['model','protocol','model_path','processor_path','training_input_sha256','fit_ids_sha256',
                  'fit_labels_sha256','ranking','gradient_at_bias','inference_bias_magnitude','k_neighborhood',
                  'primary_k','attention_query_scope','backbone_frozen']:
        if configs[0].get(field)!=configs[1].get(field):
            raise ValueError('F7/F8 change beyond fitting loss: '+field)
    expected=set(json.loads(pathlib.Path(configs[1]['fit_ids_file']).read_text()))
    labels={r['example_id']:r['reward'] for r in read_rows(configs[1]['fit_labels_file'])}
    if set(labels)!=expected:
        raise ValueError('Fit target cohort mismatch')
    old_records=list(read_rows(paths[0]));new_records=list(read_rows(paths[1]))
    old={r['example_id']:r for r in old_records}
    new={r['example_id']:r for r in new_records}
    if len(old_records)!=len(old) or len(new_records)!=len(new) or len(old)!=1942 or set(old)!=set(new) or set(new)!=expected:
        raise ValueError('Full1942 unique matching fits required')
    counts=collections.Counter()
    max_error=0.
    for eid,row in new.items():
        previous=old[eid]
        if row['status']!=previous['status'] or row['training_episode_group_sha256']!=previous['training_episode_group_sha256']:
            raise ValueError('Fit grouping/grounding status changed')
        counts[row['status']]+=1
        if row['status']=='no_grounding':
            if row['gradient'] is not None or previous['gradient'] is not None:
                raise ValueError('No-ROI convention changed')
            continue
        if row['status']!='ok':
            raise ValueError('Failed fit cannot pass audit')
        if row['native_logits']!=previous['native_logits'] or row['tracking_alignment']!=previous['tracking_alignment']:
            raise ValueError('F8 zero-bias native output or localization differs from F7')
        counts['native_logits_exact']+=1
        measured=native_cumulative_ordinal_loss(torch.tensor(row['native_logits']),labels[eid],configs[1]['model']).item()
        error=abs(measured-row['supervised_loss'])
        max_error=max(max_error,error)
        if not np.isclose(measured,row['supervised_loss'],atol=1e-5,rtol=1e-5):
            raise ValueError('Actual fit did not use the frozen native ordinal objective')
        current_gradient=np.asarray(row['gradient'])
        if not np.isfinite(current_gradient).all() or np.count_nonzero(current_gradient[:8]):
            raise ValueError('Invalid ordinal gradient or changed first8 exclusion')
        counts['gradient_arrays_differ']+=int(not np.array_equal(current_gradient,np.asarray(previous['gradient'])))
    rankings=[json.loads((p.parent/'ranking_v1.json').read_text())['ranking'] for p in paths]
    overlap={}
    for k in [8,32,48,64]:
        old_heads={(r['layer'],r['head']):r['signed_bias_direction'] for r in rankings[0][:k]}
        new_heads={(r['layer'],r['head']):r['signed_bias_direction'] for r in rankings[1][:k]}
        common=set(old_heads)&set(new_heads)
        overlap[str(k)]=dict(head_overlap=len(common),same_direction_overlap=sum(old_heads[h]==new_heads[h] for h in common))
    return dict(examples=len(new),fitting_episode_groups=done[1]['fitting_episode_groups'],counts=dict(counts),
        zero_native_logits_and_grounding_exact=True,ordinal_loss_CPU_recompute_max_absolute_error=max_error,
        ordinal_loss_CPU_tolerance=dict(atol=1e-5,rtol=1e-5),head_overlap=overlap,
        f7_records_sha256=digest(paths[0]),f8_records_sha256=digest(paths[1]),
        interpretation='training implementation/provenance only; no validation or efficacy evidence')


def main():
    audit_launch=json.loads((DEST/'f8_complete_fit_mechanism_audit_launch_v1.json').read_text())
    if digest(__file__)!=audit_launch['source_sha256']:
        raise ValueError('Frozen observer source changed')
    launch=json.loads((DEST/'f8_fit_queue_launch_v1.json').read_text())
    plan_path=DEST/'ordinal_reward_gradient_fit_frozen_plan_v1.json'
    plan=json.loads(plan_path.read_text())
    done_path=DEST/'f8_all_fit_queues_completion_v1.json'
    while not done_path.exists():
        process=pathlib.Path('/proc')/str(launch['pid'])/'cmdline'
        if not process.exists() or b'mydata_bench.auto_research_addbase.ordinal_reward_gradient_fit_queue' not in process.read_bytes():
            raise RuntimeError('F8 fit queue stopped without completion; audit cannot claim success')
        time.sleep(20)
    if json.loads(done_path.read_text())['failures']:
        create_json(DEST/'f8_complete_fit_audit_ineligible_v1.json',dict(time=timestamp(),reason='fit failures retained'))
        return
    for path,expected in plan['source_sha256'].items():
        if digest(ROOT/path)!=expected:
            raise ValueError('Frozen F8 fitting source changed')
    cases={}
    for item in plan['configurations']:
        if digest(item['path'])!=item['sha256']:
            raise ValueError('Frozen F8 fit configuration changed')
        name=item['model']+'_'+item['protocol']
        cases[name]=compare_case(name)
        print(name,cases[name]['counts'],'CPU_loss_max_error',cases[name]['ordinal_loss_CPU_recompute_max_absolute_error'],flush=True)
    create_json(DEST/'f8_complete_fit_mechanism_audit_v1.json',dict(time=timestamp(),cases=cases,
        all15_complete=len(cases)==15,source_sha256=digest(__file__),fit_plan_sha256=digest(plan_path),
        no_validation_or_test_performance_read=True,final_goal_completed=False))


if __name__=='__main__':
    main()
