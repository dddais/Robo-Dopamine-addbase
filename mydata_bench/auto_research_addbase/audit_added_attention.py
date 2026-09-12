"""Descriptive completeness, native-format and zero-control audit; append only."""
import collections,datetime,json
from .common import BASE,create_json,read_rows,timestamp


def main():
    ids={r['example_id'] for r in read_rows(BASE/'inputs/cohort.jsonl')};report={}
    for folder in sorted(BASE.iterdir()):
        if not folder.is_dir() or not (folder/'attention.jsonl').exists():continue
        cfg=json.loads((folder/'config.json').read_text())
        if cfg.get('pilot') or cfg.get('example_ids_file'):continue
        if cfg.get('model') not in {'meter','sole'}:continue
        rows=list(read_rows(folder/'attention.jsonl'));indexed={}
        for r in rows:indexed.setdefault(r['condition'],{})[r['example_id']]=r
        expected=cfg.get('conditions') or [{'name':'baseline'}]+[{'name':f'{c}_k{k}'} for k in [8,32,64] for c in ['target','wrong','low_rank']]
        conditions={}
        for c in expected:
            rr=indexed.get(c['name'],{});errors=collections.defaultdict(list)
            for eid,r in rr.items():
                if r['status']!='ok':errors[r.get('error','unspecified')].append(eid)
            conditions[c['name']]={'n_expected':len(ids),'n_attempted':len(ids&rr.keys()),
                'n_valid':sum(rr.get(i,{}).get('status')=='ok' for i in ids),
                'missing_ids':sorted(ids-rr.keys()),'errors_by_exact_message':dict(errors)}
        zero={}
        for name in indexed:
            if not name.startswith('zero_'):continue
            base=indexed.get('baseline',{});other=indexed[name];shared=sorted(ids&base.keys()&other.keys())
            different=collections.defaultdict(list)
            for eid in shared:
                a=base[eid];b=other[eid]
                for key in ['status','error','progress','progress_trajectory']:
                    if a.get(key)!=b.get(key):different[key].append(eid)
                if [s.get('raw_output') for s in a.get('steps',[])]!=[s.get('raw_output') for s in b.get('steps',[])]:different['raw_outputs'].append(eid)
            zero[name]={'n_paired':len(shared),'complete_cohort':len(shared)==len(ids),
                'all_paired_native_outputs_exact':not different,'differences':dict(different),
                'notes':'Compares valid results and matching native errors; failed-step raw traces are audited separately below.'}
        trace_errors=collections.Counter();trace_zero={}
        trace=folder/'attention_step_traces.jsonl'
        if trace.exists():
            trace_index={}
            for r in read_rows(trace):
                if r['status']!='ok':trace_errors[(r['condition'],r['step'],r.get('phase','parse'),r.get('error'))]+=1
                if r['condition']=='baseline' or r['condition'].startswith('zero_'):
                    trace_index[(r['example_id'],r['step'],r['condition'])]=r
            for name in zero:
                shared=[(eid,step) for eid,step,c in trace_index if c=='baseline' and (eid,step,name) in trace_index]
                diffs=[]
                for eid,step in shared:
                    a=trace_index[(eid,step,'baseline')];b=trace_index[(eid,step,name)]
                    fields=[key for key in ['status','error','raw_output','previous_progress','progress'] if a.get(key)!=b.get(key)]
                    if fields:diffs.append({'example_id':eid,'step':step,'fields':fields})
                trace_zero[name]={'n_paired_steps':len(shared),'differences':diffs,'all_paired_steps_exact':not diffs}
        report[folder.name]={'model':cfg['model'],'protocol':cfg['protocol'],
            'schedule':cfg.get('intervention_schedule','native_model_attention'),
            'all_expected_conditions_attempted':all(not c['missing_ids'] for c in conditions.values()),
            'n_raw_rows':len(rows),'n_unique_rows':sum(map(len,indexed.values())),
            'conditions':conditions,'zero_controls':zero,'zero_step_traces':trace_zero,
            'trace_errors':[{'condition':c,'step':s,'phase':p,'error':e,'n':n} for (c,s,p,e),n in sorted(trace_errors.items())]}
        print(folder.name,'attempted_all',report[folder.name]['all_expected_conditions_attempted'],
            'zero_exact',{c:z['all_paired_native_outputs_exact'] for c,z in zero.items()},
            'errors',sum(len(v) for c in conditions.values() for v in c['errors_by_exact_message'].values()))
    stamp=datetime.datetime.now().strftime('%Y%m%dT%H%M%S');dest=BASE/f'attention_audit_snapshot_{stamp}.json'
    create_json(dest,{'time':timestamp(),'phase':'Mainline1 descriptive audit; not new-method confirmation','experiments':report});print(dest)


if __name__=='__main__':main()
