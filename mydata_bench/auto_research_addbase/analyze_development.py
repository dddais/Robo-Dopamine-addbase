"""Offline development contrasts with completeness and both-split gates."""
import argparse,collections,datetime,json,pathlib
from .common import *
from .metrics import summarize

def contrast(candidate,baseline,continuous=False):
    key='continuous_ordinal_mae' if continuous else 'mae'
    a=candidate['overall'];b=baseline['overall']
    if a.get(key) is None or b.get(key) is None:return {'complete':False}
    delta={'complete':candidate['complete'] and baseline['complete'],'mae_delta':a[key]-b[key],'accuracy_delta':a['accuracy']-b['accuracy'],'suc_delta':candidate['by_split']['suc']['accuracy']-baseline['by_split']['suc']['accuracy'],'fail_delta':candidate['by_split']['fail']['accuracy']-baseline['by_split']['fail']['accuracy']}
    delta['all_directional_improvements']=delta['complete'] and delta['mae_delta']<0 and min(delta['accuracy_delta'],delta['suc_delta'],delta['fail_delta'])>0
    delta['point_estimate_10pp_gate']=delta['all_directional_improvements'] and delta['accuracy_delta']>=.1-1e-12
    delta['confirmation']=False
    return delta

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(RESEARCH/'development_v1'));a=p.parse_args();root=pathlib.Path(a.root)
    labels={r['example_id']:r for r in read_rows(BASE/'inputs/labels.jsonl')};report={}
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or not (folder/'sweep.jsonl').exists():continue
        cfg=json.loads((folder/'config.json').read_text());ids=json.loads(pathlib.Path(cfg['example_ids_file']).read_text());rows=collections.defaultdict(list)
        for r in read_rows(folder/'sweep.jsonl'):rows[r['condition']].append(r)
        summaries={k:summarize(rr,labels,ids) for k,rr in rows.items()};baseline=summaries.get('baseline')
        if not baseline:continue
        deltas={k:contrast(v,baseline,cfg['model']=='meter') for k,v in summaries.items() if k!='baseline'}
        for c in cfg['conditions']:
            if c['name'] not in deltas or not c.get('k'):continue
            original=summaries.get(f'original_all_k{c["k"]}')
            if original:deltas[c['name']]['versus_original_same_k']=contrast(summaries[c['name']],original,cfg['model']=='meter')
        report[folder.name]={'config':cfg,'summaries':summaries,'contrasts':deltas,'official_continuous_readout':cfg['model']=='meter'}
        valid=[(name,v) for name,v in deltas.items() if v.get('complete')]
        ordered=sorted(valid,key=lambda x:(not x[1].get('point_estimate_10pp_gate'),not x[1].get('all_directional_improvements'),x[1]['mae_delta'],-x[1]['accuracy_delta']))
        print(folder.name,'baseline_n',baseline['overall']['n_valid'],'suc/fail',baseline['by_split']['suc']['n_expected'],baseline['by_split']['fail']['n_expected'],'complete_conditions',len(valid),'baseline_mae',baseline['overall']['mae'],'baseline_acc',baseline['overall']['accuracy'])
        for name,d in ordered[:5]:print(name,json.dumps({k:v for k,v in d.items() if k!='versus_original_same_k'}))
    stamp=datetime.datetime.now().strftime('%Y%m%dT%H%M%S');path=root/f'analysis_snapshot_{stamp}.json';create_json(path,{'time':timestamp(),'phase':'development screening only, never confirmation','experiments':report});print(path)
if __name__=='__main__':main()
