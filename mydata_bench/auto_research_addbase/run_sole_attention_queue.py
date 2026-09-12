"""SOLE full-recurrence and terminal-step paired attention schedules."""
import argparse,os,subprocess,time,yaml
from .common import *


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--protocols',required=True);p.add_argument('--batch-size',type=int,default=64);p.add_argument('--schedules',default='terminal_step,all_steps');p.add_argument('--after-queue-log');p.add_argument('--preload-head-indices',action='store_true');p.add_argument('--allow-missing-context',action='store_true');a=p.parse_args()
    plan=json.loads((BASE/'sole_attention_schedules_plan_v1.json').read_text());expected_full={r['example_id'] for r in read_rows(BASE/'inputs/full.jsonl')};cohort={r['example_id'] for r in read_rows(BASE/'inputs/cohort.jsonl')}
    log=BASE/f'sole_attention_schedules_gpu{a.gpu}_events.jsonl';env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
    if a.after_queue_log:
        upstream=pathlib.Path(a.after_queue_log)
        while not (upstream.exists() and any(r.get('event') in {'queue_finished','queue_completed'} for r in read_rows(upstream))):
            append(log,{'time':timestamp(),'event':'waiting_upstream_queue','source':str(upstream)});time.sleep(30)
    def run(command,dest):
        append(log,{'time':timestamp(),'event':'launch','command':command})
        with dest.open('a') as f:proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT);code=proc.wait()
        append(log,{'time':timestamp(),'event':'exit','command':command,'pid':proc.pid,'returncode':code});return code
    for protocol in a.protocols.split(','):
        item=next(x for x in plan['experiments'] if x['protocol']==protocol);source=pathlib.Path(item['baseline_context_path'])
        while True:
            have={r['example_id'] for r in read_rows(source)} if source.exists() else set()
            free=int(subprocess.check_output(['nvidia-smi',f'--id={a.gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            audit=BASE/'sole_terminal_attention_pilot_v1/parity_and_schedule_audit_v1.json'
            pilot_ready=audit.exists() and json.loads(audit.read_text()).get('passed')
            if have>=expected_full and free>=50000 and pilot_ready:break
            append(log,{'time':timestamp(),'event':'waiting_prerequisites','protocol':protocol,'baseline_attempts':len(have),'free_mib':free,'pilot_ready':bool(pilot_ready)});time.sleep(30)
        if not pathlib.Path(item['ranking_path']).exists():
            code=run([sys.executable,'-u','-m','mydata_bench.auto_research_addbase.run','--config',item['original_config'],'--stage','rank'],BASE/f'sole_{protocol}_formal_rank_v1.log')
            if code:continue
        for schedule in a.schedules.split(','):
            entry=item['schedules'][schedule];cfg=yaml.safe_load(pathlib.Path(entry['config']).read_text());assert fingerprint(cfg)==entry['config_sha256']
            out=output_path(entry['output_dir']);out.mkdir(parents=True,exist_ok=True)
            path=out/'attention.jsonl';known={(r['example_id'],r['condition']) for r in read_rows(path)} if path.exists() else set()
            expected={(i,c['name']) for i in cohort for c in cfg['conditions']}
            if known>=expected:continue
            command=[sys.executable,'-u','-m','mydata_bench.top_eval.batch_attention','--config',entry['config'],'--batch-size',str(a.batch_size)]
            if a.preload_head_indices:command.append('--preload-head-indices')
            if a.allow_missing_context:command.append('--allow-missing-context')
            code=run(command,BASE/f'sole_{protocol}_{schedule}_formal_v1.log')
            print(protocol,schedule,code,flush=True)
    append(log,{'time':timestamp(),'event':'queue_finished'})
if __name__=='__main__':main()
