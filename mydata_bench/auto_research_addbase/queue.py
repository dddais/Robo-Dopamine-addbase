"""Sequential experiment process queue; records failures and keeps other jobs going."""
import argparse,os,pathlib,subprocess,time
from .common import *
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,required=True);ap.add_argument('--models',default='meter');ap.add_argument('--protocols',required=True);ap.add_argument('--stages',default='baseline,rank,attention');ap.add_argument('--name',required=True);ap.add_argument('--sole-batch-size',type=int,default=16);args=ap.parse_args()
 log=BASE/(args.name+'_queue_events.jsonl');env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
 for stage in args.stages.split(','):
  for model in args.models.split(','):
   for protocol in args.protocols.split(','):
    cfg=ROOT/f'mydata_bench/configs/v2_crossmodel_addbase/{model}_{protocol}_v1.yaml'
    # Check free memory before every new model load; never kill another process.
    need=14000 if model=='meter' else 25000
    while True:
     free=int(subprocess.check_output(['nvidia-smi',f'--id={args.gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
     if free>=need:break
     append(log,{'time':timestamp(),'event':'waiting_for_free_memory','gpu':args.gpu,'free_mib':free,'need_mib':need});time.sleep(30)
    if model=='sole' and stage=='baseline':
     command=[sys.executable,'-u','-m','mydata_bench.top_eval.batch_baseline','--config',str(cfg),'--batch-size',str(args.sole_batch_size)]
    elif model=='sole' and stage=='attention':
     command=[sys.executable,'-u','-m','mydata_bench.top_eval.batch_attention','--config',str(cfg),'--batch-size','16']
    else:
     command=[sys.executable,'-u','-m','mydata_bench.auto_research_addbase.run','--config',str(cfg),'--stage',stage]
    dest=BASE/f'{args.name}_{model}_{protocol}_{stage}.log'
    append(log,{'time':timestamp(),'event':'launch','command':command,'gpu':args.gpu})
    with dest.open('a') as f:
     proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
     code=proc.wait()
    append(log,{'time':timestamp(),'event':'exit','returncode':code,'command':command,'pid':proc.pid})
    print(stage,model,protocol,'exit',code,flush=True)
 append(log,{'time':timestamp(),'event':'queue_finished'})
if __name__=='__main__':main()
