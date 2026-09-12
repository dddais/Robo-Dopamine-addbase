"""Complete F7 engineering then freeze and run all15 independent fits."""
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from .common import ROOT, append, create_json, timestamp
from .robust_prepare import DEST, digest


def wait_memory(gpu, events, minimum=32000):
    while True:
        free = int(subprocess.check_output(['nvidia-smi', f'--id={gpu}', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
        if free >= minimum:
            return
        append(events, dict(time=timestamp(), event='memory_wait', free_mib=free, gpu=gpu, required_mib=minimum))
        time.sleep(20)


def worker(gpu, configurations):
    events = DEST/f'f7_fit_gpu{gpu}_events_v1.jsonl'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
    failures = []
    for item in configurations:
        name = item['model']+'_'+item['protocol']
        if digest(item['path']) != item['sha256']:
            raise ValueError('Frozen F7 fit config changed')
        wait_memory(gpu, events)
        command = [sys.executable, '-u', '-m', 'mydata_bench.auto_research_addbase.all_query_reward_gradient_fit', '--config', item['path']]
        with (DEST/f'f7_fit_{name}_stdout_v1.log').open('x') as handle:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            append(events, dict(time=timestamp(), event='launch', case=name, pid=process.pid, command=command))
            code = process.wait()
        append(events, dict(time=timestamp(), event='exit', case=name, pid=process.pid, returncode=code))
        if code:
            failures.append(dict(case=name, returncode=code))
    create_json(DEST/f'f7_fit_gpu{gpu}_completion_v1.json', dict(time=timestamp(), failures=failures,
        cases=[r['model']+'_'+r['protocol'] for r in configurations]))
    return failures


def main():
    files = ['all_query_reward_gradient_attention.py', 'all_query_reward_gradient_engineering.py',
             'all_query_reward_gradient_fit.py', 'freeze_all_query_reward_gradient_fit.py']
    hashes = {name: digest(ROOT/'mydata_bench/auto_research_addbase'/name) for name in files}
    create_json(DEST/'f7_fit_queue_plan_v1.json', dict(time=timestamp(), pid=os.getpid(), dependencies_sha256=hashes,
        queue_source_sha256=digest(__file__), fitting_only=True, validation_development_and_test_not_scheduled_here=True))
    rr_path = DEST/'all_query_reward_gradient_engineering_v1/rr_image_text/engineering_audit_v1.json'
    if not json.loads(rr_path.read_text())['passed']:
        raise ValueError('RR actual F7 engineering must be complete before launch')
    events = DEST/'f7_engineering_queue_events_v1.jsonl'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='3', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
    for model, protocol in [('qwen', 'text_video'), ('meter', 'text_image')]:
        wait_memory(3, events)
        command = [sys.executable, '-u', '-m', 'mydata_bench.auto_research_addbase.all_query_reward_gradient_engineering',
                   '--model', model, '--protocol', protocol]
        with (DEST/f'f7_engineering_{model}_{protocol}_stdout_v1.log').open('x') as handle:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            append(events, dict(time=timestamp(), event='launch', model=model, protocol=protocol, pid=process.pid, command=command))
            code = process.wait()
        append(events, dict(time=timestamp(), event='exit', model=model, protocol=protocol, pid=process.pid, returncode=code))
        if code:
            raise RuntimeError('F7 engineering failed; preserve and inspect, no fit launched')
    for name, sha in hashes.items():
        if digest(ROOT/'mydata_bench/auto_research_addbase'/name) != sha:
            raise ValueError('Frozen F7 implementation changed')
    with (DEST/'f7_fit_freeze_stdout_v1.log').open('x') as handle:
        result = subprocess.run([sys.executable, '-u', '-m', 'mydata_bench.auto_research_addbase.freeze_all_query_reward_gradient_fit'],
            cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('F7 fit freeze failed; inspect before inference')
    plan = json.loads((DEST/'all_query_reward_gradient_fit_frozen_plan_v1.json').read_text())
    a = [c for c in plan['configurations'] if c['model'] == 'rr' or c['model'] == 'qwen' and c['protocol'] in ['text_video', 'video_text', 'text_image']]
    b = [c for c in plan['configurations'] if c not in a]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(worker, 0, a), pool.submit(worker, 3, b)]
        failures = [row for job in jobs for row in job.result()]
    create_json(DEST/'f7_all_fit_queues_completion_v1.json', dict(time=timestamp(), failures=failures,
        independent_fit_only=True, all_evaluation_implementation_and_freeze_still_required=True, final_goal_completed=False))


if __name__ == '__main__':
    main()
