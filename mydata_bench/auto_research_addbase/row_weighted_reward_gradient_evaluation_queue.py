"""Durable F9 evaluation queue. Stops on prerequisite failure; never retries."""
import concurrent.futures
import json
import os
import pathlib
import subprocess
import sys
import time
from .common import ROOT, append, create_json, timestamp
from .robust_prepare import DEST, digest


def run_stage(gpu, case, module, arguments, events, env):
    while True:
        free = int(subprocess.check_output(['nvidia-smi', f'--id={gpu}', '--query-gpu=memory.free',
            '--format=csv,noheader,nounits'], text=True).strip())
        if free >= 32000:
            break
        append(events, dict(time=timestamp(), event='memory_wait', case=case, module=module, free_mib=free))
        time.sleep(20)
    stage = arguments[-1]
    command = [sys.executable, '-u', '-m', 'mydata_bench.auto_research_addbase.'+module, *arguments]
    with (DEST/f'f9_evaluation_{case}_{module}_{stage}_stdout_v1.log').open('x') as handle:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        append(events, dict(time=timestamp(), event='launch', case=case, module=module, stage=stage,
            pid=process.pid, command=command))
        code = process.wait()
    append(events, dict(time=timestamp(), event='exit', case=case, module=module, stage=stage, pid=process.pid, returncode=code))
    return code


def worker(gpu, cases, development):
    events = DEST/f'f9_evaluation_gpu{gpu}_events_v1.jsonl'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
               TOKENIZERS_PARALLELISM='false')
    failures = []
    for item in cases:
        case = item['model']+'_'+item['protocol']
        dev = development[case]
        if any(digest(record['path']) != record['sha256'] for record in [item, dev]):
            raise ValueError('Frozen evaluation configuration changed')
        stages = [('all_query_reward_gradient_development', dev['path'], 'engineering'),
                  ('all_query_reward_gradient_validation', item['path'], 'engineering'),
                  ('all_query_reward_gradient_validation', item['path'], 'validation'),
                  ('all_query_reward_gradient_development', dev['path'], 'development')]
        for module, config, stage in stages:
            code = run_stage(gpu, case, module, ['--config', config, '--stage', stage], events, env)
            if code:
                failures.append(dict(case=case, module=module, stage=stage, returncode=code))
                break
    create_json(DEST/f'f9_evaluation_gpu{gpu}_completion_v1.json', dict(time=timestamp(), failures=failures,
        cases=[r['model']+'_'+r['protocol'] for r in cases]))
    return failures


def main():
    policy_path = DEST/'f9_evaluation_policy_v1.json'
    policy = json.loads(policy_path.read_text())
    create_json(DEST/'f9_evaluation_queue_plan_v1.json', dict(time=timestamp(), pid=os.getpid(),
        source_sha256=digest(__file__), policy_sha256=digest(policy_path),
        dependencies_sha256=policy['evaluation_source_sha256'], gpus=[0, 3],
        prerequisite='all15 fit completions, then freeze validation and old234 together before evaluation',
        reserved_test_inference_authorized_by_this_queue=False, no_automatic_retries=True))
    launch = json.loads((DEST/'f9_fit_launch_v1.json').read_text())
    done_path = DEST/'f9_fit_completion_v1.json'
    while not done_path.exists():
        process = pathlib.Path('/proc')/str(launch['pid'])/'cmdline'
        if not process.exists() or b'mydata_bench.auto_research_addbase.fit_row_weighted_reward_gradient_heads' not in process.read_bytes():
            raise RuntimeError('F9 fitting queue ended without completion; inspect preserved evidence')
        time.sleep(20)
    if json.loads(done_path.read_text())['failures']:
        create_json(DEST/'f9_evaluation_queue_prerequisite_failure_v1.json', dict(time=timestamp(),
            cause='fitting queue reports failures', fitting_completion_sha256=digest(done_path)))
        raise RuntimeError('Required F9 fits failed; no evaluation launched')
    for path, sha in policy['evaluation_source_sha256'].items():
        if digest(ROOT/path) != sha:
            raise ValueError('Frozen prospective evaluation source changed: '+path)
    for module in ['freeze_row_weighted_reward_gradient_development', 'freeze_row_weighted_reward_gradient_validation']:
        with (DEST/(module+'_stdout_v1.log')).open('x') as handle:
            result = subprocess.run([sys.executable, '-u', '-m', 'mydata_bench.auto_research_addbase.'+module],
                cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError('F9 evaluation freeze failed: '+module)
    plan = json.loads((DEST/'row_weighted_reward_gradient_validation_frozen_plan_v1.json').read_text())
    dev_plan = json.loads((DEST/'row_weighted_reward_gradient_development_frozen_plan_v1.json').read_text())
    development = {r['model']+'_'+r['protocol']: r for r in dev_plan['configurations']}
    a = [c for c in plan['configurations'] if c['model'] == 'rr' or c['model'] == 'qwen' and c['protocol'] in ['text_video', 'video_text', 'text_image']]
    b = [c for c in plan['configurations'] if c not in a]
    # Freeze immediately, then wait for F7's current GPU workload to finish.
    f7_launch = json.loads((DEST/'f7_evaluation_queue_launch_v1.json').read_text())
    f7_done = DEST/'f7_all_evaluation_queues_completion_v1.json'
    while not f7_done.exists():
        process = pathlib.Path('/proc')/str(f7_launch['pid'])/'cmdline'
        if not process.exists() or b'mydata_bench.auto_research_addbase.all_query_reward_gradient_evaluation_queue' not in process.read_bytes():
            raise RuntimeError('F7 resource/comparator prerequisite stopped without completion')
        time.sleep(20)
    if json.loads(f7_done.read_text())['failures']:
        raise RuntimeError('F7 fixed comparator has stage failures; no F9 evaluation starts')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(worker, 0, a, development), pool.submit(worker, 3, b, development)]
        failures = [row for job in jobs for row in job.result()]
    create_json(DEST/'f9_all_evaluation_queues_completion_v1.json', dict(time=timestamp(), failures=failures,
        reserved_test_not_evaluated=True, final_goal_completed=False))
    if failures:
        raise RuntimeError('F9 evaluation stage failures retained; inspect before any validation label join')
    f7_launch = json.loads((DEST/'f7_evaluation_queue_launch_v1.json').read_text())
    f7_done = DEST/'f7_all_evaluation_queues_completion_v1.json'
    while not f7_done.exists():
        process = pathlib.Path('/proc')/str(f7_launch['pid'])/'cmdline'
        if not process.exists() or b'mydata_bench.auto_research_addbase.all_query_reward_gradient_evaluation_queue' not in process.read_bytes():
            raise RuntimeError('Fixed F7 comparator pipeline stopped without complete inference; inspect')
        time.sleep(20)
    if json.loads(f7_done.read_text())['failures']:
        raise RuntimeError('Fixed F7 comparator has stage failures; preserve before F9 label join')
    f8_launch = json.loads((DEST/'f8_evaluation_queue_launch_v1.json').read_text())
    f8_done = DEST/'f8_all_evaluation_queues_completion_v1.json'
    while not f8_done.exists():
        process = pathlib.Path('/proc')/str(f8_launch['pid'])/'cmdline'
        if not process.exists() or b'mydata_bench.auto_research_addbase.ordinal_reward_gradient_evaluation_queue' not in process.read_bytes():
            raise RuntimeError('Fixed F8 comparator stopped without complete inference')
        time.sleep(20)
    if json.loads(f8_done.read_text())['failures']:
        raise RuntimeError('Fixed F8 comparator stage failures retained; no F9 analysis')
    for module, arguments in [('analyze_row_weighted_reward_gradient_development', []), ('analyze_row_weighted_reward_gradient_validation', [])]:
        with (DEST/('f9_evaluation_'+module+'_stdout_v1.log')).open('x') as handle:
            result = subprocess.run([sys.executable, '-u', '-m', 'mydata_bench.auto_research_addbase.'+module, *arguments],
                cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                env=dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1'))
        if result.returncode:
            raise RuntimeError('Frozen F9 analysis failed: '+module)
    create_json(DEST/'f9_evaluation_and_analysis_completion_v1.json', dict(time=timestamp(),
        old_full846_and_once_only_reserved_test_still_required=True, final_goal_completed=False))


if __name__ == '__main__':
    main()
