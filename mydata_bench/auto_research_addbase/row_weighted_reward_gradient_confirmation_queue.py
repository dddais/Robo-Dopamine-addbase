"""Conditional once-only reserved confirmation, never a research-tuning loop."""
import concurrent.futures
import json
import os
import pathlib
import subprocess
import sys
import time
from .common import ROOT, append, create_json, timestamp
from .robust_prepare import DEST, digest
from .freeze_row_weighted_reward_gradient_confirmation import selected_confirmation_cases
from .f9_confirmation_priority import confirmation_priority


def worker(gpu, configurations):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
    events = DEST/f'f9_confirmation_gpu{gpu}_events_v1.jsonl'
    failures = []
    for cfg in configurations:
        name = cfg['model']+'_'+cfg['protocol']
        if digest(cfg['path']) != cfg['sha256']:
            raise ValueError('Frozen reserved-test configuration changed')
        for stage in ['engineering', 'confirmation']:
            while True:
                free = int(subprocess.check_output(['nvidia-smi', f'--id={gpu}', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
                if free >= 32000:
                    break
                append(events, dict(time=timestamp(), event='memory_wait', case=name, stage=stage, free_mib=free))
                time.sleep(20)
            command = [sys.executable, '-u', '-m', 'mydata_bench.auto_research_addbase.all_query_reward_gradient_confirmation',
                       '--config', cfg['path'], '--stage', stage]
            with (DEST/f'f9_confirmation_{name}_{stage}_stdout_v1.log').open('x') as handle:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
                append(events, dict(time=timestamp(), event='launch', case=name, stage=stage, pid=process.pid, command=command))
                code = process.wait()
            append(events, dict(time=timestamp(), event='exit', case=name, stage=stage, pid=process.pid, returncode=code))
            if code:
                failures.append(dict(case=name, stage=stage, returncode=code))
                break
    create_json(DEST/f'f9_confirmation_gpu{gpu}_completion_v1.json', dict(time=timestamp(), failures=failures,
        cases=[r['model']+'_'+r['protocol'] for r in configurations]))
    return failures


def main():
    policy_path = DEST/'f9_confirmation_policy_v1.json'
    policy = json.loads(policy_path.read_text())
    create_json(DEST/'f9_confirmation_queue_plan_v1.json', dict(time=timestamp(), pid=os.getpid(),
        policy_sha256=digest(policy_path), dependency_sha256=policy['source_sha256'],
        prerequisite='original846 complete gate after joint old234+validation eligibility; no weakened scope',
        fixed_once_only=True, no_automatic_retry_or_method_replacement=True))
    launch = json.loads((DEST/'f9_full_old_queue_launch_v1.json').read_text())
    completion = DEST/'f9_full_old_evaluation_and_analysis_completion_v1.json'
    while not completion.exists():
        process = pathlib.Path('/proc')/str(launch['pid'])/'cmdline'
        if not process.exists() or b'mydata_bench.auto_research_addbase.row_weighted_reward_gradient_full_old_queue' not in process.read_bytes():
            eligibility = DEST/'f9_full_old_queue_eligibility_v1.json'
            no_eligibility = eligibility.exists() and not json.loads(eligibility.read_text())['eligible']
            create_json(DEST/'f9_confirmation_queue_eligibility_v1.json', dict(time=timestamp(), eligible=False,
                reason='F9 failed joint old234/validation eligibility' if no_eligibility else 'Full846 prerequisite stopped without successful analysis',
                reserved_test_reward_inference_started=False, final_goal_completed=False))
            return
        time.sleep(20)
    for path, sha in policy['source_sha256'].items():
        if digest(ROOT/path) != sha:
            raise ValueError('Frozen final confirmation source changed: '+path)
    analysis_path = DEST/'row_weighted_reward_gradient_full_old_v1/analysis_v1.json'
    analysis = json.loads(analysis_path.read_text())
    try:
        names = selected_confirmation_cases(analysis)
    except ValueError as exc:
        create_json(DEST/'f9_confirmation_queue_eligibility_v1.json', dict(time=timestamp(), eligible=False,
            reason=str(exc), full846_analysis_sha256=digest(analysis_path), reserved_test_reward_inference_started=False))
        return
    while True:
        priority=confirmation_priority()
        if priority['state']=='released':
            break
        if priority['state']=='waiting':
            time.sleep(20)
            continue
        create_json(DEST/'f9_confirmation_queue_priority_ineligible_v1.json',dict(time=timestamp(),
            eligible=False,priority=priority,reserved_test_reward_inference_started=False,final_goal_completed=False))
        return
    create_json(DEST/'f9_confirmation_queue_eligibility_v1.json', dict(time=timestamp(), eligible=True,
        cases=names, full846_analysis_sha256=digest(analysis_path)))
    with (DEST/'f9_confirmation_freeze_stdout_v1.log').open('x') as handle:
        result = subprocess.run([sys.executable, '-u', '-m', 'mydata_bench.auto_research_addbase.freeze_row_weighted_reward_gradient_confirmation'],
            cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('Reserved-test freeze failed; inspect without changing selected method')
    plan = json.loads((DEST/'row_weighted_reward_gradient_confirmation_v1/frozen_plan_v1.json').read_text())
    a = [c for c in plan['configurations'] if c['model'] == 'rr' or c['model'] == 'qwen' and c['protocol'] in ['text_video', 'video_text', 'text_image']]
    b = [c for c in plan['configurations'] if c not in a]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(worker, 0, a), pool.submit(worker, 3, b)]
        failures = [row for job in jobs for row in job.result()]
    create_json(DEST/'f9_confirmation_inference_completion_v1.json', dict(time=timestamp(), failures=failures,
        all_frozen_cases_attempted=True, final_goal_completed=False))
    if failures:
        raise RuntimeError('Reserved inference failures retained; no score-based case removal or retry')
    with (DEST/'f9_confirmation_analysis_stdout_v1.log').open('x') as handle:
        result = subprocess.run([sys.executable, '-u', '-m', 'mydata_bench.auto_research_addbase.analyze_row_weighted_reward_gradient_confirmation'],
            cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, env=dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1'))
    if result.returncode:
        raise RuntimeError('Reserved analysis failed; preserve all results and frozen method')
    create_json(DEST/'f9_confirmation_evaluation_and_analysis_completion_v1.json', dict(time=timestamp(),
        final_user_goal_requirement_by_requirement_audit_still_required=True, final_goal_completed=False))


if __name__ == '__main__':
    main()
