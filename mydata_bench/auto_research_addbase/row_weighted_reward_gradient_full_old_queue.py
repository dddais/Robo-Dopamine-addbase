"""Conditionally schedule all846 after F9 joint old-development/validation pass."""
import concurrent.futures
import json
import os
import pathlib
import subprocess
import sys
import time
from .common import ROOT, append, create_json, timestamp
from .robust_prepare import DEST, digest
from .freeze_row_weighted_reward_gradient_full_old import eligible_cases


def worker(gpu, configurations):
    events = DEST/f'f9_full_old_gpu{gpu}_events_v1.jsonl'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
    failures = []
    for cfg in configurations:
        name = cfg['model']+'_'+cfg['protocol']
        if digest(cfg['path']) != cfg['sha256']:
            raise ValueError('Frozen full846 configuration changed')
        for stage in ['engineering', 'full846']:
            while True:
                free = int(subprocess.check_output(['nvidia-smi', f'--id={gpu}', '--query-gpu=memory.free',
                    '--format=csv,noheader,nounits'], text=True).strip())
                if free >= 32000:
                    break
                append(events, dict(time=timestamp(), event='memory_wait', case=name, stage=stage, free_mib=free))
                time.sleep(20)
            command = [sys.executable, '-u', '-m', 'mydata_bench.auto_research_addbase.all_query_reward_gradient_full_old',
                       '--config', cfg['path'], '--stage', stage]
            with (DEST/f'f9_full_old_{name}_{stage}_stdout_v1.log').open('x') as handle:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
                append(events, dict(time=timestamp(), event='launch', case=name, stage=stage, pid=process.pid, command=command))
                code = process.wait()
            append(events, dict(time=timestamp(), event='exit', case=name, stage=stage, pid=process.pid, returncode=code))
            if code:
                failures.append(dict(case=name, stage=stage, returncode=code))
                break
    create_json(DEST/f'f9_full_old_gpu{gpu}_completion_v1.json', dict(time=timestamp(), failures=failures,
        cases=[r['model']+'_'+r['protocol'] for r in configurations]))
    return failures


def main():
    files = ['all_query_reward_gradient_full_old.py', 'freeze_row_weighted_reward_gradient_full_old.py',
             'analyze_row_weighted_reward_gradient_full_old.py', 'row_weighted_reward_gradient_full_old_queue.py']
    hashes = {p: digest(ROOT/'mydata_bench/auto_research_addbase'/p) for p in files}
    create_json(DEST/'f9_full_old_queue_plan_v1.json', dict(time=timestamp(), pid=os.getpid(), dependencies_sha256=hashes,
        prerequisite='all F9 evaluation/analysis complete and same >=3 inputs per model jointly pass old234+validation',
        selection='all jointly eligible cases, no per-case k changes', original_full_denominator=846,
        no_automatic_retries=True, reserved_test_evaluation_scheduled=False))
    launch = json.loads((DEST/'f9_evaluation_queue_launch_v1.json').read_text())
    done_path = DEST/'f9_evaluation_and_analysis_completion_v1.json'
    while not done_path.exists():
        process = pathlib.Path('/proc')/str(launch['pid'])/'cmdline'
        if not process.exists() or b'mydata_bench.auto_research_addbase.row_weighted_reward_gradient_evaluation_queue' not in process.read_bytes():
            create_json(DEST/'f9_full_old_queue_prerequisite_failure_v1.json', dict(time=timestamp(),
                cause='Evaluation queue stopped without completed validation analysis; no full846 inference launched'))
            raise RuntimeError('F9 evaluation prerequisite failed; inspect preserved evidence')
        time.sleep(20)
    for p, sha in hashes.items():
        if digest(ROOT/'mydata_bench/auto_research_addbase'/p) != sha:
            raise ValueError('Prospective full846 source changed: '+p)
    candidates = sorted(DEST.glob('f9_training_validation_analysis_*.json'))
    if not candidates:
        raise ValueError('Evaluation completion has no validation analysis')
    analysis_path = candidates[-1]
    analysis = json.loads(analysis_path.read_text())
    try:
        names = eligible_cases(analysis)
    except ValueError as exc:
        create_json(DEST/'f9_full_old_queue_eligibility_v1.json', dict(time=timestamp(), eligible=False,
            reason=str(exc), analysis_file=str(analysis_path), analysis_sha256=digest(analysis_path),
            observed_joint_counts=analysis.get('same_input_joint_passing_counts'), final_goal_completed=False))
        return
    create_json(DEST/'f9_full_old_queue_eligibility_v1.json', dict(time=timestamp(), eligible=True,
        cases=names, analysis_file=str(analysis_path), analysis_sha256=digest(analysis_path), final_goal_completed=False))
    with (DEST/'f9_full_old_freeze_stdout_v1.log').open('x') as handle:
        result = subprocess.run([sys.executable, '-u', '-m', 'mydata_bench.auto_research_addbase.freeze_row_weighted_reward_gradient_full_old',
            '--analysis', str(analysis_path)], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('Full846 freeze failed; no inference launched')
    plan = json.loads((DEST/'row_weighted_reward_gradient_full_old_v1/frozen_plan_v1.json').read_text())
    a = [c for c in plan['configurations'] if c['model'] == 'rr' or c['model'] == 'qwen' and c['protocol'] in ['text_video', 'video_text', 'text_image']]
    b = [c for c in plan['configurations'] if c not in a]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(worker, 0, a), pool.submit(worker, 3, b)]
        failures = [row for job in jobs for row in job.result()]
    create_json(DEST/'f9_full_old_inference_completion_v1.json', dict(time=timestamp(), failures=failures,
        final_reserved_test_still_required=True, final_goal_completed=False))
    if failures:
        raise RuntimeError('Full846 inference errors retained; no silent retries')
    with (DEST/'f9_full_old_analysis_stdout_v1.log').open('x') as handle:
        result = subprocess.run([sys.executable, '-u', '-m', 'mydata_bench.auto_research_addbase.analyze_row_weighted_reward_gradient_full_old'],
            cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, env=dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1'))
    if result.returncode:
        raise RuntimeError('Full846 analysis failed; preserve and inspect')
    create_json(DEST/'f9_full_old_evaluation_and_analysis_completion_v1.json', dict(time=timestamp(),
        final_reserved_test_freeze_and_confirmation_still_required=True, final_goal_completed=False))


if __name__ == '__main__':
    main()
