"""Prospective F8: GPU2 synthetic engineering, freeze all15, fit all15; no retries."""
import fcntl
import json
import os
import pathlib
import subprocess
import sys
import time
import yaml
from .common import ROOT, append, create_json, timestamp
from .robust_prepare import DEST, digest


GPU = 2
CASES = [('qwen', 'text_video'), ('rr', 'image_text'), ('meter', 'text_image')]


def verify_sources(sources):
    for name, expected in sources.items():
        if digest(ROOT/name) != expected:
            raise ValueError('Frozen F8 dependency changed: '+name)


def run(module, arguments, label, sources):
    events = DEST/'f8_fit_gpu2_events_v1.jsonl'
    while True:
        free = int(subprocess.check_output(['nvidia-smi', f'--id={GPU}', '--query-gpu=memory.free',
            '--format=csv,noheader,nounits'], text=True).strip())
        if free >= 32000:
            break
        append(events, dict(time=timestamp(), event='memory_wait', case=label, free_mib=free, required_mib=32000))
        time.sleep(20)
    verify_sources(sources)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(GPU), OMP_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
    command = [sys.executable, '-u', '-m', 'mydata_bench.auto_research_addbase.'+module, *arguments]
    with (DEST/f'f8_{label}_stdout_v1.log').open('x') as handle:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        append(events, dict(time=timestamp(), event='launch', case=label, pid=process.pid, command=command))
        code = process.wait()
    append(events, dict(time=timestamp(), event='exit', case=label, pid=process.pid, returncode=code))
    return code


def main():
    lock = (DEST/'f8_fit_queue_writer.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    previous_path = DEST/'all_query_reward_gradient_fit_frozen_plan_v1.json'
    previous = json.loads(previous_path.read_text())
    sources = {}
    for item in previous['configurations']:
        if digest(item['path']) != item['sha256']:
            raise ValueError('F7 fit configuration changed before F8 launch')
        sources.update(yaml.safe_load(pathlib.Path(item['path']).read_text())['source_sha256'])
    names = ['common.py', 'robust_prepare.py', 'native_cumulative_ordinal_loss.py',
             'ordinal_reward_gradient_engineering.py', 'ordinal_reward_gradient_fit.py',
             'freeze_ordinal_reward_gradient_fit.py', 'ordinal_reward_gradient_fit_queue.py']
    for name in names:
        path = 'mydata_bench/auto_research_addbase/'+name
        sources[path] = digest(ROOT/path)
    verify_sources(sources)
    engineering_inputs = []
    for model, protocol in CASES:
        cfg_path = ROOT/f'mydata_bench/configs/v2_crossmodel/addbase_robust_frame_transport_v1_{model}_{protocol}.yaml'
        cfg = yaml.safe_load(cfg_path.read_text())
        paths = [cfg_path, pathlib.Path(cfg['engineering_ids_file']), pathlib.Path(cfg['input_file']),
                 DEST/f'frame_transport_v1/{model}_{protocol}/development/predictions.jsonl']
        engineering_inputs.append(dict(case=model+'_'+protocol, files_sha256={str(p): digest(p) for p in paths}))
    snapshots = DEST/'f8_fit_source_snapshots_v1'
    snapshots.mkdir(exist_ok=False)
    for name, sha in sources.items():
        target = snapshots/sha
        if not target.exists():
            with target.open('xb') as handle:
                handle.write((ROOT/name).read_bytes())
    create_json(DEST/'f8_fit_queue_plan_v1.json', dict(time=timestamp(), pid=os.getpid(),
        dependencies_sha256=sources, previous_fit_plan_sha256=digest(previous_path),
        source_snapshot_dir=str(snapshots), engineering_inputs=engineering_inputs,
        gpus=[GPU], no_automatic_retries=True, fitting_only=True,
        hypothesis_origin='F7 complete fit CE versus reward metric mismatch; validation not used to choose F8 parameters',
        changes_from_F7=['fitting objective only'], shared_bias_magnitude=1.,
        primary_k=48, k_neighborhood=[32, 48, 64],
        evaluation_and_reserved_test_not_scheduled=True, final_goal_completed=False))
    for model, protocol in CASES:
        record = next(r for r in engineering_inputs if r['case'] == model+'_'+protocol)
        for path, expected in record['files_sha256'].items():
            if digest(path) != expected:
                raise ValueError('Frozen synthetic engineering input changed: '+path)
        code = run('ordinal_reward_gradient_engineering', ['--model', model, '--protocol', protocol],
                   'engineering_'+model+'_'+protocol, sources)
        if code:
            create_json(DEST/'f8_fit_queue_prerequisite_failure_v1.json', dict(time=timestamp(),
                stage='synthetic_engineering', case=model+'_'+protocol, returncode=code, no_fits_launched=True))
            raise RuntimeError('F8 engineering failed; preserve and inspect')
    verify_sources(sources)
    with (DEST/'f8_fit_freeze_stdout_v1.log').open('x') as handle:
        result = subprocess.run([sys.executable, '-u', '-m',
            'mydata_bench.auto_research_addbase.freeze_ordinal_reward_gradient_fit'],
            cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('F8 fitting freeze failed; no fits launched')
    plan = json.loads((DEST/'ordinal_reward_gradient_fit_frozen_plan_v1.json').read_text())
    failures = []
    for item in plan['configurations']:
        case = item['model']+'_'+item['protocol']
        if digest(item['path']) != item['sha256']:
            raise ValueError('Frozen F8 fit config changed: '+case)
        code = run('ordinal_reward_gradient_fit', ['--config', item['path']], 'fit_'+case, sources)
        if code:
            failures.append(dict(case=case, returncode=code))
    create_json(DEST/'f8_all_fit_queues_completion_v1.json', dict(time=timestamp(), failures=failures,
        fitting_only=True, evaluation_not_scheduled=True, final_goal_completed=False))


if __name__ == '__main__':
    main()
