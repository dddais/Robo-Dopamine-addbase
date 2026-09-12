"""Require completed scientific ineligibility of both earlier families before F9 test use."""
import json
import pathlib
from .robust_prepare import DEST, digest
from .f8_confirmation_priority import confirmation_priority as f7_priority


def confirmation_priority(root=DEST):
    previous=f7_priority(root)
    if previous['state']!='released':
        return dict(previous,prior_family='F7',next_family='F9')
    eligibility=root/'f8_confirmation_queue_eligibility_v1.json'
    if eligibility.exists():
        final=json.loads(eligibility.read_text())
        if final.get('eligible') is False and final.get('reserved_test_reward_inference_started') is False:
            joint=root/'f8_full_old_queue_eligibility_v1.json'
            if joint.exists() and json.loads(joint.read_text()).get('eligible') is False:
                return dict(state='released',reason='F7 and F8 completed scientific ineligibility; shared test unclaimed',
                    F7_release=previous,evidence_sha256={str(eligibility):digest(eligibility),str(joint):digest(joint)})
            complete=root/'f8_full_old_evaluation_and_analysis_completion_v1.json'
            old=root/'ordinal_reward_gradient_full_old_v1/analysis_v1.json'
            if complete.exists() and old.exists() and json.loads(old.read_text()).get('full846_original_scope_gate') is False:
                return dict(state='released',reason='F7 ineligible and F8 completed original846 gate ineligible; test unclaimed',
                    F7_release=previous,evidence_sha256={str(eligibility):digest(eligibility),str(complete):digest(complete),str(old):digest(old)})
    launch=json.loads((root/'f8_confirmation_queue_launch_v1.json').read_text())
    process=pathlib.Path('/proc')/str(launch['pid'])/'cmdline'
    alive=process.exists() and b'mydata_bench.auto_research_addbase.ordinal_reward_gradient_confirmation_queue' in process.read_bytes()
    return dict(state='waiting' if alive else 'prerequisite_failed',f8_pid=launch['pid'],F7_release=previous,
        reason='F8 retains priority until completed scientific ineligibility; crashes never release a reserved test')
