"""Release the shared test to F8 only after F7 finishes ineligible without opening it."""
import json
import pathlib
from .robust_prepare import DEST, digest


def confirmation_priority(root=DEST):
    claim=root/'external_roboreward_reserved_v1/reward_model_confirmation_claim_v1.json'
    if claim.exists():
        record=json.loads(claim.read_text())
        return dict(state='already_claimed',family=record.get('family'),claim_sha256=digest(claim),
                    reason='One-time shared confirmation was claimed; cannot reuse it for F8')
    eligibility=root/'f7_confirmation_queue_eligibility_v1.json'
    if eligibility.exists():
        final=json.loads(eligibility.read_text())
        if final.get('eligible') is False and final.get('reserved_test_reward_inference_started') is False:
            joint=root/'f7_full_old_queue_eligibility_v1.json'
            if joint.exists() and json.loads(joint.read_text()).get('eligible') is False:
                return dict(state='released',reason='F7 completed joint old234/493 selection ineligible; test unopened',
                    evidence_sha256={str(eligibility):digest(eligibility),str(joint):digest(joint)})
            complete=root/'f7_full_old_evaluation_and_analysis_completion_v1.json'
            old=root/'all_query_reward_gradient_full_old_v1/analysis_v1.json'
            if complete.exists() and old.exists() and json.loads(old.read_text()).get('full846_original_scope_gate') is False:
                return dict(state='released',reason='F7 completed full original846 gate ineligible; test unopened',
                    evidence_sha256={str(eligibility):digest(eligibility),str(complete):digest(complete),str(old):digest(old)})
    launch=json.loads((root/'f7_confirmation_queue_launch_v1.json').read_text())
    process=pathlib.Path('/proc')/str(launch['pid'])/'cmdline'
    alive=process.exists() and b'mydata_bench.auto_research_addbase.all_query_reward_gradient_confirmation_queue' in process.read_bytes()
    return dict(state='waiting' if alive else 'prerequisite_failed',f7_pid=launch['pid'],
                reason='F7 retains priority until a completed scientific ineligibility decision; a crash does not release it')
