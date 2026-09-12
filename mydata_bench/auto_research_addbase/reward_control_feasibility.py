"""Mechanistic comparisons conditional on a constructible grounded control.

The primary method continues to use the full evaluation denominator.
Candidate failures are retained even in this conditional comparison.
"""
from .reward_validation_metrics import effect


def feasible_wrong_control(candidate, control, raw_control_rows):
    raw = {r['example_id']: r for r in raw_control_rows}
    if len(raw) != len(raw_control_rows) or [r['example_id'] for r in candidate] != [r['example_id'] for r in control]:
        raise ValueError('Unique complete paired control rows required')
    if set(raw) != {r['example_id'] for r in candidate}:
        raise ValueError('Raw control cohort differs')
    selected, reasons = [], dict(no_grounding=0, declared_geometry_infeasible=0, other_control_errors=0)
    for i, (a, b) in enumerate(zip(candidate, control)):
        row = raw[a['example_id']]
        if a['grounding_status'] != 'ok':
            reasons['no_grounding'] += 1
        elif row.get('control_geometry_infeasible', False):
            reasons['declared_geometry_infeasible'] += 1
        elif not b['valid']:
            reasons['other_control_errors'] += 1
        else:
            selected.append(i)
    result = dict(full_primary_denominator=len(candidate), feasible_control_examples=len(selected), exclusions=reasons,
        candidate_errors_not_used_for_selection=True, full_primary_effect_unaffected=True,
        estimand='conditional on grounded examples with an available equal-area disjoint control',
        other_control_runtime_errors_prevent_clean_geometry_only_interpretation=reasons['other_control_errors'] > 0)
    if selected:
        result['effect_on_same_feasible_rows'] = effect([candidate[i] for i in selected], [control[i] for i in selected])
    else:
        result['effect_on_same_feasible_rows'] = None
    return result
