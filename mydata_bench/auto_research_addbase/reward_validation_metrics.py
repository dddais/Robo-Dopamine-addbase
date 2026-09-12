"""Full-denominator mixed-grade evaluation, separate from label-free inference.

Invalid outputs count as incorrect and incur the maximum ordinal error (4).
This is a reporting penalty, never an inference-time endpoint assignment.
"""
import collections
import math
import numpy as np
from mydata_bench.protocol import progress_to_reward


THRESHOLDS = {'0.125_0.875': (.125, .875), '0.2_0.8': (.2, .8)}


def observations(rows, labels, samples, model):
    by_id = {r['example_id']: r for r in rows}
    ids = [s['example_id'] for s in samples]
    if len(set(ids)) != len(ids) or len(by_id) != len(rows) or set(by_id) != set(ids) or set(labels) != set(ids):
        raise ValueError('Require exactly the complete unique frozen cohort')
    result = []
    for sample in samples:
        eid = sample['example_id']
        truth = labels[eid]['reward']
        if isinstance(truth, bool) or truth not in (1, 2, 3, 4, 5):
            raise ValueError('Native grade 1–5 required')
        row = by_id[eid]
        valid = row.get('status') == 'ok'
        progress, prediction = row.get('progress'), row.get('native_prediction')
        if model == 'meter':
            valid = valid and isinstance(progress, (float, int)) and math.isfinite(progress) and 0 <= progress <= 1
            prediction = progress_to_reward(progress) if valid else None
            ordinal = 1 + 4 * progress if valid else None
        else:
            valid = valid and not isinstance(prediction, bool) and prediction in (1, 2, 3, 4, 5)
            ordinal = prediction if valid else None
        value = dict(example_id=eid, group=sample['training_episode_group_sha256'], task=sample['task'],
            source_subset=sample['source_subset'], grade=int(truth), valid=bool(valid), prediction=prediction if valid else None,
            grounding_status=sample['grounding_status'], baseline_fallback=bool(row.get('baseline_fallback', False)),
            absolute_error=abs(ordinal-truth) if valid else 4.,
            five_bin_absolute_error=abs(prediction-truth) if valid else 4.,
            accuracy=float(valid and prediction == truth))
        for name, (low, high) in THRESHOLDS.items():
            if truth not in (1, 5):
                hit = None
            elif model == 'meter':
                hit = float(valid and (progress < low if truth == 1 else progress >= high))
            else:
                hit = value['accuracy']
            value['endpoint_accuracy_' + name] = hit
        result.append(value)
    return result


def aggregate(obs):
    n = len(obs)
    if not n:
        return dict(n=0)
    valid = [r for r in obs if r['valid']]
    out = dict(n=n, n_valid=len(valid), n_invalid=n-len(valid), complete=len(valid) == n,
        all_grade_accuracy=float(np.mean([r['accuracy'] for r in obs])),
        ordinal_mae_full_denominator=float(np.mean([r['absolute_error'] for r in obs])),
        five_bin_mae_full_denominator=float(np.mean([r['five_bin_absolute_error'] for r in obs])),
        ordinal_mae_valid_only=float(np.mean([r['absolute_error'] for r in valid])) if valid else None,
        n_groups=len({r['group'] for r in obs}), baseline_fallback_rows=sum(r['baseline_fallback'] for r in obs),
        grounding_counts=dict(collections.Counter(r['grounding_status'] for r in obs)),
        prediction_counts=dict(collections.Counter(str(r['prediction']) if r['valid'] else 'invalid' for r in obs)),
        grade_counts=dict(collections.Counter(str(r['grade']) for r in obs)))
    for threshold in THRESHOLDS:
        key = 'endpoint_accuracy_' + threshold
        out[key] = {}
        for grade, split in [(1, 'fail'), (5, 'suc')]:
            subset = [r for r in obs if r['grade'] == grade]
            out[key][split] = dict(n=len(subset), accuracy=float(np.mean([r[key] for r in subset])) if subset else None)
    return out


def summarize_observations(obs):
    output = {'overall': aggregate(obs)}
    for field in ['task', 'source_subset', 'grade', 'grounding_status']:
        groups = collections.defaultdict(list)
        for row in obs:
            groups[str(row[field])].append(row)
        summaries = {key: aggregate(rows) for key, rows in sorted(groups.items())}
        output['by_' + field] = summaries
        if field in ['task', 'source_subset']:
            output[field + '_macro'] = {name: float(np.mean([s[name] for s in summaries.values()]))
                for name in ['all_grade_accuracy', 'ordinal_mae_full_denominator', 'five_bin_mae_full_denominator']}
    grouped = collections.defaultdict(list)
    for row in obs:
        grouped[row['group']].append(row)
    output['equal_episode_macro'] = {name: float(np.mean([aggregate(rows)[name] for rows in grouped.values()]))
        for name in ['all_grade_accuracy', 'ordinal_mae_full_denominator', 'five_bin_mae_full_denominator']}
    return output


def effect(candidate, reference):
    if [r['example_id'] for r in candidate] != [r['example_id'] for r in reference]:
        raise ValueError('Paired IDs/order mismatch')
    a, b = aggregate(candidate), aggregate(reference)
    result = dict(complete=a['complete'] and b['complete'],
        mae_delta=a['ordinal_mae_full_denominator']-b['ordinal_mae_full_denominator'],
        five_bin_mae_delta=a['five_bin_mae_full_denominator']-b['five_bin_mae_full_denominator'],
        accuracy_delta=a['all_grade_accuracy']-b['all_grade_accuracy'])
    result['endpoints'] = {t: {split: (a['endpoint_accuracy_'+t][split]['accuracy']-b['endpoint_accuracy_'+t][split]['accuracy'])
        if a['endpoint_accuracy_'+t][split]['n'] else None for split in ['suc', 'fail']} for t in THRESHOLDS}
    result['all_grade_improvement'] = result['complete'] and result['mae_delta'] < 0 and result['accuracy_delta'] > 0
    result['endpoint_non_degradation'] = all(v is not None and v >= 0 for x in result['endpoints'].values() for v in x.values())
    return result


def paired_cluster_intervals(candidate, reference, draws=20000, seed=20260910):
    """Resample whole source/content groups; paired row-weighted target.

    Marginal percentile intervals are descriptive, not simultaneous coverage.
    A centered-bootstrap one-sided p for MAE/accuracy can enter a primary-case
    IUT/Holm analysis. It tests zero improvement, not a +10pp lower bound.
    """
    effect(candidate, reference)
    groups = sorted({r['group'] for r in candidate})
    index = {g: i for i, g in enumerate(groups)}
    fields = [('mae_delta', 'absolute_error', None), ('accuracy_delta', 'accuracy', None),
              ('five_bin_mae_delta', 'five_bin_absolute_error', None)]
    fields += [(split+'_'+t+'_delta', 'endpoint_accuracy_'+t, grade)
               for t in THRESHOLDS for grade, split in [(1, 'fail'), (5, 'suc')]]
    sums, counts = np.zeros((len(groups), len(fields))), np.zeros((len(groups), len(fields)))
    for a, b in zip(candidate, reference):
        if a['group'] != b['group']:
            raise ValueError('Paired group mismatch')
        g = index[a['group']]
        for j, (_, key, grade) in enumerate(fields):
            if grade is None or a['grade'] == grade:
                sums[g, j] += a[key]-b[key]
                counts[g, j] += 1
    rng = np.random.default_rng(seed)
    chunks = []
    for start in range(0, draws, 256):
        weights = rng.multinomial(len(groups), np.full(len(groups), 1/len(groups)), size=min(256, draws-start))
        denominator = weights @ counts
        numerator = weights @ sums
        chunks.append(np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0))
    boot = np.concatenate(chunks)
    result = dict(groups=len(groups), draws=draws, seed=seed, paired=True, resampling_unit='source/content connected episode group',
        estimand='row-weighted full-denominator mean difference', marginal_intervals_not_simultaneous=True, metrics={})
    for j, (name, _, _) in enumerate(fields):
        n = counts[:, j].sum()
        if n == 0:
            result['metrics'][name] = dict(estimate=None, reason='no examples of this endpoint grade')
            continue
        point = float(sums[:, j].sum()/n)
        values = boot[np.isfinite(boot[:, j]), j]
        lo, hi = np.quantile(values, [.025, .975])
        centered = values-point
        favorable_sign = -1 if 'mae' in name else 1
        p = float((1+np.count_nonzero(favorable_sign*centered >= favorable_sign*point))/(len(values)+1))
        result['metrics'][name] = dict(estimate=point, percentile95=[float(lo), float(hi)],
            valid_bootstrap_draws=len(values), centered_bootstrap_one_sided_p_zero_improvement=p)
    return result


def holm_adjust(pvalues):
    order = sorted(pvalues, key=lambda key: (pvalues[key], key))
    result, previous = {}, 0.
    for i, key in enumerate(order):
        previous = max(previous, min(1., (len(order)-i)*pvalues[key]))
        result[key] = previous
    return result
