"""Signed benefit ranking from independent training-episode loss gradients."""
import collections
import numpy as np


def rank_reward_gradients(records, expected_episode_groups, shape, skip_layers=8):
    by_id = {row['example_id']: row for row in records}
    if len(by_id) != len(records) or set(by_id) != set(expected_episode_groups):
        raise ValueError('Full unique expected fitting rows required')
    groups = collections.defaultdict(list)
    for eid, group in expected_episode_groups.items():
        row = by_id[eid]
        if row['status']=='no_grounding':
            gradient = np.zeros(shape,dtype=np.float64)
        elif row['status']=='ok':
            gradient = np.asarray(row['gradient'],dtype=np.float64)
        else:
            raise ValueError('A required fitting gradient failed; do not silently drop it')
        if gradient.shape != tuple(shape) or not np.isfinite(gradient).all():
            raise ValueError('Invalid gradient shape or values')
        groups[group].append(gradient)
    if len(groups)<2:
        raise ValueError('At least two fitting episode groups required')
    values = np.stack([np.mean(groups[key],axis=0) for key in sorted(groups)])
    mean = values.mean(axis=0)
    se = values.std(axis=0,ddof=1)/np.sqrt(len(groups))
    strength = np.abs(mean)-1.96*se
    ranking = [dict(layer=layer,head=head,score=float(strength[layer,head]),
        mean_loss_gradient=float(mean[layer,head]),standard_error=float(se[layer,head]),
        signed_bias_direction=-1 if mean[layer,head]>0 else 1 if mean[layer,head]<0 else 0)
        for layer in range(skip_layers,shape[0]) for head in range(shape[1])]
    ranking.sort(key=lambda r:(-r['score'],-abs(r['mean_loss_gradient']),r['layer'],r['head']))
    if not any(r['signed_bias_direction'] for r in ranking):
        raise ValueError('No fitting gradient signal; do not invent a ranking')
    return dict(ranking=ranking,fitting_examples=len(records),fitting_episode_groups=len(groups),
        no_grounding_rows=sum(row['status']=='no_grounding' for row in records),
        skip_early_layers=skip_layers,score_definition='abs(mean_episode_loss_gradient)-1.96*SE_episode',
        inference_direction='negative sign of mean loss gradient',
        uses_supervised_external_fit_labels=True,uses_old_development_labels=False,uses_validation_or_test_labels=False,
        interpretation='fixed variability penalty for signed first-order benefit; not simultaneous confidence coverage or guaranteed nonlinear benefit')
