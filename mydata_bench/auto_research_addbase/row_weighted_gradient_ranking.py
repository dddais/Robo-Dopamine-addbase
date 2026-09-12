"""Sample-mean loss gradient, with episode-cluster variability for sparse head ranking."""
import collections
import numpy as np


def rank_row_weighted_gradients(records, expected_episode_groups, shape, skip_layers=8):
    by_id={row['example_id']:row for row in records}
    if len(by_id)!=len(records) or set(by_id)!=set(expected_episode_groups):
        raise ValueError('Full unique expected fitting rows required')
    grouped=collections.defaultdict(list)
    for eid,group in expected_episode_groups.items():
        row=by_id[eid]
        if row['status']=='no_grounding':
            value=np.zeros(shape,dtype=np.float64)
        elif row['status']=='ok':
            value=np.asarray(row['gradient'],dtype=np.float64)
        else:
            raise ValueError('Required fitting gradient failed; do not exclude it')
        if value.shape!=tuple(shape) or not np.isfinite(value).all():
            raise ValueError('Invalid gradient shape or values')
        grouped[group].append(value)
    count=len(grouped)
    if count<2:
        raise ValueError('At least two fitting episode groups required')
    sums=np.stack([np.sum(grouped[key],axis=0) for key in sorted(grouped)])
    sizes=np.asarray([len(grouped[key]) for key in sorted(grouped)],dtype=np.float64)
    n=len(records)
    mean=sums.sum(axis=0)/n
    residual=sums-sizes[:,None,None]*mean
    se=np.sqrt(count/(count-1.)*np.sum(residual**2,axis=0))/n
    strength=np.abs(mean)-1.96*se
    ranking=[dict(layer=layer,head=head,score=float(strength[layer,head]),
        mean_loss_gradient=float(mean[layer,head]),standard_error=float(se[layer,head]),
        signed_bias_direction=-1 if mean[layer,head]>0 else 1 if mean[layer,head]<0 else 0)
        for layer in range(skip_layers,shape[0]) for head in range(shape[1])]
    ranking.sort(key=lambda row:(-row['score'],-abs(row['mean_loss_gradient']),row['layer'],row['head']))
    if not any(row['signed_bias_direction'] for row in ranking):
        raise ValueError('No fitting gradient signal; do not invent a ranking')
    return dict(ranking=ranking,fitting_examples=n,fitting_episode_groups=count,
        no_grounding_rows=sum(row['status']=='no_grounding' for row in records),skip_early_layers=skip_layers,
        score_definition='abs(row_mean_loss_gradient)-1.96*episode_cluster_SE',
        mean_definition='sum of all fitting-row gradients / full fitting-row count',
        cluster_SE_definition='sqrt(G/(G-1)*sum_g(S_g-n_g*mu_row)^2)/N',
        inference_direction='negative sign of row mean loss gradient',
        uses_supervised_external_fit_labels=True,uses_old_development_labels=False,uses_validation_or_test_labels=False,
        interpretation='sample-weighted fitting objective with episode-cluster variability penalty; '
            'not simultaneous confidence or guaranteed finite-bias, native-score, or generalization improvement')
