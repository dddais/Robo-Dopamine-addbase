"""Once-only full reserved-test label join, with fixed comparisons and uncertainty."""
import collections
import csv
import json
import pathlib
import yaml
from .common import ROOT, create_json, read_rows, timestamp
from .robust_prepare import DEST, digest
from .reward_control_feasibility import feasible_wrong_control
from .freeze_ordinal_reward_gradient_confirmation import selected_confirmation_cases
from .reward_validation_metrics import observations, summarize_observations, effect, paired_cluster_intervals, holm_adjust


def validate_configuration_scope(configurations, selection):
    names=[r['model']+'_'+r['protocol'] for r in configurations]
    paths=[r['path'] for r in configurations]
    if len(names)!=len(set(names)) or len(paths)!=len(set(paths)):
        raise ValueError('Duplicate reserved model/protocol or configuration path')
    expected=selected_confirmation_cases(selection)
    if set(names)!=set(expected):
        raise ValueError('Reserved configurations must include every eligible old846 case exactly once')
    return names


def main():
    root = DEST/'ordinal_reward_gradient_confirmation_v1'
    plan_path = root/'frozen_plan_v1.json'
    plan = json.loads(plan_path.read_text())
    for field in ['input', 'example_ids', 'groups', 'policy', 'confirmation_claim', 'selection_analysis']:
        if digest(plan[field+'_file']) != plan[field+'_sha256']:
            raise ValueError('Frozen confirmation dependency changed: '+field)
    for path, sha in plan['source_sha256'].items():
        if digest(ROOT/path) != sha:
            raise ValueError('Frozen confirmation source changed')
    claim=json.loads(pathlib.Path(plan['confirmation_claim_file']).read_text())
    if claim.get('family')!='F8_cumulative_ordinal_signed_all_query_reward_gradient' or claim.get('policy_sha256')!=plan['policy_sha256']:
        raise ValueError('Reserved claim does not belong to the frozen F8 method and policy')
    ids = json.loads(pathlib.Path(plan['example_ids_file']).read_text())
    if len(ids) != 2831 or len(set(ids)) != 2831:
        raise ValueError('All2831 reserved IDs required')
    cfgs = plan['configurations']
    if len(cfgs) != plan['expected_cases'] or not 9 <= len(cfgs) <= 15:
        raise ValueError('All preregistered selected cases required')
    selection = json.loads(pathlib.Path(plan['selection_analysis_file']).read_text())
    validate_configuration_scope(cfgs, selection)
    loaded = []
    output_directories=set()
    for item in cfgs:
        if digest(item['path']) != item['sha256']:
            raise ValueError('Frozen reserved-test configuration changed')
        cfg = yaml.safe_load(pathlib.Path(item['path']).read_text())
        if any(cfg[field]!=item[field] for field in ['model','protocol']):
            raise ValueError('Reserved config model/protocol mismatch')
        out = pathlib.Path(cfg['output_dir'])/'confirmation'
        if str(out.resolve()) in output_directories:
            raise ValueError('Different reserved cases share an output directory')
        output_directories.add(str(out.resolve()))
        done = json.loads((out/'completion_v1.json').read_text())
        if not json.loads((out.parent/'engineering/engineering_audit_v1.json').read_text())['passed']:
            raise ValueError('Reserved-test engineering failed')
        path = out/'predictions.jsonl'
        if digest(path) != done['predictions_sha256'] or json.loads((out/'config.json').read_text()) != cfg:
            raise ValueError('Reserved predictions/config mismatch')
        rows = list(read_rows(path))
        keys = [(r['example_id'], r['condition']) for r in rows]
        expected = {(eid, c['name']) for eid in ids for c in cfg['conditions']}
        if len(keys) != len(set(keys)) or set(keys) != expected or len(keys) != done['expected_records']:
            raise ValueError('Incomplete/duplicate/unexpected reserved predictions')
        loaded.append((cfg, done, rows))
    # All cases and conditions are complete before the first test-label join.
    if digest(plan['analysis_labels_file']) != plan['analysis_labels_sha256']:
        raise ValueError('Original sealed labels changed')
    create_json(root/'performance_opening_v1.json', dict(time=timestamp(), frozen_plan_sha256=digest(plan_path),
        completed_cases=len(loaded), complete_expected_denominator=2831,
        decision='Now join all frozen outputs with labels once; no subsequent parameter/case substitution'))
    label_rows = list(read_rows(plan['analysis_labels_file']))
    labels = {r['example_id']: r for r in label_rows}
    if len(labels) != len(label_rows) or set(labels) != set(ids):
        raise ValueError('Reserved labels are not exactly all2831 unique IDs')
    group_rows = list(read_rows(plan['groups_file']))
    groups = {r['example_id']: r['episode_group_sha256'] for r in group_rows}
    if len(groups) != len(group_rows) or set(groups) != set(ids) or len(set(groups.values())) != 1172:
        raise ValueError('Reserved episode grouping scope mismatch')
    # Reuse the mixed-grade statistics' group-key interface only. This alias
    # does not make these test examples members of the fitting/validation data.
    samples = [dict(r, training_episode_group_sha256=groups[r['example_id']]) for r in read_rows(plan['input_file'])]
    policy = json.loads(pathlib.Path(plan['policy_file']).read_text())
    selection = json.loads(pathlib.Path(plan['selection_analysis_file']).read_text())
    cases, pvalues, csv_rows = {}, {}, []
    for cfg, done, rows in loaded:
        name = cfg['model']+'_'+cfg['protocol']
        if not selection['cases'][name]['full846_point_gate']:
            raise ValueError('Case was not actually eligible before test opening')
        grouped = collections.defaultdict(list)
        for row in rows:
            grouped[row['condition']].append(row)
        obs = {key: observations(values, labels, samples, cfg['model']) for key, values in grouped.items()}
        summaries = {key: summarize_observations(values) for key, values in obs.items()}
        base, zero = ({r['example_id']: r for r in grouped[key]} for key in ['baseline', 'reward_gradient_zero_k48'])
        zero_exact = all(base[eid].get('status') == zero[eid].get('status') == 'ok' and
            all(base[eid].get(k) == zero[eid].get(k) for k in ['score_logits', 'progress', 'native_prediction']) for eid in ids)
        effects, uncertainty, primary_p = {}, {}, []
        for k in [32, 48, 64]:
            target = obs[f'reward_gradient_k{k}']
            refs = {'baseline': 'baseline', 'native_generation_baseline': 'native_generation_baseline',
                    'original_same_k': f'original_all_k{k}', 'fixed_best_old846_original': cfg['best_original_condition']}
            effects[str(k)] = {key: effect(target, obs[condition]) for key, condition in refs.items()}
            uncertainty[str(k)] = {key: paired_cluster_intervals(target, obs[condition],
                draws=policy['bootstrap_draws'], seed=policy['bootstrap_seed']) for key, condition in refs.items()}
            if k == 48:
                for result in uncertainty[str(k)].values():
                    primary_p += [result['metrics'][metric]['centered_bootstrap_one_sided_p_zero_improvement']
                                  for metric in ['mae_delta', 'accuracy_delta']]
                for condition in ['raw_all_k48', 'learned_positive_k48', 'learned_readout_k48', 'reward_gradient_low_k48']:
                    effects['primary_control_'+condition] = effect(target, obs[condition])
        wrong_feasibility=feasible_wrong_control(obs['reward_gradient_k48'],obs['reward_gradient_wrong_k48'],grouped['reward_gradient_wrong_k48'])
        point_gate = zero_exact and all(
            all(value['all_grade_improvement'] for value in effects[str(k)].values()) and
            all(effects[str(k)][reference]['endpoint_non_degradation'] for reference in ['baseline', 'native_generation_baseline'])
            for k in [32, 48, 64])
        if not all(value['complete'] for value in effects['48'].values()):
            primary_p.append(1.)
        pvalues[name] = max(primary_p)
        cases[name] = dict(model=cfg['model'], protocol=cfg['protocol'], confirmation_point_gate=point_gate,
            zero_exact=zero_exact, summaries=summaries, effects=effects, paired_episode_uncertainty=uncertainty, wrong_control_feasibility=wrong_feasibility,
            primary_mae_accuracy_IUT_p=pvalues[name], predictions_sha256=done['predictions_sha256'],
            errors=done['errors'], fixed_best_old846_original=cfg['best_original_condition'], old846_gate_verified=True)
        for condition, summary in summaries.items():
            overall = summary['overall']
            csv_rows.append(dict(case=name, condition=condition, n=overall['n'], n_invalid=overall['n_invalid'],
                all_grade_accuracy=overall['all_grade_accuracy'], ordinal_mae=overall['ordinal_mae_full_denominator'],
                five_bin_mae=overall['five_bin_mae_full_denominator'], baseline_fallback_rows=overall['baseline_fallback_rows']))
        print(name, 'reserved point gate', point_gate, 'primary vs baseline', effects['48']['baseline'], flush=True)
    adjusted = holm_adjust(pvalues)
    for name, pvalue in adjusted.items():
        cases[name]['primary_mae_accuracy_IUT_Holm_p'] = pvalue
        cases[name]['confirmation_supported'] = cases[name]['confirmation_point_gate'] and pvalue <= .05
    point_counts = {model: sum(r['confirmation_point_gate'] for r in cases.values() if r['model'] == model) for model in ['qwen', 'rr', 'meter']}
    supported_counts = {model: sum(r['confirmation_supported'] for r in cases.values() if r['model'] == model) for model in ['qwen', 'rr', 'meter']}
    evidence_pass = all(n >= 3 for n in supported_counts.values())
    create_json(root/'analysis_v1.json', dict(time=timestamp(), cases=cases, passing_inputs_by_model=point_counts,
        statistically_supported_inputs_by_model=supported_counts, scientific_evidence_gate=evidence_pass,
        all_preregistered_cases_reported=True, full_reserved_denominator=2831, conservative_episode_groups=1172,
        fixed_primary_k=48, tested_k_values=[32, 48, 64], untested_k_values_not_claimed=True,
        primary_Holm_family_size=len(cases), primary_IUT_scope='MAE and all-grade accuracy against both baselines, same-k original, fixed best old846 original',
        marginal_CIs_not_simultaneous=True, centered_bootstrap_p_values_approximate=True,
        p_values_not_a_10pp_lower_bound_test=True, endpoint_non_degradation_is_a_point_check_not_a_formal_noninferiority_test=True,
        original846_10pp_scope_already_required=True, mixed_grade_test_not_substituted_for_original_target=True,
        local_sealing_not_third_party_enforced=True, pretraining_contamination_not_excluded=True,
        frozen_plan_sha256=digest(plan_path), final_goal_completed=False,
        completion_note='Scientific gate is evidence for a separate full user-goal completion audit, not automatic goal completion'))
    with (root/'condition_summary_v1.csv').open('x') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    report = ['# F8保留测试一次性确认', '',
        '全部预选输入均报告；固定k32/48/64、primary48，2831条完整分母、1172来源/内容组。', '',
        '|模型/输入|三k点门槛|primary Holm p|统计支持|MAE变化|五档总准确率变化pp|', '|---|---|---:|---|---:|---:|']
    for name, row in cases.items():
        delta = row['effects']['48']['baseline']
        report.append(f"|{name}|{row['confirmation_point_gate']}|{row['primary_mae_accuracy_IUT_Holm_p']:.6g}|{row['confirmation_supported']}|{delta['mae_delta']:.6f}|{100*delta['accuracy_delta']:.4f}|")
    report += ['', '统计支持输入数：'+json.dumps(supported_counts, ensure_ascii=False)+'。科学证据门槛：'+str(evidence_pass)+'。', '',
        '完整task/source/grade/grounding分布、两套端点阈值、同组配对不确定性和全部对照见analysis_v1.json。', '',
        'F8用独立公开训练标签选头，骨干冻结，属于监督辅助学习。全部原生评分档位保留，无ROI回退baseline。', '',
        '原846的+10pp验收与本混合等级测试分别报告。p值检验零改善，不证明+10pp置信下限；端点不下降是点估计检查。边际区间不是同时区间。本地封存并非第三方盲测，不能排除骨干预训练污染。', '',
        '最终目标是否完成仍须逐条核验原计划及交付文档；此脚本不自动宣布目标完成。']
    with (root/'REPORT_v1.md').open('x') as handle:
        handle.write('\n'.join(report)+'\n')
    print('Final reserved analysis', root/'analysis_v1.json', supported_counts, flush=True)


if __name__ == '__main__':
    main()
