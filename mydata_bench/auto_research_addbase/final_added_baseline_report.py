"""Export completed mainline-1 results without altering predictions or selection."""
import argparse
import csv
import hashlib
import json
import pathlib
from datetime import datetime, timezone

PROTOCOLS = ['image_text', 'text_image', 'text_video', 'video_text', 'interleaved']
METRICS = ['n_expected', 'n_valid', 'n_missing_or_invalid', 'mae',
           'continuous_ordinal_mae', 'accuracy',
           'endpoint_accuracy_0.125_0.875', 'endpoint_accuracy_0.2_0.8', 'mean_progress']


def load(path):
    return json.loads(pathlib.Path(path).read_text())


def source(path):
    p = pathlib.Path(path).resolve()
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def val(x, percentage=False):
    if x is None:
        return 'NA'
    return f'{100*x:.4f}%' if percentage else f'{x:.6f}'


def tablerow(name, s):
    o = s['overall']
    return (f"|{name}|{o['n_valid']}/{o['n_expected']}|{val(o['mae'])}|"
            f"{val(o['continuous_ordinal_mae'])}|{val(o['accuracy'], True)}|"
            f"{val(s['by_split']['suc'].get('accuracy'), True)}|"
            f"{val(s['by_split']['fail'].get('accuracy'), True)}|"
            f"{val(o['endpoint_accuracy_0.2_0.8'], True)}|")


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def flatten(name, stage, condition, group, task, split, value):
    return {'experiment': name, 'stage': stage, 'condition': condition,
            'group': group, 'task': task, 'split': split,
            **{m: value.get(m) for m in METRICS},
            **{f'prediction_{k}_count': value.get('prediction_distribution', {}).get(str(k), 0)
               for k in range(1, 6)}}


def main():
    parser = argparse.ArgumentParser()
    for argument in ['metrics', 'attention-audit', 'record-audit', 'macro', 'head-overlap', 'output']:
        parser.add_argument('--' + argument, required=True)
    args = parser.parse_args()
    data = load(args.metrics)
    audit = load(args.attention_audit)
    records = load(args.record_audit)
    macro = load(args.macro)
    heads = load(args.head_overlap)
    experiments = data['experiments']
    assert len(audit['experiments']) == 15
    assert records['all_scientific_duplicate_records_consistent']
    assert len(macro['reports']) == 1
    assert macro['reports'][0]['sha256'] == source(args.metrics)['sha256']
    assert len(macro['reports'][0]['summaries']) == 180
    for name, a in audit['experiments'].items():
        assert a['all_expected_conditions_attempted'], name
        assert all(c['n_expected'] == c['n_attempted'] == 846 for c in a['conditions'].values()), name
        assert all(z['complete_cohort'] and z['all_paired_native_outputs_exact']
                   for z in a['zero_controls'].values()), name
        assert all(z['all_paired_steps_exact'] for z in a['zero_step_traces'].values()), name
        assert a['n_raw_rows'] == a['n_unique_rows'], name
        assert set(a['conditions']) == set(experiments[name]['attention']), name
        for c, v in a['conditions'].items():
            assert v['n_valid'] == experiments[name]['attention'][c]['overall']['n_valid'], (name, c)
    for model in ['meter', 'sole']:
        for protocol in PROTOCOLS:
            name = f'{model}_{protocol}_v1'
            s = experiments[name]['baseline_full']
            r = records['files'][name + '/baseline.jsonl']
            assert s['overall']['n_expected'] == r['n_unique_keys'] == 1213, name
    out = pathlib.Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    rows = []; pairs = []
    for name, e in experiments.items():
        entries = [(stage, 'baseline', e[stage]) for stage in ['baseline_full', 'baseline_cohort'] if stage in e]
        entries += [('attention', c, s) for c, s in e.get('attention', {}).items()]
        for stage, condition, s in entries:
            rows.append(flatten(name, stage, condition, 'overall', '', '', s['overall']))
            for split, value in s['by_split'].items():
                rows.append(flatten(name, stage, condition, 'split', '', split, value))
            for task, value in s['by_task'].items():
                rows.append(flatten(name, stage, condition, 'task', task, '', value))
            for task, splits in s['by_task_split'].items():
                for split, value in splits.items():
                    rows.append(flatten(name, stage, condition, 'task_split', task, split, value))
            p = s['pairwise']
            pairs.append({'experiment': name, 'stage': stage, 'condition': condition,
                          **{k: v for k, v in p.items() if not isinstance(v, dict)},
                          **p['discrete_suc_minus_fail_counts']})
    write_csv(out / 'all_metrics_task_split_distribution.csv', rows)
    write_csv(out / 'all_pairwise.csv', pairs)
    header = ['|模型/输入/条件|有效/期待|五档MAE|连续ordinal MAE|总acc .125/.875|suc acc|fail acc|总acc .2/.8|',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    lines = ['# 新增baseline与attention全部完成结果', '',
             f'生成时间：{datetime.now(timezone.utc).isoformat()}。全部指标来自完成后的单一快照，原始预测和失败记录保留。', '',
             '10项native各1213次尝试；Meter五输入各846×10；SOLE五输入全七步与终步各846×11。合计147490个唯一主预测记录（不含step traces）。完成表示全部预定样本/条件已尝试，原生格式或几何失败仍明确列出。', '',
             '准确率分母包含所有期待样本，缺失/格式失败按不正确计入；MAE只对有效数值计算，须结合有效数解读。连续ordinal为|1+4p−y|，五档MAE使用既定progress分档；都不是RoboRewardBench官方23子集Overall。低阈值严格p<low，高阈值p≥high。', '',
             'Meter官方结构text_image；SOLE官方结构image_text。其余是明确输入适配。SOLE使用自身历史七步递归、greedy/max512，与官方随机解码默认不同。全七步每个条件独立递归；terminal仅末步干预，不能互相替代。', '',
             '## 原生baseline：全1213，34任务', '', *header]
    for model in ['meter', 'sole']:
        for protocol in PROTOCOLS:
            name = f'{model}_{protocol}_v1'
            lines.append(tablerow(name, experiments[name]['baseline_full']))
    lines += ['', 'Meter text_video官方sum==1类型猜测的两条越界读出保留；typed-softmax敏感性见BASE/meter_baseline_typed_readout_sensitivity_v1.json。SOLE原生text_video有9条格式失败，不从reasoning猜分。', '']
    groups = [('Meter原attention：846，28任务', 'meter', '_v1'),
              ('SOLE全七步attention：846，28任务', 'sole', '_attention_residual_v1'),
              ('SOLE终步attention补充：846，28任务', 'sole', '_terminal_residual_v1')]
    effects = []
    for title, model, suffix in groups:
        lines += ['## ' + title, '', *header]
        for protocol in PROTOCOLS:
            name = f'{model}_{protocol}{suffix}'
            entries = experiments[name]['attention']
            for condition in ['baseline', 'target_k8', 'target_k32', 'target_k64']:
                s = entries[condition]
                lines.append(tablerow(protocol + '/' + condition, s))
                if condition == 'baseline':
                    continue
                b = entries['baseline']
                metric = 'continuous_ordinal_mae' if model == 'meter' else 'mae'
                effect = {'experiment': name, 'condition': condition,
                          'target_n_valid': s['overall']['n_valid'],
                          'baseline_n_valid': b['overall']['n_valid'],
                          'complete_scores_for_point_gate': s['complete'] and b['complete'],
                          'delta_mae': s['overall'][metric] - b['overall'][metric],
                          'delta_accuracy_pp': 100*(s['overall']['accuracy']-b['overall']['accuracy']),
                          **{f'delta_{split}_accuracy_pp': 100*(s['by_split'][split]['accuracy']-b['by_split'][split]['accuracy'])
                             for split in ['suc', 'fail']}}
                effect['four_direction_10pp_point'] = (effect['complete_scores_for_point_gate'] and effect['delta_mae'] < 0 and effect['delta_accuracy_pp'] >= 10
                                                       and effect['delta_suc_accuracy_pp'] > 0 and effect['delta_fail_accuracy_pp'] > 0)
                effects.append(effect)
        lines += ['', '其余wrong、low-head与zero全部条件见CSV和源JSON；条件间格式覆盖不同，不能忽略有效样本变化。', '']
    write_csv(out / 'all_target_effects.csv', effects)
    lines += ['## 原attention预定target点的总体表现', '',
              '下列计数同时要求target与同实现baseline有效覆盖完整、MAE下降、主阈值suc/fail提高、总准确率至少+10pp。它只是已测点的描述性检查，不是新方法选型或统计显著性检验。', '',
              '|组别|通过的target点/15|具体已测点|', '|---|---:|---|']
    for title, model, suffix in groups:
        relevant = [e for e in effects if e['experiment'].startswith(model + '_') and e['experiment'].endswith(suffix)]
        if model == 'meter':
            relevant = [e for e in relevant if '_residual_' not in e['experiment']]
        accepted = [e for e in relevant if e['four_direction_10pp_point']]
        assert len(relevant) == 15
        names = ', '.join(e['experiment'] + '/' + e['condition'] for e in accepted) or '无'
        lines.append(f'|{title}|{len(accepted)}/15|{names}|')
    lines += ['', '![全部原attention target点](figures/all_target_effects.png)', '',
              '图红色为改善，蓝色为变差；每列色标各自标明单位，数值为全部已测点。', '']
    lines += ['## 全部条件尝试、错误和zero核验', '',
              '|实验|唯一行|期待行|无效条数（含全部条件）|zero配对数|zero原生输出/trace精确|',
              '|---|---:|---:|---:|---:|---|']
    for name, a in audit['experiments'].items():
        n = sum(c['n_expected'] for c in a['conditions'].values())
        invalid = sum(c['n_attempted']-c['n_valid'] for c in a['conditions'].values())
        zeros = a['zero_controls']
        lines.append(f"|{name}|{a['n_unique_rows']}|{n}|{invalid}|{sum(z['n_paired'] for z in zeros.values()) if zeros else '另见Meter工程审计'}|{'通过' if zeros else '非本全量条件'}|")
    lines += ['', '无效条数按条件累计，同一样本可能在多个条件失败；不等于独立失败视频数。源attention audit列出每个条件的所有失败ID、原生错误文字、缺失数及step trace，record audit保存逐文件SHA/唯一键/重复冲突。Meter正式十条件没有全量zero，原small-forward零干预与后续完整新方法zero核验分开报告。', '',
              '## 各任务等权macro', '',
              '全部180组task-macro见传入的独立macro JSON；其逐样本分母和有效数也在CSV。native全1213用34个任务，grounded846用28个任务；不补不存在的任务，不排除表现较差任务。', '',
              '## 具体top8与跨模型重合', '',
              'layer/head从0开始。坐标重合不代表功能等价；跨架构宽度和训练变化均限制解释。', '',
              '|模型/输入|top8|', '|---|---|']
    for name, ranking in sorted(heads['rankings'].items()):
        lines.append('|' + name + '|' + ', '.join(f"L{r['layer']}H{r['head']}" for r in ranking[:8]) + '|')
    lines += ['', '|相同输入的模型对|top8交集|top32交集|top64交集|', '|---|---:|---:|---:|']
    for name, counts in sorted(heads['same_input_cross_model_overlaps'].items()):
        lines.append('|' + name + '|' + '|'.join(str(counts[str(k)]['intersection']) for k in [8, 32, 64]) + '|')
    inputs = {key: source(getattr(args, key)) for key in ['metrics', 'attention_audit', 'record_audit', 'macro', 'head_overlap']}
    lines += ['', '## 完整文件', '',
              '- all_metrics_task_split_distribution.csv：全部180组、总体/类别/task/task×类别，两MAE、两阈值、五档预测分布。',
              '- all_pairwise.csv：全部180组同视频配对的负/0/1/2/3/4计数、平均progress差、有效/期待配对数。',
              '- all_target_effects.csv：全部45个target点相对各自同实现baseline的变化，不按表现挑点。',
              '- source_manifest.json：源文件和本报告实现SHA；原始失败/重复没有删除。', '', '源文件：', '']
    lines += [f"- {key}: `{item['path']}` (SHA256 {item['sha256']})" for key, item in inputs.items()]
    (out / 'report.md').write_text('\n'.join(lines) + '\n')
    manifest = {'time': datetime.now(timezone.utc).isoformat(), 'inputs': inputs,
                'implementation': source(__file__), 'metric_csv_rows': len(rows), 'pairwise_rows': len(pairs),
                'target_effect_rows': len(effects), 'completion_checks_passed': True,
                'outputs': {p.name: source(p) for p in out.iterdir() if p.is_file()}}
    with (out / 'source_manifest.json').open('x') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(json.dumps({'output': str(out), 'metric_rows': len(rows), 'pairwise_rows': len(pairs),
                      'target_effect_rows': len(effects)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
