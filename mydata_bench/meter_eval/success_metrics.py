"""Read-only evaluation of saved Robometer terminal success probabilities.

Uses the official example's fixed q > 0.5 rule; never fits a threshold or runs
the model. Historical predictions/configurations are inputs, never outputs.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import pathlib
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics import roc_auc_score

from .readout import canonicalize

ROOT = pathlib.Path(__file__).resolve().parents[2]
BASE = ROOT / 'results/mydata_bench/experiments_v2_addbase/session_20260908'
REFERENCE = ROOT / 'results/mydata_bench/experiments_v2_corssmodel/auto_research/session_20260908/references/robometer_official'
PROTOCOLS = ['text_video', 'video_text', 'text_image', 'image_text', 'interleaved']
NAMES = dict(zip(PROTOCOLS, ['文本 → 视频', '视频 → 文本', '文本 → 图像（官方结构）', '图像 → 文本', '交错输入']))


def read_rows(path):
    with pathlib.Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def index_rows(rows):
    indexed = {}
    for row in rows:
        key = row['example_id']
        if key in indexed:
            raise ValueError('Duplicate example in one condition: ' + key)
        indexed[key] = row
    return indexed


def metrics(scores, labels, ids):
    """Missing values count as incorrect; probability errors/AUC use valid rows."""
    ids = sorted(set(ids))
    valid = [i for i in ids if i in scores and math.isfinite(scores[i])]
    y = np.asarray([labels[i]['reward'] == 5 for i in valid], dtype=bool)
    q = np.asarray([scores[i] for i in valid], dtype=float)
    pred = q > .5
    n_suc = sum(labels[i]['reward'] == 5 for i in ids)
    n_fail = len(ids) - n_suc
    tp = int(np.sum(pred & y)); tn = int(np.sum(~pred & ~y))
    fp = int(np.sum(pred & ~y)); fn = int(np.sum(~pred & y))
    ratio = lambda a, b: float(a / b) if b else None
    suc_acc = ratio(tp, n_suc); fail_acc = ratio(tn, n_fail)
    result = dict(n_expected=len(ids), n_valid=len(valid), n_suc=n_suc, n_fail=n_fail,
                  n_invalid_suc=n_suc-int(y.sum()), n_invalid_fail=n_fail-int((~y).sum()),
                  tp=tp, tn=tn, fp=fp, fn=fn, accuracy=ratio(tp+tn, len(ids)),
                  suc_accuracy=suc_acc, fail_accuracy=fail_acc,
                  balanced_accuracy=(suc_acc+fail_acc)/2 if n_suc and n_fail else None,
                  auroc=float(roc_auc_score(y, q)) if y.any() and (~y).any() else None,
                  score_mae=float(np.mean(np.abs(q-y))) if valid else None,
                  brier=float(np.mean((q-y)**2)) if valid else None,
                  accuracy_ge_0_5=ratio(int(np.sum((q >= .5) == y)), len(ids)),
                  n_equal_0_5=int(np.sum(q == .5)),
                  n_equal_0_5_suc=int(np.sum((q == .5) & y)),
                  n_equal_0_5_fail=int(np.sum((q == .5) & ~y)))
    for split, mask in [('suc', y), ('fail', ~y)]:
        values = q[mask]
        result[split+'_mean_score'] = float(values.mean()) if len(values) else None
        result[split+'_score_quantiles'] = dict(zip(['min', 'q25', 'median', 'q75', 'max'],
            np.quantile(values, [0, .25, .5, .75, 1]).tolist())) if len(values) else None
    for low, high in [(.125, .875), (.2, .8)]:
        correct_suc = int(np.sum(y & (q >= high)))
        correct_fail = int(np.sum(~y & (q < low)))
        result[f'endpoint_{low}_{high}'] = dict(
            accuracy=ratio(correct_suc+correct_fail, len(ids)),
            suc_accuracy=ratio(correct_suc, n_suc), fail_accuracy=ratio(correct_fail, n_fail))
    return result


def summarize(rows, labels, ids, head):
    indexed = index_rows(rows)
    if set(indexed) - set(labels):
        raise ValueError('Prediction IDs are not in the frozen labels')
    scores = {}
    for key, row in indexed.items():
        if row.get('status') != 'ok':
            continue
        value = row.get('success_probability') if head == 'success' else canonicalize(row)['progress']
        if value is None or not math.isfinite(value):
            continue
        if head == 'success':
            if not 0 <= value <= 1:
                raise ValueError('Success probability outside [0,1]')
            if row.get('success_trajectory', [])[-1:] != [value]:
                raise ValueError('Success probability is not the saved terminal value')
        scores[key] = float(value)
    ids = set(ids)
    report = dict(overall=metrics(scores, labels, ids), by_task={})
    for task in sorted({labels[i]['subset'] for i in ids}):
        report['by_task'][task] = metrics(scores, labels, [i for i in ids if labels[i]['subset'] == task])
    report['task_macro_accuracy'] = float(np.mean([v['accuracy'] for v in report['by_task'].values()]))
    gaps = []; binary_gaps = []; expected_pairs = 0
    for key in sorted(ids):
        label = labels[key]; source = label['source_suc_id']
        if label['split'] != 'fail' or source not in ids:
            continue
        if labels[source]['reward'] != 5 or labels[source]['video_sha256'] != label['video_sha256']:
            raise ValueError('Invalid same-video success/failure pair')
        expected_pairs += 1
        if key in scores and source in scores:
            gaps.append(scores[source]-scores[key])
            binary_gaps.append(int(scores[source] > .5)-int(scores[key] > .5))
    report['pairwise'] = dict(expected_pairs=expected_pairs, valid_pairs=len(gaps),
        positive=sum(v > 0 for v in gaps), tied=sum(v == 0 for v in gaps), inverted=sum(v < 0 for v in gaps),
        mean_score_gap=float(np.mean(gaps)) if gaps else None,
        binary_gap_counts={str(v):binary_gaps.count(v) for v in [-1, 0, 1]})
    return report


def flat_metrics(row):
    return {key:value for key,value in row.items() if not isinstance(value, dict)}


def write_csv(path, records):
    with path.open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)


def render(report):
    pct = lambda x: f'{100*x:.2f}%'
    lines = ['## Material Passport', '', '- Origin Skill: experiment-agent', '- Origin Mode: validate',
             '- Origin Date: '+report['created_at'], '- Verification Status: ANALYZED',
             '- Version Label: meter_success_head_v1', '', '# Robometer success head 补充评价', '',
             '仅离线统计既有输出；未重跑模型、未调阈值、未修改历史结果。统计核验不等于重新验证神经前向。', '',
             '## 口径', '',
             '- 使用最后一个 `<|prog_token|>` 上已训练 success head 的 sigmoid 概率 q；与原先 progress head 的末步位置一致。q 是成功概率，不能解释为完成进度。',
             '- 主分类规则沿用官方示例：q > 0.5 为成功，q ≤ 0.5 为失败。>=0.5 敏感性另存 JSON。两套端点准确率仍为失败 q < low、成功 q ≥ high，中间值均不正确。',
             '- 全量 baseline 1213 条（407 suc / 806 fail），永远判 fail 的准确率为66.45%；846子集为268/578，永远判 fail 为68.32%。报告 balanced accuracy 与 AUROC，避免被类别比例误导。',
             '- 无效记录保留并计入准确率分母，AUROC、MAE、Brier仅在有效值上计算。success 的 MAE=mean(|q−1[suc]|)，Brier=mean((q−1[suc])²)。progress只作为同阈值代理分数对照，不能把其分数误差称为成功概率校准。',
             '- full与grounded子集分别比较；不把同视频的多条指令视为独立证据。本次不作显著性检验或独立泛化宣称。', '',
             '## Success head baseline：1213条', '',
             '| 输入 | 有效 | 总准确率 | suc | fail | 平衡准确率 | AUROC | MAE(q) | Brier |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for protocol in PROTOCOLS:
        m = report['experiments'][protocol]['baseline_full']['success']['overall']
        lines.append(f"| {NAMES[protocol]} | {m['n_valid']}/{m['n_expected']} | {pct(m['accuracy'])} | {pct(m['suc_accuracy'])} | {pct(m['fail_accuracy'])} | {pct(m['balanced_accuracy'])} | {m['auroc']:.4f} | {m['score_mae']:.4f} | {m['brier']:.4f} |")
    lines += ['', '## 同一0.5阈值：progress → success', '',
              '| 输入 | 总准确率 | suc准确率 | fail准确率 | AUROC |', '| --- | ---: | ---: | ---: | ---: |']
    for protocol in PROTOCOLS:
        row = report['experiments'][protocol]['baseline_full']; p=row['progress']['overall']; s=row['success']['overall']
        vals = [pct(p[k])+' → '+pct(s[k]) for k in ['accuracy', 'suc_accuracy', 'fail_accuracy']]
        lines.append('| '+NAMES[protocol]+' | '+' | '.join(vals)+f" | {p['auroc']:.4f} → {s['auroc']:.4f} |")
    lines += ['', '## Success head 的原端点口径：1213条', '',
              '| 输入 | 总准确率 .125/.875 | suc | fail | 总准确率 .2/.8 | suc | fail |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for protocol in PROTOCOLS:
        m=report['experiments'][protocol]['baseline_full']['success']['overall']
        vals=[pct(m[f'endpoint_{low}_{high}'][k]) for low,high in [(.125,.875),(.2,.8)] for k in ['accuracy','suc_accuracy','fail_accuracy']]
        lines.append('| '+NAMES[protocol]+' | '+' | '.join(vals)+' |')
    lines += ['', '## 原attention作用于success head：同846条', '',
              '原attention的head ranking和k保持原实验设置。此处只是读取同次前向的success输出，未重新按success监督选头。', '',
              '| 输入 | 条件 | 有效 | 总准确率 | suc | fail | AUROC |', '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for protocol in PROTOCOLS:
        for condition in ['baseline', 'target_k8', 'target_k32', 'target_k64']:
            m=report['experiments'][protocol]['attention'][condition]['overall']
            lines.append(f"| {NAMES[protocol]} | {condition} | {m['n_valid']}/{m['n_expected']} | {pct(m['accuracy'])} | {pct(m['suc_accuracy'])} | {pct(m['fail_accuracy'])} | {m['auroc']:.4f} |")
    lines += ['', 'wrong/low控制、逐task统计、概率分布、同视频指令配对以及阈值边界敏感性见 `metrics.json`。target和baseline全部有效；wrong控制的原始失败保留。', '',
              '## 统计核验（11/11）', '',
              '| 检查 | 处理与边界 |', '| --- | --- |',
              '| Simpson悖论 | 保留逐task与macro结果；不同分组的增益可能不同，不以总体替代各组表现。 |',
              '| 生态谬误 | 不从输入均值推断每条指令表现，另存同视频配对。 |',
              '| Berkson选择偏差 | 1213与grounded846分开；后者不代表未筛选总体。 |',
              '| Collider偏差 | 不用正确性筛样；grounding筛选的外推边界仍存在。 |',
              '| 基率忽略 | 明列407/806、常数fail基线、两类准确率、平衡准确率。 |',
              '| 均值回归 | 无按极值选择的新实验，不将旧数据重读视为独立重复。 |',
              '| 幸存者偏差 | 无效值计入准确率分母，配对/概率指标同时列有效数。 |',
              '| 多重搜索 | 展示全部五输入和三个target k；不报告挑选后的显著性。 |',
              '| 分析自由度 | 这是用户提出后的补充描述，主阈值取官方示例；不拟合阈值。 |',
              '| 相关与因果 | 不从AUROC或配对顺序宣称因果理解或跨数据有效。 |',
              '| 反向因果 | 仅核对固定预测与既有标签；不作因果方向推断。 |', '',
              '## 来源与复现', '',
              '官方来源：本仓库引用副本的 `robometer/models/rbm.py`（progress token上独立success head）、`robometer/evals/eval_server.py`（sigmoid）及 `scripts/example_inference_local.py`（默认0.5，严格大于）。输入、配置、源码与官方来源SHA-256均在 `metrics.json`。', '',
              '使用新目录复现统计：', '', '```bash',
              'cd /mnt/public1/dais/workspace/Robo-Dopamine-addbase',
              '/home/dais/miniconda3/envs/robo-dopamine/bin/python -m mydata_bench.meter_eval.success_metrics --output results/mydata_bench/experiments_v2_addbase/session_20260908/meter_success_head_recheck_v1',
              '```', '']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=pathlib.Path, default=BASE)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args=parser.parse_args(); base=args.base.resolve(); output=args.output.resolve()
    if not base.is_relative_to(ROOT) or not output.is_relative_to(ROOT) or output.exists():
        raise ValueError('Inputs must be in this repository and output must be a new directory in it')
    labels=index_rows(read_rows(base/'inputs/labels.jsonl'))
    full=index_rows(read_rows(base/'inputs/full.jsonl')); cohort=index_rows(read_rows(base/'inputs/cohort.jsonl'))
    if set(labels)!=set(full) or not set(cohort)<=set(full) or len(full)!=1213 or len(cohort)!=846:
        raise ValueError('Expected complete frozen full1213 and grounded846 inputs')
    for row in labels.values():
        if (row['split'],row['reward']) not in [('suc',5),('fail',1)]:
            raise ValueError('Expected binary suc/fail labels only')
    paths=[base/'inputs'/name for name in ['labels.jsonl','full.jsonl','cohort.jsonl']]
    paths += [pathlib.Path(__file__),ROOT/'mydata_bench/meter_eval/model.py',ROOT/'mydata_bench/meter_eval/runtime.py',ROOT/'mydata_bench/meter_eval/readout.py']
    paths += [REFERENCE/name for name in ['robometer/models/rbm.py','robometer/evals/eval_server.py','scripts/example_inference_local.py']]
    for protocol in PROTOCOLS:
        paths += [base/f'meter_{protocol}_v1'/name for name in ['baseline.jsonl','attention.jsonl','config.json']]
    before={str(p.relative_to(ROOT)):digest(p) for p in paths}
    report=dict(created_at=datetime.now(timezone.utc).isoformat(),threshold=.5,positive_rule='q > 0.5',
                selection='Descriptive re-scoring of existing outputs; no fitted thresholds, no neural rerun',
                success_readout='saved terminal sigmoid probability, no reconstruction from progress',
                source_sha256=before,experiments={})
    baseline_csv=[];attention_csv=[]
    expected_conditions={'baseline'}|{f'{kind}_k{k}' for kind in ['target','wrong','low_rank'] for k in [8,32,64]}
    for protocol in PROTOCOLS:
        folder=base/f'meter_{protocol}_v1';rows=read_rows(folder/'baseline.jsonl');by=index_rows(rows)
        if set(by)!=set(full):raise ValueError('Incomplete baseline rows: '+protocol)
        result=dict(baseline_full={},baseline_cohort={},attention={})
        for head in ['success','progress']:
            result['baseline_full'][head]=summarize(rows,labels,full,head)
            result['baseline_cohort'][head]=summarize(rows,labels,cohort,head)
            baseline_csv.append(dict(protocol=protocol,head=head,**flat_metrics(result['baseline_full'][head]['overall'])))
        grouped=collections.defaultdict(list)
        for row in read_rows(folder/'attention.jsonl'):grouped[row['condition']].append(row)
        if set(grouped)!=expected_conditions:raise ValueError('Unexpected attention conditions')
        for condition,values in sorted(grouped.items()):
            if set(index_rows(values))!=set(cohort):raise ValueError('Missing attention rows')
            result['attention'][condition]=summarize(values,labels,cohort,'success')
            result['attention'][condition]['raw_status_counts']=dict(collections.Counter(r['status'] for r in values))
            attention_csv.append(dict(protocol=protocol,condition=condition,**flat_metrics(result['attention'][condition]['overall'])))
        result['same_cohort_baseline_success_exact']=all(r.get('success_probability')==by[r['example_id']].get('success_probability') for r in grouped['baseline'])
        if not result['same_cohort_baseline_success_exact']:raise ValueError('Baseline readout differs between full and attention run')
        report['experiments'][protocol]=result
    if before!={str(p.relative_to(ROOT)):digest(p) for p in paths}:raise ValueError('Inputs changed during analysis')
    output.mkdir(parents=True,exist_ok=False)
    with (output/'metrics.json').open('x') as handle:
        json.dump(report,handle,ensure_ascii=False,indent=2,allow_nan=False);handle.write('\n')
    write_csv(output/'baseline.csv',baseline_csv);write_csv(output/'attention.csv',attention_csv)
    with (output/'report.md').open('x') as handle:handle.write(render(report))
    print(output)
    for protocol in PROTOCOLS:print(protocol,report['experiments'][protocol]['baseline_full']['success']['overall'])


if __name__=='__main__':
    main()
