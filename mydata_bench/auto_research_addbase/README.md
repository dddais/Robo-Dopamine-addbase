# Addbase 实验入口

优先阅读两份简短总结：[baseline 结果](../exp_plan_addbase_baseline_summary.md)、[方法探索结果](../exp_plan_addbase_method_summary.md)。目前没有满足三模型各三输入目标并通过独立确认的方法。

放宽幅度和输入数量要求后的单独记录：[区域差分法](../method_region_contrast.md)、[其他候选方法](../method_alternative_candidates.md)。前者有三模型498条结果及SOLE 60条初筛信号，后者保留F2/F6/F9等局部有效路线。

| 内容 | 入口 |
| --- | --- |
| 新增 baseline 与原 attention | `run.py`、`queue.py`、`run_sole_attention_queue.py`；SOLE 批处理在 `../top_eval/` |
| 正式新增模型配置 | [v2_crossmodel_addbase](../configs/v2_crossmodel_addbase/)：20 个正式配置 |
| Robometer success head | `../meter_eval/success_metrics.py`：从保存输出补充二分类统计；结果见[baseline总结第2.1节](../exp_plan_addbase_baseline_summary.md) |
| 跨模型与最新方法配置 | [v2_crossmodel](../configs/v2_crossmodel/)：16 个原 baseline/attention 配置、F7–F9 各 45 个 fit/development/validation 配置、4 个训练诊断配置 |
| 最新方法实现 | `all_query_reward_gradient_*`（F7）、`ordinal_reward_gradient_*`（F8）、`row_weighted_reward_gradient_*` / `fit_row_weighted_reward_gradient_heads.py`（F9） |
| 完整结果与诊断 | [F7–F9 结果](F7_F8_F9_COMPLETE_RESULTS_20260912.md)、[F9 训练诊断](F9_FIXED_FIT_FORWARD_RESULT_20260912.md) |
| 实验口径与有效性 | [续研协议](ROBUST_RESEARCH_PROTOCOL_20260909.md)、[旧方案有效性核查](METHOD_VALIDITY_AUDIT_20260910.md) |

2026-09-12 清理了 298 个历史/试跑 YAML、105 个旧探索 Python、20 个重复进度文档，均已按原路径压缩归档。[归档清单与恢复方法](../../archives/exploration_cleanup_20260912/README.md)。数据、预测、失败记录和结果目录未改动。

保留源码与配置均未改写。部分早期模块属于正式入口的依赖，或被 F7–F9 的 `source_sha256` 固定校验，因此仍在原路径；有效性核查引用的旧方法源码、正文链接到的 6 份进度记录也保留。

重新运行早期路线、旧冻结/恢复流程或历史复现文档中的旧命令前，先按归档说明恢复文件。既有配置指向历史结果目录；新实验应另建配置和输出目录。
