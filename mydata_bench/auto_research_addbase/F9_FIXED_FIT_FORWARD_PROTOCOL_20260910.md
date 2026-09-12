# F9固定训练前向诊断

## Material Passport
时间：2026-09-10T14:53:10.701624+08:00；固定训练机制诊断；无F9效果结论。

沿用之前明确选择的Qwen text_video，全部1942条／446组，以固定b1、k48比较baseline／F7／F8／F9。F7和F8的完整前向从已核验的F8诊断记录引用；F9对每条样本重新运行真实baseline和候选，要求baseline logits逐值复现已保存结果。有ROI核验全query钩子和原F8拟合定位对齐；无ROI实际调用冻结fallback并保留在全部分母。

同时报告CE、有序损失、原生准确率和MAE的样本均值、episode均值及全部1–5等级。三个方法的一阶方向都在同一F9样本均值有序梯度上计算。没有搜索幅度、k、读出或输出放大，也不根据诊断修改已冻结的F9评价门槛。

新源码f9_fit_finite_step_diagnostic.py、单独YAML、114份源码哈希及内容快照已冻结在R/f9_fit_finite_step_diagnostic_v1/frozen_plan_v1.json。启动PID2631687、GPU3、可用显存77384MiB；主F7/F8/F9队列继续。CPU聚合检查覆盖不等大小episode、无ROI完整分母、缺失／重复／失败不得生成完整均值；合成缺等级使用None避免NaN。所有训练诊断失败保留，不静默重试。

原训练梯度头对比已完整记录于R/f9_complete_fit_head_comparison_v1.json，理论边界见FINITE_INTERVENTION_OBJECTIVE_GAP_NOTE_20260910.md。当前不读任何F7/F8部分验证结果或保留测试表现；目标3未完成。

R=`results/mydata_bench/experiments_v2_corssmodel/auto_research/session_20260909_robust_v1`。
