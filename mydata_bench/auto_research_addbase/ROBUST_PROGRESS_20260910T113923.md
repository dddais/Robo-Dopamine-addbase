# 稳健研究进度 2026-09-10T11:39:23.040094+08:00

目标3未完成，F7尚在拟合阶段，全部评价设置保持冻结。当前已完成8项拟合：Qwen五输入和Meter text_video/video_text/text_image，每项1942条、1253条实际梯度、689条无ROI零贡献，无拟合错误。GPU0继续RR text_video，GPU3继续Meter image_text；三个后续条件队列正常等待。

最终确认流程已补齐并冻结：`R/f7_confirmation_policy_v1.json`，SHA `7fd403ac8bf8f251c35ca83944e6d750ad2e8be75e599a53d691b2ed14ab7611`。72份源码依赖保留内容快照，旧64份评价源码和4份full846源码未改。新增分析入口在关联任何测试标签前，检查配置的模型/输入唯一、路径唯一、预选case全部覆盖且输出目录不重复；3项CPU测试通过。固定k32/48/64、primary48、b1，所有实际选择仍必须在测试前冻结。

条件确认队列PID2571353只在前置旧234+493联合门槛及完整原846门槛均满足时运行保留测试，否则记录不符合准入并退出。完整原846要求至少三模型各三输入、两baseline下MAE下降、suc/fail上升、总+10pp、强original改良和两套Meter阈值，未放宽。保留测试检查全1–5等级、所有预选case和2831条完整分母，20000次1172整组配对bootstrap、primary IUT/Holm；它不能替代原数据集目标。保留claim仍不存在，无保留模型评分/标签关联。

本轮新增训练数据层面的证据：

1. [f7_completed_fit_zero_parity_audit_2026-09-10T033137.503889_0000.json](/mnt/public1/dais/workspace/Robo-Dopamine-addbase/results/mydata_bench/experiments_v2_corssmodel/auto_research/session_20260909_robust_v1/f7_completed_fit_zero_parity_audit_2026-09-10T033137.503889_0000.json)核对当时已完成7项，每项1253条实际梯度：F6/F7零bias的native logits及损失逐条精确一致；梯度逐条发生变化。该结果排除这些已审计拟合前向的数值变化，不证明F7有限幅度的效果。后来完成的Qwen text_image不在这份7项审计中。
2. [f7_fit_task_binding_coverage_audit_2026-09-10T033324.967133_0000.json](/mnt/public1/dais/workspace/Robo-Dopamine-addbase/results/mydata_bench/experiments_v2_corssmodel/auto_research/session_20260909_robust_v1/f7_fit_task_binding_coverage_audit_2026-09-10T033324.967133_0000.json)只读取1942条fit标签：1/2/3/4/5分别559/352/347/310/374条；同视频300组包含不同指令和不同评分，共908条，其中185组同时有1和5。因此拟合数据并非只有动作本身等级变化，也有任务区分监督；不据此保证迁移到原数据集。
3. [f7_completed_fit_directional_derivatives_2026-09-10T033423.973191_0000.json](/mnt/public1/dais/workspace/Robo-Dopamine-addbase/results/mydata_bench/experiments_v2_corssmodel/auto_research/session_20260909_robust_v1/f7_completed_fit_directional_derivatives_2026-09-10T033423.973191_0000.json)按8项已有排序计算零点方向导数，F7的训练CE下降方向通常大于F6，但两者用各自不同选头，不能作为匹配头的因果对照，也不能当作真实MAE/准确率提升。
4. [f7_qwen_fit_cross_protocol_diagnostic_2026-09-10T033617.065938_0000.json](/mnt/public1/dais/workspace/Robo-Dopamine-addbase/results/mydata_bench/experiments_v2_corssmodel/auto_research/session_20260909_robust_v1/f7_qwen_fit_cross_protocol_diagnostic_2026-09-10T033617.065938_0000.json)检查Qwen五输入，896个head中252个在所有输入的梯度均值同号，111个在最差输入下仍通过固定1.96SE波动惩罚。各输入前48头的两两重合仅6–22，梯度余弦约0.214–0.686。可作为以后研究共享选头的训练线索；没有生成新推理排名、没有修改F7、没有打开F7验证表现。

学术解释和完整性原则仍见METHOD_INTEGRITY_EXPLANATION_20260910T111808.md。493是复用选型数据，旧846是暴露开发/描述数据，保留2831尚未评分。本地封存不等于第三方盲测，模型预训练污染未排除。
