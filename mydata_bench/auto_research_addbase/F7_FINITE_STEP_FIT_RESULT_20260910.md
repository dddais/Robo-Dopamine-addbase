# F7固定幅度的完整拟合集结果

两项均使用完整1942条fit、446个episode组，全部基线真实前向，1253条有ROI、689条按冻结方法回退；无运行错误。**这是参与选头的数据，不能作为验证或最终成功证据。**

|模型/输入|组平均CE基线|F7全query|同头readout-only|F7拟合准确率变化pp|F7拟合ordinal MAE变化|
|---|---:|---:|---:|---:|---:|
|qwen_text_video|3.464337|3.007990|3.386218|-0.1545|+0.071061|
|meter_text_image|1.874768|1.732075|1.866428|+5.6128|-0.080309|

固定b1/k48下，两个输入的真实拟合CE均下降，且全query优于相同F7头与方向的readout-only控制。但Qwen的MAE反而上升，准确率略降；Meter的两项读出指标改善。准确率/MAE是看到fit CE后补充的描述性诊断，不是预定新选型门槛，不产生独立p值或新的通过声明。

逐等级CE也存在权衡：Qwen grade2的平均CE增加约0.698，grade5减少约1.167；Meter grade3/4分别增加约0.198/0.216。不能用总体CE下降声称每等级均改善。

一阶训练CE方向导数与有限b1差别明显：Qwen全query预测约−1.352、实际组平均变化−0.456；Meter预测−0.228、实际−0.143。相同头readout-only的对应预测/实际分别约−0.103/−0.078和−0.0105/−0.00834。梯度只是零点局部信息。

原F7参数、评价、完整846条件队列和保留确认门槛保持不变。后续可以单独研究与有序等级更相关的训练损失，但不得据此更改F7，或将新方法在复用数据上的改善叫作独立确认。

原始配对logits、CE、组均值、等级/来源分布、两套端点阈值和全部中间档位的拟合诊断见R/f7_fit_finite_bias_diagnostic_v1/analysis_v1.json；逐输入/标签/排名文件摘要复核见analysis_provenance_audit_v1.json。

R=`results/mydata_bench/experiments_v2_corssmodel/auto_research/session_20260909_robust_v1`。目标3尚未完成。
