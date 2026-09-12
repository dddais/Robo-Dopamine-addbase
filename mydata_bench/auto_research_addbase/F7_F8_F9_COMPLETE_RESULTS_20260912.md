# F7／F8／F9完整结果：均未满足目标3

## Material Passport

复核时间：2026-09-12T22:53:26.289760+08:00。全部实验于2026-09-10结束；本次核验完整输出和冻结来源。状态：完整阶段性分析，目标3未完成。目标目录Robo-Dopamine-addbase。

## 明确结论

三种方案均完成三模型×五输入的全部15项旧234／493评价，总计45项候选、90份完整分区预测。三种方案的原234稳定门槛通过输入数均为Qwen1、RoboReward0、Meter0；唯一通过的是Qwen video_text。与493的全等级验证联合后，三种方案的通过数均为0。它们均未进入完整846或最终2831确认，不能宣布目标3完成。

|方案|训练方式|旧234通过数（Qwen／RR／Meter）|493通过数（Qwen／RR／Meter）|同输入联合通过数|
|---|---|---|---|---|
|F7|原生类别CE梯度选头|1/0/0|0/3/0|0/0/0|
|F8|原生累计有序损失梯度选头|1/0/0|0/2/0|0/0/0|
|F9|F8每样本梯度，按样本等权聚合并使用episode簇标准误|1/0/0|0/4/0|0/0/0|

这里的“通过”包含预先固定的k32/48/64邻域、两个baseline、same-k及较强原attention对照；Meter检查两套阈值。493是复用的模型选择数据，不能称新独立确认。原数据集要求每模型至少三个输入稳定提升，不能用一个输入或混合等级数据上的改善替代。

## 有改善，但范围和比较对象受限

Qwen video_text在原234上表现最一致：F7／F8／F9的primary相对同读出baseline总准确率分别提高20.51／18.38／19.23个百分点，MAE分别下降0.2350／0.2564／0.3291。它们通过该输入的原234完整门槛，但Qwen各方案在493上的完整通过输入数都为0，因此不支持跨数据泛化成功。

RoboReward interleaved的F7 primary相对baseline总准确率提高27.78个百分点，仍未稳定超过冻结的较强原attention对照，故完整门槛失败。Meter text_video的F7在一套阈值下提高23.08个百分点，但另一套阈值下成功类下降13.16个百分点；F9对此有所改善，primary相对两套baseline阈值均通过，但仍未满足完整原attention／k邻域门槛。不能挑较好阈值或只报baseline比较。

## 全部primary描述性结果

以下每表均报告所有15项，精确门槛包含三个k及所有强对照；表中primary只便于阅读，不替代完整门槛。Meter的MAE变化使用连续ordinal MAE，准确率展示0.125／0.875阈值；另一套阈值完整保存在来源JSON。

### F7

|模型／输入|旧234 primary MAE变化|旧234准确率变化pp|493 primary MAE变化|493准确率变化pp|旧234完整通过|493完整通过|
|---|---:|---:|---:|---:|---|---|
|meter_image_text|-0.061469|+0.0000|-0.032076|-0.2028|False|False|
|meter_interleaved|-0.338194|+1.2821|-0.029758|+3.2454|False|False|
|meter_text_image|-0.436219|+12.3932|-0.037879|+2.4341|False|False|
|meter_text_video|-0.376537|+23.0769|+0.012606|+3.0426|False|False|
|meter_video_text|-0.036556|+0.0000|-0.011507|+0.2028|False|False|
|qwen_image_text|-0.209402|+14.1026|+0.077079|-1.6227|False|False|
|qwen_interleaved|-0.038462|+0.0000|-0.034483|+1.4199|False|False|
|qwen_text_image|-0.115385|+15.8120|+0.070994|+2.4341|False|False|
|qwen_text_video|-0.111111|+11.9658|+0.101420|+1.0142|False|False|
|qwen_video_text|-0.235043|+20.5128|+0.060852|-0.6085|True|False|
|rr_image_text|+0.034188|+0.0000|-0.026369|+3.4483|False|True|
|rr_interleaved|-0.457265|+27.7778|-0.012170|+2.4341|False|True|
|rr_text_image|-0.196581|+19.6581|+0.022312|+1.8256|False|False|
|rr_text_video|-0.273504|+0.0000|-0.012170|+1.4199|False|True|
|rr_video_text|+0.000000|+0.8547|+0.002028|-0.2028|False|False|

原234来源：R/f7_frame_transport_analysis_2026-09-10T065941.905827_0000.json，SHA 7b9da5056c0e6a029186485aab47f8f920bb0b8b599fe3c1304bba3c023f6fc7。
493来源：R/f7_training_validation_analysis_2026-09-10T070021.619146_0000.json，SHA 94a804545ddd51d59e39250c32f75f59e6a5e02688c4b01538cd84f7121748b1。

### F8

|模型／输入|旧234 primary MAE变化|旧234准确率变化pp|493 primary MAE变化|493准确率变化pp|旧234完整通过|493完整通过|
|---|---:|---:|---:|---:|---|---|
|meter_image_text|-0.100583|+0.0000|-0.052272|+0.0000|False|False|
|meter_interleaved|-0.341807|-1.7094|-0.040780|+2.6369|False|False|
|meter_text_image|-0.354479|+0.4274|-0.053365|+1.8256|False|False|
|meter_text_video|-0.353209|+19.6581|+0.009335|+2.4341|False|False|
|meter_video_text|-0.089721|+0.0000|-0.021798|+0.0000|False|False|
|qwen_image_text|-0.205128|+16.2393|+0.089249|-2.2312|False|False|
|qwen_interleaved|-0.076923|+2.9915|-0.044625|+1.6227|False|False|
|qwen_text_image|-0.094017|+13.2479|+0.046653|+2.4341|False|False|
|qwen_text_video|-0.111111|+9.8291|+0.070994|+1.8256|False|False|
|qwen_video_text|-0.256410|+18.3761|+0.042596|-0.2028|True|False|
|rr_image_text|-0.004274|+4.2735|-0.038540|+3.0426|False|False|
|rr_interleaved|-0.192308|+13.6752|-0.002028|+0.6085|False|False|
|rr_text_image|+0.064103|+5.1282|+0.010142|+1.0142|False|False|
|rr_text_video|-0.235043|-0.4274|-0.026369|+1.6227|False|True|
|rr_video_text|+0.000000|-0.4274|-0.010142|+1.2170|False|True|

原234来源：R/f8_frame_transport_analysis_2026-09-10T082815.042314_0000.json，SHA 9bc87a976a534fa43c8843fda16e9f61c509d36a4db35038e62371f01d97eaaa。
493来源：R/f8_training_validation_analysis_2026-09-10T082902.029046_0000.json，SHA 4bf434c03b878986aa00a6b37db11cf7c73fdb5f84e9aa48b4371fb2c643f255。

### F9

|模型／输入|旧234 primary MAE变化|旧234准确率变化pp|493 primary MAE变化|493准确率变化pp|旧234完整通过|493完整通过|
|---|---:|---:|---:|---:|---|---|
|meter_image_text|-0.107720|+0.0000|-0.053106|+0.6085|False|False|
|meter_interleaved|-0.325261|-1.7094|-0.062572|+2.2312|False|False|
|meter_text_image|-0.273434|-1.2821|-0.067093|+1.8256|False|False|
|meter_text_video|-0.345648|+13.2479|+0.001291|+3.6511|False|False|
|meter_video_text|-0.102131|+0.0000|-0.025270|+0.4057|False|False|
|qwen_image_text|-0.247863|+11.9658|+0.012170|+0.6085|False|False|
|qwen_interleaved|-0.115385|+0.8547|-0.054767|+2.0284|False|False|
|qwen_text_image|-0.162393|+8.1197|-0.012170|+3.4483|False|False|
|qwen_text_video|-0.132479|+7.2650|+0.036511|+2.4341|False|False|
|qwen_video_text|-0.329060|+19.2308|+0.038540|+0.8114|True|False|
|rr_image_text|-0.017094|+2.1368|-0.034483|+4.0568|False|False|
|rr_interleaved|-0.264957|+14.5299|-0.026369|+1.8256|False|True|
|rr_text_image|+0.034188|+7.2650|-0.016227|+2.6369|False|True|
|rr_text_video|-0.307692|+0.4274|-0.010142|+1.4199|False|True|
|rr_video_text|-0.106838|+0.4274|-0.006085|+0.8114|False|True|

原234来源：R/f9_frame_transport_analysis_2026-09-10T085752.793941_0000.json，SHA 18581423e58851a76b2ada6fc58160b8d483a199702475cb5d388c9fc92a01f4。
493来源：R/f9_training_validation_analysis_2026-09-10T085847.017281_0000.json，SHA 6ccf0b0c0161e0a309aff06393c1b0537e7df9fcd7eba2284fbe422da71b948a。

## 训练诊断解释了什么

F9在Qwen text_video的全部1942条训练数据上，把样本等权有序损失从1.173694降到1.084314，但原生准确率从38.5170%降至37.8476%，MAE从1.042739升到1.102987。F9相对F8的评分退化略小，仍未恢复baseline水平。这说明单纯换成有序损失、修正聚合权重并不足够；不能只凭训练损失变小宣称评分方法有效。该诊断只针对一个完整训练case，不能直接推广到所有模型。

F9全部1942条真实baseline精确复现缓存F8诊断，1253条实际ROI干预、689条实际fallback，0错误。全部四分支logits、损失和汇总已CPU复核。证据R/f9_fit_finite_step_diagnostic_v1/completion_provenance_audit_v1.json。理论限制见FINITE_INTERVENTION_OBJECTIVE_GAP_NOTE_20260910.md。

## 完整性、失败及数据边界

本次重新检查三种方案的冻结评价源码、90份配置、预测文件SHA、每样本×每条件的唯一完整网格，全部吻合；记录R/f7_f8_f9_complete_outcome_provenance_audit_20260912_v1.json。进程正常退出不等于每个评分条件成功：F7 Qwen interleaved的两个原生回答格式失败保留，不能重新解析成有效答案；全部条件错误和错位区域不可构造情况保留于原始输出和分析。

旧234／846已暴露，493已复用于模型选型。保留2831测试的共享claim仍不存在，三个方案因不合格均未打开测试奖励表现。失败后的方案没有用该测试调整参数，因此没有重复旧确认复用的问题；这仍不能保证新方法永不过拟合或排除公开数据的预训练污染。

三种方案使用真实训练标签选头／方向，属于有监督方法。它们保留原生读出，没有输出端点赋值；没有找到满足全部目标的最终方法，也不能以失败较少或某一漂亮数字宣布成功。

## 后续方向

当前结果否定了“只换零点损失梯度排序就足够”的假设。下一步需检验更直接的任务绑定和有限干预学习：例如保留所有等级的绝对监督，同时引入同视频不同指令的真实等级差约束，或在独立拟合分区学习受约束的有限bias。具体方案必须先核验梯度与实际评分的关系、单独冻结，再开展完整评价。此处是后续研究方向，尚未启动新的候选实验，不更改任何F7/F8/F9设置。

R=`results/mydata_bench/experiments_v2_corssmodel/auto_research/session_20260909_robust_v1`。
