# 研究进度 2026-09-10T13:28:02.865479+08:00

目标3仍未完成。原范围、至少三模型各三输入、共同k邻域、MAE/suc/fail与总+10pp要求没有降低。旧entropy/anchor结果存在确认复用风险，不能作为新的稳健成功证据。

F7：全部15项拟合完成，无拟合错误；全部零干预logits与CE逐值等于F6。GPU0/3的冻结评价进行中，已完成旧234：['qwen_image_text', 'qwen_interleaved', 'qwen_text_video', 'qwen_video_text']；493：['qwen_image_text', 'qwen_interleaved', 'qwen_text_video', 'qwen_video_text']。完整15项之前不运行本轮奖励分析。Qwen interleaved493原生生成基线有两条只输出5、违反ANSWER格式，按冻结规则保留失败；错位区域不可构造单独计数，不据此制造对照增益。

F8：三模型真实工程全部通过，完成拟合8/15：['qwen_image_text', 'qwen_interleaved', 'qwen_text_image', 'qwen_text_video', 'qwen_video_text', 'rr_text_image', 'rr_text_video', 'rr_video_text']。只改累计有序log loss，原生读出、b1、k32/48/64不变。两个完整Qwen拟合的1253有ROI原生logits与F7逐值一致，CPU重算损失最大误差分别5.722e-6、4.768e-6。83源码的评价／完整846条件方案已经冻结并启动等待；全量训练CPU审计observer也在等待。

保留2831性能尚未打开（claim存在=False）。F7有原优先权；F8没有安排保留推理，不能在F7打开后将同一集叫作新确认。两组20000次bootstrap和Holm不能消除跨轮选择偏差。

新增来源说明、训练对照结构诊断、F7/F8方法说明均已保存。训练集中确有300个同视频不同指令／等级组，支持未来配对假设，但未实现新候选，未改变当前冻结方法。
