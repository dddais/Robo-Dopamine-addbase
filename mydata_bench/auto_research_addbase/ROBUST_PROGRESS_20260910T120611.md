# 稳健研究进度 2026-09-10T12:06:11.641206+08:00

目标3未达成。F7已完成12/15项拟合，暂无拟合失败；当前剩余RR的text_image、image_text、interleaved。原评价、完整846条件评价和保留确认条件队列均保持冻结。保留test claim仍不存在，尚无F7完整效果结论。

本轮补齐了两个方面的工作。

第一，核验了两篇DRO文献的原文相关章节，并记录共享选头代理的局部性质和局限。训练最差组损失好，并不能保证测试最差组表现；更不能由零点梯度推出固定幅度1下的MAE或准确率提升。详情见[TRAINING_GRADIENT_ROBUSTNESS_NOTE_20260910.md](TRAINING_GRADIENT_ROBUSTNESS_NOTE_20260910.md)。下载/正文提取/摘要记录在R/shared_head_literature_verification_v1，未上传本地数据。Qwen五输入有111个正的最差输入惩罚后head；Meter五输入有135个。这些只用于训练诊断，未选择新推理排名或F8。

第二，在空闲GPU2/3启动了Qwen text_video和Meter官方text_image的完整1942条fit有限幅度诊断。固定k48、b1，三种条件为实际typed原生baseline、F7全query、相同F7头/方向只干预评分query。每条都实际运行baseline；689条无ROI按已冻结helper回退，全部保留。有ROI分支核对实际tracking alignment与拟合记录一致。只读fit标签，不读取493/旧234/保留测试效果，也不做参数搜索或修改F7。

拟合集的组均值统计保留真实分母、无ROI零差和不等组大小；错误不静默删除，出现必需错误就不报告完整均值。一个CPU统计边界测试通过，实际运行已产生成功记录。结果尚未完成，不能预报损失变化方向。当前状态：

```json
{
  "qwen_text_video": {
    "pid": 2577439,
    "live": true,
    "records": 1279,
    "statuses": {
      "ok": 828,
      "no_grounding": 451
    },
    "baseline_cache_mismatches": 0,
    "complete": false
  },
  "meter_text_image": {
    "pid": 2577442,
    "live": true,
    "records": 802,
    "statuses": {
      "ok": 526,
      "no_grounding": 276
    },
    "baseline_cache_mismatches": 0,
    "complete": false
  }
}
```

训练诊断源码`f7_fit_finite_bias_diagnostic.py`、2份配置和完整73份依赖已冻结在R/f7_fit_finite_bias_diagnostic_v1。完整15项拟合完成后的独立CPU审计器PID2573705也已启动，届时检查全部F6/F7零前向及三模型跨协议梯度；该审计器不启动推理、不改排名。此前11项已完成fit的逐例零前向核对均精确，每项1253条实际梯度。

R=`results/mydata_bench/experiments_v2_corssmodel/auto_research/session_20260909_robust_v1`。仍按原数据集、多模型/多输入、两阈值和强original门槛验收，不用训练诊断替代实际效果。
