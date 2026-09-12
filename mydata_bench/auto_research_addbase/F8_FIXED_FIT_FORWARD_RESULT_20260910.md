# F8固定幅度完整拟合诊断结果

全部1942条、446个episode组完成，无错误；1253条有ROI执行F8全query干预，689条真实调用冻结fallback。每条样本都重算baseline，1942条logits与F7原始诊断逐值一致。全部95项冻结依赖复核未变。

|分支|episode等权有序损失|episode等权CE|样本等权准确率%|样本等权MAE|episode等权准确率%|episode等权MAE|
|---|---:|---:|---:|---:|---:|---:|
|baseline|1.355004|3.464337|38.5170|1.042739|37.6674|1.148299|
|F7|1.213009|3.007990|38.3625|1.113800|41.2769|1.128377|
|F8|1.209026|3.022251|37.7961|1.111226|40.0014|1.128752|

F8有序损失比F7略低，但没有解决样本等权评分退化：相对baseline，准确率−0.7209个百分点，MAE+0.068486；F7对应−0.1545个百分点、+0.071061。不能由训练有序损失更低宣称评分方法更好。两者的episode等权准确率／MAE却改善，说明聚合权重会改变这项训练诊断的方向。

这仍不能证明episode权重就是退化原因：两个损失都与argmax准确率／MAE不同，有限b1也偏离零点一阶近似。两组头在同一F8有序目标上的一阶方向导数分别为F7−0.598466、F8−0.615150；真实episode有序损失变化分别约−0.141994、−0.145978。不能把局部导数当作有限干预收益。

全部1–5等级的准确率、MAE、CE和有序损失均保存在completion_v1.json，未只报grade5改善。此训练诊断没有运行验证／保留推理，不是预定新选型门槛，也不改变任何F7/F8参数。原846的+10pp及独立最终确认仍待完整流程。

证据R/f8_fit_finite_step_diagnostic_v1/qwen_text_video/records_v1.jsonl、completion_v1.json和上一级completion_provenance_audit_v1.json。原始F7结果没有改写。目标3未完成。

R=`results/mydata_bench/experiments_v2_corssmodel/auto_research/session_20260909_robust_v1`。
