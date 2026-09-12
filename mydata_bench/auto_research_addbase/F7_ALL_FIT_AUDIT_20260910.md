# F7全部拟合与零干预审计

2026-09-10 12:51（上海），三个模型各五输入、共15项F7拟合全部结束，无拟合错误；每项1942条、446个episode组。完整审计使用全部F6/F7保存的拟合记录，15项的零干预原生logits及CE均逐值相等。每项1253条有ROI的梯度因干预query范围改变；689条无ROI仍以零梯度纳入，未排除。

训练跨输入诊断：896个可选头中，Qwen五输入同号252个、最弱输入加变异惩罚后仍为正的111个；Meter分别338、135；RoboReward分别241、58。RR的58个不足覆盖此前讨论的k64。因此不能把“共同方向头”当成已有三模型统一k32/48/64解法。此诊断不改变任何排名或推理配置，也不提供有限干预、准确率或泛化保证。

冻结的F7评估已在GPU0/3启动，先工程再493与旧234；所有15项完整输出后才运行该轮分析。846条件队列和保留确认条件队列继续等待原门槛。当前只证明拟合完成和数值归因，未证明目标3成功。

证据：R/f7_all_fit_queues_completion_v1.json、R/f7_complete_fit_mechanism_audit_v1.json；源码审计observer PID2573705已正常终止。R为results/mydata_bench/experiments_v2_corssmodel/auto_research/session_20260909_robust_v1。
