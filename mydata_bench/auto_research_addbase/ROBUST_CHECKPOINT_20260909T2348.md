# 目标3续研检查点：2026-09-09 23:48（Asia/Shanghai）

目标仍为active，**尚未完成**。本轮属于实际progress：实现两种新候选、通过数值与真实前向检查、启动并完成部分全234实验、取得会改变后续方向的负结果、固定并准备外部验证数据。不能把旧共享熵结果重新计作本目标完成，也不要调用complete。

## 唯一仓库与执行约束

`/home/dais/workspace/Robo-Dopamine-addbase`（物理路径`/mnt/public1/dais/workspace/Robo-Dopamine-addbase`）。exec显式此cwd、login=false。禁止git操作；不读其它本地代码仓库；只新增，旧数据/代码版本/结果/失败日志保留。Python `/home/dais/miniconda3/envs/robo-dopamine/bin/python`。SAM3用rewardbench-sam3环境。无子代理；用户已授权auto research，不需再次问许可。

适用skill academic-research-suite已读取，experiment-agent内联执行。用户要求自主实现/迭代优先于skill中需要逐命令确认的默认建议。当前不需要重新读取整套skill。

## 当前协议与产物

- [前瞻研究协议](ROBUST_RESEARCH_PROTOCOL_20260909.md)定义抗过拟合边界、F1/F2和对照。
- 新结果根简称R：`results/mydata_bench/experiments_v2_corssmodel/auto_research/session_20260909_robust_v1`。
- 旧846全视为开发/描述数据，不能再切成独立测试。旧234用于候选开发。任何外部测试性能目前均未计算，仍需在最后候选冻结后一次性使用。
- 目标范围保留三模型各三输入、相邻已测k稳定、MAE/suc/fail改善和总+10pp、原attention改良。Meter两套阈值都报告，本轮协议要求两套成功/失败方向不反转。

## F1：逐帧概率搬运（没有分数后处理）

固定每帧一半非目标attention质量移至本帧已有目标条件分布，保持每帧视觉量及非视觉概率。所有模型/输入a=.5，k32/48/64、primary48、all_frames/all queries；原无标签raw_mass ranking。代码`frame_transport.py`、`robust_runner.py`；freeze `robust_prepare.py`。配置`addbase_robust_frame_transport_v1_*.yaml`。

3项有意义的CPU单元测试通过，真实工程每case2视频先检查zero logits/score精确、因果和逐帧质量守恒，过关才运行234×11条件。条件含native/readout baseline、原attention三k、F1三k、zero/wrong/low。

截至23:47:46，8/15case完成：Meter text_video；Qwen text_video/video_text；RR全部五输入。最后RR text_image尚未分析。已分析其它7项均失败，全部zero精确。

primary k48相对同读出baseline的代表性负结果：

|case|MAE差|总准确率差pp|suc差pp|fail差pp|
|---|---:|---:|---:|---:|
|meter text_video|-.29414|+.4274|-2.6316|+1.8987|
|qwen text_video|-.19658|-.8547|-5.2632|+1.2658|
|qwen video_text|-.04274|-3.4188|-2.6316|-3.7975|
|rr image_text|-.12393|+2.5641|0|+3.7975|
|rr interleaved|-.48718|+13.6752|-3.9474|+22.1519|
|rr text_video|-.41026|+1.2821|+3.9474|0|
|rr video_text|+.10684|-2.1368|+3.9474|-5.0633|

这说明守恒/可解释并不保证有效；不要因为MAE下降或总+10pp就宣布成功。

F1队列GPU0 PID2223138、GPU3 PID2223141已完成且completion failures为空；**不要重启**。GPU1 PID2223139和GPU2 PID2223140在23:47:46 `/proc/cmdline`核验仍活跃。分别继续Qwen剩余、Meter剩余输入。每个case分engineering/development目录，`completion_v1.json`含预测SHA与错误数。wrong几何不可行错误保留，不影响主样本分母。

## F2：F1 + 固定VCD机制

公式`2*z_F1(clean)-z_F1(corrupted)`，全部原生bins。固定权重1、diffusion_step500、sigmoid beta1000步；由视频SHA/协议/pixel字段确定seed，不用instruction或label确定noise。没有熵、旧case-specific λ/ρ/β、static参考或词表plausibility截断。

代码`frame_transport_vcd.py`、`frame_transport_vcd_runner.py`、`freeze_frame_transport_vcd.py`、`frame_transport_vcd_queue.py`。F2运行时明确替换共享runner的Runtime类，不修改旧runner源文件。复用F1所有clean原始分支，核验completion/config/source SHA；预定两条工程样本实际重算clean，要求逐logit/score精确一致，再新增noisy分支。F2每case234×15条件，额外VCD-only和原attention+同VCD三k。工程whole-zero同时a0和VCD权重0。

官方VCD源码固定commit `d6568ff81b8fd306a49e630df44f2db5c2300191`（默认分支master，main返回404，已查清）。源码和README存R/references_v1，来源SHA独立JSON；论文正文核验α1及ablation noise500。Cawley & Talbot JMLR2010选型偏差页面亦保存。2项F2数值测试通过。

F2四个耐久队列均在23:47:46核验活跃：GPU0 2235605；GPU1 2235606；GPU2 2235607；GPU3 2235608。每个先等相同GPU的F1队列终止，检查前驱真实/proc存活，不因观察超时重启。

F2 RR image_text已经完成并分析，whole-zero精确，**失败**：k48 MAE+.09829，总+2.1368pp、suc+5.2632pp、fail+.6329pp。不能只看两类准确率上升忽略MAE变坏。F2 RR interleaved已进入工程/后续开发；其它队列按空闲显存运行/等候。不要因首项失败改变已冻结F2参数，先完成整套，之后在旧开发数据上继续有机制依据的研究。

## 外部验证准备（未打开性能）

R/external_roboreward_reserved_v1：官方teetone/RoboReward test，revision `469b9af76e8539e8d2ac553b081307738fd92ca5`，2831条/23源子集，全1–5等级，license CC-BY4.0。全部样本按文件名hash顺序下载，没有按标签筛选。元数据SHA `3901cea9a7981d2e2d9fde4225a4b2ff0ebf2f4af69fc96972a2cfbe557db0cd`。

`robust_external_data.py`准备进程PID2229426仍活跃；23:45时2500/2831视频下载完成、0失败。随后自动按视频字节SHA合并重复解码8帧，保留首尾/时间戳，生成`inputs_without_labels_v1.jsonl`和`labels_sealed_v1.jsonl`，查旧1213的字节/8帧像素精确重叠。封存是本轮流程约束，不是由独立第三方或权限系统实施的盲法。推理字段剥离reward及gpt5_mini_check，媒体用opaque文件名；没有打印任何逐例测试标签或测试性能。

近重复审计v1脚本因多一个括号在启动前SyntaxError，PID2237906已终止，旧源码/日志保留。已新增修复版`robust_external_overlap_v2.py`，编译通过，PID2238216在23:47:46实际活跃，等待数据准备真实前驱。按首/中/末63bit DCT hash三帧均Hamming≤4标记潜在重叠；这不证明身份一致。输出v2 plan/audit和`independent_ids_before_grounding_v2.json`。独立子集只含已成功准备并通过重叠审计的样本，全部2831仍需另报、缺失不隐藏。稀疏审计不能排除所有同源/训练污染。

**尚未完成**：外部目标grounding/追踪、独立episode簇统计定义、最终候选/全部参数冻结和独立测试。勿提前使用外部标签帮助选候选。旧剩余322条/108视频虽无旧cohort/ranking重叠，却有baseline暴露且无grounding，不能当完全未见数据。

外部grounding可复用目标仓库`grounding/parser.py`的InstructionParser（/home/dais/workspace/model/Qwen3-4B-Instruct-2507）、`grounding/sam3.py`的SAM3Grounder（/home/dais/workspace/model/sam3，threshold.3、mask.5、top_n20）及base.py选择/track方法；不要直接运行旧pipeline写回其它仓库输出。它尚未实现/排队，须新增输出和明确无grounding时baseline回退或coverage统计，不按成功与否筛样本。样本来自多种任务，不能声称外部SAM3机器结果经过人工验证。

## 下一步与检查命令

1. 先查真实/proc、各queue event与completion、各stdout最近进度；当前process snapshot为R/checkpoint_20260909T1548_v1.json，不能仅依该历史snapshot判存活。
2. 对新增完成case做`python -u -m mydata_bench.auto_research_addbase.robust_analyze`及`--family f2`；工具默认完整234，不分析未完成case。报告和旧original开发最优参考都只新增；`old_original_development_reference_v1.json`已冻结11case的更强原方法比较，来源全是旧开发，0科学字段冲突。F2还要求改善VCD-only和same-k VCD-original，防止误归因。
3. 等外部数据及重叠审计完成，检查0漏/重复、完整expected与所有失败、SHA；继续实现无标签grounding和最终验证流程，但暂不开性能。
4. 根据完整开发证据继续提出新机制，限制选型自由度、保存全部失败。目标目前没有达标候选；不能结束为完成、改成较容易的指标或把控制失败删除。最终仍需旧指定数据范围达标及新独立证据。

此处的运行命令中python均指上述robo-dopamine绝对路径。全部实验都还在已授权范围，不问许可、不调用子代理。
