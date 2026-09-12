# 主线要求与证据清单（2026-09-09 15:05）

本文件用于最后逐项验收。当前明确**未完成全部目标**；完成状态必须由结果文件、实际进程和对应统计核实，不能从计划或工程通过推断。

|要求|当前证据|状态/剩余动作|
|---|---|---|
|只在Robo-Dopamine-addbase工作、增量改动、保护旧数据|common.py限制新结果输出路径，launch保存cwd/config/source SHA；源预测与失败版本保留；未对已有仓库执行git|持续约束；最终检查新增产物路径与有效性注释|
|新增Meter与SOLE的五输入native baseline|BASE/metrics_snapshot_20260909T124540.json及主summary完整表；10个case各1213次尝试|完成；SOLE text_video保留9条格式错误，Meter typed敏感性分列|
|新增模型的官方输入、prompt、递归/训练heads与token对齐|meter_eval/top_eval，官方公开reference clone、processor/zero/recurrence审计，mydata_bench/check.md对应说明|已核验；官方结构与greedy/processor/parser偏差已披露|
|两个新增模型五输入完整attention及原方法/区域/低heads/zero控制|Meter五输入846×10全部尝试；SOLE image_text/text_image/text_video完整七步846×11全部尝试|未完成：SOLE interleaved/video_text完整七步仍运行|
|SOLE终步分解|五输入846×11全部尝试，attention_audit_snapshot_20260909T133441.json|补充已完成；不能代替上项完整七步|
|MAE、两阈值准确率、suc/fail、每task/分布、same-video配对|metrics.py的完整snapshot与paired_analysis；原始score/trace保留|已完成部分数据齐全；待剩余mandatory/确认结束后生成最终完整快照|
|具体top8及跨模型top8/32/64重合|主summary的20行top8表，BASE/all_four_models_head_overlap_v1.json|完成；仅head坐标重合，不宣称功能等价|
|主线2文献调研与原因分析|LITERATURE_AND_HYPOTHESES_20260908.md、references/source_verification.jsonl及根auto-explore追加|已完成主要调研；PAI/PASTA链接纠正，错误不相关检索排除，来源与机制边界均说明|
|提出、实现、复验attention改良|视觉守恒、对称区域、readout-query、时间差分、小半径、native调节、KL预算；每轮prospective_plan/config/source/SHA和负结果|已有完整开发证据；不能据此写最终达标|
|至少三模型各三输入，同参相邻k四项改善且总+10pp|candidate_inventory_20260909T144254.json；confirmation_candidate_selection_20260909T144256.json；真实--validate-only通过9case|仅开发准入通过，确认仍未启动|
|禁止端点hard coding/按样本真值选推理参数|全部native bins保留；推理与labels离线分析分离；区域/低heads/温度/同强度对照；selected_development_distributions_20260909T1500_v1.json|目前九个primary仍预测30–133条中间档；分布只是补充证据，不能单独证明无泄漏|
|独立确认、固定参数与不确定性|confirmation_analysis_protocol_v1.json、freeze_confirmation.py、analyze_confirmation.py；20000视频簇/IUT/Holm与缺失规则已确定|待已启动RR小半径234复验结束后一次性冻结全部case、运行498确认；完整报告失败与边界|
|最终文档|mydata_bench/exp_plan_addbase_summary.md、根auto-explore的最终有效方案位置|持续追加中；须在主线1/3完成后写自洽的最终结论、限制与复现实验命令|

BASE=`results/mydata_bench/experiments_v2_addbase/session_20260908`。
RESEARCH=`results/mydata_bench/experiments_v2_corssmodel/auto_research/session_20260908`。

额外审计待办：SOLE text_image数值读出初筛曾短暂双写，110个已审计重复键的科学字段完全相同，冲突0；完成后必须对全文件再审计。旧错误百分号token版本、static-cache/CUDA-graph分岔工程、原生格式错误均保持排除或覆盖率注释，不可因后续结果改善而删除。

确认通过后，仍需给出最终冻结设置在完整可grounding cohort上的描述性汇总；独立确认与包含开发/ranking重叠视频的描述性结果不能混为一谈。未grounding的原始样本不擅自当作方法已覆盖。


2026-09-09 15:33状态追加：九项498确认已一次性冻结并启动；完整846的remaining114也已冻结并串接调度。准确路径、冻结设置及等待/完成后审计清单见上述主summary最新段落；原表的“确认尚未启动”为15:05历史快照，不代表当前状态。全部主线仍未完成。


### 2026-09-09 15:34：SOLE text_image最终重复审计完成

完整文件审计`sole_score_development_v2/sole_text_image/completed_score_audit_20260909T153343.json`：原始2450神经支路行包含2340唯一键，全部有效；110重复键科学字段冲突0；sweep为15000唯一键、全部有效、无重复。此前运行中审计的待办已由最终文件审计关闭，没有删除原始重复记录，也没有把重复行当独立样本。observed/static两组zero各60/60的logits与progress精确一致。

60条baseline的201规范格式总概率最小0.9999082，中位0.9999990，均>.99；最大1.00000018保留为浮点累加误差而未裁剪。联合MAP与自身native greedy有一条差异（suc/ljx_lfz_task_1_1/20：83%对85%）。这是读出差异，不能归因于attention。剩余video_text/interleaved/text_video仍需完成screen；text_image完成不意味着SOLE新方法达标。


### 2026-09-09 16:23：全部Meter邻居的五档MAE补查

`confirmation_v1/development_all_neighbor_five_bin_mae_audit_v1.json`仅检查开发234，确认性能未读取。Meter三个输入×三个k的五档MAE均相对两套baseline下降：image_text约−1.1410/−1.1111/−1.0769；text_video约−0.6838/−0.6752/−0.6453；video_text约−0.7350/−0.7607/−0.6880。原连续ordinal MAE改善与此相容。最终确认/846总结需逐邻居同时核对并给出两种MAE，不能只写primary或只挑较好口径。既有确认门槛/参数不改；任何更严格的全邻居五档检查须与原冻结point gate分列。


## 2026-09-09 17:01：SOLE五输入数值screen与最终审计全部完成

`sole_score_development_v2/analysis_snapshot_20260909T165904.json`覆盖同一60条/24视频簇上的全部五种输入。每输入39个神经支路与250个总读出均完成（104个完整target候选），**没有任何target候选通过四向/+10pp及同k原方法改良门槛**。下表报告每输入“最大总准确率增益”的单个实际候选，连同其余指标，避免拼接不同点。

|输入|该点MAE变化|总准确率变化|suc变化|fail变化|完整通过点数/target点数|
|---|---:|---:|---:|---:|---:|
|sole_image_text|-0.085333|1.6667pp|0.0000pp|2.5000pp|0/104|
|sole_interleaved|0.046667|10.0000pp|0.0000pp|15.0000pp|0/104|
|sole_text_image|0.868667|10.0000pp|15.0000pp|7.5000pp|0/104|
|sole_text_video|0.613333|33.3333pp|-20.0000pp|60.0000pp|0/104|
|sole_video_text|-0.002667|43.3333pp|-15.0000pp|72.5000pp|0/104|

text_video最大总增益+33.3333pp时MAE反而+0.613333、suc−20pp；video_text最大总增益+43.3333pp时suc−15pp，MAE仅−0.002667。因此不能以总准确率上升掩盖其他验收失败。这些是固定自身前六步与末步reasoning的201规范数字读出实验，不能推广为所有可能的SOLE全推理干预都无效。

最后两个输入审计：video_text的`completed_score_audit_20260909T165829.json`与text_video的`completed_score_audit_20260909T165602.json`均为2340/2340唯一有效支路、15000/15000唯一有效读出，无重复或科学冲突；observed/static zero各60/60精确。baseline规范格式总概率最小分别0.99988976/0.99987147；前者一条native/MAP差异（fail/ljx_lfz_task_5_1/10：7%/8%），后者无差异。五输入全部baseline的规范格式概率均>.99，不存在旧错误百分号token版本的低格式概率问题。最大概率约1+2.12e−7的浮点累加误差保留原值。

三项此前已完成的审计和110个text_image重复键处理继续有效：重复不计作独立样本，无原数据删除。第4模型在本轮数值对比家族中未取得稳定改良，应作为方法边界保留；三模型×三输入目标仍由冻结的Meter/Qwen/RR九项确认检验。

GPU2的SOLE text_video与GPU0的SOLE video_text数值任务及补充均正常结束，新Qwen text_video/video_text确认按预存资源计划自动启动。**主线1的SOLE interleaved/video_text完整七步attention还在运行**，数值screen完成不能替代这两项mandatory实验。


## 2026-09-09 19:36：共享熵方案通过观测门槛，独立性限制不变

完整结果见`SHARED_ENTROPY_RESULT_20260909T1932.md`，唯一最终观测方案为`RESEARCH/shared_entropy_evaluation_v1/frozen_manifest_v1.json`。开发234、重复498、全846均有全部9项×3个k通过两baseline四向/+10pp、same-k原方法及primary开发预选原方法检查；另逐个summary用严格算术复核全部81点，见`direct_final_acceptance_audit_v1.json`。不再选择参数。

原v1独立确认8/9通过仍保持False；anchor v1及entropy单独选型v1的失败均保留。共享β是看到entropy v1失败后提出的开发规则，虽参数仍按234选取，重复498与full846也不能被称新独立确认。无新确认性p；原IUT/Holm不能转移给最终方案。11类偏差处理继续适用，尤其多重搜索、选择偏差、微/宏口径、总体与任务异质性、错误对照可行子集、关联与因果区分。

新whole controls全部9项target/zero精确，但RR image_text相对wrong的suc略低，RR text_video相对wrong的总/fail更低，不能宣称所有控制指标支配。完整846的最小suc改善2.2388pp、重复498为1.8987pp，需如实披露余量；114分区仍有失败且未剔除。原模型分布保留中间档，标签未进入推理/新评分生成，熵只改变全bin证据的标量强度。

两个SOLE mandatory全七步仍运行；全量baseline最终表/重复科学记录审计尚待完成。整体goal不能提前标complete。


## 2026-09-09 20:16：主验收与第二阈值的范围

最终共享熵方案的已通过结论明确对应预定主指标：连续Meter采用0.125/0.875五档端点阈值，Qwen/RR采用原生1–5输出。另按计划完整统计0.2/0.8；Meter text_video在此第二阈值下成功类准确率下降（primary全846为58.9552%→44.0299%，重复498为56.3291%→44.9367%），其总/fail仍提高。因此不能称两套阈值所有指标均改善，也不能声称阈值无关的稳定性。两baseline指原生生成和同读出baseline，与两阈值是不同概念。全部第二阈值邻居结果存于`shared_entropy_evaluation_v1/secondary_threshold_descriptive_audit_v1.json`，没有据此重选参数或改写主门槛。

173项最终冻结来源核对已通过，主线1的两项SOLE全七步继续推进；整体goal仍未完成。


## 2026-09-09 21:07：最终逐项关闭状态

|计划要求|最终状态与证据|
|---|---|
|目标目录、旧结果保护、增量实现|全程Robo-Dopamine-addbase；正式/失败数据保留，冻结来源173项通过，未作本地git操作|
|Meter/SOLE五输入native|10项×1213尝试，SOLE text_video9格式失败保留|
|五输入ranking与具体heads|10项ranking完成，四模型20组top8与30组top8/32/64交集已表列|
|Meter五输入原attention|5×846×10全部尝试|
|SOLE五输入完整七步attention|5×846×11全部尝试，最后两项于20:55/20:58完成|
|SOLE终步补充|5×846×11全部尝试；没有替代完整七步|
|check.md中的输入、query/key、ranking、scope、wrong与递归|实现与数值证据见最终总报告；官方结构和解码/processor偏差明示|
|MAE/双阈值/suc/fail/task/分布/pairwise|最终metrics+180 macro；完整15840行指标和180配对CSV；最终方法1980 task/split CSV|
|调研/原因分析|15篇文献矩阵及指标原定义核验完成，准确区分机制背景与本轮提案|
|理论—实现—迭代与负结果|原确认、anchor、单独熵、共享熵结果各自冻结保留，控制/消融完备|
|三模型各三输入、已测k邻域主指标验收|开发/复用498/full846全部9×3通过；完整846最小总+14.1844pp|
|端点hard coding禁令|全原生bin、无标签评分生成、公式/zero/来源/直接算术与置换核验；没有端点映射|
|最终文档|FINAL_RESEARCH_REPORT_20260909.md、主summary完整baseline最终段、根计划最终状态与复现指南均已写入|

不附加虚假的完成结论：新的独立泛化、两套阈值所有类均提升、SOLE有效、所有任务与排序指标改善均未成立。主线验收范围是预定主指标的本数据观测结果；限制不是被删除的结果。机器完成清单为`RESEARCH/final_completion_audit_v1.json`。
