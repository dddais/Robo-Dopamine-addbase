# Auto research 执行日志（2026-09-08 启动）

## 授权与边界

- 唯一实现目录：`/home/dais/workspace/Robo-Dopamine-addbase`（物理路径 `/mnt/public1/dais/workspace/Robo-Dopamine-addbase`）。
- 执行 `exp_plan_addbase_auto_explore.md`；其中引用的 `mydata_bench/exp_plan_addbase.md` 不存在，使用仓库根目录现有 `exp_plan_addbase.md`。
- 只新增代码/配置/结果；原始数据、既有结果只读；不对本地仓库运行 git 命令。已有 `io.provenance()` 内含 git rev-parse，新增执行器不得调用它。
- 官方代码通过网络取得并保存在目标目录内，不读取其他本地代码库。授权数据集、grounding/cohort 和模型权重为只读输入。
- 用户已经授权无人值守实验与改进迭代，无需重复询问执行许可；涉及必须另行确认的工作跳过。
- baseline 优先使用 GPU 0–2 空余资源，探索可用 GPU 0–3；不干预他人进程。

## 预先固定的验收标准

1. 两个新增模型：官方输入 baseline，以及五类输入的 baseline/attention，top-k=8/32/64、wrong-region 与 low-rank-head 对照；完整汇总至 `mydata_bench/exp_plan_addbase_summary.md`。
2. 文献来源逐一核验，记录官方实现、机制与适用边界；未核实不能充作证据。
3. 改进方法至少在四模型中的三模型、每模型至少三输入协议上同时 MAE 下降、suc 与 fail 准确率提高、总准确率提高至少 **10 个百分点**（同时报告相对增幅以覆盖原文歧义）。不通过标签、ID、文件名、endpoint hard coding 或标签驱动后处理生成分数。
4. 无缺失的同 cohort 比较，invalid 不隐藏；按视频簇切分，ranking/development/confirmation 的泄漏与选择偏差明确披露。原始全 cohort 复现与独立确认分开报告。
5. 连续模型既报告 1+4p 连续 MAE，也报告五级离散映射 MAE；阈值 0.125/0.875 与 0.2/0.8 分别报告。保存 per-task、split 分布、同视频配对差及 head 重合度。

## 初始观察

- 本目录尚无 results；新增两个模型执行器尚不存在。
- 权重已具备。Robometer 是带自定义 progress/success/preference heads 的 Qwen3-VL-4B，不能当普通文本生成模型加载后丢弃 heads。
- SOLE-R1 是 Qwen3-VL-8B；官方契约含首帧、前帧、当前帧及上一时刻预测进度，不能直接当普通 RoboReward 1–5 输出。
- 旧结果存在强 bias 导致整体分数下移、suc 受损、解析失败后条件级联缺失及 ranking/评测重叠等问题，新增 runner 必须逐条件隔离错误。

## 当前状态

三个主线均进行中；尚无新的数值结果，尚未声称验收成功。

## 2026-09-08 22:24 前后：执行进展

- 原始数据转换为只读引用 + 新增帧缓存；407 个真实视频内容 hash、1213 条全量、846 cohort、34 ranking。快照 `inputs/manifest.json` 和切分 `inputs/splits.json` 已冻结。
- 14 篇计划内文献核实标题/摘要；主要机制论文正文已读取。额外 VCD 核实；一次无关数学文献检索明确排除。研究报告已写入独立文档。
- 新增 Robometer loader 严格加载所有参数和训练 token；官方 baseline 1213 条已完成，无无效条目，结果见 summary。
- SOLE 官方 prompt 与 montage 两个纯函数从新 clone 的官方代码经 AST 筛选导入；未安装新依赖，不导入完整 vLLM 程序。原先字符串系统消息在 processor 中报错，新增 runtime 改为等价 text content list；失败与修复 smoke 文件各自保留。
- 4 个 attention 数值不变量测试通过。ranking 需要同时把 language model 的共享 text config 设为 eager 才能保证模型生成因果 mask，仅更改 attention child 会导致 SDPA elide mask，此隐患已在正式 ranking 前修复。
- Qwen/RR 首次 smoke 的 video 总像素预算误按单帧设置，4 个 temporal patch 合计仅 16 visual tokens。该 smoke 留作调试记录、不能作为正式结果；正式开发设置为 `8*50176` 的总视频预算，保持单帧空间预算与 image 输入相当。
- 视觉守恒算法仅在该新方法中使用 float32 additive mask，以降低 bfloat16 校正量舍入误差；原 bias 分支不变。
- 初筛计划已在任何新方法开发结果产生前冻结，SHA-256（canonical-json 函数定义）`1c87d6a43df954304f782469a15a9cefaa87be765c1a11efbb56a63f8324b0e4`，60 条 / 24 视频组，28 条件，Qwen/RR/Meter × 5 输入。确认集498条未用于参数选择。
- 当前长作业：GPU0 SOLE五种输入批量baseline队列；GPU1 Meter其余四输入baseline→ranking→attention队列；GPU2 Meter官方ranking→attention队列；GPU3 三模型五输入development初筛队列。只监控自己的进程，不触碰原有其他任务。
- 工作仍未完成，goal保持active；不能因作业已排队就认为全部验收完成。

## 2026-09-08 22:35 前后：GPU backend 修复与补跑

- 开发初筛首例 Qwen text→video 完成 1680 次条件尝试，但其中 1080 个守恒条件因 CUDA cuDNN backend 拒绝 float32 additive mask / bfloat16 query 而失败；另有 600 个 baseline/original/target-only 有效输出。下一例 video→text 记录830次尝试后，主动停止本研究队列 PID1950756 与其子进程PID1951445；先核验 `/proc/<pid>/cwd` 为本目标目录，再仅终止自己的开发进程。其它 GPU 作业未停止。
- 保留所有失败行与日志，不把零 exit code 当作有效实验完成。修复：守恒分支显式使用 PyTorch math SDPA，支持 float32 mask 与 bfloat16 query；原 bias 和 target-only 分支保持原 backend。
- 修复后 CUDA 单/批量注意力均有限值；完整 Qwen text→video、image→text 生成验证通过，原模型、bias=0、守恒bias=0 均返回相同分数，守恒 b=1.5 正常运行。真实视频192 visual tokens、图像384 visual tokens，确认视频空间预算修正生效。
- CPU 测试增加真实微型 Qwen3-VL text model 的 ranking 因果性测试，以及逐样本区域不同的 batch/single 等价测试；共6项通过。
- 用 `--retry-errors` 补跑已有失败/未完成项，原有有效行不覆盖，源代码哈希随每次 launch 保存；初筛样本、条件、选择规则和冻结配置 SHA 均不改动。

## 2026-09-08 22:40–22:45：官方读出与零干预审计

- Robometer 官方 `eval_server.py:576` 明确先 `.detach().cpu().float()` 再对十个 progress bins 作 softmax 加权。初版 runner 的保存字段 `progress` 在 bf16 中计算；其原始 `native_progress_logits` 已完整保存。因此新增 `meter_eval/readout.py` 在离线统计中从原始 logits 重构官方 CPU float32 读出，绝不修改原始 JSONL。全量官方 baseline 最大进度差0.0044543，3个离散分档改变，总准确率仍31.57%；主指标从此以该官方重构为准，旧快照保留但已被精度审计更正。
- 官方两阶段 Qwen processor 路径与当前 direct processor 的 token IDs、image grids、1472-token 序列完全一致；像素张量平均绝对差6.84e-6，最大0.007843（少量 resize 舍入差），已保存 `meter_official_processor_audit_v1.json`。这是官方输入结构复现，非所有像素逐 bit 复现，最终文档需披露。
- SOLE batch attention pilot v1 的8个结果（2样本×4条件）均有效，但无干预与显式zero-bias干预的递归进度不一致（0.31 vs0.25；0.04 vs0.25）。这说明长CoT会放大后端数值差异，不能把该pilot的非零改变量直接当成attention的因果效应。
- 正在进行独立pilot v2：基线与所有干预条件统一为显式causal mask + PyTorch math SDPA，并加入 bias=0、preserve=0 的完整7步一致性检查。SOLE正式attention尚未启动，必须先通过这一检查。官方标准baseline仍独立按默认backend保留。
- 开发队列第一轮补跑发现wrapper CLI虽接收retry-errors却没有传给子进程，导致早期失败项被跳过。已停止自己的wrapper/child、修正参数传递并重启；有效结果不覆盖，真正的补跑已出现新增守恒条件成功记录。此为执行故障，不作为效果失败判断。

## 2026-09-08 23时后：完整Meter结果、SOLE优化和后续候选

- Meter五输入baseline、ranking已完成。text_image和text_video的846×10原attention条件全部尝试完毕；仅wrong-region各7/10条几何不可行，原始target与low-head无缺失。已追加汇总、全部分布和具体top8；其它3输入attention继续。
- Qwen video_text完成60×28初筛，preserve_last b3 k64点估计MAE−0.2、总准确率+11.67pp、suc+20pp、fail+7.5pp；冻结234条扩展计划，与text_video一样补k24/40/48、零/区域/低head对照。确认集未触碰。
- 新增paired_analysis按完整视频内容簇bootstrap及cluster sign-flip，而非按指令独立重采样。Qwen text_video初筛k32的总准确率+15pp，95%簇bootstrap区间[6.35,25]pp；fail+5pp的区间含0。只属筛选后的探索统计，不能算独立确认。
- SOLE同时间panel、同面积、无重叠wrong-region映射已通过合成几何测试。队列现已把SOLE attention路由至batch_attention，避免未经backend修正的旧singleton路径。
- 全math batch64工程pilot速度偏慢，保留运行记录；新增native_residual可选引擎：原生SDPA输出加上选中heads上的FP32概率差乘V。实数算术与steering等价；低精度残差加法是明确版本差异。CPU dense对照≤1e−6，零差与未选中heads精确一致。正在独立8条、五条件、七步推理pilot；非零bias不会冒称与全math逐bit相同。未据任何标签来选工程引擎。

## 2026-09-08 23:24–23:32：首个全开发集结论与数值对照

- Qwen text→video全234条开发集完成：baseline MAE1.4573，总准确率14.957%。preserve last b3 k24/32/40/48/64分别总准确率+6.84/+7.69/+8.97/+6.84/+4.27个百分点，均MAE下降、两类准确率提高，但**均未达到10pp门槛**。不能把60条筛选最大值当作确认。簇bootstrap和全部结果保存于development_full_v1/qwen_text_video。
- 同一全开发集，旧math替换引擎的zero_preserve k32/k64分别使5/3条原始回答变化，其中k32一条suc从5变3。这说明短回答也存在backend数值混杂；旧候选非零结果不能全归因于区域干预。此前仅2条GPU零对照的通过不代表全cohort数值等价。本轮新增native_residual保留原生输出并仅加选中heads的概率差×V，零差精确为0，以解决该已观测问题。
- native_residual工程检查在Qwen/RR/Meter各2条上全部7支路成功，零bias/零preserve的所有评分logits与baseline逐项精确一致，生成式模型共同ANSWER格式前缀读出的argmax亦与native greedy一致；Meter保持官方十bin期望。原始branch与工程审计永久保存score_engineering_v1，未读取这些样本标签作工程选择。
- 第二候选score_development_v1已在运行前冻结：同60条/24视频簇，三模型×五输入，21个神经分支、53个读出；对称正负视觉守恒b3，k32/64，all/last scope，λ=.5/1/2/4及wrong/low controls。所有1–5或十bin都保留，不做端点映射捷径。接受前必须同时胜过同读出baseline、native generation baseline和原强bias，并扩展234条，再冻结确认方案。
- Qwen image→text初筛在all/last b3 k32/64均出现同时正向+10pp。另冻结234条native_residual扩展、k24/32/40/48/64及23条件，在GPU3与原始初筛队列共用空余显存。Qwen video→text旧引擎全开发集在GPU2继续，用零对照量化其数值限制。
- SOLE稀疏引擎8条×五条件pilot完成，两种零条件全部七步轨迹及raw reasoning与native相同；64条真实top64压力pilot进行中。旧全math batch64中baseline/zero共128条成功、全部相同，随后仅停止自己的工程PID1955966以释放资源，未完成preserve工程分支保留、不进入科学结果。此停止不涉及其它用户任务。

## 2026-09-09 00时后：双对照读出与SOLE完整执行队列

- SOLE真实top64的全七步pilot v2完成64×5=320条，baseline、zero-bias、zero-preserve、original-bias、preserve全部有效；零bias与零preserve的64条全轨迹和raw reasoning一致。32条终步pilot的6条件共192条全部有效，前六步均逐项继承自身native baseline，最后一步两种零干预也精确相同。几何wrong-region在这32条上全部可构造。source snapshots记录各实现版本。
- 优化native_residual静态区域缓存：不再对每个新decode长度重复逐样本CUDA索引赋值；区域张量仅CPU构建一次、传GPU一次，随后补零。张量内容/公式不变，既有数值测试通过。
- `sole_attention_schedules_plan_v1.json`冻结五输入×两种schedule×11条件。**终步对照是补充，不能代替完整七步干预**。GPU1负责image_text/interleaved，GPU2负责text_image/video_text，GPU0负责text_video；等待各自native baseline与ranking，使用≥50GB空余显存门槛。GPU2等待score_development队列结束；GPU0等待其剩余native baseline队列结束。所有主线仍未完成。
- SOLE原生首输入继续完成原batch16作业；仅停止旧scheduler PID1950219（未停止正在运行的baseline子进程1950224），剩余四输入在首输入结束后以batch64启动。各输入的batch size和每次launch明确记录；attention对照自己重算相同batch的baseline，不混减不同算术batch的输出。
- 新增统计回归检查：整组样本均匀复制不改变video-cluster bootstrap区间；零效应区间为0且p=1；缺失样本不能通过完整cohort门槛；Holm校正单调。均通过。
- 核对官方`robometer/models/utils.py:18`发现其convert_bins_to_continuous以整序列sum==1猜测是否已为概率；FP32原始baseline text_video有2条误触发（logits含负值），其它多帧构造还有若干“仅最后一行sum==1但整序列不为概率”。旧baseline统计继续忠实保留官方整序列转换；不改原始结果。
- Counterfactual合成向量明确是logits，必须softmax，不能被偶然sum==1重解释成概率。新score分支保存全序列logits并标记typed softmax；native_generation_baseline仍使用官方读出。接受前需胜过两者，排除读出修复的收益冒充attention收益。此修正在Meter正式score开发运行前完成，两个工程样本不受影响；构造sum==1且含负值的回归测试已验证。
- “确认集未触碰”应精确理解为：确认的标签/新方法性能未参与开发选型。原始全cohort baseline与部分label-free SOLE工程形状/零对照会覆盖其中视频；这些只做基线描述或数值验证，没有把其带标签新方法效果用于选择。后续确认仍需预先冻结最终参数并单独报告这层边界。


## 2026-09-09 00:50后：原生评分 query 的局部证据对比（前瞻冻结）

- 现有全序列视觉守恒对比在 Qwen text_video 全234开发样本已通过点估计门槛与相邻 k；Meter 初筛 text_video 也有双类正向+10pp，但其 text_image 常以成功类损失换取失败类收益；video_text 改善 MAE 却不提高端点准确率。RR interleaved 初筛虽双类+10pp，但未同时优于原强bias的 MAE/准确率，不能计作改良验收。
- 机制假设：在每一层所有 query 上改注意力，会同时修改任务文字与视觉 token 表示，因而影响后续评分所依赖的任务语义。候选第三轮只在原生评分 query 行重新分配视觉注意力：Meter 使用最后一个训练好的 prog token；Qwen/RR 使用共同 ANSWER 格式前缀末 token。Meter prog token 后还有聊天模板尾部，因此不能把 last_prompt 误当作 trained readout。
- 候选公式仍为全原生 bins 的 z*=z0+gamma*(z(+b)-z(-b))/(2b)，b={1.5,3}，gamma={3,6,12,24}，k={32,64}，all_frames/last_frame。较小有限差分半径检验非线性损伤。native生成、同读出baseline、零干预、原强bias以及同帧等面积wrong/low-head对照均保留。此为待检验推论，不把 PASTA/ASCD 的先例当作本任务有效性证明。
- `score_readout_development_v1/prospective_plan.json` 在读取本候选性能前冻结三模型×五输入，每case 29个神经支路、77个读出，同60条/24完整视频组。真实模型先做两条不读标签的零logits与实际query命中审计，通过才启动对应60条；确认498标签/新方法指标仍未用于选型。CPU测试验证未选中query和head精确不变、零残差精确不变、预加载索引与原索引实现相同。
- Meter text_video 原第二候选已扩展到完整234开发样本，包含k24/32/40/48/64、两scope及控制；GPU2执行中。
- Qwen text_video第二候选 k32/40/48/64、lambda4的簇bootstrap四项方向区间均支持改善（k24成功类区间触0），总准确率分别+20.51/+30.34/+30.77/+27.78pp；这是开发筛选后的探索区间，不能替代独立确认。文件 `score_development_full_v1/qwen_text_video/paired_k_neighborhood_v1.json`。
- SOLE原native residual 64条×5条件完整七步零对照已重新逐行核验并保存 `sole_batch_attention_residual_pilot_v2/full_zero_control_parity_audit_v1.json`。两种零干预全部8点轨迹、7段raw reasoning与baseline完全一致。
- CUDA static cache审计：HF static与手写static eager逐token一致，eager与CUDA graph一致，但dynamic与static发生数值分岔；没有启用static做正式实验。进一步发现主要开销是Python索引列表在每次decode的重复设备传输。仅预加载索引、保持原native dynamic cache，8条真实top64非零preserve在image_text/text_video逐token完全一致，用时44.3→16.2s、25.6→15.5s（共享GPU时序测量）；完整七步复核进行中。加速选型只据数值一致性和速度，不读效果标签。


### 2026-09-09 00:58：开发集扩展与控制结论（尚未独立确认）

- Qwen text→video，对称区域证据对比在完整234条开发集（76 suc/158 fail）中维持有效。lambda4下，k24/32/40/48/64的总准确率改善分别+15.38/+20.51/+30.34/+30.77/+27.78个百分点，全部MAE下降且两类准确率上升。k40/48的同k原强bias补充已经完成，改良对照门槛也通过。共同前缀读出与native生成3/234条不一致；全部234条零干预logits精确不变。
- k32/64新方案相对原强bias：MAE−0.2863/−0.3547，总准确率+14.53/+27.78pp，suc+2.63/+15.79pp，fail+20.25/+33.54pp。相对低排名heads的同机制对照也全部四项正向。
- 同时间等面积wrong-region在229/234条可构造。k32/64目标相对wrong的总准确率分别+11.79/+16.59pp，成功类+52.05/+47.95pp；失败类分别−7.05/+1.92pp。这支持任务相关区域信息的作用，同时表明错误区域也可能通过整体分数偏低改善fail，不能只以fail或总准确率单独归因。完整簇bootstrap与控制可行子集见 `score_development_full_v1/qwen_text_video/paired_region_and_original_controls_v1.json`。
- **RR video→text 的60条初筛改善没有在234条中达到门槛**：完整baseline MAE1.1966，accuracy56.41%；目前最佳双类方案last b3 k32 lambda2为MAE−0.1410，accuracy仅+5.13pp，suc+7.89pp，fail+3.80pp。不能将它计入稳定有效输入。其它已冻结候选和新readout-query候选继续。
- Meter baseline typed敏感性和SOLE 64条完整七步零干预审计均已补齐。SOLE index预加载在8条×两输入非零干预上保持原dynamic cache与原始token结果完全一致，完整七步工程pilot继续；static CUDA graph仍仅为工程实验，未替换正式结果。

以上均不构成最终验收；确认498的新方法性能尚未用于选型。尚需完成SOLE其它输入baseline/全部完整七步attention、三模型各至少三输入的完整开发与冻结确认，以及最终完整cohort汇总。


## 2026-09-09 11:17–11:30：夜间结果、索引加速接续与时间对比候选

- 上一目标阶段属于实际进展：新增原生readout-query候选、通过数值测试，完成相关开发实验，并补充簇配对统计和文档。此次恢复先检查了/proc真实进程与结果，未依据过期会话重启已完成任务。
- Meter全部五输入baseline1213、ranking34、原attention846×10均已尝试完成，错误仅来自wrong-region几何不可行。SOLE五输入baseline全量均已尝试；image_text/text_image/interleaved/video_text全部1213有效，text_video1204有效/9原生格式失败，保留raw文本。完整七步attention的image_text/text_image已完成，其余继续。
- 新增baseline总表、逐task/分布/两阈值/pairwise完整JSON引用和全部四模型×五输入top8，已追加主summary。`all_four_models_head_overlap_v1.json`记录同输入跨模型top8/32/64坐标交集，不将不同架构的同坐标当成功能等价。
- SOLE预加载head索引的完整七步pilot40条（8样本×5条件，包括非零original/preserve）与原实现逐段raw输出、轨迹完全一致；审计 `sole_dynamic_preloaded_full_pilot_v1/full_recurrence_index_parity_audit_v1.json` passed。保留native dynamic cache，未采用static CUDA graph。
- 仅停止已核验本目录的scheduler PID1962398/1962406，未终止其当前terminal子进程；全七步video_text转GPU3，interleaved在GPU1等前驱释放显存后继续，GPU0在当前text_video全七步后补terminal。新调用显式记录preload-head-indices。
- SOLE text_video的terminal原先因5个cohort样本缺完整有效native history而整个作业拒绝启动。新增显式allow-missing-context：这些样本各条件记录上下文不可用错误，仍计入预期846分母，其余样本各自重算终步配对baseline。不会捏造previous progress、从推理文字猜分或以标签补值。官方rewardgen的宽松字符串parser和随机采样默认与本实验greedy/严格numeric parser差异继续披露。
- 第二候选完整234开发：Qwen image_text、video_text也出现四向改善且+10pp的点；Meter text_video稳定覆盖k24/32/40/48/64，RR text_image稳定覆盖多个k。RR video_text未过10pp。Qwen另外两输入主要在k64改善，前瞻补测k56/60/68/72（lambda2/3/4、原强bias同k控制），不把单个k64直接当稳定范围。
- 第三候选readout-query已跑完3×5×60。15个case均未通过原strong-bias改良筛选；仅RR text_image有2个相对baseline的四向+10pp点，仍未同时优于original。限制评分query不够，不能将其记作有效方案。
- 发现筛选设计的天花板：Meter text_image/interleaved在60条中suc baseline为100%，而完整234为94.74%。因此60条不可能严格提高suc。后续只允许“suc不降且保持100%、fail提高、MAE下降、总+10pp”的候选进入234扩展；最终234/确认仍必须严格提升suc，验收标准不变。回查前两轮，在这个放宽的初筛转扩展规则下也无额外合格Meter案例，故没有重命名既有失败为成功。
- 第四候选`score_temporal_development_v1`前瞻冻结：z*=z0+lambda*[(z+observed−z−observed)−(z+static−z−static)]。static仅在内存中重复首帧像素，保留原native输入token、grid、mask、时间戳；grounding对应实际显示的首帧像素。首帧不被强制赋零，全部5/10native bins保留。VCD形式z0+alpha*(z0−zstatic)作为无region控制。原已完成observed支路按源文件SHA复用，不重写原结果；每case新增18个static神经支路。
- 新时间差分数学测试验证静态视频一致时退化为baseline，零干预精确不变，保留全部bins并拒绝错误维度。Meter/RR各2条不读标签的实际static-zero、输入结构等价工程检查passed，随后分别GPU2/GPU3运行15case初筛。新方法确认498性能尚未读取、未参与选型。


## 2026-09-09 12:20–12:40：Qwen邻域、RoboReward时间扩展与SOLE原生格式修复

- Qwen第二候选在完整234开发集补齐k64邻域：image_text在last/b3/lambda4下k56/60/64/68/72均四项改善、总增幅超过10pp；video_text在同参数k60/64/68通过（k56/72未通过）。text_video原有k24/32/40/48/64通过。统一primary k64及相邻k60/68可作为开发候选，但尚无确认结论。相关前瞻补充及源记录位于score_development_full_v1各输入目录。
- RoboReward时间对比text_image全234已完成：例如all/k40/lambda3，MAE−0.6282、总准确率+54.27pp、suc+10.53pp、fail+75.32pp；多个相邻k通过。interleaved全234仍运行。新增text_video初筛all/k32与k64/lambda4均总+18.33pp、两类上升并胜过原bias；video_text初筛all/k64/lambda1为总+16.67pp、两类上升且胜过原bias。因此前瞻冻结这两个输入的234扩展、k24/32/40/48/56/64/72、lambda1/2/3/4和原bias/wrong/low/VCD对照；接续GPU2已授权队列。没有据初筛直接宣布稳定有效。
- 第五轮score_local_radius_development_v1冻结15case，b0.5/1.5全query、中心差分gain3/6/12/24，分开检验有限差分半径与前轮readout-query限制因素。Meter text_image已完成，未过门槛；其余继续。
- SOLE新增terminal answer distribution适配器固定每个样本自身前六步历史及末步原生reasoning，保留−100..100全部201个整数百分比，并用prefix-free token trie的联合概率评估，计入负号、数字长度与结束符；无端点映射。它不替代主线完整七步attention。
- 无标签工程v2虽然零对照精确，但随后格式审计发现native20%/15%的合法候选总概率仅约4e−8/8e−7，属于实现错误：单独编码%得到id4，原生完整回答的%</answer>则把%</融合为id52508。旧sole_score_development_v1在validity_annotations.jsonl明确整体排除；原始错误结果完整保留。仅停止已确认本目录的旧SOLE score进程1986614/1987484/1987487，未动其它任务。
- 修复为先tokenize完整原生结束标签，再依据offset_mapping截取到覆盖百分号的完整canonical token边界。合成融合token测试、201整数保留测试、符号联合概率测试、真实小Qwen缓存对照及时间对比测试共7项通过；真实SOLE tokenizer的201候选均以52508结束、prefix-free。新的工程v3及sole_score_development_v2使用新目录，不覆盖旧输出。真实v3首样本baseline合法格式总概率0.999949、MAP15%与native15%一致，零对照logits精确；两样本全部支路审计通过后才自动进入60条五输入初筛。
- 主线1当前SOLE text_video全七步846×11已尝试完成（9217有效/89错误，待按条件分类）；video_text与interleaved全七步、text_video终步仍运行。不能将“所有baseline完成”写作“主线1完成”。确认498的新方法性能仍未参与选型。


## 2026-09-09 12:43：原生完成置信度调节的区域证据对比（第六候选，前瞻冻结）

- 已观测的主要失败模式是统一放大区域方向降低了fail的误报，却同时破坏suc完成度。一个可检验的解释是，高置信度的完成判断已经综合了物体与任务关系，统一强干预会擦除这些证据；不能从这个解释直接宣称有效。
- 新候选只调节干预方向的幅度：z*=z0+lambda*(1−q_native)^power*D，D取原对称区域证据，或扣除静态外观敏感性的时间区域证据。q_native由当前样本的模型原生输出计算：Meter为checkpoint已有、单独训练的task-success head；Qwen/RR为全部五档分布的归一化期望。这里q是模型预测，不是数据真值；推理代码不读取标签、suc/fail路径语义或配对样本。
- 全部原生5/10分档保留，q既不替代progress也不决定输出端点，无硬阈值、无端点映射、无额外训练。D=0时任何q都严格退化为baseline。它是一项可选调节，不将Meter辅助head的存在误写成跨模型相同架构。
- score_native_confidence_development_v1/prospective_plan.json在计算本候选性能前冻结：三模型×五输入，k32/64，all/last，lambda4/8/16，power1/2，两种D；同60条/24视频组。原始不调节方案、native/readout baseline、原强bias、wrong/low controls均保留。只复用有完整神经支路的已完成源文件，并记录SHA；尚未结束的源实验等待完成后再做相同冻结读出，不重复神经推理。
- 10项数值/缓存测试通过，包括全五档对置信度均有贡献、零干预精确、连续无阈值调节、缺置信度拒绝计算，以及既有SOLE原生token与temporal对比测试。最终仍需234与独立498同时通过严格两类改善和总+10pp；初筛成功类100%的天花板规则不改变最终验收。


### 2026-09-09 12:51：同强度对照修正归因，冻结234扩展

同lambda的uniform_strength_matched_supplement_v1对照表明：Meter image_text/video_text在不加置信度调节、lambda16时也能通过60条初筛，且MAE略优于调节版本。image_text all/k64/lambda16为MAE−1.1149、总+28.33pp、suc+30pp、fail+27.5pp；video_text all/k64/lambda16为MAE−0.9336、总+38.33pp、suc+5pp、fail+55pp。**因此这些Meter初筛收益目前应归于更大的区域证据放大幅度，不能归于辅助成功head或置信度调节。**

Meter image_text/video_text/text_video的234扩展已前瞻冻结在score_native_confidence_development_full_v1：k24/32/40/48/56/60/64/68/72，lambda4/8/12/16，不调节与power1/2完全配对，原bias、wrong、low及零对照。前两输入all scope，text_video last scope；每case38神经支路/194读出，已复用同条件源输出并校验无重复数值差异。GPU0按可用显存接续，确认集仍未运行。

RoboReward image_text则只有置信度调节的时间对比通过初筛，匹配lambda的无调节对照均未过四向+10pp门槛；best last/k64/lambda8/power1为MAE−0.85、总+31.67pp、suc+10pp、fail+42.5pp。该输入也冻结234扩展，含相同k/幅度/调节消融和两类区域控制，GPU3等待自身Qwen时间初筛队列完成后执行。所有这些是开发探索，需观察独立视频组扩展，不因best60结果宣布成功。


### 2026-09-09 13:05：读出温度控制与确认分析边界

`native_temperature_development60_control_v1.json`显示，单独放大native logits并不能解释Meter image_text/video_text的区域方法收益：image_text的tau8/16虽然使suc准确率+30pp，但fail仍不改善、MAE反而增加约0.2；video_text在tau2/4/8/16都没有准确率收益且MAE上升。text_video则有明确读出校准收益：tau8总+28.33pp、suc+45pp、fail+20pp，但连续MAE仅−0.0829，远小于区域方法约−0.9。因此最终必须保留temperature-only对照，不能把所有端点改善都归于attention信息增益。text_image/interleaved的温度控制也能在保持初筛suc100%时增加fail约25–27.5pp；这只是native读出控制，不计作attention方案。

`confirmation_analysis_protocol_v1.json`已冻结统计规则及498条/179视频簇边界，尚未选定最终manifest、未启动确认。完整开发候选由candidate_inventory按相同超参数的连续已测k整理，缺失/失败的中间k不能被跨过算稳定；同时描述原强bias在开发k网格的MAE/准确率包络。统计方法、11/11偏差类型核验与未完成项见STATISTICAL_VALIDATION_20260909.md。

Qwen时间初筛15case现已全部完成，其中qwen_video_text all/k64/lambda1的60条MAE−0.7667、总+33.33pp、suc+25pp、fail+37.5pp，已冻结234扩展并于GPU2启动；原non-temporal三输入候选仍保留。SOLE201整数读出的有效v2初筛继续，原有完整七步两个输入继续；目标仍active。


### 2026-09-09 13:12：Qwen相邻k的不确定性

新增qwen_image_text与qwen_video_text的paired_k64_neighborhood_lambdas3_4_v1.json，对k60/64/68、lambda3/4全234做20000次完整视频簇统计。image_text的lambda3/k64成功类改善14.47pp、95%区间[2.53,26.92]pp；但k60仅+1.32pp、区间跨0，k68也跨0。video_text的多项MAE/suc区间同样跨0，尽管这些点都通过开发点估计门槛。不能把“同参数相邻k点估计均正向”写作“每项区间都证明稳定”。

因此保留原候选与其限制，继续检验已通过60条筛选的时间对比：Qwen video_text全234已运行；image_text all-scope的234扩展也已冻结（k24/32/40/48/56/64/72、lambda1/2/3/4，含原bias/wrong/low/VCD控制），队列等待自身Qwen video_text任务结束后执行。image_text复用6912条同条件分支，重复源无数值差异。此追加仅依据开发信息，不读取确认结果。


## 2026-09-09 13:18：KL预算控制的区域证据（第七候选，前瞻冻结）

- 现有开发信息显示，固定raw-logit gain在Meter、Qwen和不同输入中所需尺度不同；较大的lambda改善Meter部分输入，但个别Qwen相邻k的成功类不确定性仍较大。提出用同一个后验分布变化量来定义干预强度，而不是假设各模型raw logits可直接比较。
- p_lambda=softmax(z0+lambda*D)，D仍为对称区域证据或扣除首帧静态参考的时间区域证据。选择lambda∈[0,64]中满足KL(p0||p_lambda)≤delta的最大值；delta={0.25,0.5,1,2,4} nats。对D做p0加权中心化后，KL等于logsumexp(z0+lambda*D_centered)−logsumexp(z0)，在lambda≥0上单调，使用64步二分；局部曲率为Var_p0[D]。整个计算只读当前样本原生分数向量，不用标签、输出端点或额外success head。
- 全native bins保留；D为常量或delta0时精确返回baseline。最大gain64是预先固定的数值/噪声放大上限，不把近零方向无限放大来强行满足预算。每个结果保存实际KL、有效gain、是否触顶、方向方差。预算不一定能达到，此时必须如实报告。该方法可能仍放大无用方向，因此必须保留wrong/low对照，不能从数学约束直接断言有效。
- score_kl_development_v1全三模型×五输入冻结同60/24视频簇，k32/64、all/last、region/temporal。增加delta0数值对照，以及同KL预算、仅以baseline logits为方向的temperature-only对照。无新神经前向、无新训练；对原始完整源文件只读引用并核验SHA，输出在新目录。
- 10项组合/置信度/KL数值测试通过，包含KL独立定义复算达到目标、全bins保留、常量/零方向原样返回、score整体平移不改变后验、极小方向受gain上限限制、同KL温度保持argmax，以及旧temperature/temporal/置信度接口。当前尚未读取本候选的性能，确认集仍未执行。

补充负结果：RR interleaved时间对比全234仅k72的lambda2/3/4通过同k原bias改良；k24/32虽双类及总+10pp，但不如对应原bias，k64的MAE也差于原bias。不存在三个相邻k的通过范围，因此不计为稳定有效输入。原bias最优k24/32也优于k72候选，不能用挑选原方法较弱k来宣称总体改良。


### 2026-09-09 13:41：SOLE百分比范围的来源与原生越界值

重新核对官方论文正文（references/2603.28730_plain.txt第414–416行）明确给出p_t∈[−100,100]；官方system prompt要求signed integer percentage（rewardgen/sole.py第42–46行）。因此201点读出覆盖的是论文规定范围内的规范整数值，不是模型整套词表或所有可能的不规范字符串。

无标签遍历五输入原baseline全部递归数值后，未发现小数百分比；image_text的8491次递归数值中有1次原生101%，其余输入均落在论文范围。该101%是模型生成的越界数值，旧native baseline及上下文原样保留，未裁剪或改写。201点候选读出与原生生成作为不同读出并列，效果需同时胜过两套baseline；不把范围约束或格式修复算作steering收益。完整定位见sole_native_percentage_contract_audit_v1.json。


### 2026-09-09 13:45：KL及大幅度对照的复核与新234扩展

KL初筛三模型×五输入已全部完成；all_cases_numerical_audit_v1.json中全部delta0对照与baseline精确一致，预算或gain上限越界均为0。部分弱方向达到lambda64上限却未达到目标KL，已逐条记录，不能把它们冒称实际KL完全匹配。Meter image_text/video_text/text_video、Qwen image_text/video_text/text_video初筛均有通过点；RR仅text_image/text_video有少量点，KL本身没有形成三模型各三输入的结论。

再次做等强度归因：Meter video_text的KL temporal-last/k64/delta4实际gain中位25.64，范围11.26–64。新增固定16/32/64对照发现，temporal-last/k64/lambda64同样带来suc+25pp，并且fail+72.5pp、总+56.67pp、MAE−0.9705，优于该KL点的部分指标。因此这里仍不能将收益归因于KL自适应；新的有希望因素是更强的末帧时间证据。

已冻结score_kl_development_full_v1/meter_video_text：k56/64/72，last scope，region/temporal配对，uniform16/32/64与KL1/2/4配对，原bias、wrong/low、静态/零、温度/同KL温度均保留。28个神经支路/95读出，复用1080条无标签源结果，GPU2执行中。

对已有完整开发神经计划另追加posterior_budget_and_large_gain_supplement_v1.json：Meter三输入允许uniform32/64与KL预算；Qwen/RR相应已筛选输入补KL及同KL温度控制。它们只复用已经冻结的分支，不新增神经工作。所有补充需在默认combine结束后运行materialize_supplements --folder F，再做分析；该工具校验全部ledger哈希。RR text_image及Qwen text_video补充已完成，Qwen video_text刚完成神经分支且正在补读出。

完整234上的KL并未普遍优于固定幅度：Qwen text_video的delta1/2/4都在k24/32/40/48/64通过；RR text_image只有少量零散k通过，其固定时间对比仍更稳定。最终选择必须服从开发证据与独立确认，不因新公式更整齐就宣称它更有效。


## 2026-09-09 14:10：新增完整开发证据与确认前的缺口

完整234条开发数据中，Robometer现有image_text、video_text、text_video三种输入具备同参数相邻k范围；Qwen此前三输入的区域方案保留；RoboReward目前只有text_image和text_video两输入达到开发稳定范围，第三输入仍待正在运行的image_text/video_text验证。尚不满足确认启动门槛，新方法确认498推理未启动。

|模型/输入|方案（b3）|k|连续或原生MAE变化|总准确率变化|suc变化|fail变化|
|---|---|---:|---:|---:|---:|---:|
|Meter image_text|区域差分，all，lambda64|64|−1.0173|+43.16pp|+34.21pp|+47.47pp|
|Meter video_text|区域差分，last，lambda64|64|−0.8263|+55.56pp|+21.05pp|+72.15pp|
|RR text_video|时间区域差分，all，lambda4|64|−0.7308|+16.24pp|+7.89pp|+20.25pp|

Meter两个新输入的k56/64/72全部通过开发四指标和同k原bias改良门槛，五档MAE也下降。20000视频簇bootstrap中这些点四项方向区间均支持改善，但这是经过开发选型后的描述性区间，不能代替确认。RR text_video的suc区间跨零（k64：−3.85至+19.23pp）；应保留成功类不确定性。candidate_inventory_20260909T140623.json中RR text_video的完整改良范围为k48/56/64/72；仅对baseline通过的其它点不能混入改良范围。

Meter image_text/video_text的纯native温度控制未复制收益：image_text温度tau16仅总+7.26pp且连续MAE更差；video_text所有已测温度仍为0端点准确率且MAE更差。固定lambda64已能改善两个输入，不能把新增收益归因于KL预算、置信度调节或时间扣除。video_text时间版本提高更多suc，但区域版本MAE及fail更好，属于权衡而非普遍优势。

Qwen video_text时间候选在全234的k64/72有强效果但此前只有两个连续已测点；已于计算前冻结k68中点（同all/b3/lambda1，另列KL .25/.5敏感性与observed-only、同k原bias），运行run_temporal_midpoint_supplement。它是开发内追加，不能称独立确认或把未测整数区间写为已验证。

全部新证据见score_native_confidence_development_full_v1/meter_image_text、score_kl_development_full_v1/meter_video_text及score_temporal_development_full_v1/rr_text_video内paired_full_development_evidence_20260909T1408_v1.json；分析快照及原始输出均保留。SOLE两种完整七步attention仍运行，终步补充不能替代。


## 2026-09-09 14:34：确认工具准备、调度接续与Qwen时间结果

新增freeze_confirmation.py与analyze_confirmation.py，尚未实际冻结或启动确认。真实8case admission检查逐项通过完整234、相同配方邻域、精确零logits/分数、same-k原bias及native读出门槛；总覆盖仍因RoboReward只有两输入而被明确拒绝。另用真实已测k56/64/72跳过k60/68的例子核验邻域检查会拒绝跳点。确认配置构建在内存中核验所有分支依赖齐全，未写confirmation_v1。可审计证据为confirmation_admission_rejection_audit_v1.json和confirmation_admission_real_data_audit_v2.json；draft_v1是拒绝测试材料，绝不是最终选型。

开发Qwen text_video的原k24/40/48对照补充早已冻结并执行，但未链接到后建的analysis_supplements ledger。此次核对原launch内完整内容/SHA和combination事件后，仅追加现有补充的链接；没有生成新参数或改写旧数据。首次工具自检因此遇到KeyError，修复元数据链接后通过。

Qwen image_text时间候选234全部完成，lambda2在all/b3下k24/32/40/48/56/64/72均通过改良门槛；lambda1也有范围，但某些k不优于开发最优原bias。k24/32/40的lambda1四项95%簇bootstrap方向区间均支持改善；lambda2的k40 suc区间仍跨0。Qwen video_text补测k68完成：时间all/b3/lambda1的k64/68/72均通过（总+42.31/+39.74/+39.74pp），k68 suc+9.21pp；部分suc区间仍跨0，不能只因MAE大降就宣称成功类已充分证实。源为analysis_snapshot_20260909T141730.json和各目录paired_temporal_completed_neighborhood_v1.json。

RoboReward video_text时间候选全234最高双类总增幅8.55pp（k64/lambda1），未达10pp。较大lambda恶化，因而前瞻冻结gain_sensitivity_full_development_supplement_v1.json，测试更小uniform幅度、原生置信度调节、KL预算与匹配控制；402个离线读出，零新增神经前向。开发内仍可探索，确认集不参与。

约14:14核查发现数个旧工具会话的调度父进程已消失，子模型仍活着或刚完成；退出原因不可得，不能归为科学/模型失败。新增resume_preserved_score_queue.py通过/proc核对完整config路径和cwd，保留存活模型，完成后仅补combine及冻结supplements，随后执行未启动配置。四个durable恢复队列2015070/2015071/2015072/2015078以独立会话启动，详见durable_recovery_launches_20260909T1420_v1.json。没有发信号给模型，没有重跑完成的预测。Qwen image_text的同类恢复已完成，RR video_text恢复已完成。SOLE mandatory两个完整七步仍在原进程运行。


## 2026-09-09 14:48：三模型开发准入通过，确认尚未启动

RoboReward image_text的完整234扩展完成。相同last/b3/lambda16/power1时间配方在k60/64/68的MAE变化为−0.1966/−0.2436/−0.2436，总准确率+12.82/+14.53/+14.10pp，suc均+3.95pp，fail+17.09/+19.62/+18.99pp。全量零bias/static-zero logits及分数精确不变；与原生生成有1条读出差异，因此两个baseline继续并列。

suc的20000视频簇bootstrap区间为−2.67至+10.96pp，三点均存在成功类不确定性；不能将3/76的成功类净增益写成充分确证。与同lambda16不调节时间版本相比，primary k64 MAE−0.3547、总+10.68pp、suc+38.16pp、fail−2.53pp，调节主要保护原生成功判断。相同observed-only gated读出原配置未生成，已在读其结果前补充冻结并从现有observed分支计算，不假装已有结果。

confirmation_candidate_selection_20260909T144256.json是按显式最弱余量规则提出的9case开发方案；freeze_confirmation --validate-only已经实际通过：Meter三输入、Qwen三输入、RR三输入，各234完整、同参数连续已测三k、same-k原bias及两个baseline/zero审计均符合。此命令没有写confirmation_v1、没有启动确认。先完成已启动的RR video_text b1.5全234复验与最终材料审阅，再一次性冻结全部确认case；不读取确认结果边跑边增删primary案例。

Meter新增text_video幅度扩展也已完成234，多套固定幅度存在宽k范围；当前最弱余量规则选择last/b3/lambda12、k40/48/56。目录名带confidence不意味着最终方案使用success head，当前选择的Meter三个输入均不使用置信度调节。代码中temperature的tau是logit乘数（相当于1/T），不是直接把物理温度T增大；KL温度对照对应1+effective_gain倍logit，最终报告需使用这一明确含义。

SOLE余下数值读出screen并行调度仅改变GPU分配：image_text原进程1993315不动，text_image→video_text在GPU0（scheduler2026510），interleaved→text_video在GPU2（2026511），image_text完成后的汇总由2026509负责。此前等待调度器2015078仅在确认没有子进程时终止，native模型均保留。主线1的两个完整七步仍在1980518/1984884执行。


## 2026-09-09 14:55：SOLE重复启动事件与原始数值审计

检查新进程发现，原先消失的是shell2003974，其Python子进程2003975仍存活。它在旧image_text完成后又启动scheduler2029484及text_image worker2029488，与已授权的2026520同config重复。这是调度核查遗漏，不能隐瞒或按两个独立实验计数。

已逐一核对cwd、完整命令、父子关系后停止2003975/2029484/2029488，仅停止旧调度链和多余实例；保留2026520及其合法新调度器2026510。所有原始行保留。停止后的前缀审计共有110个重复(example_id,condition)键，全部status、201点score_logits、progress、native_percent及raw/error等科学字段相同，冲突0；不把耗时/记录时间当作数值差异。证据在sole_score_development_v2/sole_text_image/duplicate_overlap_audit_20260909T1455_v1.json。原worker完成后仍须再做全文件重复/覆盖审计，不能仅据此快照提前宣布整个文件无冲突。

SOLE image_text规范201点初筛60条已完成，baseline/zero及static-zero logits与分数精确相等；原生生成与MAP读出2条不同，两个baseline分开。lambda<=4区域/时间候选未通过四项+10pp。为检验回答在固定原生reasoning之后的原生logit尺度，已对五输入统一前瞻冻结固定lambda8/16/32/64与KL .25/.5/1/2/4读出扩展（每输入144条件），保留全部201规范整数和匹配区域/低head对照，零新增神经前向。此补充不替代任何完整七步实验。


## 2026-09-09 15:24：一次性冻结并启动独立确认
最后已启动的RR video_text小半径b1.5完整234开发复验已完成。最大总准确率增益为7.692308pp，对应`contrast_target_local_last_frame_b1.5_k24_gamma12`；其MAE/suc/fail变化分别为-0.213675 / 5.263158pp / 8.860759pp。没有新增满足同参三个相邻k准入的输入，全部负结果保留于`results/mydata_bench/experiments_v2_corssmodel/auto_research/session_20260908/score_local_radius_development_full_v1`。
最终选型文件为`confirmation_candidate_selection_20260909T151434.json`。完整九项真实`--validate-only`通过后，已一次性生成`confirmation_v1/frozen_manifest_v1.json`及全部九个YAML；随后再次核验实际配置的所有分支依赖和名称唯一性，再启动确认。此前仅预览脚本用了假设的original k32，在Meter video_text源配置中不存在而报错；该假设未进入冻结配置，真实准入/实际依赖检查均通过。
|模型/输入|冻结primary|邻域k|
|---|---|---|
|meter/image_text|`contrast_target_uniform_large_region_all_frames_b3_k68_lambda64`|64,68,72|
|meter/text_video|`contrast_target_uniform_matched_region_last_frame_b3_k48_lambda12`|40,48,56|
|meter/video_text|`contrast_target_uniform_matched_temporal_last_frame_b3_k64_lambda64`|56,64,72|
|qwen/image_text|`contrast_target_temporal_all_frames_b3_k40_lambda2`|32,40,48|
|qwen/text_video|`contrast_target_last_frame_b3_k48_lambda4`|40,48,64|
|qwen/video_text|`contrast_target_midpoint_kl_temporal_all_frames_b3_k68_delta0.5`|64,68,72|
|rr/image_text|`contrast_target_confidence_temporal_last_frame_b3_k64_lambda16_power1`|60,64,68|
|rr/text_image|`contrast_target_all_frames_b3_k48_lambda2`|40,48,56|
|rr/text_video|`contrast_target_temporal_all_frames_b3_k56_lambda4`|48,56,64|

每项包含两个baseline、same-k原方法、开发最小MAE原方法、匹配primary的zero/wrong/low，以及预定温度/模块消融。新增primary零干预用于匹配实际k与scope；时间方案同时含static-zero。九项同时冻结，后续无论结果好坏均报告，不能删掉失败项或用确认结果调整参数。
确认498来自179个完整视频内容簇，与开发234及ranking视频不交叉。已有baseline和无标签工程曾接触少量确认像素；新方法确认性能未用于这次选型。因此称独立于新方法开发选型的确认集，不能写“从未接触像素”。
GPU0运行RR三项，GPU1运行Qwen三项，GPU3运行Meter三项；调度记录`confirmation_v1/durable_queue_launch_v1.json`。队列按空闲显存启动，排他写锁保护输出，失败不会静默重跑。SOLE video_text/interleaved完整七步与剩余数值screen继续运行。当前仅确认方案已冻结/开始推理，**最终验收仍未完成**。


## 2026-09-09 15:33：冻结完整cohort补齐与完成后审计

已在未读取确认性能的情况下冻结`frozen_cohort_completion_v1/frozen_manifest_v1.json`：固定同一九项方案和全部三个邻居，补跑remaining114，然后将开发234、确认498、ranking同视频重叠114逐样本汇总为grounded846。三部分ID不重叠；114部分可能包含ranking所用视频，因此846只能作描述性汇总，不能替代498确认。所有九项无论确认成功或失败都会保留在完整汇总中，不通过确认结果筛掉case。

开发源文件SHA与所需全部condition×234覆盖已在补跑前核验。补跑按GPU0/RR、GPU1/Qwen、GPU3/Meter排在各自确认三项全部完成之后；`durable_followup_launch_v1.json`记录等待队列PID。`analyze_frozen_cohort.py`按冻结来源SHA拼接既有结果并检查重复/交叉、两套baseline、同k与开发选定原方法、zero、两阈值、逐task/分布和same-video配对；不会重复推理已经有相同冻结结果的732条。

新增`audit_sole_screen.py`对完成后的score_branches与sweep分别审计完整期待键、重复键、科学字段冲突和201格式概率。image_text已完成最终审计：神经支路2340/2340唯一键有效，离线读出15000/15000唯一键有效；sweep原始19924行的重复键科学字段冲突为0，统计仍每条样本/condition一次。baseline60条的规范格式总概率全部>.99，两条greedy和联合MAP相差1个百分点，均已保留。text_image的先前双写最终审计将在完成所有补充读出后自动执行，不用运行中快照代替。

`completion_analyses_v1/plan.json`固定九项完成后分析任务：SOLE四项最终审计、五输入screen完整分析、九项498确认、完整846描述汇总、主线1最终attention审计和metrics快照。只有真实完成标记出现才运行，对所有结果同样处理；失败分析记录退出码，原始结果不改动。最终研究结论仍需根据这些产物核验后写定。


### 2026-09-09 15:34：SOLE text_image最终重复审计完成

完整文件审计`sole_score_development_v2/sole_text_image/completed_score_audit_20260909T153343.json`：原始2450神经支路行包含2340唯一键，全部有效；110重复键科学字段冲突0；sweep为15000唯一键、全部有效、无重复。此前运行中审计的待办已由最终文件审计关闭，没有删除原始重复记录，也没有把重复行当独立样本。observed/static两组zero各60/60的logits与progress精确一致。

60条baseline的201规范格式总概率最小0.9999082，中位0.9999990，均>.99；最大1.00000018保留为浮点累加误差而未裁剪。联合MAP与自身native greedy有一条差异（suc/ljx_lfz_task_1_1/20：83%对85%）。这是读出差异，不能归因于attention。剩余video_text/interleaved/text_video仍需完成screen；text_image完成不意味着SOLE新方法达标。


### 2026-09-09 15:45：SOLE三个完成数值screen的完整负结果

`results/mydata_bench/experiments_v2_corssmodel/auto_research/session_20260908/sole_score_development_v2/analysis_snapshot_20260909T154533.json`已汇总image_text、text_image、interleaved各60条和全部250个读出（包含预先声明的大尺度/KL补充）。三个输入均没有满足完整四向/+10pp及同k原方法改良的候选。

|输入|完整target候选数|通过改良门槛数|全部target中的最大总准确率增益|
|---|---:|---:|---:|
|sole_image_text|104|0|1.67pp|
|sole_interleaved|104|0|10.00pp|
|sole_text_image|104|0|10.00pp|

最大总增益列仅说明搜索结果上限，不代表该点同时降低MAE或提高suc/fail；不能把不同点的最优指标拼成一个方案。SOLE目前采用固定自身前六步和末步reasoning的数值读出干预，这些screen不能替代主线1的完整七步attention。完整七步video_text/interleaved与两个剩余视频数值screen继续运行。
interleaved最终审计`completed_score_audit_20260909T154210.json`显示2340支路键、15000读出键全部有效，无重复或科学冲突；observed/static zero各60/60精确，native greedy与联合MAP全部一致，规范数字格式概率最小0.9993461。


### 2026-09-09 16:03：确认分析完整性补充

确认启动后、尚未读取新方法确认性能时的代码检查发现：manifest虽记录全部输入SHA，原分析入口尚未重新核验这些SHA。已给`analyze_confirmation`增加四个冻结输入文件与metrics/paired_analysis/analyze_development三个统计模块的SHA检查；给846汇总增加冻结输入与三个partition IDs的SHA检查。全部实际核验通过。该改动只在文件不匹配时拒绝分析，不改变任何模型参数、评分、指标、门槛、重采样次数/种子或case选择。新分析源码内容SHA和变更理由保存在`confirmation_v1/analysis_integrity_checks_supplement_v1.json`；新结果另记录实际分析源码SHA。原冻结manifest与旧源码快照均保留。

复核与复现入口集中于`mydata_bench/auto_research_addbase/REPRODUCTION_GUIDE_20260909.md`。完整实现补充清单含141个Python文件，原冻结21个推理相关文件仍全部相同；补充清单不冒称在确认开始前就已预注册。


## 2026-09-09 16:35：Qwen调度按后续空余GPU重新分配

资源安排由原先Qwen三个输入都在GPU1串行，改为：当前qwen_image_text模型PID2038007继续在GPU1完成；qwen_text_video待GPU2的SOLE text_video数值screen及其补充结束后启动；qwen_video_text待GPU0的SOLE video_text数值screen及其补充结束后启动。四卡均为A100-SXM4-80GB。这样利用后续释放的资源，并减轻SOLE mandatory全七步interleaved与Qwen队列长期共享GPU1的压力。

事先保存`RESEARCH/confirmation_qwen_gpu_handoff_v1/plan.json`，核对cmdline/cwd/真实模型子进程后，仅停止旧Qwen调度父2037993和旧Qwen 114等待父2039244；未给模型2038007发送信号，接管后仍正常运行。恢复父2062735（上层2062732）只等待该模型真实完成，再做combine，避免重复前向或双写。两个新等待父2062733/2062734分别负责GPU2/GPU0后续输入。各输入确认完成后在自己的GPU接续已冻结114补跑。

RR/Meter原确认父2037992/2037994和114等待父2039243/2039245继续运行。完成后分析守护2040764按各case的真实completion/combination标记执行，仍适用于此次分配。新确认任务可能使同一GPU事件日志出现多个独立队列的事件；完成判断必须按case与输出目录，不能仅把某个queue_completed当成该GPU所有工作已完成。

此次变更只调整未来资源调度。九个case、每项三个k、所有分支/读出/对照、数据ID、ranking和统计口径不改；重新核验原冻结21个推理文件SHA全部相同。全部调度动作、等待条件、PID和阶段退出码都在handoff目录保存，原manifest与预测不修改。


## 2026-09-09 16:45：首项冻结独立确认结果（RR image_text）

`confirmation_v1/rr_image_text/individual_case_analysis_v1.json`为已预声明九项中的第一项完整分析。九项参数在首次确认前已共同冻结；此时其余八项尚未完成，不删case、不调整后续参数，总体三模型验收及跨九项Holm仍待全部完成。

原始15438/15438支路有效，无重复或冲突；四组zero（observed/static、k32/primary k64）各498/498的logits和progress精确一致。两个baseline有5条分数差异，因此仍分别比较。

|k|对native baseline的MAE变化|总准确率变化|suc变化|fail变化|两baseline与同k原方法点门槛|
|---|---:|---:|---:|---:|---|
|60|-0.148594|10.2410pp|1.8987pp|14.1176pp|通过|
|64|-0.164659|11.4458pp|3.7975pp|15.0000pp|通过|
|68|-0.152610|10.4418pp|3.1646pp|13.8235pp|通过|

primary k64：原生baseline MAE0.833333→0.668675，总准确率64.0562%→75.5020%，suc58.2278%→62.0253%，fail66.7647%→81.7647%。primary也同时胜过开发选定的单个最小MAE原方法设置，所以本case的严格原方法点检查通过。
20000视频簇bootstrap的primary变化95%区间：MAE[−0.29541,−0.03725]，总准确率[+6.5606,+16.4635]pp，suc[−1.2821,+9.0909]pp，fail[+8.5162,+21.5976]pp。suc区间跨0，四方向IUT p=0.121044；尚未做全部九项Holm，校正也不会把该p变小。因此可以说三个已测k在独立确认点估计上通过，**不能说四项方向均有联合统计显著证据，也不能说总增益的95%下界超过10pp**。k60仅比10pp门槛高0.24096pp，边界余量应如实保留。

primary对全部498均可行的对照：相对wrong，总+17.4699pp、MAE−0.307229；相对low-head，总+9.2369pp、MAE−0.182731；相对仅observed区域差分，总+10.0402pp、MAE−0.192771，其中suc相同、fail+14.7059pp。相对同λ16的uniform不调节方案，总+15.0602pp、suc+52.5316pp、fail−2.3529pp，符合调节保护高原生完成度判断并牺牲少量fail的权衡；q仍是原生归一化评分期望，不是已校准正确性概率。温度控制不改变离散argmax，未复制attention增益。控制比较作为机制描述，不另冒称已完成所有控制检验的多重校正。

primary分布仍含37个2档、11个3档、16个4档（共64个中间档），1/5档为312/122；全部五档保留，无输出端点映射。315个同视频配对的negative/0/1/2/3/4计数由原生baseline的43/25/9/25/42/171变为17/59/11/12/32/184。负序减少，但ties增加，严格正差配对数247→239，不能笼统声称所有pairwise指标都提高；平均progress差0.616667→0.646032。


## 16:45–16:50 接续更新：首项确认完成且点门槛通过

RR image_text已全部完成，`confirmation_v1/rr_image_text/individual_case_analysis_v1.json`是首次真实读取确认性能。九项保持冻结，其余八项未完成；该case三个k60/64/68均过两baseline+same-k original点门槛，primary也胜开发选定原方法，strict_original=True。相对native总增益10.24096/11.44578/10.44177pp，suc+1.8987/+3.7975/+3.1646pp，fail+14.1176/+15.0000/+13.8235pp。primary MAE0.833333→0.668675，suc95%CI[-1.2821,+9.0909]pp跨0，IUT p=.121044；尚无九项Holm/总体验收。其余详细CI、全部控制、分布/配对见主summary16:45段。

主primary相对wrong/low/observed-only总+17.4699/+9.2369/+10.0402pp；相对同强度uniform总+15.0602pp，但fail−2.3529pp而suc+52.5316pp。315配对负序43→17、ties25→59，正差数247→239，平均gap上升；不能泛称全部pairwise指标改善。

新增audit_score_engineering.py（无标签原始支路审计）真实跑RR image_text：15438/15438唯一有效，无重复冲突；四zero各498精确；native/readout分数差5条。对应completed_branch_engineering_audit_v1.json。后续完成case可使用该脚本补查原始支路冲突，最终analyze_confirmation主要审计sweep。

发现并修正分析metadata：共享开发contrast函数硬带confirmation:false，首次独立case输出也带入该历史字段。最终分析wrapper已移除该字段，改标evaluation_phase=frozen_confirmation，接受与否仍用单独acceptance布尔。所有首case数字对比实际重新计算并精确相同，metrics.py/paired_analysis.py等未改，推理参数未改。原individual_case_analysis_v1.json保留，通过confirmation_v1/analysis_metadata_correction_v1.json解释旧字段；新分析源码SHA/快照已保存。不要把旧内部confirmation:false误读为该case点验收失败。

RR队列已启动text_image，新模型PID2064357（父2037992）。Qwen image_text模型2038007继续，接管父2062735/上层2062732在等待；后续Qwen两个等待父2062733/2062734按16:35新安排。Meter image_text尚在跑，约440+/498。SOLE video_text/text_video numerical约41/60；mandatory两个全七步仍运行。实际进度/PID请复核。


## 2026-09-09 16:55：Meter image_text冻结独立确认完成

`confirmation_v1/meter_image_text/individual_case_analysis_v1.json`覆盖全部498条。三个冻结邻居通过两baseline及同k原方法点门槛，primary k68也胜开发选定原方法；主参数没有根据确认中k64较高的总准确率重新选择。

|k|连续ordinal MAE|五档MAE|总准确率（0.125/0.875）|suc准确率|fail准确率|总准确率（0.2/0.8）|
|---|---:|---:|---:|---:|---:|---:|
|baseline|2.255305|2.351406|0.0000%|0.0000%|0.0000%|0.0000%|
|64|1.008300|0.959839|51.0040%|34.8101%|58.5294%|61.2450%|
|68|1.042449|1.004016|46.9880%|33.5443%|53.2353%|60.6426%|
|72|1.064509|1.068273|43.9759%|27.2152%|51.7647%|57.2289%|

baseline的0%端点准确率需要作为绝对参照披露：这是image→text自定义适配，498条输出落在17个3档和481个4档，不能把相对大增益误解为官方输入已经达到同样水平。primary真实端点准确率46.9880%，第二阈值60.6426%；连续与全部三个邻居五档MAE均下降。
primary相对native的95%视频簇区间：连续MAE变化[−1.31318,−1.10820]，总准确率变化[+42.4063,+51.6395]pp，suc[+26.3804,+41.0256]pp，fail[+46.8208,+59.7734]pp。四方向IUT Monte Carlo p=1/20001≈0.000050；全九项Holm结果仍等待其余case。

原始8964个期待支路键全部尝试、无重复冲突；8952有效，12个错误来自6条样本的wrong正负分支几何不可行。两个zero（k32与primary k68）各498/498精确，native/readout baseline分数完全一致。主方法与原方法等必需结果完整；wrong对照只在明确492条可行子集上比较，primary的连续MAE比wrong低0.526540，总准确率高41.4634pp。相对low-head的全498比较为MAE−1.163881、总+43.1727pp。
相同强度的native-logit温度及KL温度均未复制目标区域的失败类收益；各温度下fail准确率仍为0%，primary为53.2353%。这支持区域证据方向的作用，仍不能将本适配下的收益推广成所有输入或真实物理任务的因果理解。

primary五档分布为215/179/11/29/64，中间档219条。315配对的negative/0/1/2/3/4由baseline的0/313/2/0/0/0变为7/126/29/23/86/44，mean progress gap从0.006525到0.387053；排序解除了大量平分，同时出现7个负序，保留完整分布。


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


## 2026-09-09 17:17：逐邻居补充与任务异质性

新增`summarize_frozen_report.py`，只消费已完成的冻结分析JSON，可处理单case、全九项确认及846汇总索引；不读新性能作选择，不改冻结推理/统计源码。它输出所有邻居的连续ordinal与五档/原生MAE、两套准确率、两baseline变化、每task方向、全部控制分布和task-macro。已真实运行Meter/RR image_text两项，输出各自`readable_supplement_v1.{json,md}`；两项全部邻居五档MAE也下降。最终全体结果仍须运行此补充。

独立确认498实际上包含**27个有样本的task**，不是原1213中的34个task全部都有。这里的task-macro按这27个task等权；不将缺席task补零，也不冒称论文23子集Overall。Meter image_text primary五档MAE的micro为2.351406→1.004016，task-macro为2.282299→1.146488；27task的五档MAE为24改善/3平/0恶化，连续ordinal MAE为26改善/0平/1恶化（task2_3），准确率为26改善/1平/0恶化。

RR image_text primary的task-macro MAE为1.242741→0.843389，与micro .833333→.668675同向，但27task中MAE是17改善/0平/10恶化，准确率20改善/2平/5恶化。准确率恶化task为task2_2、task2_3、task2_4、task2_5、task3_7；完整MAE恶化名单和逐task变化在补充JSON。总体通过点门槛不代表每个task改善。以上是描述性异质性补充，不新增逐task显著性选择或改变验收条件。


## 2026-09-09 17:20：RR text_image确认完成，点门槛通过但严格原方法比较失败

完整498确认报告：`RESEARCH/confirmation_v1/rr_text_image/individual_case_analysis_v1.json`。同参k40/48/56都通过两套baseline四向/+10pp与same-k原方法MAE/总准确率改良；相对native的总增益分别+27.51004/+30.92369/+34.53815pp，suc+8.22785/+1.89873/+3.79747pp，fail+36.47059/+44.41176/+48.82353pp，MAE变化−.578313/−.622490/−.650602。primary固定k48，未换成确认表现更高的邻居。

primary原生MAE1.540161→.917671，总准确率19.0763%→50.0000%，suc50.0000%→51.8987%，fail4.7059%→49.1176%；同读出baseline的MAE1.536145、准确率相同，两个baseline分数有2条差异。primary的suc MAE反而从1.063291增至1.316456：虽然端点正确数79→82，仍有27条suc被预测为1，不能把总MAE下降写成两类MAE都下降（冻结要求为总MAE下降与两类准确率提高）。task-macro MAE1.666165→1.023691。

primary相对native的95%视频簇区间：MAE变化[−.747036,−.498105]，总准确率[+25.85859,+35.95977]pp，suc[−6.87500,+10.66730]pp，fail[+37.92049,+50.75075]pp。四方向IUT p=.388080596，未达到联合统计显著；全九项Holm仍待其余case。

**严格原方法比较为False。** primary对same-k原bias k48是MAE−.078313、总+2.20884pp，但相对开发预先选定的原bias k24是MAE+.006024、总−4.41767pp（原k24 MAE .911647、总准确率54.4177%），suc+3.16456pp而fail−7.94118pp。因此该case只通过冻结的same-k改良点门槛，不能声称胜过事先选定的较优原方案；不根据确认表现换k来消除这个失败。

工程审计8964/8964期待键全部尝试、8954有效，10个错误仅为5条wrong正负支路几何不可行；没有重复或科学冲突。两个zero各498精确。wrong比较明确使用493条可行子集，primary对wrong的MAE−.809331、总+43.81339pp；对low全498为MAE−.477912、总+27.10843pp。

primary五档分布194/155/29/20/100，中间档204条。315配对negative/0/1/2/3/4由22/94/24/92/78/5变为14/69/35/33/91/73，mean progress gap .342857→.515079。完整控制、逐task/split与预测分布仍在报告，不选择性删除低收益或失败对照。


## 2026-09-09 17:30：Qwen image_text/text_video独立确认完成

两项的所有冻结邻居都通过两baseline与same-k原方法点门槛，primary也都胜过开发预先选定的原方法（strict_original=True）。仍保持primary为各自中间k，不根据确认最高准确率重选。五项已完成case的点门槛均通过；尚未完成全部九项、Holm或整体主线验收。

|输入|k|primary|MAE|总准确率|suc准确率|fail准确率|对native总变化|
|---|---:|---|---:|---:|---:|---:|---:|
|image_text|native|False|1.451807|23.8956%|56.9620%|8.5294%|+0.00000pp|
|image_text|32|False|0.861446|59.0361%|67.0886%|55.2941%|+35.14056pp|
|image_text|40|True|0.845382|59.4378%|68.3544%|55.2941%|+35.54217pp|
|image_text|48|False|0.877510|60.6426%|59.4937%|61.1765%|+36.74699pp|

qwen_image_text primary的95%视频簇区间（准确率为比例单位）为`{"mae": [-0.7336065573770492, -0.4816955684007707], "accuracy": [0.30916030534351147, 0.40227870760199225], "suc_accuracy": [0.026139546727782025, 0.2], "fail_accuracy": [0.4093563235010603, 0.5259978828510935]}`；对两baseline取较大的四方向IUT p=0.007999600，Holm待全九项。native/readout分数不同数为14，两baseline分别比较。

原始支路15438/15438全部尝试，15428有效，无重复或科学冲突；zero各498精确。错误仅为wrong正负支路几何不可行：`{"confirmation_wrong_target_all_frames_b3_k40_plus": {"ValueError('No equal-area disjoint same-frame wrong control')": 5}, "confirmation_wrong_target_all_frames_b3_k40_minus": {"ValueError('No equal-area disjoint same-frame wrong control')": 5}}`。全部默认读出的22893个有效结果和15个预期错误传播均可从保存的原生logits/冻结配方精确重算，差异0。

primary五档分布`{"1": 203, "2": 85, "3": 34, "4": 50, "5": 126}`，中间档169条；315配对分布`{"negative": 7, "0": 56, "1": 35, "2": 29, "3": 70, "4": 118}`，mean progress gap=0.604762。完整两baseline/全部邻居/task-macro/异质性与控制见各case的`readable_supplement_v1.json`及原报告。

|text_video|native|False|1.413655|15.2610%|42.4051%|2.6471%|+0.00000pp|
|text_video|40|False|0.931727|51.0040%|68.3544%|42.9412%|+35.74297pp|
|text_video|48|True|0.937751|49.5984%|67.7215%|41.1765%|+34.33735pp|
|text_video|64|False|0.923695|50.0000%|65.8228%|42.6471%|+34.73896pp|

qwen_text_video primary的95%视频簇区间（准确率为比例单位）为`{"mae": [-0.6113362450243255, -0.3422653116929475], "accuracy": [0.28865979381443296, 0.39759036144578314], "suc_accuracy": [0.15527950310559005, 0.35], "fail_accuracy": [0.3219814241486068, 0.44846796657381616]}`；对两baseline取较大的四方向IUT p=0.000049998，Holm待全九项。native/readout分数不同数为9，两baseline分别比较。

原始支路8964/8964全部尝试，8956有效，无重复或科学冲突；zero各498精确。错误仅为wrong正负支路几何不可行：`{"confirmation_wrong_target_last_frame_b3_k48_plus": {"ValueError('No equal-area disjoint same-frame wrong control')": 4}, "confirmation_wrong_target_last_frame_b3_k48_minus": {"ValueError('No equal-area disjoint same-frame wrong control')": 4}}`。全部默认读出的15924个有效结果和12个预期错误传播均可从保存的原生logits/冻结配方精确重算，差异0。

primary五档分布`{"1": 151, "2": 178, "3": 14, "4": 21, "5": 134}`，中间档213条；315配对分布`{"negative": 6, "0": 81, "1": 27, "2": 20, "3": 86, "4": 95}`，mean progress gap=0.553968。完整两baseline/全部邻居/task-macro/异质性与控制见各case的`readable_supplement_v1.json`及原报告。


### 无标签分数组合复算的范围

新增`audit_score_recombination.py`对已完成case的所有默认读出，重新执行冻结的raw-logit组合、原生readout、置信度缩放及KL诊断，只比较工程字段，不读取labels。前三项Meter image_text、RR image_text、RR text_image分别精确重现15918/23406/15921个有效读出，以及18/0/15个原支路无效造成的读出错误，差异均0；Qwen两项同样通过。它是离线组合的实际确定性复算，不等于重新跑了一次神经推理。各case输出`completed_recombination_audit_v1.json`，尚未完成的case待补。

RR image_text主方案名义lambda16，实际因原生评分期望调节，498条有效gain中位13.457202、均值10.665494、最小.00001067、最大15.999813；不能把它写成每条固定lambda16。其baseline五档规范格式概率最小.999977；RR text_image最小.999990。浮点累计略大于1（最大1.000001907）的原始值保留，不裁剪。此概率是输出格式/选项质量，不是正确性置信度。


## 2026-09-09 17:36：Meter text_video确认通过，温度对照揭示读出校准成分

`confirmation_v1/meter_text_video/individual_case_analysis_v1.json`覆盖498条。三个冻结邻居k40/48/56都通过两baseline、same-k原方法点门槛，primary k48也胜开发选定原方法（strict_original=True）。全部邻居的五档MAE另外也下降；primary保持k48，不换成确认准确率更高的k40。

|条件|连续ordinal MAE|五档MAE|总准确率|suc准确率|fail准确率|总准确率0.2/0.8|
|---|---:|---:|---:|---:|---:|---:|
|native baseline|1.535920|1.570281|5.4217%|5.6962%|5.2941%|31.1245%|
|同读出baseline|1.516293|1.572289|5.2209%|5.0633%|5.2941%|30.9237%|
|k40|.998509|.907631|56.4257%|34.1772%|66.7647%|68.6747%|
|primary k48|1.002114|.943775|51.2048%|37.9747%|57.3529%|66.2651%|
|k56|1.027894|.945783|49.1968%|36.0759%|55.2941%|63.0522%|

三个邻居对native总增益+51.00402/+45.78313/+43.77510pp。primary相对native的95%视频簇区间：连续MAE[−.663166,−.406421]，总准确率[+40.66798,+50.91659]pp，suc[+23.12500,+41.33333]pp，fail[+45.05814,+58.85886]pp，IUT p=.000049998；九项Holm待剩余三项。native与typed baseline一条分数不同，两个口径独立比较，不能把typed修正计为attention收益。

**温度控制能够复制很大一部分端点准确率增益。** native logits乘tau2/4/8/16的总准确率分别36.5462/49.5984/53.2129/54.8193%，对应连续MAE1.305854/1.173296/1.121456/1.108130。tau8/16的总准确率高于primary 51.2048%，而primary连续MAE更低。tau16的suc51.2658%、fail56.4706%，相比primary体现成功类更高、失败类略低的权衡；KL温度delta4同样总55.0201%、连续MAE1.110961。因此此输入的准确率改善不能全部归因于区域信息，不能声称attention在所有指标优于温度校准。温度不包含attention干预，不计作主方法跨模型成功，但其已冻结的完整结果必须报告。

primary分布230/139/20/37/72，中间档196条。315配对negative/0/1/2/3/4由11/94/59/119/32/0变为23/81/33/42/78/58，mean progress gap .361086→.415573；负序从11增至23，不能泛称配对判别全面改善。

工程方面8466期待原始键全部尝试，无重复冲突，两个zero各498精确；wrong几何失败对应8条原支路错误、12个读出错误。15426有效读出与12个预期错误传播均从raw logits和冻结配方精确复算，差异0。完整逐邻居/逐task/两阈值与控制见`readable_supplement_v1.{json,md}`及原报告。

Qwen text_video的remaining114也已完成：1482/1482原始支路有效、零干预114精确，native/readout差1条；1140/1140默认读出精确复算，无冲突。它仍属于含ranking重叠的描述性补全集，不能增加独立确认样本量。


## 2026-09-09 17:41：剩余114补跑转GPU2与自动raw审计

GPU2的Qwen text_video确认/114均完成后，只剩其它程序约20.6GiB占用，四卡均A100-SXM4-80GB。保存`RESEARCH/cohort_gpu2_handoff_20260909T1740_v1/plan.json`及launch/events，核对/proc命令、cwd和空children后，仅停止原114等待父2039243与2039245；**没有中断RR/Meter/Qwen/SOLE任何正在推理的模型，也没有停止确认父2037992/2037994**。六个Meter/RR 114case均确认此前从未launch，全部仍使用原冻结manifest/YAML/ranking。原冻结推理文件SHA再次全部一致。

新Meter114上层父2081412，在GPU2依次运行image_text/text_video/video_text；新RR114上层父2081413，等Meter image_text114完成后在GPU2依次运行image_text/text_image/text_video，避免同时模型加载。每个队列仍逐项检查实时显存，实际模型PID见该目录events。无需等待同模型全部498确认完成才对固定114前向：样本集不交叉，参数已冻结，114性能不会用于选型。这样可在确认尚运行时提前补齐数据，并在Meter确认结束后让GPU3只执行SOLE full。Qwen image_text114仍在GPU1，video_text确认/114仍按原接管计划GPU0执行。

新增自动raw审计守护PID2081699：`RESEARCH/final_raw_audits_v1/{plan.json,launch.json,events.jsonl}`。对确认九项+remaining114九项共18个case，已有14份audit完成，剩余22份工程/重组合审计将在各case的真实completion/combination标记出现后逐项执行。没有标签选择，不跑新的神经前向；任何错误保留退出码且不会静默重试。

为复用既有守护，`run_completion_analyses.py`只增量添加可选`--plan-root`，默认行为相同；原PID2040764及原九任务plan没有重启或修改。该调度脚本不属于冻结推理21文件，也不属于被冻结的metrics/paired_analysis/analyze_development三个统计实现。最终仍需检查两个watcher均正常、收取全体确认与846、最终BASE；再手动生成macro、`summarize_frozen_report`全邻居补充及图表并查看图片。


## 2026-09-09 17:52：Qwen三个输入全部通过冻结确认点门槛

最后完成的Qwen video_text三个邻居k64/68/72均通过两baseline、same-k原方法改良，primary k68也胜开发预先选定原方法（strict_original=True）。因此Qwen已有三个输入各三个冻结邻居的独立确认点估计支持；全九项与主线总体验收仍未完成。

|条件|原生MAE|总准确率|suc准确率|fail准确率|对native总增益|
|---|---:|---:|---:|---:|---:|
|native_generation_baseline|1.473896|28.5141%|51.8987%|17.6471%|+0.00000pp|
|baseline|1.483936|29.1165%|51.8987%|18.5294%|+0.60241pp|
|contrast_target_kl_temporal_all_frames_b3_k64_delta0.5|1.040161|59.2369%|67.0886%|55.5882%|+30.72289pp|
|primary k68|1.018072|59.0361%|67.7215%|55.0000%|+30.52209pp|
|contrast_target_kl_temporal_all_frames_b3_k72_delta0.5|1.026104|58.2329%|65.8228%|54.7059%|+29.71888pp|

primary对native的95%视频簇区间（准确率单位为比例）：`{"mae": [-0.5520181322187879, -0.35772357723577236], "accuracy": [0.2587251220114003, 0.3518133538655926], "suc_accuracy": [0.08333333333333333, 0.23417721518987342], "fail_accuracy": [0.31420744391065325, 0.43352601156069365]}`。四方向IUT p=0.000149993，全九项Holm待最后Meter/RR各一项。

KL调节在498条primary上全部实际达到预算.5 nats，gain上限64的触顶数为0。有效gain最小.200812、中位.852356、均值1.400577、p90=2.623946、最大40.662422；这一实际变化不能等同固定lambda，也不证明相对真值的误差被KL直接约束。

原始15438期待键全部尝试，15410有效，无重复冲突；28个错误来自14条wrong正负支路几何不可行，wrong比较可行样本484/498。四个zero各498精确，native/readout分数差17条。22866个有效读出与42个预期错误传播均可精确复算，差异0。

native_generation_baseline五档分布`{"1": 63, "2": 85, "3": 153, "4": 75, "5": 122}`，315配对`{"negative": 25, "0": 69, "1": 44, "2": 71, "3": 74, "4": 32}`、mean progress gap=0.394444。
contrast_target_midpoint_kl_temporal_all_frames_b3_k68_delta0.5五档分布`{"1": 210, "2": 49, "3": 67, "4": 26, "5": 146}`，315配对`{"negative": 22, "0": 59, "1": 14, "2": 47, "3": 42, "4": 131}`、mean progress gap=0.560317。

全部控制、task-macro、各task方向和三个邻居见该case的individual_case_analysis_v1.json及readable_supplement_v1.{json,md}。confirmation补充不用于重新调整配置；该case的remaining114已经按原Qwen GPU0接管计划自动接续。


## 2026-09-09 18:09：RR text_video冻结独立确认失败，不能宣布三模型×三输入通过

全部498推理/读出完成、zero与重复审计通过，但`confirmation_case_point_acceptance=False`，strict_original也为False。三个冻结邻居对两套baseline均满足四向/+10pp：相对native的k48/56/64总增益+15.66265/+11.44578/+13.25301pp，MAE变化−.728916/−.775100/−.769076，suc+3.79747/+3.79747/+4.43038pp，fail+21.17647/+15.00000/+17.35294pp。

**失败来自primary k56对same-k原方法的MAE。** 新方法MAE1.32730923695，original k56为1.32530120482，变化+.00200803213，即498条绝对误差总和多1；虽然总准确率比original高13.85542pp，仍未满足预先冻结的MAE严格下降条件。不能因为只差1分而改容差、换成k48/k64或只报告对native的正结果。k48/k64的same-k MAE分别下降.014056/.050201；这也不能替代要求三个冻结邻居都通过。

开发预先选定的original k40更严格对照同样失败：primary MAE+.018072、总准确率+12.248996pp，体现MAE/端点准确率权衡。primary保持k56，官方记录中的scope为**all_frames**；任何简写将其写成last_frame均以冻结YAML和真实condition `contrast_target_temporal_all_frames_b3_k56_lambda4`为准，不更改运行设置。

primary原生baseline→新方法：MAE2.102410→1.327309，总准确率19.8795%→31.3253%，suc62.6582%→66.4557%，fail0%→15%。同读出baseline准确率20.2811%、suc63.9241%，也单独过四向点门槛。成功类MAE由.683544增至.911392，因此不能把总MAE下降写成两类MAE均下降。

primary对native的95%视频簇区间（准确率单位为比例）为`{"mae": [-0.9127007198228129, -0.6365420546916335], "accuracy": [0.07889546351084813, 0.1512770137524558], "suc_accuracy": [-0.050314465408805034, 0.12804878048780488], "fail_accuracy": [0.1069364161849711, 0.195906432748538]}`；对两baseline的四向IUT p=.337733113，成功类改善不确定，九项Holm仍待最后Meter case。

原始支路15438/15438全部尝试，15410有效，无重复或科学冲突；各zero498精确，native/readout分数差6条。全部22866有效读出与42个预期错误传播精确复算，差异0。

当前完成8项，7项通过冻结point、1项失败；RR只有image_text/text_image两项通过point，其中text_image不胜开发选定原方法。**无论最后Meter结果如何，原九项方案都不能取得“独立确认三模型各三输入均通过冻结验收”的结论。** 原确认manifest、参数和所有失败永久保留；完整846的描述性汇总及SOLE mandatory全七步仍继续。后续新方案如使用已见过结果的498，只能明确标为确认后探索/重复使用验证集，不得冒称新的独立确认。


## 2026-09-09 18:16：确认后的新探索——加入原方法的完整评分分布参照

原九项冻结确认已经出现RR text_video的严格同k MAE失败；这些数据、参数、primary和验收不变，后续不会把新方案的结果回写为v1确认成功。按用户要求继续进行理论—实现—实验迭代，同时明确证据阶段：后续498只能叫重复使用验证集上的探索，不能叫新的独立确认。

提出并在计算本候选性能前保存`post_confirmation_anchor_development_v1/prospective_plan_v1.json`及源码SHA。令z_parent为原冻结区域/时间证据方案，z0为同读出无干预分布，z_original为相同k的原强bias分布，新增：

`z_anchor = z_parent + ρ (z_original − z0)`，`ρ∈{0,.25,.5,.75,1}`。

等价于原生分布与原方法分布在logit空间作凸组合，再施加原来的区域证据项；固定原η时，对全部评分档位有`p_anchor(j)∝p0(j)^(1−ρ) p_original(j)^ρ exp(ηD(j))`。该代数表达没有指定某个端点。对ρ的log概率导数是`(z_original−z0)_j−E_p(z_original−z0)`；它描述原方法提供的整体评分方向，不保证任何真值指标单调改善。原强bias可能保留更好的序数信息，也可能再次损害suc，因此两类、两baseline、原方法和整个k邻域仍需一起检验。分布代数可对照本轮已核实的ASCD/VCD机制背景，不能据此冒称独立校准的Bayes证据或引用文献证明本任务效果。

本轮仅在development234上运行：保留原九个模型/输入、每项三个k及中间primary，原radius/scope/gain/置信度/KL配置均不变；新增加的唯一变量是每case共用一个ρ。全部九项、所有五个ρ都保存，ρ0必须精确重现原父方案；先完成无标签评分组合，再join开发labels做评价。所有父sweep的SHA逐项核验，只读复用，不做新神经前向，不改冻结的21个推理文件和三个统计模块。

选型需所有三个k完整、对两baseline四向/+10pp、对同k原方法MAE/总准确率改良、全部邻居五档MAE下降，且primary胜开发预先选定的原方法。在合格ρ中先最大化最弱同k原方法MAE改良余量，再最大化最弱同k准确率余量，平局取较小ρ。**这是观察到v1确认失败后的开发决策，不是最初的预注册，也没有消除后来复用498的适应性偏差。** 当前计划已冻结并实际启动开发CPU组合，尚未宣称新方法成功。


## 2026-09-09 18:41：后续探索设置冻结，补充完整方案控制

开发234上的所有9case×5个ρ×3个k已完成；ρ0在9×3×234=6318个父方案读出上精确一致，全部新候选评分先无标签生成再分析。9项均存在满足该轮开发门槛的ρ；依预先保存的余量规则，选中：

|case|冻结ρ|原primary k（不变）|
|---|---:|---:|
|meter_image_text|1|68|
|meter_text_video|0|48|
|meter_video_text|1|64|
|qwen_image_text|0|40|
|qwen_text_video|0|48|
|qwen_video_text|0|68|
|rr_image_text|0|64|
|rr_text_image|.25|48|
|rr_text_video|.25|56|

开发的最弱same-k原方法MAE改良余量：Meter image_text由ρ0的.520877到ρ1的.524129，Meter video_text由.496034到.512527，RR text_image由.085470到ρ.25的.119658，RR text_video由.047009到.064103。所有ρ结果均在`post_confirmation_anchor_development_v1/{case}/development_analysis_v1.json`，包括RR text_video的ρ.75/1不合格；没有只保留获选行。

完整评估已经一次性冻结于`post_confirmation_anchor_evaluation_v1/frozen_manifest_v1.json`。守护PID2098349，目录`post_confirmation_anchor_evaluation_watcher_v1`，等全部原始9项确认与9项114完成后统一执行`anchor_evaluation.py`。它生成所有选中设置在development234、reused_validation498、ranking_overlap_remainder114上的评分，再统一join labels，分别报告三分区和full846；ρ不再调整。后续498明确为reused validation，不是新的独立确认，原v1失败不回写。

方法解释的两条新增边界：ρ>0时加入原强bias的响应，**只有对称证据支路保持局部视觉总量守恒，不能再把整个后续方案称为全部支路守恒**。一般情况下，父方案的KL预算也不约束额外加ρ项后的整体分布；实际选中带KL的Qwen video_text为ρ0，故其原.5预算保持原样。非零ρ部署时比父区域/时间方案各增加一条original-target模型前向，即区域4条、时间6条；ρ0仍为3/5条。

为了不把“固定正确目标anchor、仅换证据部分”的条件控制误当完整wrong/low控制，又前瞻冻结并启动`post_confirmation_anchor_whole_controls_v1/frozen_manifest_v1.json`。仅四个非零ρcase各498，5条真实支路：matched baseline duplicate、original-target duplicate、original-bias zero、original-bias wrong、original-bias low heads。GPU0父2101487运行Meter两项，GPU2父2101488运行RR两项。没有修改原冻结推理文件，21项SHA已再次核验相同。

分析协议`post_confirmation_anchor_whole_controls_v1/analysis_protocol_v1.json`要求新旧baseline/原target logits与progress精确一致，original-bias zero精确，全target组合重现主评估、全zero重现baseline。结合已保存的父wrong/low证据，可以同时替换两部分为错误区域或低heads，构成完整方案控制；几何不可行按明确子集比较，不删主方法样本。新原始支路及组合审计8项，加全方案分析1项，守护PID2103136，目录`post_confirmation_anchor_whole_control_analyses_v1`。若任何parity失败，保存工程记录并不计算其效应解释。

截至本记录，原114九项全部已完成；原确认仅Meter video_text尚在运行（18:38约371/498），SOLE interleaved/video_text完整七步仍未结束。新方案的完整评估尚未运行，不声称通过最终目标。


## 2026-09-09 19:08：原确认、完整846及第一次确认后探索均已完成

原独立确认九项全部完成；8项通过冻结点门槛，RR text_video失败。Meter与Qwen各三输入通过；RR只有两输入，且text_image不胜开发预选原方法，故三模型整体目标未达。以下MAE对Meter用连续ordinal口径，对Qwen/RR用原生1–5；全部邻居的五档MAE也均比两个baseline下降。

|case|点门槛|严格原方法|MAE变化vs native|总增益pp|suc增益pp|fail增益pp|IUT p|九项Holm p|
|---|---|---|---:|---:|---:|---:|---:|---:|
|meter_image_text|True|True|-1.212856|46.98795|33.54430|53.23529|0.00005000|0.00044998|
|meter_text_video|True|True|-0.533805|45.78313|32.27848|52.05882|0.00005000|0.00044998|
|meter_video_text|True|True|-0.804366|44.77912|18.35443|57.05882|0.00005000|0.00044998|
|qwen_image_text|True|True|-0.606426|35.54217|11.39241|46.76471|0.00799960|0.03199840|
|qwen_text_video|True|True|-0.475904|34.33735|25.31646|38.52941|0.00005000|0.00044998|
|qwen_video_text|True|True|-0.455823|30.52209|15.82278|37.35294|0.00014999|0.00074996|
|rr_image_text|True|True|-0.164659|11.44578|3.79747|15.00000|0.12104395|0.36313184|
|rr_text_image|True|False|-0.622490|30.92369|1.89873|44.41176|0.38808060|0.67546623|
|rr_text_video|False|False|-0.775100|11.44578|3.79747|15.00000|0.33773311|0.67546623|

Holm仅针对预声明primary对两个baseline的四方向IUT；它不检验总准确率增益95%下界>10pp，也不能替代同k原方法的改良判断。RR三项的联合方向均无校正后显著证据。

最后Meter video_text：native端点准确率0%，497条预测3档、1条4档；primary k64绝对总准确率44.7791%，suc18.3544%、fail57.0588%，第二阈值54.2169%。三个k56/64/72连续MAE分别1.252926/1.265222/1.357778，五档MAE1.204819/1.228916/1.347390。315配对中负序由0增至39、平分312降至119，mean progress gap .011800增至.245616，不能宣称pairwise全部改善。四组zero各498精确，14940期待支路全部尝试，22398有效读出及12个无效传播精确复算。

原full846结果为Meter/Qwen各三项、RR text_image/text_video两项通过；RR image_text k60总增益只有9.21986pp，k64/k68为10.87470/10.16548pp，不能删除k60或移动已冻结邻域。原full846的RR text_video通过也不能覆盖其498确认失败。

第一次确认后anchor探索：保持原九项、三k、primary及其它参数，仅以234开发集选择ρ。重复498上9项均通过same-k点门槛，但严格开发预选原方法仅Meter/Qwen六项与RR image_text通过；RR text_image的primary对预选original总准确率持平，RR text_video的MAE持平，严格下降/上升不能改成不劣。完整846只有Meter/Qwen六项与RR text_image通过，因此这一新版本也未完成整体目标。

|anchor case|ρ|846点门槛|498重复验证点门槛|846主MAE|846总准确率|846 suc|846 fail|498主MAE|498总准确率|
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
|meter_image_text|1|True|True|1.088400|44.4444%|32.0896%|50.1730%|1.036388|47.9920%|
|meter_text_video|0|True|True|1.013422|51.7730%|37.3134%|58.4775%|1.002114|51.2048%|
|meter_video_text|1|True|True|1.240332|46.3357%|21.2687%|57.9585%|1.243225|46.7871%|
|qwen_image_text|0|True|True|0.916076|56.5012%|64.5522%|52.7682%|0.845382|59.4378%|
|qwen_text_video|0|True|True|0.979905|48.4634%|64.9254%|40.8304%|0.937751|49.5984%|
|qwen_video_text|0|True|True|0.997636|58.3924%|66.4179%|54.6713%|1.018072|59.0361%|
|rr_image_text|0|False|True|0.684397|75.2955%|62.6866%|81.1419%|0.668675|75.5020%|
|rr_text_image|0.25|True|True|0.934988|52.9551%|50.7463%|53.9792%|0.899598|54.4177%|
|rr_text_video|0.25|False|True|1.336879|30.1418%|63.0597%|14.8789%|1.309237|30.5221%|

anchor完整846的RR image_text仍因k60总+9.21986pp失败；RR text_video三个k相对matched baseline的suc变化为−1/268、−1/268、0，均不满足严格提高。该114剩余分区只有8任务且与ranking视频重叠，不能删掉它来声称在完整846稳定有效。

四项完整控制的新旧baseline/原target、original-zero、全target重现与全zero全部精确通过。完整wrong/low控制同时替换区域证据与anchor参照；下表在明确可行子集上比较，主方法仍保留全部498。

|case|控制|可行/498|primary−control MAE|总pp|sucpp|failpp|
|---|---|---:|---:|---:|---:|---:|
|meter_image_text|full_anchored_wrong|492/498|-0.530601|42.47967|32.89474|46.76471|
|meter_image_text|full_anchored_low_rank|498/498|-1.169572|43.97590|20.88608|54.70588|
|meter_video_text|full_anchored_wrong|490/498|-0.148597|9.79592|12.00000|8.82353|
|meter_video_text|full_anchored_low_rank|498/498|-0.890571|45.38153|16.45570|58.82353|
|rr_text_image|full_anchored_wrong|493/498|-0.849899|47.46450|49.67320|46.47059|
|rr_text_image|full_anchored_low_rank|498/498|-0.495984|29.71888|22.15190|33.23529|
|rr_text_video|full_anchored_wrong|484/498|-0.357438|-3.09917|58.90411|-29.88166|
|rr_text_video|full_anchored_low_rank|498/498|-0.813253|16.66667|26.58228|12.05882|

RR text_video相对whole-wrong的MAE更低且suc更高，但总准确率低3.09917pp、fail低29.88166pp，说明控制也体现明显类别权衡，不能说正确区域在所有指标支配wrong。Meter video_text相对whole-wrong总增益9.79592pp，勿四舍五入称+10pp。

结果入口（RESEARCH下）：原确认`confirmation_v1/confirmation_analysis_20260909T190153.json`；原846 `frozen_cohort_completion_v1/full_cohort_analysis_20260909T190156/summary.json`；anchor `post_confirmation_anchor_evaluation_v1/summary_v1.json`；完整控制`post_confirmation_anchor_whole_controls_v1/whole_method_control_analysis_v1.json`。上述三份主结果均已生成全部case/邻居readable_supplement与task_macro补充。确认task-macro分母27、full846分母28。

原确认图为`confirmation_v1/figures_20260909T190421/primary_effects.{png,svg}`；anchor图为`post_confirmation_anchor_evaluation_v1/figures_20260909T190347/all_selected_neighbor_effects.{png,svg}`，已逐图查看。原图早一版图例遮挡末行，保留早版并仅修正新图的图例位置，数字与推理未改。anchor图只有全部k的观测点，无独立确认区间。

18项原确认/remaining114的36份raw审计汇总于`final_raw_audits_v1/all_eighteen_case_audit_index_v1.json`，全部确定性读出精确、科学字段冲突0；四个新控制另8份审计亦通过。这些不等于独立神经复现；四个新控制的真实重复前向parity另有上述证据。

两个mandatory SOLE全七步仍未结束，主线1继续。三模型方案继续研究，不修改上述已冻结版本或把已见498/846重新称独立确认。


# 共享熵调节方案：已冻结的实验结果（2026-09-09 19:32）

本轮获得了满足数据点估计验收的方案：Meter、Qwen、RoboReward各三个输入，在完整grounded846与重复验证498上，所列三个已测k均对两种baseline做到MAE下降、suc/fail准确率提高、总准确率至少+10个百分点，并改善同k原attention方法；primary也胜开发预选原方法。**这是多轮探索后的观测结论，不是新的独立确认。** 原九项独立确认失败永久保留，不能回写成成功。主线1仍待SOLE video_text/interleaved两个完整七步实验结束，总任务尚未完成。

## 方法及固定设置

令z0为同读出原生全档位logits，z_parent为原冻结区域/时间证据方案，z_original为相同k的原强bias输出，B为原生档位数，h=H(softmax(z0))/log(B)。最终使用：

`z_final = z0 + s h^β (z_parent − z0) + ρ (z_original − z0)`。

其中`z_parent−z0=η[(z_obs,+−z_obs,−)−α(z_static,+−z_static,−)]`。α=0为区域差分，α=1为扣除首帧重复参考的时间区域差分；η按原开发配置固定。全部原生bin参与softmax/读出，不把预测硬映射到端点。熵只缩放区域证据，其标量对档位置换对称；原parent本身仍可能使用完成度置信度，不能把整个ordinal模型称为置换不变。熵不是正确性标签，过度自信的错误可能被保留。

RoboReward三个输入共用β=.5，通过开发234的共同最弱余量选择；各输入的s和ρ仍由同一开发集选择。Meter/Qwen保留先前开发选定的六项设置。共享β是在上一轮RR text_video验证失败后提出的结构约束，因此即使没有用新设置的498/846指标选参数，研究整体仍属于确认后适应性探索。

|模型/输入|三个已测k（primary居中）|s|β|ρ|
|---|---|---:|---:|---:|
|meter_image_text|64,68,72|1|0|1|
|meter_text_video|40,48,56|1|0|0|
|meter_video_text|56,64,72|1|0|1|
|qwen_image_text|32,40,48|1|0|0|
|qwen_text_video|40,48,64|1|0|0|
|qwen_video_text|64,68,72|1|0|0|
|rr_image_text|60,64,68|1|0.5|0|
|rr_text_image|40,48,56|4|0.5|0.25|
|rr_text_video|48,56,64|4|0.5|0|

原parent参数均不变：Meter image_text为region/all_frames/b3/λ64，text_video为region/last_frame/b3/λ12，video_text为temporal/last_frame/b3/λ64；Qwen image_text为temporal/all_frames/b3/λ2，text_video为region/last_frame/b3/λ4，video_text为temporal/all_frames/b3/KL=.5（gain cap64）；RR image_text为temporal/last_frame/b3/λ16×(1−q_native)，text_image为region/all_frames/b3/λ2，text_video为temporal/all_frames/b3/λ4。RR text_video不是last_frame。精确分支名和完整参数见九份最终YAML配方。

上述k稳定性仅支持所列三个已测点，未宣称每一个未测整数k都通过。Meter官方输入text_image未入选；其三个成功输入属于适配输入。Qwen的video_text及RR公开HELM请求形式text_video有官方输入来源。SOLE作为第四模型的五输入数值screen没有通过，保留为边界。

## 完整grounded846：299视频簇、28任务

下表为每个固定primary；所有三个k的完整指标均保存在readable补充和原分析。MAE对Meter为连续ordinal值，对Qwen/RR为原生1–5值；Meter五档MAE另表。百分数前后为native baseline→新方案。

|case|MAE native→新|总准确率 native→新|suc native→新|fail native→新|总增益pp|
|---|---:|---:|---:|---:|---:|
|meter_image_text|2.258014→1.088400|0.0000%→44.4444%|0.0000%→32.0896%|0.0000%→50.1730%|44.44444|
|meter_text_video|1.515647→1.013422|5.6738%→51.7730%|5.2239%→37.3134%|5.8824%→58.4775%|46.09929|
|meter_video_text|2.072061→1.240332|0.0000%→46.3357%|0.0000%→21.2687%|0.0000%→57.9585%|46.33570|
|qwen_image_text|1.431442→0.916076|22.8132%→56.5012%|55.9701%→64.5522%|7.4394%→52.7682%|33.68794|
|qwen_text_video|1.375887→0.979905|15.6028%→48.4634%|43.6567%→64.9254%|2.5952%→40.8304%|32.86052|
|qwen_video_text|1.452719→0.997636|26.5957%→58.3924%|52.9851%→66.4179%|14.3599%→54.6713%|31.79669|
|rr_image_text|0.819149→0.598109|64.4208%→79.3144%|59.3284%→62.6866%|66.7820%→87.0242%|14.89362|
|rr_text_image|1.476359→0.848700|18.2033%→61.3475%|49.6269%→61.9403%|3.6332%→61.0727%|43.14421|
|rr_text_video|2.109929→1.283688|19.7400%→35.5792%|62.3134%→65.6716%|0.0000%→21.6263%|15.83924|
## 重复验证498：179视频簇、27任务

下表为每个固定primary；所有三个k的完整指标均保存在readable补充和原分析。MAE对Meter为连续ordinal值，对Qwen/RR为原生1–5值；Meter五档MAE另表。百分数前后为native baseline→新方案。

|case|MAE native→新|总准确率 native→新|suc native→新|fail native→新|总增益pp|
|---|---:|---:|---:|---:|---:|
|meter_image_text|2.255305→1.036388|0.0000%→47.9920%|0.0000%→33.5443%|0.0000%→54.7059%|47.99197|
|meter_text_video|1.535920→1.002114|5.4217%→51.2048%|5.6962%→37.9747%|5.2941%→57.3529%|45.78313|
|meter_video_text|2.069588→1.243225|0.0000%→46.7871%|0.0000%→18.3544%|0.0000%→60.0000%|46.78715|
|qwen_image_text|1.451807→0.845382|23.8956%→59.4378%|56.9620%→68.3544%|8.5294%→55.2941%|35.54217|
|qwen_text_video|1.413655→0.937751|15.2610%→49.5984%|42.4051%→67.7215%|2.6471%→41.1765%|34.33735|
|qwen_video_text|1.473896→1.018072|28.5141%→59.0361%|51.8987%→67.7215%|17.6471%→55.0000%|30.52209|
|rr_image_text|0.833333→0.610442|64.0562%→78.3133%|58.2278%→61.3924%|66.7647%→86.1765%|14.25703|
|rr_text_image|1.540161→0.833333|19.0763%→61.8474%|50.0000%→60.7595%|4.7059%→62.3529%|42.77108|
|rr_text_video|2.102410→1.271084|19.8795%→35.9438%|62.6582%→66.4557%|0.0000%→21.7647%|16.06426|

所有27个candidate点在上述两集合均通过matched-readout baseline和same-k原方法门槛，不只是表中的native比较。完整846含开发234与ranking重叠114；该114仅8任务，多项RR单独在114仍失败，不能说每个任务或每个分区都改善。该分区没有从总体分母删除。开发与确认都只有27个实际任务，且任务集合并不完全相同；full846为28任务，原始1213为34任务。

## Meter的五档MAE及第二阈值

|case/集合|五档MAE baseline→新|0.2/0.8总准确率 baseline→新|
|---|---:|---:|
|meter_image_text/full_cohort|2.355792→1.061466|0.0000%→56.8558%|
|meter_image_text/reused_validation|2.351406→0.995984|0.0000%→61.4458%|
|meter_text_video/full_cohort|1.562648→0.945626|33.6879%→65.2482%|
|meter_text_video/reused_validation|1.570281→0.943775|31.1245%→66.2651%|
|meter_video_text/full_cohort|1.996454→1.208038|0.0000%→56.2648%|
|meter_video_text/reused_validation|1.997992→1.206827|0.0000%→55.0201%|

Meter全部三个邻居的连续MAE和五档MAE均分别下降；五档结果未代替连续结果。两套阈值的suc/fail与逐任务统计在JSON中全量保留。image_text/video_text的native端点准确率为0，需同时看绝对准确率，不能将很大的相对增益解释为已接近完美。

## 消融、审计与限制

原v1在498为8/9点门槛、三模型整体False；第一次anchor版本在498为9/9点门槛但严格原方法失败两项，full846仅7/9；单独熵选型版本在498/full846均8/9，RR text_video的s=4、β=0损害suc。以上文件和参数全部保留，不以本轮成功改写它们。共享β候选的开发最弱协议余量分别为β0=.0282051、β.5=.0366972、β1=.0244648；β2因RR text_video没有完整合格设置而不准入。新版本没有改primary、删除k、删失败样本或放宽严格不等式。

九项whole-target对评估primary的498条重算和whole-zero回baseline均精确；来自已有真实神经支路与已通过的零干预、原bias重复前向审计，**本轮组合审计没有再次进行九项神经前向**。非零ρ三项的完整original wrong/low来自此前实际补跑；其余ρ0无需该分支。正确区域与wrong/low的比较均列明可行子集，主分母仍498。

|case|完整控制|可行/498|MAE变化|总pp|sucpp|failpp|
|---|---|---:|---:|---:|---:|---:|
|meter_image_text|whole_wrong|492/498|-0.530601|42.47967|32.89474|46.76471|
|meter_image_text|whole_low|498/498|-1.169572|43.97590|20.88608|54.70588|
|meter_text_video|whole_wrong|494/498|-0.595657|29.95951|35.71429|27.35294|
|meter_text_video|whole_low|498/498|-0.615897|47.38956|27.84810|56.47059|
|meter_video_text|whole_wrong|490/498|-0.148597|9.79592|12.00000|8.82353|
|meter_video_text|whole_low|498/498|-0.890571|45.38153|16.45570|58.82353|
|qwen_image_text|whole_wrong|493/498|-0.470588|32.45436|59.47712|20.29412|
|qwen_image_text|whole_low|498/498|-0.532129|27.91165|16.45570|33.23529|
|qwen_text_video|whole_wrong|494/498|-0.572874|17.40891|58.44156|-1.17647|
|qwen_text_video|whole_low|498/498|-0.520080|34.33735|35.44304|33.82353|
|qwen_video_text|whole_wrong|484/498|-0.289256|26.03306|38.35616|20.71006|
|qwen_video_text|whole_low|498/498|-0.212851|19.07631|22.78481|17.35294|
|rr_image_text|whole_wrong|498/498|-0.192771|11.84739|-0.63291|17.64706|
|rr_image_text|whole_low|498/498|-0.198795|11.24498|1.26582|15.88235|
|rr_text_image|whole_wrong|493/498|-0.833671|39.55375|41.83007|38.52941|
|rr_text_image|whole_low|498/498|-0.447791|23.49398|22.78481|23.82353|
|rr_text_video|whole_wrong|484/498|-0.140496|-13.22314|43.83562|-37.86982|
|rr_text_video|whole_low|498/498|-0.696787|13.45382|15.82278|12.35294|

RR image_text相对wrong的suc低1/158，RR text_video相对wrong的总准确率与fail更低，但MAE更低且suc更高。错误区域干预可能造成偏向失败的分布，不能据此声称正确区域全面支配或已建立因果完成度理解。配对负序、平分、正差和逐任务异质性也必须随整体指标一起解读。

熵组合的实际234×9组primary向量已做固定档位置换、共同logit平移、精确零点和原方案退化检查；最大置换误差2.77e−13，零点/退化差异0。数值容差仅用于浮点代数核验，未用于放宽任何性能门槛。全部原生评分档位保留，标签仅在评分保存后参与离线汇总。

部署每次评分的基础区域/时间证据需3/5条前向，ρ>0另加一条original-target。最终Meter image_text需4条、video_text需6条，RR text_image需4条；其他区域/时间设置仍3/5条。熵由已得z0计算，不额外调用模型。仅证据支路有视觉总量守恒；非零ρ的整个方法不是全部支路守恒。Qwen video_text保持s1/β0/ρ0，原KL=.5约束不变；一般新缩放后不能借用父方案KL预算。

参数来自逐轮探索，效应仅在本数据分布与grounded子集观察到。498已经重复使用，full846含开发/ranking，因此不提供新的确认性p值，不把原IUT/Holm转移给新方案。原独立确认中RR的成功类改善CI跨0等不确定性仍应保留。

## 可追溯入口

RESEARCH=`results/mydata_bench/experiments_v2_corssmodel/auto_research/session_20260908`。

- 最终manifest/summary：`shared_entropy_evaluation_v1/{frozen_manifest_v1,summary_v1}.json`；每case `analysis_v1.json`含全部分区、两类/每task/分布/配对/对照。
- 全部邻居及macro：同目录`all_nine_readable_supplement_v1.{json,md}`、`all_nine_task_macro_v1.json`。
- 完整控制：`shared_entropy_whole_controls_v1/{frozen_protocol_v1,analysis_v1}.json`。
- 开发共享选型：`shared_entropy_development_v1/{prospective_plan_v1,joint_selection_v1}.json`；原64组候选全部在`entropy_evidence_development_v1`。
- 九份最终配方：`mydata_bench/configs/v2_crossmodel/addbase_shared_entropy_final_v1_*.yaml`；SHA注册表`shared_entropy_evaluation_v1/final_recipe_registry_v1.json`。它们是方法配方，明确引用真实parent神经配置，不冒充可直接交给score_branches的配置。
- 核心代码：`entropy_development.py`的`entropy_scaled_logits`；独立文件`shared_entropy_followup.py`只做共享选型及对新输出目录的调度，复用已冻结的`entropy_evaluation.py`/`entropy_whole_controls.py`，不改旧源码、参数或输出。
- 图：`shared_entropy_evaluation_v1/figures_20260909T192735/all_selected_neighbor_effects.{png,svg}`，是全部k的观测点图，不是置信区间。

后续只完成剩余mandatory SOLE实验、最终baseline统计及总报告；不再对已达点门槛的最终九项调参。


# 最终固定方法的成分消融（2026-09-09 19:48）

此分析在最终方案已固定后，按先存协议分别移除已启用的熵调节、额外幅度或原方法参照。没有再次选参数；未启用的模块不另造重复移除条件。全部评分先保存，再读取标签；九项primary在846条上的重算均精确、全部变体有效。数据继续属于探索后重复使用。

下表为最终方法减去消融，MAE负值更好、准确率正值更好。Meter用连续ordinal MAE；RoboReward用原生1–5。

|case|集合|移除模块|MAE变化|总pp|sucpp|failpp|
|---|---|---|---:|---:|---:|---:|
|meter_image_text|full_cohort|without_original_anchor|-0.006139|1.06383|0.00000|1.55709|
|meter_image_text|reused_validation|without_original_anchor|-0.006061|1.00402|0.00000|1.47059|
|meter_video_text|full_cohort|without_original_anchor|-0.020760|1.53664|0.37313|2.07612|
|meter_video_text|reused_validation|without_original_anchor|-0.021997|2.00803|0.00000|2.94118|
|rr_image_text|full_cohort|without_entropy|-0.086288|4.01891|0.00000|5.88235|
|rr_image_text|reused_validation|without_entropy|-0.058233|2.81124|-0.63291|4.41176|
|rr_text_image|full_cohort|without_entropy|-0.089835|0.35461|10.07463|-4.15225|
|rr_text_image|full_cohort|without_extra_amplitude|-0.100473|15.13002|2.23881|21.10727|
|rr_text_image|full_cohort|without_original_anchor|-0.010638|0.82742|-1.86567|2.07612|
|rr_text_image|reused_validation|without_entropy|-0.076305|-0.60241|7.59494|-4.41176|
|rr_text_image|reused_validation|without_extra_amplitude|-0.074297|12.65060|0.00000|18.52941|
|rr_text_image|reused_validation|without_original_anchor|-0.006024|0.60241|-3.16456|2.35294|
|rr_text_video|full_cohort|without_entropy|-0.021277|-0.35461|5.22388|-2.94118|
|rr_text_video|full_cohort|without_extra_amplitude|-0.169031|7.56501|-4.10448|12.97578|
|rr_text_video|reused_validation|without_entropy|-0.034137|0.00000|3.79747|-1.76471|
|rr_text_video|reused_validation|without_extra_amplitude|-0.148594|7.02811|-5.06329|12.64706|

1. **熵门控不保证所有指标同升。** RR image_text在full846加入熵后MAE−.086288、总+4.0189pp、fail+5.8824pp，suc不变；重复498则suc少1/158。它在本例主要保护原模型较确定的失败预测，不能称成功类也因熵增加。
2. **RoboReward text_video的成功类保护解释了组合通过。** 固定s4/ρ0时，加入β.5在full846使suc+5.22388pp、MAE−.021277，但fail−2.94118pp、总−.35461pp；重复498的suc+3.79747pp、总持平。去额外倍数则suc更高，却少了整体MAE/失败类收益。两个组件形成权衡，不能把最终总增益全部归因于熵。
3. **RoboReward text_image同时有幅度与置信度权衡。** 固定其它设置，加熵在full846使suc+10.07463pp、fail−4.15225pp、MAE−.089835；重复498总准确率反而低.60241pp。额外幅度带来更大的总/fail收益；ρ参照仅带来较小MAE/总体改善，并牺牲部分suc。
4. **Meter的anchor贡献较小。** full846中image_text/video_text增加原方法参照分别总+1.06383/+1.53664pp，远小于其相对native的全部收益。主要收益不能归给这个末加模块；原区域/时间证据和大幅度读出才是此前控制所检验的主成分。
5. 本表是固定算法组件的输出效应，不能证明物理任务完成度的因果识别或独立泛化。原论文、输入与分区边界、正确/错误区域控制、预测与配对分布仍须合并解读。

协议与原始数据：`RESEARCH/final_postprocess_ablations_v1/{frozen_protocol_v1,analysis_v1}.json`及各case `scores.jsonl`/`generation_audit_v1.json`；源码`final_postprocess_ablations.py`单独新增，最终方法及其全部冻结源码未改。

最终共享参数的代数核验补充：`shared_entropy_evaluation_v1/algebraic_invariance_audit_v1.json`保留八个未变case既有核验并新测RR text_video的β.5。全九项的最大档位置换浮点误差为3.33955e−13，精确zero/退化检查全通过；前一方法报告2.77e−13对应先前entropy v1参数的测试。性能门槛仍使用严格不等式，没有引入此数值容差。


## 2026-09-09 20:08：最终方法的任务差异、分布与配对补充

新增 `RESEARCH/final_descriptive_tables_v1/report.md`、`report.json`、`per_task_split_metrics.csv`（1980行）及 `task_effects.png/svg`。全部九个固定primary、846/498两集合、native/final、task/suc/fail均保留；图已目视核验。这是冻结后的描述性分析，不再选参。

完整846中，九项task-macro MAE与准确率均改善，但RR image_text仍有10/28任务MAE变差、5/28任务准确率变差；Qwen text_video有7/28任务MAE变差。不能把总体通过写成每任务有效。MAE的micro与task等权macro分开，后者不是官方23公开子集Overall。

543个同视频配对中，九项平均progress差均增加，但排序改善不全面：Meter image_text/text_video/video_text的负序数分别3→15、23→30、0→57；RR image_text虽负序73→30，平分却47→96。全部负/0/1/2/3/4计数和两类五档分布均在补充中。消除大量中档和平分有助于端点准确率，同时可能新增错误排序，不能用平均差值掩盖这一边界。

当前仅主线1的两个SOLE完整七步实验尚未结束；不以该补充或终步实验替代。

时间注记：上段标题20:08为书写误差；实际追加时间由本轮工具UTC时间核对为2026-09-09 20:00（上海）。内容和产物不变，保留原段并在此纠正。


## 2026-09-09 20:10：最终方法第二阈值的成功类权衡

主验收使用预定五档端点阈值0.125/0.875；两套阈值均完整报告，不把0.2/0.8结果隐去。补充 `shared_entropy_evaluation_v1/secondary_threshold_descriptive_audit_v1.json` 固定全部Meter三输入、三个k、四分区和两baseline，未重新选参。

|输入/集合（固定primary）|0.2/0.8总准确率native→final|suc native→final|fail native→final|
|---|---:|---:|---:|
|meter_image_text/full_cohort|0.0000%→56.8558%|0.0000%→38.4328%|0.0000%→65.3979%|
|meter_image_text/reused_validation|0.0000%→61.4458%|0.0000%→41.1392%|0.0000%→70.8824%|
|meter_text_video/full_cohort|33.6879%→65.2482%|58.9552%→44.0299%|21.9723%→75.0865%|
|meter_text_video/reused_validation|31.1245%→66.2651%|56.3291%→44.9367%|19.4118%→76.1765%|
|meter_video_text/full_cohort|0.0000%→56.2648%|0.0000%→26.8657%|0.0000%→69.8962%|
|meter_video_text/reused_validation|0.0000%→55.0201%|0.0000%→22.1519%|0.0000%→70.2941%|

**Meter text_video在0.2/0.8下有成功类损失：** full846的三个k suc分别−17.1642/−14.9254/−16.7910pp；重复498为−13.9241/−11.3924/−15.1899pp。总/fail仍大幅改善。其它两个Meter输入在此阈值仍双类提高。不得把主门槛通过表述为两套阈值四项均通过或阈值无关的稳定性。这里保持原冻结主指标，新增敏感性说明；该权衡不经重新选参掩盖。

本次来源审计 `final_frozen_lineage_audit_v1.json` 已实际通过173项核对，涵盖21个原冻结推理文件、指标和后处理源码、输入分区、实际配置/ranking、开发与父评分、九配方和控制来源。首版汇总审计误要求控制失败计数也为0，保留了exit1及源码SHA记录；修正为保留几何失败并检查主方案完整，未改任何冻结实现或评分。


## 2026-09-09 21:07：最终方案与三条主线完成状态

全部预定baseline／attention实验、文献调研和方法迭代已完成，最终结论统一见[独立研究总报告](FINAL_RESEARCH_REPORT_20260909.md)及[新增baseline完整总结](../exp_plan_addbase_summary.md)。只在Robo-Dopamine-addbase操作，保留旧数据、原始预测、失败版本和历史记录。

**最终观测方案：原生全档位的区域／时间对比分布，加固定的熵幅度调节和可选原方法参照。** 公式为`z_final=z0+s*h^β*(z_parent−z0)+ρ*(z_original_same_k−z0)`，`h=H(softmax(z0))/log(B)`；parent、k及全部参数由唯一`results/mydata_bench/experiments_v2_corssmodel/auto_research/session_20260908/shared_entropy_evaluation_v1/frozen_manifest_v1.json`和九份`addbase_shared_entropy_final_v1_*.yaml`确定。所有评分bin保留，没有端点硬映射或按真值选分。

|模型|通过主指标观测门槛的输入|已测k邻域|
|---|---|---|
|Robometer-4B|image_text / text_video / video_text|64,68,72 / 40,48,56 / 56,64,72|
|Qwen3-VL-8B|image_text / text_video / video_text|32,40,48 / 40,48,64 / 64,68,72|
|RoboReward-8B|image_text / text_image / text_video|60,64,68 / 40,48,56 / 48,56,64|

开发234、复用498、完整grounded846的全部9项×3k均满足两baseline下MAE下降、suc/fail准确率提高、总准确率至少+10pp及原attention改良；primary也胜开发预选原方法。846上全部邻居对两baseline的最小总增益14.1844pp。支持所列三个已测点，未断言未测整数k都通过。

这个完成结论限定为**预定主指标的本数据观测验收**：Meter主阈值0.125/0.875，Qwen/RR原生1–5。原独立确认v1整体失败，后续498已复用，不宣称新的独立确认；0.2/0.8的Meter text_video成功类下降，未取得两套阈值全部指标同时改善。SOLE未通过新方法，Meter官方text_image也未进入成功输入；任务/配对/完整wrong控制的权衡均保留。

主线1：10个native各1213尝试；Meter五输入各846×10；SOLE五输入full与terminal各846×11均已结束。正式40文件561517条记录无重复/科学冲突；全部SOLE full/terminal zero结果精确。主预测147490条、437条错误保留。最终快照：`results/mydata_bench/experiments_v2_addbase/session_20260908/metrics_snapshot_20260909T205912.json`，最终图表与15840行指标CSV在`results/mydata_bench/experiments_v2_addbase/session_20260908/final_added_baseline_report_v1/`。

主线2：14篇计划文献及VCD已核验，RoboReward MAE原定义补充核对，链接更正与理论分析完整保存。主线3：理论—实现—完整开发—原独立确认失败—后续独立版本探索—固定最终结果／控制／消融闭环已完成。原失败不能被最终观测成功回写。
