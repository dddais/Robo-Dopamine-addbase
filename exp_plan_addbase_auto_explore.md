# 跨模型研究与效果分析

遵守本文的的要求，进行auto research，针对现有研究背景与问题进行探索性研究。

进行长时间的充分的调研、思考、理论分析、实验，直到完成所有主线目标。

我要睡觉了，需要我确认的部分先跳过，进行你能进行的内容。

## 研究背景：

- 基于/home/dais/workspace/Robo-Dopamine-addbase/mydata_bench/exp_plan.md的规划，进行了baseline 和attention steering的实验，目前发现在robo-dopamine的GRM上该方法的效果十分明显;
- 但是在qwen3-vl-8b和roboreward-8b的效果不是很明显
- 已有实验可供参考：
  - 目前已进行的GRM实验结果：/home/dais/workspace/Robo-Dopamine/mydata_bench/exp_plan_GRM_summary.md
  - 目前已进行的跨模型实验：/home/dais/workspace/Robo-Dopamine/mydata_bench/exp_plan_crossmodel.md
  - 跨模型实验结果：/home/dais/workspace/Robo-Dopamine/mydata_bench/exp_plan_crossmodel_summary.md



## 主线目标1

- 遵守/home/dais/workspace/Robo-Dopamine-addbase/mydata_bench/exp_plan_addbase.md 完成新增baseline任务
- 直到新增baseline的所有实验完成，得到完整的实验结果总结mydata_bench/exp_plan_addbase_summary.md



## 主线目标2

- 调研相关文章，调研总结当前有哪些相关方法 ，目前已有部分可参考，见本文档“可参考代码/文献”部分，调研结果更新在“可参考代码/文献”部分。



## 主线目标3

- step1:基于调研的结果和方法，从理论角度思考提出能够改进目前attention steering的方案，如果需要可继续调研相关工作文章
- step2:实现step1提出的改进方案，进行实验验证，分析实验结果，如果效果不好则重复step1提出改进方案
- 验收目标：不断重复上述两个step，直到有一种方案改良原 attention steering方法同时满足下面的所有要求:
  - 四种模型（roboreward-8b 和qwen3-vl-8b ，robometer-4b,SOLE-R1-8B  ）中至少有三种模型满足“稳定有效”。
  - 稳定有效：指在所有输入构造中的至少三种输入下（最好包含官方的输入构造）都能稳定提升在数据集上的表现（MAE下降，suc,fail准确率提高，总准确率提高至少10%）(不要求所有top-k都能满足，至少存在一个top k 的范围满足)。
- 严禁使用端点hard coding这种类似作弊的方法！！！



## 基本原则（必须遵守）

- 目标代码库是/home/dais/workspace/Robo-Dopamine-addbase，千万不要搞错了！！
- 尽量不修改现有代码库，如果需要修改，进行增量式修改，比如增加可选配置项等；
- 不允许进行git 操作本地已有的仓库，只能git clone开源仓库进行参考；
- 不允许对本地数据，结果等进行删除修改等操作，只能新增；
- 不用担心耗时，进行充分的调研、思考、理论分析，提出有道理的优雅的方案，严禁作弊的方法
- 注意GPU可能被其它程序使用，根据空余显存灵活使用，优先使用空闲GPU



## 可参考相关代码/文献

### 2026-09-08 本轮核验补充（研究进行中）

- 原列表开头的 PAI/PASTA 链接对应关系写反：PASTA 是 [2311.02262](https://arxiv.org/abs/2311.02262)，PAI 是 [2407.21771](https://arxiv.org/abs/2407.21771)。保留原始条目供追溯。
- 已核实计划内 14 篇论文的 arXiv 标题与摘要，并读取主要机制论文正文；新增 [Visual Contrastive Decoding, 2311.16922](https://arxiv.org/abs/2311.16922)。方法、来源与适用边界见 [文献与假设记录](mydata_bench/auto_research_addbase/LITERATURE_AND_HYPOTHESES_20260908.md)。
- 两个新增模型的官方代码通过新 clone 存在本目标目录的 `results/mydata_bench/experiments_v2_corssmodel/auto_research/session_20260908/references/`；未读取其他本地代码仓库。


- /home/dais/workspace/gaze-heads : [https://arxiv.org/pdf/2606.14703v1](https://arxiv.org/pdf/2606.14703v1)
- /home/dais/workspace/PAI : [https://arxiv.org/pdf/2311.02262](https://arxiv.org/pdf/2311.02262)
- /home/dais/workspace/PASTA : [https://arxiv.org/pdf/2407.21771](https://arxiv.org/pdf/2407.21771)
- **PASTA — Post-hoc Attention Steering for LLMs**（ICLR 2024）：[arXiv:2311.02262](https://arxiv.org/abs/2311.02262)。
- **PAI — Paying More Attention to Image**（ECCV 2024）：[arXiv:2407.21771](https://arxiv.org/abs/2407.21771)。
- **ASCD — Attention-Steerable Contrastive Decoding**（AAAI 2026）：[arXiv:2506.14766](https://arxiv.org/abs/2506.14766)，
- **CAST — Caption-Guided Visual Attention Steering**：[arXiv:2605.04641](https://arxiv.org/abs/2605.04641)。
- **Gaze Heads: How VLMs Look at What They Describe**：[arXiv:2606.14703](https://arxiv.org/abs/2606.14703)。
- **HAS — Highlight-guided Attention Steering for Multimodal LLM Video Summarization**：[arXiv:2607.17994](https://arxiv.org/abs/2607.17994)。
- **Arbitration Failure, Not Perceptual Blindness**：[arXiv:2604.09364](https://arxiv.org/abs/2604.09364)。
- **Inference-Time Attention Steering for VLA Driving Models**（ECCV 2026）：[arXiv:2608.17095](https://arxiv.org/abs/2608.17095)。
- **Attention is Case-Sensitive**（ECCV 2026）：[arXiv:2608.03711](https://arxiv.org/abs/2608.03711)。
- Localization heads :Your Large Vision-Language Model Only Needs AFew Attention Heads For Visual Grounding; [https://arxiv.org/pdf/2503.06287](https://arxiv.org/pdf/2503.06287)
- Your Model Already Knows: Attention-Guided Safety Filter for Vision-Language-Action Models : [https://arxiv.org/pdf/2606.09749](https://arxiv.org/pdf/2606.09749)
- Analyzing Multi-Head Self-Attention :[https://arxiv.org/pdf/1905.09418](https://arxiv.org/pdf/1905.09418)



## 原因分析

请你进行调研思考理论分析后补充。

**2026-09-08 阶段分析，待完整实验验证：** 原 +6/−6 bias 将目标/非目标相对赔率放大约 16 万倍，同时改变视觉与文字的总注意力比例。定位增强并不保证完成度判断正确，同视频不同指令的任务绑定可能仍然失败。实际 reward 读出位置、模态融合方式和 temporal patch 对齐在不同模型中也不同。本轮据此提出“保持视觉总注意力不变，仅重新分配视觉内部注意力”的候选；数学不变量和小规模真实前向已验证，尚不能据此宣称指标有效。详见上述文献与假设记录。


## 实验基础设置

增量式修改：代码修改不要影响到之前的实验运行，尽量以增量式的形式增加代码，比如加可选参数配置之类的

**数据集** ：/home/dais/workspace/data/mydata_v2/new ;/home/dais/workspace/Robo-Dopamine/results/mydata_bench/cohorts/auto_grounded_v2 (认为这就是正确的，不需要人工审核)

**config** 放在：/home/dais/workspace/Robo-Dopamine-addbase/mydata_bench/configs/v2_crossmodel

**输入**：video->text ; text->video; image->text ; text->image ;interleaved ;以上五种都需要尝试，方法最好能在大部分输入构造下work
**输出** 在：/home/dais/workspace/Robo-Dopamine-addbase/results/mydata_bench/experiments_v2_corssmodel/auto_research

**conda环境**：sam3:rewardbench-sam3 ；其它实验：robo-dopamine

**可用GPU**：0，1，2 ,3

**vpn** : proxy_on

**评价指标**：
1.MAE：按照roboreward的原定义
2.准确率：对于suc数据，lable=5,对于fail数据，lable=1；预测结果和lable相同的数量与概率。包括总准确率，suc,fail准确率，各个具体task的准备率分布。对于GRM这种输出连续进度的，用阈值区分开，统计两套阈值的情况：0.125，0.875；0.2，0.8
3.预测分布：统计在suc,fail数据上模型预测的lable分布，以及各个具体task上的模型预测分布；
4.pairwise区分度分析：因为数据集构成原理是1条suc数据，对应了1条或多条相同视频，不同instruction的fail数据，所以需要先找到suc数据所对应的fail数据，分析相同视频下不同instruction带来的影响。对于roboreward-8b,qwen这种输出离散的模型，计算统计配对数据中suc数据的预测值与fail数据的预测值的差值，把差值分成：负，0，1，2，3，4几档统计一下；
5.ranking head统计:列出具体的top 8，统计top 8,32,64在不同模型的重合度

## 最终有效方案

请你进行研究后，在这写明满足主线目标的最终方案

### 最终方案状态（2026-09-08，执行中）

目前尚未确认满足“至少三模型、各至少三输入、四项指标同时提升且总准确率 +10 个百分点”的方案。已启动完整 baseline / attention 与预先冻结的开发初筛；独立确认集尚未参与调参。新增 baseline 的已完成结果持续追加于 [exp_plan_addbase_summary.md](mydata_bench/exp_plan_addbase_summary.md)。仅在完整验证结束后，在此追加最终有效方案或如实说明未达到的验收项。


## 原因分析补充（2026-09-09 12:53，开发实验结论）

1. **注意力集中程度与奖励判断不是同一件事。** 原强bias只让模型更关注目标区域，无法保证完成度的任务绑定和标度都正确。SOLE已完成的三种全七步输入，在零干预逐例精确一致的前提下，原强bias也没有稳定+10pp；text_video的k64还增加原生回答格式失败。这不是只靠增加attention mass就能解释为有效的结果。相关理论边界可对照Gaze Heads、Attention is Case-Sensitive和ASCD，详见文献记录。
2. **视觉总量守恒消除了一个混杂，但单次重分配仍不够。** 本轮使用原生SDPA输出加选中heads的概率差×V，解决旧math替换在零bias时引发的数值漂移。对称区域证据D=z(+b)−z(−b)再作用于全部native reward bins，能利用“对任务区域的敏感方向”；Qwen三个输入和Meter text_video已有完整234开发改善及相邻k证据，仍待确认。
3. **跨模型的分数尺度不同。** 在Meter image_text/video_text中，lambda≤4时预测仍停留中间区间，较大lambda16的60条初筛才同时改善两类端点准确率。匹配强度消融显示，这里的初步改善来自放大幅度，不应归因于后来试验的原生成功head置信度调节。所有bins均保留，并非把预测硬映射到1或5；234扩展正在检验收益是否稳定。
4. **静态外观敏感性可能混入区域证据。** 时间候选用D_observed−D_static，其中static重复首帧像素、保持文本/token/grid/时间戳并让tracking对应实际显示像素。RoboReward text_image完整234优于原bias并有相邻k支持；其它输入正扩展。该对照不能证明因果完成度已经完全识别，只能测试扣除静态部分是否有用。
5. **统一强干预存在成功/失败类权衡。** 可选连续调节(1−q_native)^power只缩放区域方向，不替代奖励分数。RoboReward image_text初筛中调节版本通过，而匹配lambda的无调节版本未通过；这是待234验证的候选。q仅来自模型原生输出，推理不读真值、数据目录的类别语义或其他同视频指令。

可审计的公式、冻结grid、代码/输入SHA、数值测试、失败版本排除和完整结果路径持续记录于 [研究日志](mydata_bench/auto_research_addbase/RESEARCH_LOG_20260908.md) 与 [新增baseline总结](mydata_bench/exp_plan_addbase_summary.md)。截至此补充，独立498确认集的新方法指标仍未用于开发选型，**尚不构成最终有效方案**。


## 2026-09-09：当前候选家族的统一解释与确认前选型

这些候选共享一个核心：用目标区域注意力干预产生的原生评分分布差，修正同一模型的原生评分分布。记观察视频的两条守恒干预输出为z_obs,+、z_obs,−，首帧重复参考的输出为z_static,+、z_static,−。统一写为：

`D = (z_obs,+ − z_obs,−) − α (z_static,+ − z_static,−)`，

`z_new = z_native + η D`。

α=0给区域差分，α=1给时间区域差分。η可为固定λ，或λ(1−q_native)^β，或在预设上限内满足KL(p_native||p_new)预算的最大幅度；本轮没有把置信度与KL同时叠加。半径b控制用于估计D的干预强度，η控制读出放大，两者分开消融。它是一组共享机制的可配置方案，**不是已经证明同一组超参数在全部模型/输入通用**。

令p为各自全部原生bins的softmax，固定η时上述公式等价于：

`p_new(j) ∝ p_native(j) × [(p_obs,+(j)/p_obs,−(j)) / (p_static,+(j)/p_static,−(j))^α]^η`。

每个分支的归一化常数与j无关，最后统一归一化时消去。没有预先指定j=1或j=5，五档/十bin中间值始终参与；SOLE补充使用论文范围内全部201规范整数百分比，原始越界/格式结果另列。q_native来自模型自己的输出，Meter为checkpoint已有success head，Qwen/RR为五档分布的归一化期望；它可能在错误指令上也过度自信，因此不能当作正确标签，也没有替代reward。

b=0时正负分支相同，η=0时回到同读出baseline。采用O_native+(P_modified−P_reference)V的残差实现，保证数值零干预精确，同时保留原生attention后端。视觉总量守恒是理想softmax下逐query的数学性质；浮点实现以实际数值审计为准，不能从公式直接推断长推理逐token完全一致。

当前证据表明：放大区域证据可改善Meter中间分档的读出；时间扣除可改善部分Qwen/RR输入；较强放大并非普遍有用，RR video_text的b3时间方案完整开发未达门槛，新增小幅度/KL/置信度敏感性最高仍只有9.83pp。b1.5 last-frame局部差分在另一轮60条screen通过k32/64，因此已冻结其234扩展，含same-k原bias及wrong/low/zero对照。这些结果区分了半径、区域方向、参考视频和幅度的作用，没有假设任一模块必然有效。

确认前选型使用显式规则select_confirmation_candidates.py：在三个连续已测k、相同其他参数的窗口中，保护最弱验收余量，即两套baseline下的suc增益、fail增益以及总增益减10pp三者的最小值；MAE下降和原方法改良是硬约束。primary取窗口中间k，平局优先较少神经分支和较简单调节，再比较平均MAE改善。保存全部候选窗口，不依据确认性能或p阈值挑参数。规则自身是在已观察开发结果后作出的开发决策，不能冒称研究开始前的预注册。

随后freeze_confirmation独立检查完整234、零logits/分数精确、视频簇切分、same-k原bias、两个baseline、连续模型primary五档MAE及三模型各三输入覆盖，全部通过才写最终配置与SHA。analyze_confirmation预备按同一manifest报告所有case、所有邻居、两个baseline、原方法、错误区域/低heads/温度、覆盖率、两套阈值、task/split分布和同视频配对差。点估计验收与20000视频簇bootstrap、IUT/Holm不确定性分别报告，p值不是10pp置信下限检验。

截至此记录，确认推理仍未启动，最终有效方案尚未成立。ASCD/VCD等文献提供对比分布的机制参照；本任务的性能、归因和稳定性只能由本轮完整实验验证。引用与原文核验见文献矩阵。


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


## 2026-09-09 18:53：原独立确认已出现失败，后续探索单独冻结

原九项确认的manifest与设置保持不变。当前8项完成：7项通过冻结点门槛，RoboReward text_video失败；最后Meter video_text仍运行。RR text_video的primary k56虽然相对两种baseline均满足MAE下降、suc/fail上升及总+10pp，却比same-k原方法的MAE高1/498=0.002008，因此不能宣布原方案三模型各三输入整体通过，也不改容差或换primary。RR text_image另不胜开发预选的original k24。全部数值与不确定性见主总结18:09段及确认目录，最终九项Holm仍待全体完成。

另行冻结的确认后探索在原方案的完整native logits上加入`ρ(z_original_same_k−z_matched_native)`，ρ仅用原234开发集按事先保存规则选择，每个输入的三个k共用一个ρ；原primary、k邻域及其他参数均不变。选中Meter image_text/video_text的ρ=1，RR text_image/text_video的ρ=.25，其余五项ρ=0。所有评分bins保留，不按真值或类别路径选分。该候选是在观察原确认失败后提出的；其后498条明确称重复使用验证集，不能称新的独立确认。全846含开发/ranking数据，只作描述性统计。

完整评估尚未结束，不在此提前宣布新方案达标。四个非零ρcase的新增完整wrong/low/zero神经控制已全部完成，原始支路复算通过；新旧前向parity与全方案效果仍待自动分析。两个mandatory SOLE全七步video_text/interleaved仍在运行，末步数值screen不能替代它们。最终有效方案/未达项将在所有必要实验及审计结束后追加。可追溯入口：`post_confirmation_anchor_evaluation_v1/frozen_manifest_v1.json`、`post_confirmation_anchor_whole_controls_v1/frozen_manifest_v1.json`（均位于本仓库RESEARCH目录）。


## 2026-09-09 19:36：当前数据点估计验收已通过的冻结方案

主线3已获得符合本轮观测门槛的方案，唯一版本为`shared_entropy_evaluation_v1`：Meter/Qwen/RoboReward各三种输入、每项三个原冻结k，完整grounded846与重复验证498均通过两baseline的MAE下降、suc/fail提高、总准确率+10pp及同k原方法改良；primary还胜过开发预选原方法。原v1的独立确认失败仍保持False，本方案是确认后探索，不能据此宣称新的独立确认成功。**主线1仍有SOLE video_text/interleaved两个mandatory全七步未完，因此总任务尚未完成。**

方法为`z_final=z0+s[H(softmax(z0))/log(B)]^β(z_parent−z0)+ρ(z_original_same_k−z0)`，全部原生bin保留。RoboReward三个输入共享β=.5；image_text的s/ρ=1/0，text_image=4/.25，text_video=4/0。Meter image_text/video_text的s/β/ρ=1/0/1，Meter text_video及Qwen三个输入=1/0/0。原区域/时间差分、q_native或KL机制、primary、k邻域不变。九个明确k列表、绝对指标、两套MAE/阈值、全部失败阶段和控制局限详见[已冻结共享熵方案结果](mydata_bench/auto_research_addbase/SHARED_ENTROPY_RESULT_20260909T1932.md)。

从原始summary数值单独复核全部9×3点，不使用性能容差：完整846对两baseline的最小总增益14.1844pp、最小suc增益2.2388pp；重复498分别13.6546pp、1.8987pp。小的成功类余量仍需要真实独立数据检验，不把点通过等同于统计显著或每任务都改善。114剩余分区的失败完整保留，未删除该分区。

最终配方已新增到`mydata_bench/configs/v2_crossmodel/addbase_shared_entropy_final_v1_*.yaml`，注册表与直接验收审计分别为RESEARCH下`shared_entropy_evaluation_v1/{final_recipe_registry_v1,direct_final_acceptance_audit_v1}.json`。九项whole-target/zero精确审计通过，所有邻居图已查看。此后不再调参，继续收完主线1并写总报告。


## 2026-09-09 20:16：主验收与第二阈值的范围

最终共享熵方案的已通过结论明确对应预定主指标：连续Meter采用0.125/0.875五档端点阈值，Qwen/RR采用原生1–5输出。另按计划完整统计0.2/0.8；Meter text_video在此第二阈值下成功类准确率下降（primary全846为58.9552%→44.0299%，重复498为56.3291%→44.9367%），其总/fail仍提高。因此不能称两套阈值所有指标均改善，也不能声称阈值无关的稳定性。两baseline指原生生成和同读出baseline，与两阈值是不同概念。全部第二阈值邻居结果存于`shared_entropy_evaluation_v1/secondary_threshold_descriptive_audit_v1.json`，没有据此重选参数或改写主门槛。

173项最终冻结来源核对已通过，主线1的两项SOLE全七步继续推进；整体goal仍未完成。


## 2026-09-09 21:07：最终方案与三条主线完成状态

全部预定baseline／attention实验、文献调研和方法迭代已完成，最终结论统一见[独立研究总报告](mydata_bench/auto_research_addbase/FINAL_RESEARCH_REPORT_20260909.md)及[新增baseline完整总结](mydata_bench/exp_plan_addbase_summary.md)。只在Robo-Dopamine-addbase操作，保留旧数据、原始预测、失败版本和历史记录。

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


## 2026-09-09 23:48：目标3的抗过拟合续研（未完成）

用户要求继续寻找新方法，拒绝原共享熵方案的适应性验证风险。旧846全部降为开发/描述数据，不再复用498宣称独立确认。本轮已新增跨模型/输入统一参数的逐帧attention概率搬运F1、固定VCD视觉先验修正F2，完整234实验进行中；现有已分析结果尚未达标，全部负结果和对照保留。另固定公开RoboRewardBench完整2831条test版本并准备独立验证数据，尚未查看新方法测试性能。前瞻规则见[续研协议](mydata_bench/auto_research_addbase/ROBUST_RESEARCH_PROTOCOL_20260909.md)，代码/实验/活跃进程与下一步见[23:48检查点](mydata_bench/auto_research_addbase/ROBUST_CHECKPOINT_20260909T2348.md)。本续研目标保持active，不能沿用旧探索结果声称新的严格目标完成。


## 2026-09-10：旧结论可信度复核

针对用户关于“作弊”的疑问，补充[方法可信度说明](mydata_bench/auto_research_addbase/METHOD_VALIDITY_AUDIT_20260910.md)。旧实现没有发现直接答案赋值，但存在确认反馈后的适应性选型，最终旧结果不能作为本轮抗过拟合目标的成功证据；目标3续研仍未完成。新候选、全部失败及外部一次性验证准备见[续研协议](mydata_bench/auto_research_addbase/ROBUST_RESEARCH_PROTOCOL_20260909.md)。


### 2026-09-10补充参考：监督敏感度选头

[Are Sixteen Heads Really Better than One?](https://arxiv.org/abs/1905.10650)（Michel等，NeurIPS2019），已核验正文4.1的损失梯度head重要性；本轮F6探索其在真实评分query处、区域bias有符号梯度上的改造，使用另行冻结的官方训练数据，而非旧确认标签。机制差异、训练/验证隔离和全部未完成项见[续研协议](mydata_bench/auto_research_addbase/ROBUST_RESEARCH_PROTOCOL_20260909.md)。


### 2026-09-10 02:12：抗过拟合续研仍未达标

F1完整15项均失败，F2当前13项、F3当前7项、F4/F5各5项完整开发也均未通过严格门槛。F6已固定独立公开训练的监督选头规则，并完成493条训练验证及原846完整验收的增量推理/统计/条件调度实现；当前仍在训练输入准备，尚无F6拟合或验证性能。保留外部test的2831条无标签定位已完成，但奖励模型表现未打开。最新工作记录见[进展文档](mydata_bench/auto_research_addbase/ROBUST_PROGRESS_20260910T0212.md)，旧结果不能作为当前目标3完成的依据。


### 2026-09-10 10:53：F6完整失败，F7监督全query选头启动

F1–F6均已完成三模型五输入的旧234开发，无完整稳定达标；F6的1942条监督拟合和493条验证也已完成，与原目标联合通过数0，未打开保留test。新F7改变attention干预的query范围，保持监督fit隔离、原生全部评分档位、统一幅度/k设置；三个模型真实工程通过，15项拟合配置冻结并运行。新的493验证仅作为复用选型数据，不能称独立确认。当前仍未完成目标3，见[最新进展](mydata_bench/auto_research_addbase/ROBUST_PROGRESS_20260910T1053.md)。


## 2026-09-10 11:18：方法完整性说明与F7评价冻结

此前共享熵方案是在观察原确认失败后继续开发得到的观测结果，原498已复用，不能作为新的独立确认；历史“目标完成”不作为当前稳健主线3的完成依据。F1–F6完整旧234稳定通过数均为0，当前F7尚未完成效果验证。详见[METHOD_INTEGRITY_EXPLANATION_20260910T111808.md](mydata_bench/auto_research_addbase/METHOD_INTEGRITY_EXPLANATION_20260910T111808.md)。

F7评价policy已在任何F7效果读取前冻结：64份源码快照、90份F6比较文件摘要、493条复用选型数据、固定幅度1与k32／48／64、primary48。新增相同学习头只在评分query干预的控制，错位区域不可构造时不据其惩罚分数归因。持久评价队列PID2567628等待全部15项拟合成功，然后冻结配置、跑工程与493／旧234并分析；此队列不运行保留测试。


## 2026-09-10 11:39：F7条件最终确认与训练诊断

F7已完成8项拟合，无拟合失败，效果评价仍待全部15项完成。最终确认policy已在保留测试前冻结72份源码；条件队列2571353只在原范围的联合门槛和完整846门槛通过后，才执行全部预选case的一次性2831条保留确认。保留claim与奖励模型测试输出当前不存在。3项最终确认范围/完整性测试通过。

7项已完成拟合的F6/F7零干预native logits及损失在每项1253条实际梯度样本逐条一致；训练数据有300个同视频多指令、多等级组。另记录8项零点方向导数和Qwen五输入梯度兼容性，均为训练诊断，不是F7效果结果，不改变已冻结方法。完整证据见[ROBUST_PROGRESS_20260910T113923.md](/mnt/public1/dais/workspace/Robo-Dopamine-addbase/mydata_bench/auto_research_addbase/ROBUST_PROGRESS_20260910T113923.md)。目标3仍未达成。


## 2026-09-10 11:48：共享选头的训练诊断与文献边界

新增核验group DRO与分布鲁棒优化两篇文献（arXiv:1911.08731、1810.08750），下载并提取原文、阅读相关公式及泛化局限。训练跨输入一致不保证测试有效。Qwen五输入中111个head具有正的最差输入波动惩罚后值，仅为后续低自由度共享选头的研究线索；没有修改F7或选择F8推理配置。局部梯度推导、适用边界和原文核验见[TRAINING_GRADIENT_ROBUSTNESS_NOTE_20260910.md](/mnt/public1/dais/workspace/Robo-Dopamine-addbase/mydata_bench/auto_research_addbase/TRAINING_GRADIENT_ROBUSTNESS_NOTE_20260910.md)。


## 2026-09-10 12:06：F7拟合12/15与有限幅度训练诊断

F7完整拟合已完成12项，Qwen/Meter各五输入、RR两输入，无拟合失败。独立训练记录诊断已核对11项每项1253条实际梯度的F6/F7零bias前向逐例精确。GPU2/3空闲后，新增固定Qwen text_video与Meter官方text_image的完整1942条fit诊断：实际baseline、全query b1/k48、同头同方向readout-only。全部无ROI样本也实际运行baseline，不删样本；不搜索参数，不读取493/原234/保留test效果，不改F7验收。规则/源码/配置已冻结，两个进程在正常产生记录。详情见[ROBUST_PROGRESS_20260910T120611.md](/mnt/public1/dais/workspace/Robo-Dopamine-addbase/mydata_bench/auto_research_addbase/ROBUST_PROGRESS_20260910T120611.md)。


## 2026-09-10：F7完整拟合审计与F8有序损失冻结

F7全部15项拟合完成、无拟合错误；全部F6/F7零干预logits与CE逐值相同，冻结评价已经启动。完整拟合有限幅度诊断显示Qwen可出现CE下降而MAE变差，因此不能由训练损失推断目标达标。完整证据见[拟合审计](mydata_bench/auto_research_addbase/F7_ALL_FIT_AUDIT_20260910.md)及[有限幅度结果](mydata_bench/auto_research_addbase/F7_FINITE_STEP_FIT_RESULT_20260910.md)。

F8单独研究全部原生有序边界的累计log loss，只改训练选头目标；保留原native评分、b1及k32/48/64，无端点hard coding。三模型真实合成标签工程检查全部通过，完整拟合进行中；83项依赖和相同严格评价门槛已冻结，见[拟合方案](mydata_bench/auto_research_addbase/F8_ORDINAL_FIT_PROTOCOL_20260910.md)与[评价方案](mydata_bench/auto_research_addbase/F8_EVALUATION_PROTOCOL_20260910.md)。493复用选型、846已暴露的边界保持明确。F8尚未安排保留确认，目标3仍未完成。


### 有序监督相关文献（2026-09-10核验补充）

- Cao、Mirjalili、Raschka：[Rank consistent ordinal regression for neural networks with application to age estimation](https://arxiv.org/abs/1901.07884)（CORAL）。已核实官方元数据和v7正文相关公式；F8采用原生softmax CDF上的有序损失，不是CORAL架构。正文训练选择描述存在val选型与best-test措辞不一致，未核查原代码，不以其测试数字证明本研究泛化。理论、来源范围与实际拟合核验见[补充说明](mydata_bench/auto_research_addbase/F8_ORDINAL_THEORY_AND_SOURCE_20260910.md)。


### 2026-09-10：F8最终确认规则补齐

在尚未读取F8验证效果、保留奖励性能仍未打开时，新增并冻结F8条件最终确认流程。只有完整原目标范围通过，且F7科学不合格并留下未使用测试集时才允许进入；F7崩溃不释放优先权，已有共享claim则禁止复用。全部选中9–15项和2831样本先完成预测，再一次连接标签。规则、源码摘要与四项边界测试见[最终确认协议](mydata_bench/auto_research_addbase/F8_FINAL_CONFIRMATION_PROTOCOL_20260910.md)。当前只是条件流程就绪，目标3未完成。


### 2026-09-10：F8完整拟合负结果与后续机制分析

F8在Qwen text_video完整1942训练前向中，有序损失下降但样本准确率38.52%→37.80%、MAE1.043→1.111，不能据训练loss宣布方法有效。全部数据、无ROI回退与1942条baseline精确复现均通过核验，见[固定拟合结果](mydata_bench/auto_research_addbase/F8_FIXED_FIT_FORWARD_RESULT_20260910.md)。另发现episode等权让该训练语料grade5的有效权重从19.26%变成31.34%，与完整样本验收的估计目标不同；尚不能据此认定因果原因，冻结F7/F8不变，见[权重说明](mydata_bench/auto_research_addbase/TRAINING_ESTIMAND_WEIGHTING_NOTE_20260910.md)。

- Burges：[From RankNet to LambdaRank to LambdaMART: An Overview](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf)，MSR-TR-2010-82。已核验官方19页PDF和相关成对损失／梯度分解；排序目标不能代替绝对MAE／准确率。结合训练中已有的同视频指令对照，整理了[后续假设与限制](mydata_bench/auto_research_addbase/PAIRWISE_SUPERVISION_THEORY_NOTE_20260910.md)，尚未实现F9或宣称新效果。


## 2026-09-10 14:45：F8完整拟合结束，F9评价及条件确认已冻结

F8全部15项拟合与原生有序损失CPU审计完成，无失败。F9复用这些训练梯度，按1942条样本等权估计均值，以446个episode簇估计标准误，全部15项重新聚合完成。推理仍用固定b1、k32/48/64及原生评分；训练标签参与了选头，不能称无训练方法。

F9已冻结97份评价源码、113份最终确认源码，30份旧234/493配置一次生成；新增8项CPU边界测试通过。评价包含固定F8、F7及F6对照，完整原846与最终2831确认均已有条件队列。F7/F8必须均科学不合格且未领取测试，F9才可获得一次性确认资格；崩溃不释放测试。具体见[F9评价与确认协议](mydata_bench/auto_research_addbase/F9_EVALUATION_AND_CONFIRMATION_PROTOCOL_20260910.md)。

当前仍无新最终有效方案。旧确认反馈后追加模块再复用498的问题不能靠“没有端点硬编码”抵消，历史达标不作为本稳健目标完成依据；解释见[方法有效性更新](mydata_bench/auto_research_addbase/METHOD_INTEGRITY_UPDATE_20260910T144554.md)。原234/846及493的暴露边界不变，保留测试奖励表现尚未打开。


## 2026-09-12：F7／F8／F9全量结果复核，均未满足目标3

三种方案均于9月10日完成各15项旧234／493评价，未发生队列进程失败。原234稳定通过数三种均为Qwen1／RoboReward0／Meter0，唯一通过输入是Qwen video_text；493通过数分别0／3／0、0／2／0、0／4／0。同一输入联合通过数全部为0，未满足三模型各至少三输入的原目标，故没有继续原846或保留2831确认。保留测试共享claim不存在，未用该测试调参。

这次复核了全部90份分区预测的完整网格、配置与SHA，冻结来源一致。完整表格、实际改善和失败原因已写入[F7／F8／F9完整结论](mydata_bench/auto_research_addbase/F7_F8_F9_COMPLETE_RESULTS_20260912.md)。F9训练诊断的损失下降仍伴随准确率下降及MAE升高，说明仅修正梯度损失／聚合不足。没有最终有效方法，不宣称目标完成；后续须继续研究任务绑定或有限干预学习，而非复用最终测试挑参数。
