# 自动研究最终总结（2026-09-09 21:05，上海时间）

本轮在 **Robo-Dopamine-addbase** 完成了计划中的新增baseline、五输入attention实验、文献调研、方法迭代和最终结果审计。得到一套在本数据预定主指标上满足“三模型各三输入、三个已测k均稳定改善”的方法。**这个结论属于多轮探索后的观测结果；原独立确认整体失败，最终方案未获得新的独立泛化证明。** 第二阈值、部分任务和配对排序仍有权衡，不能把达标范围扩大。

## 完成状态与口径

|主线|最终状态|直接证据|
|---|---|---|
|1：新增Robometer与SOLE baseline/attention|全部预定实验已完成尝试，失败保留|[完整新增baseline报告](../../results/mydata_bench/experiments_v2_addbase/session_20260908/final_added_baseline_report_v1/report.md)|
|2：文献调研与原因分析|14篇计划文献及新增VCD已核验，补充RoboReward指标原定义|[文献与理论记录](LITERATURE_AND_HYPOTHESES_20260908.md)|
|3：提出、实现、迭代改良|Meter/Qwen/RoboReward各三个输入，在开发234、复用498及完整grounded846均通过预定主指标与原方法比较|[最终冻结结果](../../results/mydata_bench/experiments_v2_corssmodel/auto_research/session_20260908/shared_entropy_evaluation_v1/summary_v1.json)|

“+10%”按更严格的 **+10个百分点** 执行。连续Meter的预定主准确率采用0.125/0.875端点阈值，低端p<0.125、高端p≥0.875；同时完整报告0.2/0.8敏感性。Qwen/RoboReward使用原生1–5输出。对Meter同时报告连续ordinal MAE=mean(|1+4p−y|)与五档MAE；离散模型MAE按RoboReward的1–5绝对误差定义。自定义task等权macro另列，不等于官方RoboRewardBench的23子集Overall。

准确率分母始终含全部期待样本；无效输出计为不正确，MAE只在有效数值上计算并列覆盖率。没有删除失败预测、几何不可行控制、早期无效工程版本或后续失败方法。

|集合|样本|视频内容簇|实际任务数|suc/fail|
|---|---:|---:|---:|---:|
|原始数据|1213|407|34|407/806|
|grounded完整cohort|846|299|28|268/578|
|开发|234|86|27|76/158|
|原确认，后续复用验证|498|179|27|158/340|
|ranking重叠剩余|114|34|8|34/80|

846=234+498+114，包含开发和ranking重叠数据，只作完整cohort的描述性统计。各分区按eligible视频内容簇划分；开发与原确认视频不交叉，但原确认性能后来被多轮研究复用。grounded可组成543个suc−fail配对，498中315个；缺eligible suc的fail仍计入总体指标。未grounded的367条未被宣称获得新方法改善。

## 新增baseline与原attention的结论

10个native模型／输入组合各完成1213次尝试；Meter五输入各846×10条件；SOLE五输入的完整七步与终步补充各846×11条件。合计147490个唯一主预测记录，437个错误记录保留。计入单步轨迹后，40个正式文件共561517条记录，重复键与科学字段冲突均为0。所有五输入ranking完成，并列出四模型×五输入的20组具体top8及30组同输入模型对的top8/32/64交集。

|模型/输入|有效/1213|五档MAE|连续ordinal MAE|主阈值总准确率|suc|fail|
|---|---:|---:|---:|---:|---:|---:|
|meter/image_text|1213/1213|2.309151|2.226536|0.0000%|0.0000%|0.0000%|
|meter/text_image|1213/1213|1.975268|1.954338|31.5746%|92.1376%|0.9926%|
|meter/text_video|1213/1213|1.718054|1.683867|4.3693%|3.4398%|4.8387%|
|meter/video_text|1213/1213|1.997527|2.063455|0.0000%|0.0000%|0.0000%|
|meter/interleaved|1213/1213|1.986810|1.968878|31.1624%|92.1376%|0.3722%|
|sole/image_text|1213/1213|1.437758|1.475746|42.4567%|21.1302%|53.2258%|
|sole/text_image|1213/1213|1.488871|1.517857|30.8326%|4.6683%|44.0447%|
|sole/text_video|1204/1213|1.833056|1.831927|9.0684%|17.4447%|4.8387%|
|sole/video_text|1213/1213|1.863149|1.862160|8.9860%|22.6044%|2.1092%|
|sole/interleaved|1213/1213|1.527617|1.534246|27.0404%|6.6339%|37.3449%|

原attention在Meter的15个target点、SOLE全七步15点和SOLE终步15点中，均无一个点同时满足完整有效覆盖、MAE下降、suc/fail提高和总+10pp。SOLE全七步text_image k64虽总+7.9196pp，仍未达到10pp；新完成的video_text k64总准确率10.2837%→8.6288%、suc28.3582%→11.1940%，且有1个格式失败。interleaved各k的fail均下降。定位增强不能直接等同奖励判断改善。

Meter官方结构是text_image；image_text/video_text两个适配的native主端点准确率为0，后续较大改善必须结合绝对正确率理解。SOLE官方image_text采用first/previous/current拼图、自身历史7步绝对进度递归；没有把进度当增量累加，也不把上一预测固定成0。greedy/max512与官方随机解码默认不同，输入适配及processor像素舍入差异均已披露。

实现对照check.md已核对：原attention覆盖选中heads的全部有效query行；干预作用于实际映射的视觉key列；ranking排除前8层，以任务区域raw mass排名并保存其它质量指标，Meter使用训练progress-token query、生成模型用末prompt query。视频tubelet按贡献帧映射区域，SOLE拼图按padding/列位置/resize变换区域；wrong区域保持同帧或同面板、等面积且不相交。scope与读出位置按配置明确记录。

正式SOLE采用保留native attention的残差实现和dynamic cache；数值分岔的旧math/static-cache/CUDA-graph版本没有进入正式结果。全部10个SOLE full/terminal的846个zero结果均与各自baseline精确一致，包含原生失败；最后两个full各5922个单步trace亦精确。Meter正式十条件没有全量zero，已完成的工程zero与新方法完整zero证据分开报告。

错误边界：attention中293条为wrong几何不可行，80条为原生数字answer格式失败，55条为terminal缺完整native历史；另有native text_video的9条失败。SOLE full text_video target k64有效809/846（37个格式失败）；video_text target k32/k64各845/846。原生101%等越界值保留。Meter官方sum==1类型猜测导致text_video两条越界读出，另列typed-softmax敏感性，没有覆写官方结果。

全部两种MAE、两套阈值、每task/split五档分布、配对和head表见[完整报告](../../results/mydata_bench/experiments_v2_addbase/session_20260908/final_added_baseline_report_v1/report.md)、[15840行指标CSV](../../results/mydata_bench/experiments_v2_addbase/session_20260908/final_added_baseline_report_v1/all_metrics_task_split_distribution.csv)、[180组配对CSV](../../results/mydata_bench/experiments_v2_addbase/session_20260908/final_added_baseline_report_v1/all_pairwise.csv)及[180组task-macro](../../results/mydata_bench/experiments_v2_addbase/session_20260908/task_macro_final_v1.json)。full与terminal分别比较各自同实现baseline，不把它们之间的差异全归因于干预时机。

![全部原attention target点](../../results/mydata_bench/experiments_v2_addbase/session_20260908/final_added_baseline_report_v1/figures/all_target_effects.png)

## 文献、机制与最终方法

[PASTA](https://arxiv.org/abs/2311.02262)与[PAI](https://arxiv.org/abs/2407.21771)提示head选择和视觉／文本权衡；[Gaze Heads](https://arxiv.org/abs/2606.14703)、[Attention is Case-Sensitive](https://arxiv.org/abs/2608.03711)等说明注意力集中与任务表现可能分离；[ASCD](https://arxiv.org/abs/2506.14766)和[VCD](https://arxiv.org/abs/2311.16922)提供输出分布对比的机制参照。原计划PAI/PASTA链接倒置已纠正。每篇机制、正文核验与适用限制见文献矩阵；未将本轮精确公式或效果冒称为这些论文的既有结论。

原+6/−6 bias将目标／其它视觉token的相对赔率放大exp(12)≈162755，并改变视觉相对文本的总量。本轮先在当前hook的Q/K下保持视觉总注意力，仅重新分配视觉内部注意力；再用正负干预产生的原生评分差作为方向，必要时扣除首帧重复参考中的响应。该时间差分是模型输入上的交互对比，不是对物理成功状态的随机因果干预。

最终公式如下，z均为同一模型完整原生评分档位的logits，h为原生分布的归一化熵：

```text
D = (z_observed,+ − z_observed,−) − α (z_static,+ − z_static,−)
z_parent = z0 + η D
h = H(softmax(z0)) / log(B)
z_final = z0 + s h^β (z_parent − z0) + ρ (z_original_same_k − z0)
```

α=0为区域差分，α=1为时间区域差分。η保留各parent的固定幅度、原生完成度调节或KL预算；最终s/β/ρ只调整其证据幅度和原方法分布参照。全部原生五档或十bin保留；推理及评分生成不读取真值、类别路径语义或同视频其它指令的分数，不将输出硬映射到1或5。标签只用于开发选型与离线评价。

|模型/输入|三个已测k（中间为primary）|parent：区域/时间，scope，η|s|β|ρ|
|---|---|---|---:|---:|---:|
|meter_image_text|64,68,72|区域，all_frames，64|1|0|1|
|meter_text_video|40,48,56|区域，last_frame，12|1|0|0|
|meter_video_text|56,64,72|时间，last_frame，64|1|0|1|
|qwen_image_text|32,40,48|时间，all_frames，2|1|0|0|
|qwen_text_video|40,48,64|区域，last_frame，4|1|0|0|
|qwen_video_text|64,68,72|时间，all_frames，KL=.5 / gain cap64|1|0|0|
|rr_image_text|60,64,68|时间，last_frame，16×(1−q_native)|1|0.5|0|
|rr_text_image|40,48,56|区域，all_frames，2|4|0.5|0.25|
|rr_text_video|48,56,64|时间，all_frames，4|4|0.5|0|

全部parent半径b=3。RoboReward三个输入共用β=.5；共享指数规则是在上一轮失败后提出，再根据原234开发候选的最弱余量选择。它是适应性探索，不能因为参数数值仍由234选择就重新宣称498独立。只支持所列三个已测k，不宣称区间内每个未测整数也通过；这是共享机制的可配置方法，不是全部模型共用同一套参数。

基础区域／时间方案分别需要3／5条模型支路；ρ>0多一条原original-target。最终Meter image_text/video_text为4／6条，RR text_image为4条，其余为3／5条。熵不增加神经前向。仅对称证据支路保留局部视觉总量守恒；ρ>0的整个方法不能称所有分支守恒。Qwen video_text实际s1/β0/ρ0，原KL预算保持；其它额外缩放不能自动继承父预算。

## 最终主指标的观测结果

全部9项×3个k在开发234、复用498和完整846均通过两个baseline的四向/+10pp、same-k原attention改良以及primary对开发预选原attention的严格比较。独立算术核对81个分区×邻居点使用严格不等式，没有性能容差。相对两baseline的全部已测点，完整846最小总增益14.1844pp、最小suc增益2.2388pp；复用498分别13.6546pp、1.8987pp。

以下是固定primary在完整grounded846上的数值；Meter的MAE列用连续ordinal，Qwen/RR用原生1–5。所有邻居、五档MAE、第二阈值和复用498绝对值均在[共享熵结果详表](SHARED_ENTROPY_RESULT_20260909T1932.md)。

|case|MAE native→final|总准确率 native→final|suc native→final|fail native→final|总增益pp|
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

两个baseline分别是原生生成／官方读出与同格式同读出的无干预评分，和“两套阈值”是不同概念。Meter所有邻居的五档MAE也下降。Meter text_video的0.2/0.8总准确率虽33.6879%→65.2482%，suc却58.9552%→44.0299%；复用498的suc56.3291%→44.9367%。因此**没有取得两套阈值所有指标同时改善的结论**。

![全部最终邻居的观测效果](../../results/mydata_bench/experiments_v2_corssmodel/auto_research/session_20260908/shared_entropy_evaluation_v1/figures_20260909T192735/all_selected_neighbor_effects.png)

## 失败阶段、不确定性与归因

|阶段|498点门槛/严格原方法|完整846点门槛/严格原方法|解释|
|---|---|---|---|
|原冻结独立确认v1|8/9；7/9|8/9；详见原分析|RR text_video独立确认失败；完整846的RR image_text邻居失败|
|第一次anchor|9/9；7/9|7/9；7/9|完整846只有RR text_image加Meter/Qwen六项通过|
|按输入单独选择熵参数|8/9；8/9|8/9；8/9|RR text_video的β0方案损害suc|
|最终共享熵|9/9；9/9|9/9；9/9|复用数据上的探索结果|

原RR text_video primary k56在498对same-k原attention的MAE多1/498=.002008，严格下降失败；原RR text_image虽同k通过，却不胜开发预选original k24。这些失败未通过换primary、删k、改容差或重写原manifest消除。原846的RR image_text k60总增益9.21986pp，也保持失败。

原确认按179个视频内容簇进行20000次bootstrap／符号翻转，跨9个primary作IUT/Holm。Meter三个输入与Qwen text_video的Holm p≈.000450，Qwen video_text≈.000750，Qwen image_text≈.031998；RR三项约.363132/.675466/.675466，成功类方向仍有不确定性。这些p检验改善方向，不检验+10pp的置信下界，也不能转移给最终新方案。最终方法没有新的确认性p值。

固定成分消融表明，收益不能全部归于熵。RR text_video固定s4时，加熵在846使suc+5.2239pp、MAE−.021277，但fail−2.9412pp、总−.3546pp；额外幅度带来相反的部分权衡。RR text_image加熵使suc+10.0746pp、fail−4.1522pp；Meter的原方法参照仅贡献总+1.0638/+1.5366pp。全部消融均在最终参数固定后运行，没有据此再选参。

9项完整wrong/low/zero控制使用已审计的真实神经支路；额外original参照所需的wrong/low来自此前实际补跑。498条whole-target重算与whole-zero均精确，但这不是重新跑了9项神经前向。RR text_video的正确区域相对wrong总准确率低13.2231pp、fail低37.8698pp而MAE/suc更好；RR image_text相对wrong的suc少1/158。正确区域没有在所有指标支配控制，不能宣称已证明物理任务完成度的因果理解。

任务与排序也有边界：九项task-macro MAE/准确率均改善，但RR image_text仍有10/28任务MAE变差、5/28任务准确率变差。543个配对的平均progress差均增加，但Meter三输入负序分别3→15、23→30、0→57；RR image_text负序减少而平分47→96。所有分布和任务均保留，不能把平均改善写成每任务或全部排序指标改善。

SOLE五输入的201规范整数读出screen各104个target候选均未达完整门槛。它固定自身前六步和末步reasoning，只覆盖论文规定−100..100规范格式，不等于完整词表或全部未来SOLE方法。新增全七步实验也未给出三输入稳定+10pp。Meter官方text_image未成为最终成功输入；Qwen video_text与RR公开HELM text_video有官方请求顺序来源，但固定帧数和推理设置仍有适配。最终方案不支配官方输入的每项指标，也未在机器人在线闭环或新任务上验证。

## 交付、复核与后续使用

- 最终完整方法：[shared_entropy_evaluation_v1/frozen_manifest_v1.json](../../results/mydata_bench/experiments_v2_corssmodel/auto_research/session_20260908/shared_entropy_evaluation_v1/frozen_manifest_v1.json)，[九配方注册表](../../results/mydata_bench/experiments_v2_corssmodel/auto_research/session_20260908/shared_entropy_evaluation_v1/final_recipe_registry_v1.json)。配置在`mydata_bench/configs/v2_crossmodel/addbase_shared_entropy_final_v1_*.yaml`。
- 复现流程与环境：[REPRODUCTION_GUIDE_20260909.md](REPRODUCTION_GUIDE_20260909.md)。这些最终YAML是方法配方；先用其引用的parent配置在新目录生成神经支路，再调用固定的全bin后处理与原生读出，不能直接当旧runner配置或覆盖本轮结果。
- 完成后的baseline源：[metrics](../../results/mydata_bench/experiments_v2_addbase/session_20260908/metrics_snapshot_20260909T205912.json)、[attention audit](../../results/mydata_bench/experiments_v2_addbase/session_20260908/attention_audit_snapshot_20260909T205847.json)、[40文件一致性审计](../../results/mydata_bench/experiments_v2_addbase/session_20260908/final_record_consistency_audit_v1.json)。
- 最终工程与算术：[173项冻结来源核对](../../results/mydata_bench/experiments_v2_corssmodel/auto_research/session_20260908/shared_entropy_evaluation_v1/final_frozen_lineage_audit_v1.json)、[81点直接验收](../../results/mydata_bench/experiments_v2_corssmodel/auto_research/session_20260908/shared_entropy_evaluation_v1/direct_final_acceptance_audit_v1.json)、[代数核验](../../results/mydata_bench/experiments_v2_corssmodel/auto_research/session_20260908/shared_entropy_evaluation_v1/algebraic_invariance_audit_v1.json)。最大档位置换浮点误差3.33955e−13，零点/父方案退化精确；此容差未用于性能门槛。
- 归因与统计：[固定消融](FINAL_POSTPROCESS_ABLATIONS_20260909T1948.md)、[原确认与anchor](FROZEN_AND_ANCHOR_RESULTS_20260909T1908.md)、[统计解释与11类偏差检查](STATISTICAL_VALIDATION_20260909.md)。
- 分布与任务：[完整报告](../../results/mydata_bench/experiments_v2_corssmodel/auto_research/session_20260908/final_descriptive_tables_v1/report.md)、[1980行最终task/split CSV](../../results/mydata_bench/experiments_v2_corssmodel/auto_research/session_20260908/final_descriptive_tables_v1/per_task_split_metrics.csv)、[第二阈值全部Meter邻居](../../results/mydata_bench/experiments_v2_corssmodel/auto_research/session_20260908/shared_entropy_evaluation_v1/secondary_threshold_descriptive_audit_v1.json)。
- 逐项完成检查：[final_completion_audit_v1.json](../../results/mydata_bench/experiments_v2_corssmodel/auto_research/session_20260908/final_completion_audit_v1.json)。历史阶段记录继续保留于[新增baseline总结](../exp_plan_addbase_summary.md)和[研究日志](RESEARCH_LOG_20260908.md)。

本轮推荐保留这套冻结配置作为下一轮新数据验证的候选。要形成独立泛化结论，应在未用于本次探索的新视频／新任务上检验，并继续检查成功类、第二阈值及任务排序；不能把现有498重新切分后称新的独立确认。这一后续建议不改变本轮已完成的实验与观测验收范围。

## Material Passport

- 模式：academic-research-suite / experiment-agent validate，内联执行，无子代理。
- 验证状态：ANALYZED；真实实验、零干预、重复前向控制、确定性组合和来源哈希已按所列范围核验。没有宣称最终方法获得独立全神经复现或新的独立统计确认。
- 来源：目标仓库代码与本轮冻结结果、用户指定的只读数据／grounding／权重、已核验公开论文。没有读取其它本地代码仓库；无本地git操作，旧数据和结果保留。
- 报告由AI执行并整理；人工审阅状态未标记为已完成。

符号补充：最终RR image_text的q_native是原生五档分布的归一化评分期望，不是答案正确性的概率；归一化熵同样不保证预测正确。全方案zero同时将证据和original参照归零，才精确回到baseline；仅去掉证据而保留非零ρ，会留下参照项。


## 2026-09-10：结论适用范围更正与续研状态

用户要求继续寻找避免此前过拟合风险的新方法后，本报告的旧方案不再计为当前目标3成功。这里记录的通过属于多轮研究选择后、复用数据上的主指标观测结果；原始独立确认失败，最终配方没有新的独立泛化证明。具体泄漏核查、确认复用和第二阈值问题见[方法可信度说明](METHOD_VALIDITY_AUDIT_20260910.md)。主线1、2的已完成实验/调研记录保留；主线3续研仍未完成，不再把旧观测达标表述为已证明稳定通用改进。新方案和外部一次性验证安排见[续研协议](ROBUST_RESEARCH_PROTOCOL_20260909.md)。
