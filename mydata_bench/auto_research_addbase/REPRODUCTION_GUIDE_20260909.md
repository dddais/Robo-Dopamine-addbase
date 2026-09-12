# 本轮研究的复核与复现入口

本文件提供可追溯入口，不表示所有实验已经结束。当前结果状态以主总结最新记录、各输出目录completion/combination事件和最终分析文件为准。仅操作Robo-Dopamine-addbase，保留旧文件，不执行git。

## 环境与实现

- 项目：`/home/dais/workspace/Robo-Dopamine-addbase`，物理路径`/mnt/public1/dais/workspace/Robo-Dopamine-addbase`。
- Python：`/home/dais/miniconda3/envs/robo-dopamine/bin/python`，3.10.20；PyTorch2.8.0+cu128；Transformers4.57.0。
- BASE：`results/mydata_bench/experiments_v2_addbase/session_20260908`。
- RESEARCH：`results/mydata_bench/experiments_v2_corssmodel/auto_research/session_20260908`。
- 确认前配置、模型路径、ranking SHA、输入文件SHA、核心源码SHA及内容快照入口均在`RESEARCH/confirmation_v1/frozen_manifest_v1.json`。模型推理使用本地既有权重，不在本轮另作训练。
- 确认启动后补充保存全部141个`mydata_bench` Python文件的清单：`confirmation_v1/implementation_source_inventory_20260909T1555.json`。其141文件清单是补充快照，不冒称全部在启动前预注册；其中原冻结的21个推理相关文件已核对，全部未变。内容按SHA存于BASE/source_snapshots。

## 参数与队列

最终九项候选来源为`confirmation_candidate_selection_20260909T151434.json`，准入检查和唯一冻结记录在`confirmation_v1/frozen_manifest_v1.json`。每个case有模型/输入、primary、三个相邻已测k、完整branch与combination配方、原方法比较设置。配置在`mydata_bench/configs/v2_crossmodel/addbase_confirmation_v1_*.yaml`。主方法共享守恒区域有限干预差分；具体参考、强度和k因模型/输入而异，不能宣称同一组超参数全模型通用。

运行顺序为`score_branches`保存原生全部bins与工程记录，`combine_scores`只按冻结配方离线组合，`analyze_confirmation`最后读取labels计算性能。labels不用于神经推理或组合。队列入口`run_confirmation_queue`按显存和指定case执行，每个输出目录带排他写锁。恢复时应同时检查/proc里的真实子进程、配置路径、cwd和completion事件；父shell/工具句柄消失不意味着模型进程已消失。

完整846的设置在`frozen_cohort_completion_v1/frozen_manifest_v1.json`；只对remaining114补神经前向，直接使用冻结开发234/确认498的相同条件结果。开发/确认/ranking重叠部分在分析中分别列出；846区间为描述性结果。

## 完成后的离线复核

以下命令要求相应输入已完成，会新增带时间戳的分析文件，不改原始预测。

```bash
cd /home/dais/workspace/Robo-Dopamine-addbase
PYTHON_RESEARCH=/home/dais/miniconda3/envs/robo-dopamine/bin/python
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$PYTHON_RESEARCH" -m mydata_bench.auto_research_addbase.analyze_confirmation
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$PYTHON_RESEARCH" -m mydata_bench.auto_research_addbase.analyze_frozen_cohort
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$PYTHON_RESEARCH" -m mydata_bench.auto_research_addbase.audit_added_attention
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$PYTHON_RESEARCH" -m mydata_bench.auto_research_addbase.metrics
```

SOLE数值screen的最终审计入口是`audit_sole_screen --folder <已完成case目录>`，必须等默认与补充读出全部完成。报告分别统计原始行、唯一键、科学字段冲突、完整覆盖、zero和201规范格式概率。已有重复行不删除；计量单位是唯一example_id/condition。

确认图表入口：`render_confirmation --analysis <已完成confirmation_analysis时间戳JSON>`，导出九项全部primary及两套baseline的点估计和95%视频簇区间。点估计门槛、三k邻域门槛与Holm/IUT分别报告，图中的95%区间不是同时置信区间。

## 新的一次神经复现

独立重新运行应使用新的结果目录，保留本轮manifest作为来源，并为新输出路径建立对应配置/来源SHA；不能覆盖本轮目录。现有队列在完成目录上会跳过已完成任务，再次调用旧配置不能宣称取得一份独立复现。输入、采样帧、模板、模型权重、attention后端、递归历史、readout、解析和软件版本均需保持一致。SOLE正式七步使用native dynamic cache和预加载索引；曾产生数值分岔的static-cache/CUDA-graph版本不用于正式结论。

研究中产生的原生格式失败、无法构造的wrong-region对照、未grounded数据及早期无效tokenizer版本都有单独边界。保持其原始记录，不从reasoning猜分，不补真值，不通过端点映射修饰指标。


## 2026-09-09 17:41：剩余114补跑转GPU2与自动raw审计

GPU2的Qwen text_video确认/114均完成后，只剩其它程序约20.6GiB占用，四卡均A100-SXM4-80GB。保存`RESEARCH/cohort_gpu2_handoff_20260909T1740_v1/plan.json`及launch/events，核对/proc命令、cwd和空children后，仅停止原114等待父2039243与2039245；**没有中断RR/Meter/Qwen/SOLE任何正在推理的模型，也没有停止确认父2037992/2037994**。六个Meter/RR 114case均确认此前从未launch，全部仍使用原冻结manifest/YAML/ranking。原冻结推理文件SHA再次全部一致。

新Meter114上层父2081412，在GPU2依次运行image_text/text_video/video_text；新RR114上层父2081413，等Meter image_text114完成后在GPU2依次运行image_text/text_image/text_video，避免同时模型加载。每个队列仍逐项检查实时显存，实际模型PID见该目录events。无需等待同模型全部498确认完成才对固定114前向：样本集不交叉，参数已冻结，114性能不会用于选型。这样可在确认尚运行时提前补齐数据，并在Meter确认结束后让GPU3只执行SOLE full。Qwen image_text114仍在GPU1，video_text确认/114仍按原接管计划GPU0执行。

新增自动raw审计守护PID2081699：`RESEARCH/final_raw_audits_v1/{plan.json,launch.json,events.jsonl}`。对确认九项+remaining114九项共18个case，已有14份audit完成，剩余22份工程/重组合审计将在各case的真实completion/combination标记出现后逐项执行。没有标签选择，不跑新的神经前向；任何错误保留退出码且不会静默重试。

为复用既有守护，`run_completion_analyses.py`只增量添加可选`--plan-root`，默认行为相同；原PID2040764及原九任务plan没有重启或修改。该调度脚本不属于冻结推理21文件，也不属于被冻结的metrics/paired_analysis/analyze_development三个统计实现。最终仍需检查两个watcher均正常、收取全体确认与846、最终BASE；再手动生成macro、`summarize_frozen_report`全邻居补充及图表并查看图片。


## 方法部署与计算范围补充

一次新的评分在区域差分模式下需要原生读出、目标正干预、目标负干预三条模型支路；时间区域差分需要再加首帧重复参考的正负干预，共五条。相邻k扫描、native generation、zero/wrong/low和温度等在本轮用于选择或验证，不能全部算成单个最终配方的必要在线支路。RR image_text的q由已需的原生五档分布计算，Qwen video_text的KL幅度也只由这些评分向量计算，无额外模型训练。

本轮主要实现仍顺序运行支路，未验证专门的在线并行服务吞吐或时延优化。`completed_recombination_audit_v1.json`保留逐支路实际耗时分布，但实验期间GPU共享、模型加载和输入预处理的范围不同，不能把这些计时当作控制条件一致的算法速度基准。部署还需要与指令匹配的grounding及冻结head ranking；本轮使用用户明确接受的grounded cohort，不据离线模型评分结果宣称已经完成在线机器人闭环验证。

全部已选配方保留原生评分bin；改变输出分布依赖区域干预产生的方向，不使用标签、suc/fail路径语义、同视频其它instruction的预测或人为端点映射。统计分析阶段才join标签与同视频配对关系。无标签确定性组合重算的通过，说明保存的raw logits与声明配方一致；它不单独证明grounding正确、因果理解成立或神经前向跨平台逐bit可复现。


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


## 2026-09-09 19:36：最终观测方法入口

最终方法注册表`RESEARCH/shared_entropy_evaluation_v1/final_recipe_registry_v1.json`指向九份`mydata_bench/configs/v2_crossmodel/addbase_shared_entropy_final_v1_*.yaml`，含parent神经配置、SHA、每个k的精确parent组合与s/β/ρ。这些YAML是方法配方，不是score_branches原生runner配置；新神经复现须先把所引用parent配置指向全新的输出目录并保留ranking/input/源码关系，再对得到的parent/native/original向量调用`entropy_development.entropy_scaled_logits`，用原`score_readout`读取。不可覆盖本轮任何结果。

`shared_entropy_followup.py`复用冻结的`entropy_evaluation.run`及`entropy_whole_controls.run`，仅将模块输出根设为新版本目录；未改其源码或旧版本参数。旧参数方案的失败留在独立目录。该版本的`evaluate`/`controls`在现有结果目录上会拒绝重跑，不能把再次调用已完成目录当作独立神经复现。

最终汇总`shared_entropy_evaluation_v1/summary_v1.json`，每case含四个分区与全量任务/分布/配对；完整controls在`shared_entropy_whole_controls_v1/analysis_v1.json`，来自已审计的真实支路而非新九项神经运行。两任务watcher`shared_entropy_completion_watcher_v1/events.jsonl`均记录returncode0。配方、数据选择与独立性边界见`SHARED_ENTROPY_RESULT_20260909T1932.md`。


## 2026-09-09 21:08：全部完成后的精确复核入口

两个SOLE mandatory full均已完成。`completion_analyses_v1`九个任务与`final_baseline_record_audit_watcher_v1`的记录任务均exit0且watcher_completed。无本轮推理/调度进程继续运行。最终源固定为：

- `BASE/metrics_snapshot_20260909T205912.json`
- `BASE/attention_audit_snapshot_20260909T205847.json`
- `BASE/final_record_consistency_audit_v1.json`
- `BASE/task_macro_final_v1.json`（180组）
- `BASE/final_added_baseline_report_v1/`（15840指标行、180配对组、45target点、完整top8/重合表、PNG/SVG，图已目视核验）

本轮实际执行了如下新增离线工具；再执行时，`--output`必须使用新的文件／目录。它们拒绝覆盖原结果，且需要全部完成的输入。

```bash
PYTHON_RESEARCH=/home/dais/miniconda3/envs/robo-dopamine/bin/python
"$PYTHON_RESEARCH" -m mydata_bench.auto_research_addbase.task_macro_summary --sources <最终metrics.json> --output <新的macro.json>
"$PYTHON_RESEARCH" -m mydata_bench.auto_research_addbase.final_added_baseline_report --metrics <最终metrics.json> --attention-audit <最终attention_audit.json> --record-audit <最终record_audit.json> --macro <同一metrics的macro.json> --head-overlap <BASE/all_four_models_head_overlap_v1.json> --output <新报告目录>
"$PYTHON_RESEARCH" -m mydata_bench.auto_research_addbase.render_added_final --report-root <新报告目录>
```

尖括号为需替换的路径占位符，不直接作为shell参数执行。冻结方法的配置和神经复现流程仍按上节九份最终recipe；本报告工具只复核与导出已有实验，不产生新的独立神经复现。

`audit_final_frozen_lineage`的173项核验已通过；首版报告检查误把控制的几何失败也要求为0，失败记录保存在`shared_entropy_evaluation_v1/final_frozen_lineage_audit_engineering_attempt_v1.json`。修正后的检查保留控制错误，只要求最终主方案完整，没有改任何冻结源码或分数。

最终主门槛使用.125/.875。第二阈值.2/.8下Meter text_video有suc下降，详见`secondary_threshold_descriptive_audit_v1.json`。原确认失败和后续数据复用依然有效；最终阅读入口为[独立总报告](FINAL_RESEARCH_REPORT_20260909.md)。整轮机器完成清单`RESEARCH/final_completion_audit_v1.json`检查主线范围与所有必需产物，并显式保持独立泛化、两阈值全面提升等未成立状态为False。
