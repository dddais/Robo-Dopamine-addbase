# 新增 baseline 与 attention 实验总结

> 2026-09-08 启动；本文件按完成事件追加。**研究正在执行，尚未完成全部主线，尚无最终有效方案。** 不将部分样本或最优筛选点当作验收成功。

## 协议与产物

- 唯一目标代码库：`/home/dais/workspace/Robo-Dopamine-addbase`。
- 依据根目录 `exp_plan_addbase.md` 执行。auto-explore 所写 `mydata_bench/exp_plan_addbase.md` 不存在，已记录路径差异。
- 新增执行器：`mydata_bench/meter_eval`、`mydata_bench/top_eval`；共同执行/指标代码：`mydata_bench/auto_research_addbase`。原有评估代码未改动。
- 配置：`mydata_bench/configs/v2_crossmodel_addbase`；结果：`results/mydata_bench/experiments_v2_addbase/session_20260908`。
- 全数据 1,213 条（407 suc / 806 fail），grounding cohort 846 条（268 suc / 578 fail），ranking 34 条。
- 主线 3 另以视频内容 SHA-256 排除 ranking 重叠后固定开发集 234 条、确认集 498 条。初筛 60 条来自开发集的 24 个完整视频组。主线 1 的 846 全 cohort 复现与主线 3 的独立确认分开解释。
- 全部推理只接收任务文字与解码后的图像/视频；真值只由离线 metrics 代码加入。原始分数、无效输出和控制条件错误均保留。

## 新模型的输入构造

| 模型 | 输入 | 具体实现 |
|---|---|---|
| Robometer | text→image（官方） | 官方任务 prompt，8 个均匀帧，每帧后跟训练过的 `<\|prog_token\|>`，原始 progress/success heads 读出 |
| Robometer | image→text | 8 张图后放任务文字，再放最终 progress token，保证读出能看到任务；这是自定义适配 |
| Robometer | interleaved | 官方逐帧结构加 Frame 序号提示；自定义适配 |
| Robometer | text→video / video→text | 8 源帧构成原生 video tokens，保留各 temporal patch 对应的源帧索引，最终 progress token 读出；自定义适配 |
| SOLE-R1 | image→text（官方输入） | 官方 external-only 的 first / previous / current 三列拼图与原始 prompt；8 帧共 7 次递归，previous 使用本条件自己的上一预测 |
| SOLE-R1 | text→image | 相同拼图与上下文，交换图文顺序 |
| SOLE-R1 | interleaved | 三个历史时刻各为独立图像，以 FIRST / PREVIOUS / CURRENT 文本对应 |
| SOLE-R1 | text→video / video→text | 三个历史时刻各重复两次，保持各自的 temporal patch；按同一七步递归评估 |

SOLE 为可复现的配对实验使用 greedy / max_new_tokens=512，官方当前 RewardGen 的默认随机参数为 temperature=1 / top_p=.9 / top_k=50 / max_tokens=200。因此“官方输入”不等同于逐参数复现随机解码。首时刻 0 是官方上下文定义；末时刻分数完全来自模型，不按成功/失败标签设为 0 或 1。

## 已完成的全量 baseline（2026-09-08 22:18）

连续模型的 `连续 MAE` 为 `mean(|1+4p−label|)`；`离散 MAE` 用仓库原五级阈值映射后计算。准确率是端点精确匹配，低端 `p<0.125`、高端 `p≥0.875`。所有分母为预定样本数。

| 模型 / 输入 | 集合 | n / 有效 | 连续 MAE | 离散 MAE | 总准确率 | suc | fail | 0.2/0.8 总准确率 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Robometer / 官方 text→image | 全量 | 1213 / 1213 | 1.9547 | 1.9777 | 31.57% | 92.14% | 0.99% | 37.02% |
| Robometer / 官方 text→image | 同 grounding cohort | 846 / 846 | 1.7648 | 1.7719 | 31.09% | 95.15% | 1.38% | 37.23% |

该 baseline 强烈偏向高进度，不能只依据 suc 准确率判断有效。注意力干预必须同时改善 suc 与 fail，而不是把总体分数整体压低。完整 per-task、两套阈值、分布和配对统计在不可覆盖的 `metrics_snapshot_20260908T221840.json` 中；该快照里其它未完成实验标为 incomplete，不用于比较。

## 已完成的实现验证

- Robometer、SOLE 权重检查：missing/unexpected/mismatched keys 均为空；保留全部训练 heads。
- 两个模型五种输入的单样本 forward / generation 均已跑通；SOLE 最初 system 内容类型异常保存在 `sole_smoke_v1`，修复后 `sole_smoke_v2` 五种输入均成功。
- Robometer 五种输入的 ranking、原始 bias、守恒 bias smoke 已通过；零 bias 与其对应基线一致。
- 四项 CPU attention 不变量测试通过：零干预、未来因果遮罩、未选 head 不变、视觉质量与非视觉概率守恒。
- ranking 使用前 8 层之外的 head；Robometer query 为末个训练 progress token，生成模型为末 prompt token。必须显式保留 ranking 的因果遮罩。
- wrong-region 与 low-rank 的任何失败仅影响该条件，不再级联跳过其他条件。
- SOLE 16 样本批量试跑完成全部 7 步、16/16 有效，约 87 秒；全部中间推理文字与上一预测均已落盘。

## 待完成

其它输入全量 baseline、新增模型全部 ranking / attention / controls、top-8 与 top-8/32/64 重合度汇总，以及跨模型改进方案的独立确认仍在执行。最终结论须以完整结果追加，不把此阶段文件当作最终交付。

文献与机制分析见 [LITERATURE_AND_HYPOTHESES_20260908.md](auto_research_addbase/LITERATURE_AND_HYPOTHESES_20260908.md)，执行边界与日志见 [RESEARCH_LOG_20260908.md](auto_research_addbase/RESEARCH_LOG_20260908.md)。

## 2026-09-08 22:40 精度审计更正与新增全量结果

**以下表格采用官方 float32 expected-bin 读出，优先于前面的初始快照。** 官方 Robometer 把原始 progress logits 转到 CPU float32 后再 softmax 加权。初版保存的派生 `progress` 使用 bf16，最大差0.0044543，导致3条的五级分档变化。原始 logits 均已保存；新离线读出重构不需要重新跑模型，也不改写任何原始预测。总准确率的主要结论不变。

| Robometer 输入 | 全量有效数 | 连续 MAE | 离散 MAE | 总准确率 | suc | fail | 0.2/0.8 总准确率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| text→image（官方结构） | 1213 | 1.9543 | 1.9753 | 31.57% | 92.14% | 0.99% | 36.93% |
| text→video | 1213 | 1.6839 | 1.7181 | 4.37% | 3.44% | 4.84% | 25.97% |
| video→text | 1213 | 2.0635 | 1.9975 | 0.00% | 0.00% | 0.00% | 0.00% |
| image→text | 1213 | 2.2265 | 2.3092 | 0.00% | 0.00% | 0.00% | 0.00% |

图像在文字之前的两个适配中，预测集中于中间进度，端点准确率为0，不能把它解释为模型在这些输入上没有视觉信息。更直接的解释是训练 progress-token 读出契约与输入顺序变化不匹配；之后的干预需与各自同输入基线比較。

新增计算审计：官方两阶段 processor 与当前 direct processor 的文本token和图像grid相同；少量resize舍入使像素张量平均绝对差6.84e-6、最大0.007843。因此这里准确称“官方输入结构”，不声称像素逐bit一致。

SOLE batch attention的初始零bias审计发现长推理对backend舍入敏感；正式attention将使用基线/干预统一的显式causal mask与math SDPA，并在独立pilot通过完整递归的零干预一致性后才启动。初始pilot保持为工程验证记录，不进入效果表。

来源：`results/mydata_bench/experiments_v2_addbase/session_20260908/metrics_snapshot_20260908T224024.json`。该快照中尚未完成的interleaved和SOLE只用于进度监控，不纳入上述全量表。

## 2026-09-08 23:09 快照：Robometer 五输入完整 baseline

来源：`results/mydata_bench/experiments_v2_addbase/session_20260908/metrics_snapshot_20260908T230917.json`。五种输入各1213条，无缺失/无效；下表连续MAE均从原始十-bin logits按官方CPU float32重新读出。

| 输入 | 连续MAE | 五档MAE | 总准确率 | suc准确率 | fail准确率 |
|---|---:|---:|---:|---:|---:|
| text_image | 1.9543 | 1.9753 | 31.57% | 92.14% | 0.99% |
| text_video | 1.6839 | 1.7181 | 4.37% | 3.44% | 4.84% |
| video_text | 2.0635 | 1.9975 | 0.00% | 0.00% | 0.00% |
| image_text | 2.2265 | 2.3092 | 0.00% | 0.00% | 0.00% |
| interleaved | 1.9689 | 1.9868 | 31.16% | 92.14% | 0.37% |

完整的两套阈值、每个task/split的预测分布、同视频配对差和head重合度均在该JSON快照中。输入变化影响很大：官方text→image及interleaved多数输出高进度，image→text与video→text则落在中间分档，严格端点准确率为0。**不能把这种格式失配解释为端点阈值可以随意改动。**

### 已完成的同846条cohort原始attention结果

| 输入 | 条件 | 连续MAE | 总准确率 | suc准确率 | fail准确率 |
|---|---|---:|---:|---:|---:|
| text_image | baseline | 1.7645 | 31.09% | 95.15% | 1.38% |
| text_image | target_k8 | 1.0348 | 26.48% | 74.63% | 4.15% |
| text_image | target_k32 | 1.0219 | 23.88% | 63.43% | 5.54% |
| text_image | target_k64 | 1.1867 | 21.39% | 55.60% | 5.54% |
| text_image | low_rank_k8 | 1.7709 | 30.61% | 95.15% | 0.69% |
| text_image | low_rank_k32 | 1.7444 | 29.91% | 93.28% | 0.52% |
| text_image | low_rank_k64 | 1.8132 | 29.43% | 92.91% | 0.00% |
| text_video | baseline | 1.5156 | 5.67% | 5.22% | 5.88% |
| text_video | target_k8 | 1.2215 | 8.04% | 9.33% | 7.44% |
| text_video | target_k32 | 1.0987 | 18.68% | 1.12% | 26.82% |
| text_video | target_k64 | 1.1837 | 12.06% | 1.87% | 16.78% |
| text_video | low_rank_k8 | 1.5493 | 5.67% | 5.22% | 5.88% |
| text_video | low_rank_k32 | 1.5360 | 4.49% | 4.48% | 4.50% |
| text_video | low_rank_k64 | 1.5585 | 2.36% | 5.60% | 0.87% |

原强bias在官方输入的MAE下降，但suc与总准确率下降，不满足主线3。text→video k32总准确率+13.00个百分点，但suc从5.22%降至1.12%，同样不满足。wrong-region分别有7/10条视频指令的几何匹配不可行，各k均保留显式错误；后续区域因果对照必须使用各自可行的同样本子集。

### 实际top-8（层与头索引均从0开始）

- Meter text_image: L19H21, L19H10, L18H30, L21H16, L19H17, L19H16, L22H15, L22H26
- Meter text_video: L19H21, L19H10, L19H23, L19H17, L19H16, L22H26, L22H15, L21H16
- Meter video_text: L22H15, L21H11, L12H18, L20H29, L19H15, L14H17, L18H30, L19H21
- Meter image_text: L19H21, L22H15, L21H11, L21H16, L18H30, L22H2, L21H19, L19H15
- Meter interleaved: L19H21, L19H10, L18H30, L22H15, L19H17, L21H16, L19H16, L22H26

SOLE五输入完整baseline/attention仍未全部完成。此节只确认Robometer已完成项，不代表主线1或主线3已完成。

## 2026-09-09：SOLE官方结构完整baseline

1213条全量推理完成，记录无推理/解析错误。使用官方external-only首帧/前帧/当前帧拼图与系统/用户prompt，每个视频执行7次递归、上一进度均由模型自身给出。Greedy max512及8个源帧的偏离已在方法节说明；本输入batch16。

| 范围 | N | 连续MAE | 五档MAE | 总准确率 | suc准确率 | fail准确率 |
|---|---:|---:|---:|---:|---:|---:|
| full | 1213 | 1.4757 | 1.4378 | 42.46% | 21.13% | 53.23% |
| cohort | 846 | 1.3753 | 1.3180 | 46.69% | 29.10% | 54.84% |

完整task/split分布、两套端点阈值及same-video配对差保存在`results/mydata_bench/experiments_v2_addbase/session_20260908/sole_official_complete_baseline_metrics_v1.json`。另外四输入的native baseline正在batch64队列；各自attention会另算相同batch的无干预对照。五种输入的完整七步attention与终步分解均已冻结，仍未全部完成。

### 对此前“全部有效”的精度限定

此前Meter表格中“无无效记录”指runner未抛异常且数值有限。进一步核验发现官方progress转换的sum==1类型猜测会把text_video中2条含负数的logits误当概率，读出3.6979及1.0661，均超出[0,1]。这些官方API兼容结果保持不动，不能把它们理解为合法进度。对应typed softmax值为0.7456及0.5594；后续追加typed读出敏感性，并在新方法中同时比较native API与typed-readout baseline。


### 2026-09-09：Meter 官方概率猜测的完整敏感性检查

从同一份原始十-bin logits 重构，逐输入核对完整1213条。默认 baseline 继续复现官方整序列 CPU float32 转换；typed 列显式对已知 logits 作 softmax，仅为读出敏感性，不计作 attention 改进。

|输入|不同条数|官方连续MAE|typed连续MAE|官方准确率|typed准确率|
|---|---:|---:|---:|---:|---:|
|text_image|0|1.954338|1.954338|31.5746%|31.5746%|
|text_video|2|1.683867|1.674138|4.3693%|4.2869%|
|video_text|0|2.063455|2.063455|0.0000%|0.0000%|
|image_text|0|2.226536|2.226536|0.0000%|0.0000%|
|interleaved|0|1.968878|1.968878|31.1624%|31.1624%|

只有 text_video 的两条 logits 误触发官方 sum==1 概率猜测；逐任务、两阈值、分布与配对统计均保存在 `results/mydata_bench/experiments_v2_addbase/session_20260908/meter_baseline_typed_readout_sensitivity_v1.json`。不覆盖任何预测记录。


### 2026-09-09 00:58：开发集扩展与控制结论（尚未独立确认）

- Qwen text→video，对称区域证据对比在完整234条开发集（76 suc/158 fail）中维持有效。lambda4下，k24/32/40/48/64的总准确率改善分别+15.38/+20.51/+30.34/+30.77/+27.78个百分点，全部MAE下降且两类准确率上升。k40/48的同k原强bias补充已经完成，改良对照门槛也通过。共同前缀读出与native生成3/234条不一致；全部234条零干预logits精确不变。
- k32/64新方案相对原强bias：MAE−0.2863/−0.3547，总准确率+14.53/+27.78pp，suc+2.63/+15.79pp，fail+20.25/+33.54pp。相对低排名heads的同机制对照也全部四项正向。
- 同时间等面积wrong-region在229/234条可构造。k32/64目标相对wrong的总准确率分别+11.79/+16.59pp，成功类+52.05/+47.95pp；失败类分别−7.05/+1.92pp。这支持任务相关区域信息的作用，同时表明错误区域也可能通过整体分数偏低改善fail，不能只以fail或总准确率单独归因。完整簇bootstrap与控制可行子集见 `score_development_full_v1/qwen_text_video/paired_region_and_original_controls_v1.json`。
- **RR video→text 的60条初筛改善没有在234条中达到门槛**：完整baseline MAE1.1966，accuracy56.41%；目前最佳双类方案last b3 k32 lambda2为MAE−0.1410，accuracy仅+5.13pp，suc+7.89pp，fail+3.80pp。不能将它计入稳定有效输入。其它已冻结候选和新readout-query候选继续。
- Meter baseline typed敏感性和SOLE 64条完整七步零干预审计均已补齐。SOLE index预加载在8条×两输入非零干预上保持原dynamic cache与原始token结果完全一致，完整七步工程pilot继续；static CUDA graph仍仅为工程实验，未替换正式结果。

以上均不构成最终验收；确认498的新方法性能尚未用于选型。尚需完成SOLE其它输入baseline/全部完整七步attention、三模型各至少三输入的完整开发与冻结确认，以及最终完整cohort汇总。


### 2026-09-09：新增模型五输入 baseline 已全部尝试

完整统计来源：`results/mydata_bench/experiments_v2_addbase/session_20260908/metrics_snapshot_20260909T111857.json`。每格准确率分母包括所有预期样本；MAE仅对有效数值报告，缺失/格式错误单列。连续MAE为 |1+4p−y|，五档MAE由统一分档器计算。

|模型|输入|有效/1213|连续MAE|五档MAE|总准确率|suc|fail|
|---|---|---:|---:|---:|---:|---:|---:|
|meter|text_image|1213/1213|1.9543|1.9753|31.57%|92.14%|0.99%|
|meter|text_video|1213/1213|1.6839|1.7181|4.37%|3.44%|4.84%|
|meter|video_text|1213/1213|2.0635|1.9975|0.00%|0.00%|0.00%|
|meter|image_text|1213/1213|2.2265|2.3092|0.00%|0.00%|0.00%|
|meter|interleaved|1213/1213|1.9689|1.9868|31.16%|92.14%|0.37%|
|sole|text_image|1213/1213|1.5179|1.4889|30.83%|4.67%|44.04%|
|sole|text_video|1204/1213|1.8319|1.8331|9.07%|17.44%|4.84%|
|sole|video_text|1213/1213|1.8622|1.8631|8.99%|22.60%|2.11%|
|sole|image_text|1213/1213|1.4757|1.4378|42.46%|21.13%|53.23%|
|sole|interleaved|1213/1213|1.5342|1.5276|27.04%|6.63%|37.34%|

Meter 默认仍使用已披露的官方 CPU float32 整序列概率猜测读出，typed-softmax敏感性另列。SOLE采用官方image→text montage/prompt/7步绝对进度递归与确定性greedy；官方默认随机采样和宽松字符串parser未复刻。text→video的9条失败来自原生格式错误（其中部分伴随token截断），保留raw文本，不提取推理中的数字来填补终端分数。

完整cohort846、逐task、suc/fail分布、两套阈值、same-video配对分档均在该JSON中。当前全七步attention的image→text、text→image已完成；其余输入与终步补充仍在运行，不能将本节视为主线1全部结束。

|模型/输入|具体top8（layer/head从0开始）|
|---|---|
|meter_text_image|L19H21、L19H10、L18H30、L21H16、L19H17、L19H16、L22H15、L22H26|
|meter_text_video|L19H21、L19H10、L19H23、L19H17、L19H16、L22H26、L22H15、L21H16|
|meter_video_text|L22H15、L21H11、L12H18、L20H29、L19H15、L14H17、L18H30、L19H21|
|meter_image_text|L19H21、L22H15、L21H11、L21H16、L18H30、L22H2、L21H19、L19H15|
|meter_interleaved|L19H21、L19H10、L18H30、L22H15、L19H17、L21H16、L19H16、L22H26|
|sole_text_image|L22H2、L19H17、L10H12、L24H31、L26H0、L22H0、L24H13、L21H16|
|sole_text_video|L10H12、L19H23、L21H16、L15H25、L22H0、L10H13、L14H29、L8H22|
|sole_video_text|L26H26、L28H16、L31H15、L10H12、L27H16、L31H29、L31H9、L34H0|
|sole_image_text|L24H13、L29H1、L30H8、L34H0、L31H15、L16H15、L24H28、L8H22|
|sole_interleaved|L22H0、L19H23、L10H12、L22H2、L19H28、L10H13、L15H25、L31H15|
|qwen_text_image|L21H25、L21H16、L19H31、L22H15、L19H23、L20H15、L19H28、L21H29|
|qwen_text_video|L21H16、L21H25、L19H23、L20H15、L10H13、L22H15、L19H28、L19H31|
|qwen_video_text|L25H10、L24H13、L12H18、L21H16、L21H18、L24H12、L12H20、L13H17|
|qwen_image_text|L24H13、L21H16、L12H18、L20H15、L21H25、L23H13、L27H16、L19H28|
|qwen_interleaved|L20H15、L19H23、L19H28、L21H25、L21H31、L21H27、L21H18、L23H13|
|rr_text_image|L22H15、L19H28、L19H31、L22H5、L21H16、L19H10、L20H15、L19H17|
|rr_text_video|L22H15、L26H31、L19H28、L19H23、L21H25、L21H16、L19H17、L20H15|
|rr_video_text|L18H30、L12H18、L20H15、L21H16、L21H27、L14H7、L19H28、L13H17|
|rr_image_text|L19H28、L21H16、L19H16、L20H15、L22H15、L19H0、L18H30、L19H31|
|rr_interleaved|L19H28、L20H15、L19H31、L26H31、L20H13、L23H13、L19H23、L24H13|

四模型同输入的top8/32/64交集计数见 `results/mydata_bench/experiments_v2_addbase/session_20260908/all_four_models_head_overlap_v1.json`。这是head坐标重合，不代表功能等价。


## 2026-09-09 12:20–12:40：Qwen邻域、RoboReward时间扩展与SOLE原生格式修复

- Qwen第二候选在完整234开发集补齐k64邻域：image_text在last/b3/lambda4下k56/60/64/68/72均四项改善、总增幅超过10pp；video_text在同参数k60/64/68通过（k56/72未通过）。text_video原有k24/32/40/48/64通过。统一primary k64及相邻k60/68可作为开发候选，但尚无确认结论。相关前瞻补充及源记录位于score_development_full_v1各输入目录。
- RoboReward时间对比text_image全234已完成：例如all/k40/lambda3，MAE−0.6282、总准确率+54.27pp、suc+10.53pp、fail+75.32pp；多个相邻k通过。interleaved全234仍运行。新增text_video初筛all/k32与k64/lambda4均总+18.33pp、两类上升并胜过原bias；video_text初筛all/k64/lambda1为总+16.67pp、两类上升且胜过原bias。因此前瞻冻结这两个输入的234扩展、k24/32/40/48/56/64/72、lambda1/2/3/4和原bias/wrong/low/VCD对照；接续GPU2已授权队列。没有据初筛直接宣布稳定有效。
- 第五轮score_local_radius_development_v1冻结15case，b0.5/1.5全query、中心差分gain3/6/12/24，分开检验有限差分半径与前轮readout-query限制因素。Meter text_image已完成，未过门槛；其余继续。
- SOLE新增terminal answer distribution适配器固定每个样本自身前六步历史及末步原生reasoning，保留−100..100全部201个整数百分比，并用prefix-free token trie的联合概率评估，计入负号、数字长度与结束符；无端点映射。它不替代主线完整七步attention。
- 无标签工程v2虽然零对照精确，但随后格式审计发现native20%/15%的合法候选总概率仅约4e−8/8e−7，属于实现错误：单独编码%得到id4，原生完整回答的%</answer>则把%</融合为id52508。旧sole_score_development_v1在validity_annotations.jsonl明确整体排除；原始错误结果完整保留。仅停止已确认本目录的旧SOLE score进程1986614/1987484/1987487，未动其它任务。
- 修复为先tokenize完整原生结束标签，再依据offset_mapping截取到覆盖百分号的完整canonical token边界。合成融合token测试、201整数保留测试、符号联合概率测试、真实小Qwen缓存对照及时间对比测试共7项通过；真实SOLE tokenizer的201候选均以52508结束、prefix-free。新的工程v3及sole_score_development_v2使用新目录，不覆盖旧输出。真实v3首样本baseline合法格式总概率0.999949、MAP15%与native15%一致，零对照logits精确；两样本全部支路审计通过后才自动进入60条五输入初筛。
- 主线1当前SOLE text_video全七步846×11已尝试完成（9217有效/89错误，待按条件分类）；video_text与interleaved全七步、text_video终步仍运行。不能将“所有baseline完成”写作“主线1完成”。确认498的新方法性能仍未参与选型。


## 2026-09-09 12:50：已完成SOLE完整七步attention的核验

三个输入已完成846×11全部尝试，以下均为每个条件独立完整七步递归；baseline与干预使用同一batch实现，不与不同batch的旧native结果混减。全量逐任务、分布、same-video配对及两套阈值见 `results/mydata_bench/experiments_v2_addbase/session_20260908/metrics_snapshot_20260909T124540.json`。

|输入|条件|有效/846|连续MAE|五档MAE|总准确率|suc准确率|fail准确率|0.2/0.8总准确率|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|image_text|baseline|846/846|1.3814|1.3416|44.33%|30.97%|50.52%|54.96%|
|image_text|target_k8|846/846|1.4079|1.3735|43.03%|30.97%|48.62%|54.85%|
|image_text|target_k32|846/846|1.5358|1.4905|42.67%|23.88%|51.38%|50.59%|
|image_text|target_k64|846/846|1.3385|1.3050|46.45%|32.09%|53.11%|56.74%|
|image_text|zero_bias_k64|846/846|1.3814|1.3416|44.33%|30.97%|50.52%|54.96%|
|text_image|baseline|846/846|1.4437|1.4125|30.97%|6.34%|42.39%|45.39%|
|text_image|target_k8|846/846|1.4196|1.3759|31.68%|10.07%|41.70%|43.74%|
|text_image|target_k32|846/846|1.3924|1.3522|32.74%|8.21%|44.12%|45.51%|
|text_image|target_k64|846/846|1.3661|1.3168|38.89%|9.33%|52.60%|49.88%|
|text_image|zero_bias_k64|846/846|1.4437|1.4125|30.97%|6.34%|42.39%|45.39%|
|text_video|baseline|841/846|1.8844|1.8894|10.05%|23.88%|3.63%|17.02%|
|text_video|target_k8|844/846|1.8478|1.8555|10.87%|25.75%|3.98%|17.49%|
|text_video|target_k32|839/846|1.9071|1.9154|9.34%|19.03%|4.84%|19.15%|
|text_video|target_k64|809/846|1.8952|1.9011|9.46%|22.01%|3.63%|16.78%|
|text_video|zero_bias_k64|841/846|1.8844|1.8894|10.05%|23.88%|3.63%|17.02%|

`attention_audit_snapshot_20260909T123909.json`已对全部已完成SOLE零条件逐例核验status、错误、终值、完整轨迹、raw reasoning和单步trace：零bias与baseline完全一致，包括text_video的相同原生格式失败。text_video共89条错误中72条为严格numeric answer格式失败、17条为wrong-region几何不可行；target_k64有37/846条原生格式失败。准确率分母保留846，MAE仅有效样本计算，不能忽略覆盖率差异。image_text/text_image各21条错误仅为wrong-region几何不可行。

这些原强bias结果没有达到稳定+10pp要求，主要作用是补齐主线1并为新方案提供同实现对照。interleaved/video_text完整七步和text_video终步仍运行；本节不是主线1完成声明。


### 2026-09-09 12:51：同强度对照修正归因，冻结234扩展

同lambda的uniform_strength_matched_supplement_v1对照表明：Meter image_text/video_text在不加置信度调节、lambda16时也能通过60条初筛，且MAE略优于调节版本。image_text all/k64/lambda16为MAE−1.1149、总+28.33pp、suc+30pp、fail+27.5pp；video_text all/k64/lambda16为MAE−0.9336、总+38.33pp、suc+5pp、fail+55pp。**因此这些Meter初筛收益目前应归于更大的区域证据放大幅度，不能归于辅助成功head或置信度调节。**

Meter image_text/video_text/text_video的234扩展已前瞻冻结在score_native_confidence_development_full_v1：k24/32/40/48/56/60/64/68/72，lambda4/8/12/16，不调节与power1/2完全配对，原bias、wrong、low及零对照。前两输入all scope，text_video last scope；每case38神经支路/194读出，已复用同条件源输出并校验无重复数值差异。GPU0按可用显存接续，确认集仍未运行。

RoboReward image_text则只有置信度调节的时间对比通过初筛，匹配lambda的无调节对照均未过四向+10pp门槛；best last/k64/lambda8/power1为MAE−0.85、总+31.67pp、suc+10pp、fail+42.5pp。该输入也冻结234扩展，含相同k/幅度/调节消融和两类区域控制，GPU3等待自身Qwen时间初筛队列完成后执行。所有这些是开发探索，需观察独立视频组扩展，不因best60结果宣布成功。


## 2026-09-09 13:40：终步补充全量完成与可视化

SOLE五输入的terminal-step补充均已完成846×11全部尝试；attention_audit_snapshot_20260909T133441.json显示，各输入全量零bias与baseline的status、错误、终值、完整轨迹、raw reasoning及单步trace均精确一致。text_video终步76条错误由55条缺native历史（5样本×11条件）、15条wrong-region几何不可行和6条新增原生格式错误构成；native缺历史不伪造、不补分。它们仍计入expected分母。

主线1尚余SOLE interleaved/video_text的完整七步，不能将已完成的终步补充替代完整递归。

![新增两模型原生baseline四项指标](../results/mydata_bench/experiments_v2_addbase/session_20260908/figures/added_native_baselines_v1.png)

图的可编辑矢量版：`results/mydata_bench/experiments_v2_addbase/session_20260908/figures/added_native_baselines_v1.svg`。源为metrics_snapshot_20260909T124540.json。可见Meter在image_text/video_text的两类端点准确率均为0，但连续MAE不是满损失，说明其native分数主要停在中间；输入顺序与读出标度必须与注意力定位质量分别分析。


### 2026-09-09 13:41：SOLE百分比范围的来源与原生越界值

重新核对官方论文正文（references/2603.28730_plain.txt第414–416行）明确给出p_t∈[−100,100]；官方system prompt要求signed integer percentage（rewardgen/sole.py第42–46行）。因此201点读出覆盖的是论文规定范围内的规范整数值，不是模型整套词表或所有可能的不规范字符串。

无标签遍历五输入原baseline全部递归数值后，未发现小数百分比；image_text的8491次递归数值中有1次原生101%，其余输入均落在论文范围。该101%是模型生成的越界数值，旧native baseline及上下文原样保留，未裁剪或改写。201点候选读出与原生生成作为不同读出并列，效果需同时胜过两套baseline；不把范围约束或格式修复算作steering收益。完整定位见sole_native_percentage_contract_audit_v1.json。


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


### 2026-09-09：样本、视频簇和配对分母的精确范围

`frozen_cohort_completion_v1/partition_pairing_coverage_audit_v1.json`核验：原始1213条为407视频簇，407 suc/806 fail；grounded846为299视频簇，268 suc/578 fail；开发234为86簇、76/158；确认498为179簇、158/340；ranking重叠114为34簇、34/80。划分时同一视频在eligible grounded cohort内的全部条目一起分配，没有跨开发/确认/ranking视频泄漏。

“完整视频簇”指eligible grounded条目在切分和bootstrap时整体保留，不能误写成原始1213中的该视频所有条目都有可用grounding。grounded中31个视频簇缺eligible suc，确认中为21簇。故grounded可配对fail是543/578，确认可配对fail是315/340；这些fail仍参与总体/分类准确率，只在要求对应suc也存在的same-video配对统计中没有配对。原始806个fail在全1213中均有对应suc。


### 2026-09-09 15:45：SOLE三个完成数值screen的完整负结果

`results/mydata_bench/experiments_v2_corssmodel/auto_research/session_20260908/sole_score_development_v2/analysis_snapshot_20260909T154533.json`已汇总image_text、text_image、interleaved各60条和全部250个读出（包含预先声明的大尺度/KL补充）。三个输入均没有满足完整四向/+10pp及同k原方法改良的候选。

|输入|完整target候选数|通过改良门槛数|全部target中的最大总准确率增益|
|---|---:|---:|---:|
|sole_image_text|104|0|1.67pp|
|sole_interleaved|104|0|10.00pp|
|sole_text_image|104|0|10.00pp|

最大总增益列仅说明搜索结果上限，不代表该点同时降低MAE或提高suc/fail；不能把不同点的最优指标拼成一个方案。SOLE目前采用固定自身前六步和末步reasoning的数值读出干预，这些screen不能替代主线1的完整七步attention。完整七步video_text/interleaved与两个剩余视频数值screen继续运行。
interleaved最终审计`completed_score_audit_20260909T154210.json`显示2340支路键、15000读出键全部有效，无重复或科学冲突；observed/static zero各60/60精确，native greedy与联合MAP全部一致，规范数字格式概率最小0.9993461。


### 2026-09-09 16:17：MAE定义与论文Overall聚合方式的核验

新读取[RoboReward论文v2](https://arxiv.org/html/2601.00675v2)：Section3明确`MAE=(1/N)Σ|r_hat−r|`，预测与标签均为1..5；本轮离散模型的逐样本MAE与此一致。Table1的Overall另外以RoboRewardBench各子集的MAE作group-wise聚合，不能把其23个公开benchmark子集与本数据的34个自定义task混为同一评测。正文、下载时间与SHA存于`RESEARCH/references/roboreward_mae_definition_check_20260909T1615`，来源账本已追加。

当前确认的主指标一直是按样本平均的原生1..5 MAE；连续模型另外报告`|1+4p−y|`，这项连续适配不能冒称论文原生离散输出。为完整呈现聚合敏感性，新增`task_macro_summary.py`从既有逐task统计计算等权task-macro MAE/准确率，不修改已冻结metrics模块、主门槛或参数选择。整task无有效预测时macro MAE记为undefined，不删除该task来制造完整结果；逐task有效数和原始完整性同时保留。

初次真实执行针对BASE/metrics_snapshot_20260909T124540.json，得到180项等权task摘要，文件`BASE/task_macro_snapshot_20260909T1617_v1.json`；其中历史未完成条件仍按原覆盖率标识。最终BASE、确认498和完整846结果完成后都应再生成对应macro补充，并与micro并列、明确各cohort实际出现的task数。这是描述性补充，不在读取确认结果后换用较有利的聚合方式来决定是否通过。

输入来源也需精确措辞：Meter text_image为checkpoint官方结构，所选三个Meter候选输入均为适配；RoboReward现有代码将text_video说明为公开HELM请求顺序，video_text为model card引用的视频推理例顺序，没有据此宣称存在已公开的独立官方benchmark evaluator。本轮固定8帧与attention计算预算，不能把输入顺序相同称为全套官方推理参数复现。


### 全1213新增baseline的task等权MAE补充

对应`BASE/task_macro_snapshot_20260909T1617_v1.json`；每项都有34个自定义task。下表所有MAE列区分原生1..5适配与连续进度适配，并不等同RoboRewardBench的23子集Overall。

|模型/输入|有效/预期|逐样本五档MAE|task等权五档MAE|逐样本连续MAE|task等权连续MAE|
|---|---:|---:|---:|---:|---:|
|meter_image_text_v1|1213/1213|2.3092|2.1706|2.2265|2.1256|
|meter_interleaved_v1|1213/1213|1.9868|2.0399|1.9689|2.0302|
|meter_text_image_v1|1213/1213|1.9753|2.0379|1.9543|2.0271|
|meter_text_video_v1|1213/1213|1.7181|1.8830|1.6839|1.8843|
|meter_video_text_v1|1213/1213|1.9975|1.9986|2.0635|2.0367|
|sole_image_text_v1|1213/1213|1.4378|1.7220|1.4757|1.7448|
|sole_interleaved_v1|1213/1213|1.5276|1.7229|1.5342|1.7184|
|sole_text_image_v1|1213/1213|1.4889|1.7015|1.5179|1.7221|
|sole_text_video_v1|1204/1213|1.8331|1.8725|1.8319|1.8679|
|sole_video_text_v1|1213/1213|1.8631|1.9206|1.8622|1.9158|

SOLE text_video的9条原生格式错误仍保留；其MAE列基于各组有效分数，不应脱离1204/1213覆盖率解读。所有准确率仍用预期样本分母；完整task详细有效数在补充JSON中。


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


## 2026-09-09 17:46：相同498样本上的新增模型五输入原生背景

为避免把某个较弱适配输入的改善写成全输入或官方行为的全面改良，新增`confirmation_v1/added_native_baseline_context_v1.json`。它只从确认启动前已经完成的原生baseline1213中抽取相同冻结498个ID，无新方法推理、不改冻结比较基线、不重选输入。SOLE的MAE为既定五档适配；连续ordinal另列。

|模型/输入|有效/498|五档MAE|连续ordinal MAE|总准确率|suc准确率|fail准确率|
|---|---:|---:|---:|---:|---:|---:|
|meter_image_text|498|2.351406|2.255305|0.0000%|0.0000%|0.0000%|
|meter_text_image|498|1.797189|1.793348|31.1245%|94.3038%|1.7647%|
|meter_interleaved|498|1.835341|1.824718|30.7229%|94.9367%|0.8824%|
|meter_text_video|498|1.570281|1.535920|5.4217%|5.6962%|5.2941%|
|meter_video_text|498|1.997992|2.069588|0.0000%|0.0000%|0.0000%|
|sole_image_text|498|1.337349|1.388835|45.3815%|33.5443%|50.8824%|
|sole_text_image|498|1.445783|1.466667|28.3133%|6.3291%|38.5294%|
|sole_interleaved|498|1.455823|1.460643|27.9116%|4.4304%|38.8235%|
|sole_text_video|495|1.862626|1.848323|10.4418%|24.0506%|4.1176%|
|sole_video_text|498|1.921687|1.905060|9.0361%|24.6835%|1.7647%|

Meter官方text_image的原生suc准确率94.3038%、fail1.7647%，总31.1245%；当前新方案image_text primary总46.9880%、suc33.5443%、fail53.2353%。尽管总准确率与MAE有所改善，跨输入比较牺牲了很多成功类正确率，因此不能声称新方案支配官方输入的每个指标。严格四向增益是在每个冻结候选各自相同输入baseline上检验的。协议变换也不只是纯顺序置换：Meter的训练progress token位置、SOLE的拼图/多图/视频构造都具有明确适配边界。

SOLE text_video在该498子集里有3个原生格式失败，MAE用495有效预测、准确率分母仍498；它们没有通过猜数或真值补齐。SOLE官方image_text原生总准确率45.3815%，为第四模型边界提供绝对参照，数值screen失败不能据此推广为所有未来SOLE方法无效。


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


## 2026-09-09 17:58：数据分区实际task覆盖的完整核对

新增`frozen_cohort_completion_v1/partition_task_coverage_v1.json`，对冻结labels SHA核验后按原始ID统计：

|集合|样本|视频簇|实际有样本的task|
|---|---:|---:|---:|
|原始全量|1213|407|34|
|grounded全cohort|846|299|28|
|development|234|86|27|
|confirmation|498|179|27|
|ranking重叠remaining|114|34|8|

原始34task中，task3_5、task3_9、task4_4、task4_5、task5_5、task5_9没有进入grounded846；这个覆盖限制必须与846/1213的样本覆盖一起披露。开发集另外不含task2_3，确认集另外不含task3_2；两者虽各27task，任务集合不完全相同。remaining114只覆盖8task，完整逐task/suc/fail数量在JSON中。这里的task数量不等于独立视频数量。

此前“每项34task”的baseline macro表只适用于其明确列出的原始1213 baseline，不能扩展解释为grounded/attention/confirmation同样覆盖34task。最终846 task-macro的分母应为28，确认为27；缺席task不补零，也不通过删除表现差的task改变macro。该补充只核对数据覆盖，不触发改参、换case或修改原始分区。


## 2026-09-09 18:09：RR text_video冻结独立确认失败，不能宣布三模型×三输入通过

全部498推理/读出完成、zero与重复审计通过，但`confirmation_case_point_acceptance=False`，strict_original也为False。三个冻结邻居对两套baseline均满足四向/+10pp：相对native的k48/56/64总增益+15.66265/+11.44578/+13.25301pp，MAE变化−.728916/−.775100/−.769076，suc+3.79747/+3.79747/+4.43038pp，fail+21.17647/+15.00000/+17.35294pp。

**失败来自primary k56对same-k原方法的MAE。** 新方法MAE1.32730923695，original k56为1.32530120482，变化+.00200803213，即498条绝对误差总和多1；虽然总准确率比original高13.85542pp，仍未满足预先冻结的MAE严格下降条件。不能因为只差1分而改容差、换成k48/k64或只报告对native的正结果。k48/k64的same-k MAE分别下降.014056/.050201；这也不能替代要求三个冻结邻居都通过。

开发预先选定的original k40更严格对照同样失败：primary MAE+.018072、总准确率+12.248996pp，体现MAE/端点准确率权衡。primary保持k56，官方记录中的scope为**all_frames**；任何简写将其写成last_frame均以冻结YAML和真实condition `contrast_target_temporal_all_frames_b3_k56_lambda4`为准，不更改运行设置。

primary原生baseline→新方法：MAE2.102410→1.327309，总准确率19.8795%→31.3253%，suc62.6582%→66.4557%，fail0%→15%。同读出baseline准确率20.2811%、suc63.9241%，也单独过四向点门槛。成功类MAE由.683544增至.911392，因此不能把总MAE下降写成两类MAE均下降。

primary对native的95%视频簇区间（准确率单位为比例）为`{"mae": [-0.9127007198228129, -0.6365420546916335], "accuracy": [0.07889546351084813, 0.1512770137524558], "suc_accuracy": [-0.050314465408805034, 0.12804878048780488], "fail_accuracy": [0.1069364161849711, 0.195906432748538]}`；对两baseline的四向IUT p=.337733113，成功类改善不确定，九项Holm仍待最后Meter case。

原始支路15438/15438全部尝试，15410有效，无重复或科学冲突；各zero498精确，native/readout分数差6条。全部22866有效读出与42个预期错误传播精确复算，差异0。

当前完成8项，7项通过冻结point、1项失败；RR只有image_text/text_image两项通过point，其中text_image不胜开发选定原方法。**无论最后Meter结果如何，原九项方案都不能取得“独立确认三模型各三输入均通过冻结验收”的结论。** 原确认manifest、参数和所有失败永久保留；完整846的描述性汇总及SOLE mandatory全七步仍继续。后续新方案如使用已见过结果的498，只能明确标为确认后探索/重复使用验证集，不得冒称新的独立确认。


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


## 2026-09-09 21:07：全部主线收尾完成，最终阅读入口

本文件前文为逐时追加的历史记录。当前所有新增baseline与attention预定实验已完成尝试，文献与方法研究、最终表格、图和审计已齐。独立可读总报告：[FINAL_RESEARCH_REPORT_20260909.md](auto_research_addbase/FINAL_RESEARCH_REPORT_20260909.md)。

最终方法在预定主阈值0.125/0.875下实现Meter、Qwen、RoboReward各三输入、各三个已测k的观测验收；开发234、复用498和完整grounded846全部通过两baseline与原方法比较。原独立确认失败保持不变，最终数据已用于适应性探索；未获得新独立泛化证明。0.2/0.8下Meter text_video成功类下降，任务/排序和官方输入也有边界，均见总报告。

SOLE interleaved于20:55:40、video_text于20:58:16完成全七步846×11；全量attention审计、metrics与40文件记录审计均returncode0。两个守护已结束，当前没有本轮推理/调度任务继续运行。下面追加由最终快照生成的完整baseline报告；原始历史结果不覆盖。

# 新增baseline与attention全部完成结果

生成时间：2026-09-09T12:59:57.714614+00:00。全部指标来自完成后的单一快照，原始预测和失败记录保留。

10项native各1213次尝试；Meter五输入各846×10；SOLE五输入全七步与终步各846×11。合计147490个唯一主预测记录（不含step traces）。完成表示全部预定样本/条件已尝试，原生格式或几何失败仍明确列出。

准确率分母包含所有期待样本，缺失/格式失败按不正确计入；MAE只对有效数值计算，须结合有效数解读。连续ordinal为|1+4p−y|，五档MAE使用既定progress分档；都不是RoboRewardBench官方23子集Overall。低阈值严格p<low，高阈值p≥high。

Meter官方结构text_image；SOLE官方结构image_text。其余是明确输入适配。SOLE使用自身历史七步递归、greedy/max512，与官方随机解码默认不同。全七步每个条件独立递归；terminal仅末步干预，不能互相替代。

## 原生baseline：全1213，34任务

|模型/输入/条件|有效/期待|五档MAE|连续ordinal MAE|总acc .125/.875|suc acc|fail acc|总acc .2/.8|
|---|---:|---:|---:|---:|---:|---:|---:|
|meter_image_text_v1|1213/1213|2.309151|2.226536|0.0000%|0.0000%|0.0000%|0.0000%|
|meter_text_image_v1|1213/1213|1.975268|1.954338|31.5746%|92.1376%|0.9926%|36.9332%|
|meter_text_video_v1|1213/1213|1.718054|1.683867|4.3693%|3.4398%|4.8387%|25.9687%|
|meter_video_text_v1|1213/1213|1.997527|2.063455|0.0000%|0.0000%|0.0000%|0.0000%|
|meter_interleaved_v1|1213/1213|1.986810|1.968878|31.1624%|92.1376%|0.3722%|35.2844%|
|sole_image_text_v1|1213/1213|1.437758|1.475746|42.4567%|21.1302%|53.2258%|52.0198%|
|sole_text_image_v1|1213/1213|1.488871|1.517857|30.8326%|4.6683%|44.0447%|44.4353%|
|sole_text_video_v1|1204/1213|1.833056|1.831927|9.0684%|17.4447%|4.8387%|16.4880%|
|sole_video_text_v1|1213/1213|1.863149|1.862160|8.9860%|22.6044%|2.1092%|14.5919%|
|sole_interleaved_v1|1213/1213|1.527617|1.534246|27.0404%|6.6339%|37.3449%|43.6109%|

Meter text_video官方sum==1类型猜测的两条越界读出保留；typed-softmax敏感性见BASE/meter_baseline_typed_readout_sensitivity_v1.json。SOLE原生text_video有9条格式失败，不从reasoning猜分。

## Meter原attention：846，28任务

|模型/输入/条件|有效/期待|五档MAE|连续ordinal MAE|总acc .125/.875|suc acc|fail acc|总acc .2/.8|
|---|---:|---:|---:|---:|---:|---:|---:|
|image_text/baseline|846/846|2.355792|2.258014|0.0000%|0.0000%|0.0000%|0.0000%|
|image_text/target_k8|846/846|2.016548|2.061152|0.0000%|0.0000%|0.0000%|0.0000%|
|image_text/target_k32|846/846|1.802600|1.824065|0.0000%|0.0000%|0.0000%|0.0000%|
|image_text/target_k64|846/846|1.709220|1.721155|0.0000%|0.0000%|0.0000%|0.0000%|
|text_image/baseline|846/846|1.769504|1.764461|31.0875%|95.1493%|1.3841%|37.2340%|
|text_image/target_k8|846/846|1.111111|1.034774|26.4775%|74.6269%|4.1522%|65.1300%|
|text_image/target_k32|846/846|1.092199|1.021852|23.8771%|63.4328%|5.5363%|62.4113%|
|text_image/target_k64|846/846|1.163121|1.186681|21.3948%|55.5970%|5.5363%|37.3522%|
|text_video/baseline|846/846|1.562648|1.515647|5.6738%|5.2239%|5.8824%|33.6879%|
|text_video/target_k8|846/846|1.268322|1.221511|8.0378%|9.3284%|7.4394%|45.2719%|
|text_video/target_k32|846/846|1.164303|1.098695|18.6761%|1.1194%|26.8166%|58.8652%|
|text_video/target_k64|846/846|1.260047|1.183704|12.0567%|1.8657%|16.7820%|51.4184%|
|video_text/baseline|846/846|1.996454|2.072061|0.0000%|0.0000%|0.0000%|0.0000%|
|video_text/target_k8|846/846|1.942080|1.962846|0.0000%|0.0000%|0.0000%|0.0000%|
|video_text/target_k32|846/846|1.888889|1.874084|0.0000%|0.0000%|0.0000%|0.0000%|
|video_text/target_k64|846/846|1.847518|1.805592|0.0000%|0.0000%|0.0000%|0.0000%|
|interleaved/baseline|846/846|1.796690|1.791059|30.6147%|95.5224%|0.5190%|35.1064%|
|interleaved/target_k8|846/846|1.109929|1.055076|25.5319%|75.0000%|2.5952%|60.4019%|
|interleaved/target_k32|846/846|1.109929|1.067918|22.6950%|61.9403%|4.4983%|52.9551%|
|interleaved/target_k64|846/846|1.145390|1.135415|20.4492%|55.5970%|4.1522%|36.9976%|

其余wrong、low-head与zero全部条件见CSV和源JSON；条件间格式覆盖不同，不能忽略有效样本变化。

## SOLE全七步attention：846，28任务

|模型/输入/条件|有效/期待|五档MAE|连续ordinal MAE|总acc .125/.875|suc acc|fail acc|总acc .2/.8|
|---|---:|---:|---:|---:|---:|---:|---:|
|image_text/baseline|846/846|1.341608|1.381418|44.3262%|30.9701%|50.5190%|54.9645%|
|image_text/target_k8|846/846|1.373522|1.407943|43.0260%|30.9701%|48.6159%|54.8463%|
|image_text/target_k32|846/846|1.490544|1.535839|42.6714%|23.8806%|51.3841%|50.5910%|
|image_text/target_k64|846/846|1.304965|1.338534|46.4539%|32.0896%|53.1142%|56.7376%|
|text_image/baseline|846/846|1.412530|1.443735|30.9693%|6.3433%|42.3875%|45.3901%|
|text_image/target_k8|846/846|1.375887|1.419574|31.6785%|10.0746%|41.6955%|43.7352%|
|text_image/target_k32|846/846|1.352246|1.392435|32.7423%|8.2090%|44.1176%|45.5083%|
|text_image/target_k64|846/846|1.316785|1.366052|38.8889%|9.3284%|52.5952%|49.8818%|
|text_video/baseline|841/846|1.889417|1.884376|10.0473%|23.8806%|3.6332%|17.0213%|
|text_video/target_k8|844/846|1.855450|1.847820|10.8747%|25.7463%|3.9792%|17.4941%|
|text_video/target_k32|839/846|1.915375|1.907128|9.3381%|19.0299%|4.8443%|19.1489%|
|text_video/target_k64|809/846|1.901112|1.895179|9.4563%|22.0149%|3.6332%|16.7849%|
|video_text/baseline|846/846|1.884161|1.887660|10.2837%|28.3582%|1.9031%|14.5390%|
|video_text/target_k8|846/846|1.874704|1.881986|12.2931%|29.8507%|4.1522%|16.9031%|
|video_text/target_k32|845/846|1.862722|1.873231|12.2931%|23.5075%|7.0934%|17.8487%|
|video_text/target_k64|845/846|2.000000|2.012166|8.6288%|11.1940%|7.4394%|14.1844%|
|interleaved/baseline|846/846|1.430260|1.452293|31.3239%|9.3284%|41.5225%|44.4444%|
|interleaved/target_k8|846/846|1.366430|1.381087|31.6785%|14.1791%|39.7924%|47.2813%|
|interleaved/target_k32|846/846|1.384161|1.422128|32.5059%|14.9254%|40.6574%|46.9267%|
|interleaved/target_k64|846/846|1.531915|1.546336|32.0331%|12.3134%|41.1765%|43.0260%|

其余wrong、low-head与zero全部条件见CSV和源JSON；条件间格式覆盖不同，不能忽略有效样本变化。

## SOLE终步attention补充：846，28任务

|模型/输入/条件|有效/期待|五档MAE|连续ordinal MAE|总acc .125/.875|suc acc|fail acc|总acc .2/.8|
|---|---:|---:|---:|---:|---:|---:|---:|
|image_text/baseline|846/846|1.325059|1.377825|47.0449%|30.5970%|54.6713%|54.7281%|
|image_text/target_k8|846/846|1.258865|1.312766|48.4634%|34.7015%|54.8443%|58.1560%|
|image_text/target_k32|846/846|1.410165|1.447376|45.1537%|27.6119%|53.2872%|54.3735%|
|image_text/target_k64|846/846|1.417258|1.456643|41.8440%|31.3433%|46.7128%|52.2459%|
|text_image/baseline|846/846|1.412530|1.446998|31.0875%|6.7164%|42.3875%|45.0355%|
|text_image/target_k8|846/846|1.420804|1.440709|30.2600%|8.2090%|40.4844%|45.2719%|
|text_image/target_k32|846/846|1.397163|1.441844|31.9149%|6.3433%|43.7716%|44.6809%|
|text_image/target_k64|846/846|1.405437|1.435697|32.5059%|6.3433%|44.6367%|45.1537%|
|text_video/baseline|841/846|1.862069|1.863306|10.5201%|25.3731%|3.6332%|17.1395%|
|text_video/target_k8|841/846|1.888228|1.892747|10.4019%|24.6269%|3.8062%|16.4303%|
|text_video/target_k32|841/846|1.885850|1.894507|10.2837%|22.7612%|4.4983%|15.9574%|
|text_video/target_k64|836/846|1.860048|1.867081|7.9196%|16.0448%|4.1522%|14.5390%|
|video_text/baseline|846/846|1.887707|1.882648|10.2837%|27.9851%|2.0761%|16.3121%|
|video_text/target_k8|846/846|1.897163|1.884870|11.1111%|27.9851%|3.2872%|16.6667%|
|video_text/target_k32|846/846|1.877069|1.885437|12.1749%|30.2239%|3.8062%|17.8487%|
|video_text/target_k64|846/846|1.923168|1.938771|9.1017%|20.8955%|3.6332%|13.4752%|
|interleaved/baseline|846/846|1.437352|1.455130|29.7872%|7.8358%|39.9654%|44.4444%|
|interleaved/target_k8|846/846|1.379433|1.390922|31.9149%|13.0597%|40.6574%|46.5721%|
|interleaved/target_k32|846/846|1.397163|1.421655|31.6785%|14.5522%|39.6194%|45.5083%|
|interleaved/target_k64|846/846|1.379433|1.413428|33.9243%|15.6716%|42.3875%|45.8629%|

其余wrong、low-head与zero全部条件见CSV和源JSON；条件间格式覆盖不同，不能忽略有效样本变化。

## 原attention预定target点的总体表现

下列计数同时要求target与同实现baseline有效覆盖完整、MAE下降、主阈值suc/fail提高、总准确率至少+10pp。它只是已测点的描述性检查，不是新方法选型或统计显著性检验。

|组别|通过的target点/15|具体已测点|
|---|---:|---|
|Meter原attention：846，28任务|0/15|无|
|SOLE全七步attention：846，28任务|0/15|无|
|SOLE终步attention补充：846，28任务|0/15|无|

![全部原attention target点](../results/mydata_bench/experiments_v2_addbase/session_20260908/final_added_baseline_report_v1/figures/all_target_effects.png)

图红色为改善，蓝色为变差；每列色标各自标明单位，数值为全部已测点。

## 全部条件尝试、错误和zero核验

|实验|唯一行|期待行|无效条数（含全部条件）|zero配对数|zero原生输出/trace精确|
|---|---:|---:|---:|---:|---|
|meter_image_text_v1|8460|8460|21|另见Meter工程审计|非本全量条件|
|meter_interleaved_v1|8460|8460|21|另见Meter工程审计|非本全量条件|
|meter_text_image_v1|8460|8460|21|另见Meter工程审计|非本全量条件|
|meter_text_video_v1|8460|8460|30|另见Meter工程审计|非本全量条件|
|meter_video_text_v1|8460|8460|30|另见Meter工程审计|非本全量条件|
|sole_image_text_attention_residual_v1|9306|9306|21|846|通过|
|sole_image_text_terminal_residual_v1|9306|9306|15|846|通过|
|sole_interleaved_attention_residual_v1|9306|9306|18|846|通过|
|sole_interleaved_terminal_residual_v1|9306|9306|15|846|通过|
|sole_text_image_attention_residual_v1|9306|9306|21|846|通过|
|sole_text_image_terminal_residual_v1|9306|9306|15|846|通过|
|sole_text_video_attention_residual_v1|9306|9306|89|846|通过|
|sole_text_video_terminal_residual_v1|9306|9306|76|846|通过|
|sole_video_text_attention_residual_v1|9306|9306|20|846|通过|
|sole_video_text_terminal_residual_v1|9306|9306|15|846|通过|

无效条数按条件累计，同一样本可能在多个条件失败；不等于独立失败视频数。源attention audit列出每个条件的所有失败ID、原生错误文字、缺失数及step trace，record audit保存逐文件SHA/唯一键/重复冲突。Meter正式十条件没有全量zero，原small-forward零干预与后续完整新方法zero核验分开报告。

## 各任务等权macro

全部180组task-macro见传入的独立macro JSON；其逐样本分母和有效数也在CSV。native全1213用34个任务，grounded846用28个任务；不补不存在的任务，不排除表现较差任务。

## 具体top8与跨模型重合

layer/head从0开始。坐标重合不代表功能等价；跨架构宽度和训练变化均限制解释。

|模型/输入|top8|
|---|---|
|meter_image_text|L19H21, L22H15, L21H11, L21H16, L18H30, L22H2, L21H19, L19H15|
|meter_interleaved|L19H21, L19H10, L18H30, L22H15, L19H17, L21H16, L19H16, L22H26|
|meter_text_image|L19H21, L19H10, L18H30, L21H16, L19H17, L19H16, L22H15, L22H26|
|meter_text_video|L19H21, L19H10, L19H23, L19H17, L19H16, L22H26, L22H15, L21H16|
|meter_video_text|L22H15, L21H11, L12H18, L20H29, L19H15, L14H17, L18H30, L19H21|
|qwen_image_text|L24H13, L21H16, L12H18, L20H15, L21H25, L23H13, L27H16, L19H28|
|qwen_interleaved|L20H15, L19H23, L19H28, L21H25, L21H31, L21H27, L21H18, L23H13|
|qwen_text_image|L21H25, L21H16, L19H31, L22H15, L19H23, L20H15, L19H28, L21H29|
|qwen_text_video|L21H16, L21H25, L19H23, L20H15, L10H13, L22H15, L19H28, L19H31|
|qwen_video_text|L25H10, L24H13, L12H18, L21H16, L21H18, L24H12, L12H20, L13H17|
|rr_image_text|L19H28, L21H16, L19H16, L20H15, L22H15, L19H0, L18H30, L19H31|
|rr_interleaved|L19H28, L20H15, L19H31, L26H31, L20H13, L23H13, L19H23, L24H13|
|rr_text_image|L22H15, L19H28, L19H31, L22H5, L21H16, L19H10, L20H15, L19H17|
|rr_text_video|L22H15, L26H31, L19H28, L19H23, L21H25, L21H16, L19H17, L20H15|
|rr_video_text|L18H30, L12H18, L20H15, L21H16, L21H27, L14H7, L19H28, L13H17|
|sole_image_text|L24H13, L29H1, L30H8, L34H0, L31H15, L16H15, L24H28, L8H22|
|sole_interleaved|L22H0, L19H23, L10H12, L22H2, L19H28, L10H13, L15H25, L31H15|
|sole_text_image|L22H2, L19H17, L10H12, L24H31, L26H0, L22H0, L24H13, L21H16|
|sole_text_video|L10H12, L19H23, L21H16, L15H25, L22H0, L10H13, L14H29, L8H22|
|sole_video_text|L26H26, L28H16, L31H15, L10H12, L27H16, L31H29, L31H9, L34H0|

|相同输入的模型对|top8交集|top32交集|top64交集|
|---|---:|---:|---:|
|image_text_meter_vs_qwen|1|10|24|
|image_text_meter_vs_rr|3|11|27|
|image_text_meter_vs_sole|0|1|5|
|image_text_qwen_vs_rr|3|18|46|
|image_text_sole_vs_qwen|1|4|14|
|image_text_sole_vs_rr|0|4|9|
|interleaved_meter_vs_qwen|0|10|24|
|interleaved_meter_vs_rr|0|11|20|
|interleaved_meter_vs_sole|0|7|19|
|interleaved_qwen_vs_rr|4|25|53|
|interleaved_sole_vs_qwen|2|7|19|
|interleaved_sole_vs_rr|2|8|17|
|text_image_meter_vs_qwen|2|13|27|
|text_image_meter_vs_rr|4|13|23|
|text_image_meter_vs_sole|2|7|14|
|text_image_qwen_vs_rr|5|25|49|
|text_image_sole_vs_qwen|1|10|24|
|text_image_sole_vs_rr|2|11|22|
|text_video_meter_vs_qwen|3|14|22|
|text_video_meter_vs_rr|4|15|29|
|text_video_meter_vs_sole|2|7|15|
|text_video_qwen_vs_rr|6|23|42|
|text_video_sole_vs_qwen|3|8|17|
|text_video_sole_vs_rr|2|7|16|
|video_text_meter_vs_qwen|1|6|18|
|video_text_meter_vs_rr|2|7|21|
|video_text_meter_vs_sole|0|1|10|
|video_text_qwen_vs_rr|3|13|37|
|video_text_sole_vs_qwen|0|8|14|
|video_text_sole_vs_rr|0|2|11|

## 完整文件

- all_metrics_task_split_distribution.csv：全部180组、总体/类别/task/task×类别，两MAE、两阈值、五档预测分布。
- all_pairwise.csv：全部180组同视频配对的负/0/1/2/3/4计数、平均progress差、有效/期待配对数。
- all_target_effects.csv：全部45个target点相对各自同实现baseline的变化，不按表现挑点。
- source_manifest.json：源文件和本报告实现SHA；原始失败/重复没有删除。

源文件：

- metrics: `/mnt/public1/dais/workspace/Robo-Dopamine-addbase/results/mydata_bench/experiments_v2_addbase/session_20260908/metrics_snapshot_20260909T205912.json` (SHA256 972dd0fc73e99067fc83b42426e1da0034eb6e5287a9892814e2faa527bd5354)
- attention_audit: `/mnt/public1/dais/workspace/Robo-Dopamine-addbase/results/mydata_bench/experiments_v2_addbase/session_20260908/attention_audit_snapshot_20260909T205847.json` (SHA256 21386e5d020987e116a3aa16c98115eb17009947f443c119f3822f02fec81ac5)
- record_audit: `/mnt/public1/dais/workspace/Robo-Dopamine-addbase/results/mydata_bench/experiments_v2_addbase/session_20260908/final_record_consistency_audit_v1.json` (SHA256 81a83e077573316fe306f0013dd44dd2f892331e107284096bced6bd70ef360a)
- macro: `/mnt/public1/dais/workspace/Robo-Dopamine-addbase/results/mydata_bench/experiments_v2_addbase/session_20260908/task_macro_final_v1.json` (SHA256 82895be2c5295d4acd92891b25ffcd12b73d4c91961e62afeaa561d1bf7c22ae)
- head_overlap: `/mnt/public1/dais/workspace/Robo-Dopamine-addbase/results/mydata_bench/experiments_v2_addbase/session_20260908/all_four_models_head_overlap_v1.json` (SHA256 872b69e21e0ca32cef4f032e4ae0b5eee5a51cd07e7d3afa7ca8ad392cbab66d)


## 稳健研究状态更新 2026-09-10T13:28:02.865479+08:00

目标3仍未完成。F7全部15项独立拟合与零干预审计完成，冻结评价进行中；F8有序损失方案完成三模型真实工程检查，当前8/15项拟合结束。F8评价／完整846条件流程已冻结83项源码并启动等待，保留2831奖励性能仍未打开；旧498复用不能作为新独立确认。详见[最新进度](auto_research_addbase/ROBUST_PROGRESS_20260910T132802.md)、[F8评价方案](auto_research_addbase/F8_EVALUATION_PROTOCOL_20260910.md)和[理论／来源说明](auto_research_addbase/F8_ORDINAL_THEORY_AND_SOURCE_20260910.md)。不能提前宣布新方法稳定有效。
