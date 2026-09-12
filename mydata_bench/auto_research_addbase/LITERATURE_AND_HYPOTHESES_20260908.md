# 文献核验与方法假设（2026-09-08）

本文件是实验前机制分析，**不是实验成功结论**。来源网页、下载时间、SHA-256 与摘要保存于 `results/mydata_bench/experiments_v2_corssmodel/auto_research/session_20260908/references/source_verification.jsonl`。公开网络检索未上传任何本地视频、标注或研究材料。14 篇计划内论文的 arXiv 标题与摘要已核验；方法描述另依据可取得的论文 HTML 正文。额外检索 `2403.14588` 实际为数学论文 *Exponential Networks for Linear Partitions*，与本项目无关，**排除**，不将检索命中等同于相关证据。

| 论文 | 核实的机制 | 本研究中的用途与边界 |
|---|---|---|
| [Robometer, 2603.02115](https://arxiv.org/abs/2603.02115) | 从机器人轨迹比较和逐帧监督学习 progress / success / preference heads | 必须保留 checkpoint heads、训练新增 special tokens、逐帧 progress-token 读出；普通生成式适配会改变方法 |
| [SOLE-R1, 2603.28730](https://arxiv.org/abs/2603.28730) | 视频语言推理输出当前任务进度；官方代码递归输入自身上一时刻预测 | 官方复现使用 first/previous/current 拼图和原始系统/用户提示，递归更新上一预测，不能把每一步 previous 都设为 0，也不能把进度当增量累加 |
| [PASTA, 2311.02262](https://arxiv.org/abs/2311.02262) | 对未强调 token 的注意力乘 alpha 再归一化；依据少量多任务表现 profile 特定 heads | head 选择依赖干预效用，而 raw visual mass 仅是相关性代理；全 head 干预可能破坏模型 |
| [PAI, 2407.21771](https://arxiv.org/abs/2407.21771) | 推理时沿已有图像注意力方向增强视觉权重，考虑语言先验与图像之间的平衡 | 不能推论“提高图像权重总会改善奖励判断”；与 PASTA 的前两条原始链接应对调 |
| [ASCD, 2506.14766](https://arxiv.org/abs/2506.14766) | attention steering 与 contrastive decoding 结合；text-centric heads 正向调节、关键视觉 token 负向干预 | 明确提示，增加全部视觉注意力并非唯一方向；可作为独立对比解码后续候选 |
| [CAST, 2605.04641](https://arxiv.org/abs/2605.04641) | 从 caption 查询识别敏感 heads，并干预 head 输出向量；多查询集成提高稳定性 | 它不是简单 bbox 的 pre-softmax bias；可后续借鉴查询集成与输出空间方向 |
| [Gaze Heads, 2606.14703](https://arxiv.org/abs/2606.14703) | 通过图文叙述位置相关性识别少量 heads；因果干预能改变所描述区域 | “改变看哪里”不等于“提高任务成功判别”；head 选择和区域对照必要 |
| [HAS, 2607.17994](https://arxiv.org/abs/2607.17994) | 连续视频 highlight 分布、逐 query 加性 steering、可训练稀疏 head gates；在 held-out 校准集选策略 | 支持连续时域引导和稀疏剂量控制；其摘要任务/训练校准结果不能直接外推到机器人奖励 |
| [Arbitration Failure, 2604.09364](https://arxiv.org/abs/2604.09364) | 视觉信号已被编码，输出仍可能服从语言先验；通过多位置 hidden-state intervention 检验 arbitration | 若保持视觉总质量后仍不改善，要检查决策融合和任务绑定，而非只继续增强 bbox |
| [VLA Driving Steering, 2608.17095](https://arxiv.org/abs/2608.17095) | 推理时控制 VLA 驾驶模型的注意力（标题/摘要已核实） | 任务为驾驶轨迹而非 reward calibration；细节需正文进一步核查 |
| [Attention is Case-Sensitive, 2608.03711](https://arxiv.org/abs/2608.03711) | 文本大小写的显著性影响注意力（标题/摘要已核实） | prompt 表面变化是潜在干扰，需固定模板；不能把任务相关提示变更误归因于 mask |
| [Localization Heads, 2503.06287](https://arxiv.org/abs/2503.06287) | 少量 text-to-image attention heads 有稳定视觉定位能力，冻结模型即可读出区域 | 定位 head 与评价 head 可能不同；raw mass 和去面积偏差的 enrichment 排序应分开比较 |
| [Attention-Guided Safety Filter, 2606.09749](https://arxiv.org/abs/2606.09749) | 从 VLA 内部注意力得到安全过滤信号（标题/摘要已核实） | 只作为内部特征可用性的旁证，尚不是本任务 steering 的直接有效性证据 |
| [Analyzing Multi-Head Self-Attention, 1905.09418](https://arxiv.org/abs/1905.09418) | heads 功能分化，少数重要 heads 承担主要功能；其余可剪枝 | 支持选择性干预及 low-rank 对照，不能直接迁移 NLP head 索引到 VLM |
| [VCD, 2311.16922](https://arxiv.org/abs/2311.16922) | 对比原始视觉输入与扰动视觉输入的输出分布，减轻单模态先验偏差 | 非 attention-only 候选，可在守恒 steering 不足时检验；必须保留无对比与错误区域对照 |

## 机制分析

1. **模态总量与区域分配耦合。** 原方案给目标区域 +6、其余视觉 token −6。目标与非目标的相对赔率乘上 `exp(12)≈162,755`，而视觉相对于文本的总权重也变化。因此它同时改变“看哪个区域”和“视觉/语言融合比例”，不能只解读为目标增强。旧实验普遍压低分数、fail 增益与 suc 损伤共存，符合这一潜在混杂，但尚不构成因果证明。
2. **目标存在不等于目标交互成功。** 该数据包含同视频、不同指令的失败样本。把注意力集中在一个未被操作但确实存在的目标物体，可以增强正确定位，却仍不能识别指令是否完成。应保留手爪、关系目标与时间变化的证据，警惕过度压制非 bbox 场景。
3. **rank 代理与模型读出差异。** 末 prompt 的 bbox raw mass 可能优先选视觉定位 heads；Robometer 实际读出位置是训练过的 `<|prog_token|>`，SOLE 是长推理后的 `<answer>`，与原 GRM 的进度解码不相同。新实现针对实际读出位置排名，并另存 raw mass、visual mass、excess mass、visual enrichment。
4. **causal mask 与模板语义。** 当视觉在任务文字之前，视觉 token 本身无法看到指令。Robometer 若把 progress token 也留在任务之前，预测甚至完全看不到任务；因此 image→text 适配将训练 progress token 放到任务之后，并明确标为非官方变体。官方 text→image 保留每帧读出。五种输入不应伪装成严格仅交换顺序的同构干预。
5. **时空对齐。** 原生视频每两个源帧汇入一个 temporal patch；bbox 需结合真正贡献该 patch 的帧。SOLE 拼图需先按官方 384-pixel padding，再按三列偏移和整图 resize 变换 bbox。所有映射均记录源帧索引和最终像素框。

## 预先提出的第一候选：视觉质量守恒的区域重分配

对于某一 layer/head/query 的原始 logits `z_j`，视觉列集合 `V`、目标列 `T⊂V`。令 `u_j=+b`（目标），`u_j=−b`（其他视觉），非视觉为 0。仅对视觉列施加：

`z'_j = z_j + u_j − [logsumexp_{k∈V}(z_k+u_k) − logsumexp_{k∈V}(z_k)]`。

非视觉 logits 保持原值。由定义，视觉部分 softmax 分母的指数和严格不变，故视觉总注意力、每个非视觉 token 的概率、因果遮罩均保持不变，只有视觉内部区域分配改变。每个 query 单独计算校正；没有读取标签、文件名类别、已有真值或输出端点。

与原 bias 相比，这个候选把一个机制变量隔离出来。数学不变量已在 CPU 合成注意力上验证，真实 Robometer 前向的 normalizer 数值误差约 `2e−6`；**这不是效果验收**。后续需在冻结开发集检验剂量 `b`、top-k 与时域，选定后才进入独立确认集。若多数 suc/fail 未同时改善，则保留负结果并进入下一个机制假设。

## 实验忠实度说明

- Robometer loader 覆盖所有 checkpoint 参数，包括未用于逐帧评分的 preference/similarity/frame_pool heads；实际加载 `missing_keys=[]`、`unexpected_keys=[]`、`mismatched_keys=[]`。
- SOLE 官方输入契约采用 external-only 分支、8 个均匀源帧、7 次递归预测；有多视角也先统一使用 front，以便与其他三模型的视觉信息预算比较。官方代码确实支持 external-only。
- 为配对比较可复现，SOLE 使用 greedy decoding、最大 512 个新 token；官方当前 RewardGen 默认是 temperature=1、top_p=.9、top_k=50、max_tokens=200。因此称“官方输入”，不冒称“逐参数官方随机解码复现”。
- 本轮尚未获得满足三模型×三输入的有效方案，不预先承诺实验一定成功。

## 正文核验补充与第二候选（2026-09-08 23时后）

- **VLA Driving Steering 2608.17095**：补充取得15页PDF及40,330字符正文，下载与SHA记录追加在source_verification.jsonl。方法为有界视觉token additive bias，YOLO检测后映射token；论文用配对零bias和逐调用注入审计。关键限制：其部署栈的CoC推理走无显式mask的fused kernel，注入实际上未触达推理；因此“推理不变”不是推理鲁棒性的证据。轨迹偏向被关注车辆也不等于正确驾驶行为。这直接支持本研究对hook实际触达、因果mask和零计算对照的审计。
- **Attention is Case-Sensitive 2608.03711**：正文4.4与补充B.2明确讨论attention–performance divergence和reasoning buffer。大小写增强注意力可能改善也可能损害准确率，VLM中还会改变视觉/文字总分配；不能据定位图变清楚就断言reward更好。
- **Attention-Guided Safety Filter 2606.09749**：正文方法是active-object attention heads + 实时tracker + CBF/QP外部安全过滤。作用对象是运动规划安全，不能直接当作reward steering已有效的证据。
- **Analyzing Multi-Head Self-Attention 1905.09418**：正文以LRP衡量head相关性，并以可微L0 gate学习剪枝。raw attention mass不等价于因果重要性，适合以替代ranking/low-head对照检验。

第一候选在Qwen两个视频构造的60条开发初筛出现正向结果，已进入234条全开发集扩展，仍未验证跨三模型。text→image最好的同时双类正向结果只有总准确率+5个百分点。为后续机制迭代预先定义第二候选：**对称区域证据对比**。

对原生评分logits保存三条支路：无干预z0、正向区域重分配z+、反向区域重分配z−。在保持全部原生分档的条件下，组合z*=z0+λ(z+−z−)，使用原生argmax（Qwen/RoboReward全部1–5选项）或原生十bin softmax期望（Robometer），不将分数强制设成1/5、不读标签或任务ID、不按类别使用不同公式。理论上对称差分减少共同的格式/语言先验分量，突出对任务区域证据的响应；是否能区分“物体存在”和“指令实际完成”必须由实验而非此公式证明。

生成模型必须固定共同的原生格式前缀`ANSWER: `后读取五个合法数字的logits，并增加同读出无attention baseline，检查它与原生greedy生成的分数一致性；否则不能把读出变化归因于attention。开发阶段可离线比较预先记录的λ及top-k邻域，所有分支logits永久保存；确认集只运行冻结方案。错误区域、低排名heads、λ=0、b=0及原方法为必要对照。第二候选尚未实现/运行，不能视为结果。

### 第二候选实现状态更新

上节“尚未实现/运行”是提出时状态。23:28已完成三模型小规模原生读出/零干预工程验证，冻结`score_development_v1/prospective_plan.json`并启动开发实验。λ集合为{0.5,1,2,4}，原生无干预分支等同λ=0；正/负支路为b=±3的视觉质量守恒重分配。当前只在开发集做选择，尚无跨模型成功结论。

旧第一候选在Qwen text→video全234条的最佳总准确率增幅8.97pp，未满足10pp。旧实现的零bias也在少量样本上改变回答，新增稀疏残差引擎用于隔离这种已观测的后端混杂。后续必须结合零对照解释，不能仅从先前CPU公式不变量推断完整模型逐bit等价。

### 对称对比的分布解释与读出边界

`softmax(z0 + λ(z+−z−))`等价于对全部原生类别使用`p0(y)·[p+(y)/p−(y)]^λ`再归一化，因为各分支log-normalizer只贡献与y无关的常数。这是条件证据比重加权，不是预先选择端点；所有五个reward值或十个progress bins始终参与。λ=0回到同读出baseline，b=0时正负支路相同而退回baseline。差分也会放大错误区域和不可靠heads造成的误差，故实际区域、低head、相同读出及原始generation四类对照都必须报告。

Robometer官方工具会以sum==1推测输入是否已经是概率；本地原始text_video baseline有2条logits偶然误触发，已保留官方结果并另记typed softmax敏感性。合成的counterfactual值具有明确logit类型，新方法对这类值始终使用softmax；它与native baseline读出差异通过单独typed-readout baseline控制，不将读出bug修复当作attention因果收益。


## 2026-09-09 00:50后：原生评分 query 的局部证据对比（前瞻冻结）

- 现有全序列视觉守恒对比在 Qwen text_video 全234开发样本已通过点估计门槛与相邻 k；Meter 初筛 text_video 也有双类正向+10pp，但其 text_image 常以成功类损失换取失败类收益；video_text 改善 MAE 却不提高端点准确率。RR interleaved 初筛虽双类+10pp，但未同时优于原强bias的 MAE/准确率，不能计作改良验收。
- 机制假设：在每一层所有 query 上改注意力，会同时修改任务文字与视觉 token 表示，因而影响后续评分所依赖的任务语义。候选第三轮只在原生评分 query 行重新分配视觉注意力：Meter 使用最后一个训练好的 prog token；Qwen/RR 使用共同 ANSWER 格式前缀末 token。Meter prog token 后还有聊天模板尾部，因此不能把 last_prompt 误当作 trained readout。
- 候选公式仍为全原生 bins 的 z*=z0+gamma*(z(+b)-z(-b))/(2b)，b={1.5,3}，gamma={3,6,12,24}，k={32,64}，all_frames/last_frame。较小有限差分半径检验非线性损伤。native生成、同读出baseline、零干预、原强bias以及同帧等面积wrong/low-head对照均保留。此为待检验推论，不把 PASTA/ASCD 的先例当作本任务有效性证明。
- `score_readout_development_v1/prospective_plan.json` 在读取本候选性能前冻结三模型×五输入，每case 29个神经支路、77个读出，同60条/24完整视频组。真实模型先做两条不读标签的零logits与实际query命中审计，通过才启动对应60条；确认498标签/新方法指标仍未用于选型。CPU测试验证未选中query和head精确不变、零残差精确不变、预加载索引与原索引实现相同。
- Meter text_video 原第二候选已扩展到完整234开发样本，包含k24/32/40/48/64、两scope及控制；GPU2执行中。
- Qwen text_video第二候选 k32/40/48/64、lambda4的簇bootstrap四项方向区间均支持改善（k24成功类区间触0），总准确率分别+20.51/+30.34/+30.77/+27.78pp；这是开发筛选后的探索区间，不能替代独立确认。文件 `score_development_full_v1/qwen_text_video/paired_k_neighborhood_v1.json`。
- SOLE原native residual 64条×5条件完整七步零对照已重新逐行核验并保存 `sole_batch_attention_residual_pilot_v2/full_zero_control_parity_audit_v1.json`。两种零干预全部8点轨迹、7段raw reasoning与baseline完全一致。
- CUDA static cache审计：HF static与手写static eager逐token一致，eager与CUDA graph一致，但dynamic与static发生数值分岔；没有启用static做正式实验。进一步发现主要开销是Python索引列表在每次decode的重复设备传输。仅预加载索引、保持原native dynamic cache，8条真实top64非零preserve在image_text/text_video逐token完全一致，用时44.3→16.2s、25.6→15.5s（共享GPU时序测量）；完整七步复核进行中。加速选型只据数值一致性和速度，不读效果标签。


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


## 2026-09-09 12:43：原生完成置信度调节的区域证据对比（第六候选，前瞻冻结）

- 已观测的主要失败模式是统一放大区域方向降低了fail的误报，却同时破坏suc完成度。一个可检验的解释是，高置信度的完成判断已经综合了物体与任务关系，统一强干预会擦除这些证据；不能从这个解释直接宣称有效。
- 新候选只调节干预方向的幅度：z*=z0+lambda*(1−q_native)^power*D，D取原对称区域证据，或扣除静态外观敏感性的时间区域证据。q_native由当前样本的模型原生输出计算：Meter为checkpoint已有、单独训练的task-success head；Qwen/RR为全部五档分布的归一化期望。这里q是模型预测，不是数据真值；推理代码不读取标签、suc/fail路径语义或配对样本。
- 全部原生5/10分档保留，q既不替代progress也不决定输出端点，无硬阈值、无端点映射、无额外训练。D=0时任何q都严格退化为baseline。它是一项可选调节，不将Meter辅助head的存在误写成跨模型相同架构。
- score_native_confidence_development_v1/prospective_plan.json在计算本候选性能前冻结：三模型×五输入，k32/64，all/last，lambda4/8/16，power1/2，两种D；同60条/24视频组。原始不调节方案、native/readout baseline、原强bias、wrong/low controls均保留。只复用有完整神经支路的已完成源文件，并记录SHA；尚未结束的源实验等待完成后再做相同冻结读出，不重复神经推理。
- 10项数值/缓存测试通过，包括全五档对置信度均有贡献、零干预精确、连续无阈值调节、缺置信度拒绝计算，以及既有SOLE原生token与temporal对比测试。最终仍需234与独立498同时通过严格两类改善和总+10pp；初筛成功类100%的天花板规则不改变最终验收。


## 2026-09-09 13:18：KL预算控制的区域证据（第七候选，前瞻冻结）

- 现有开发信息显示，固定raw-logit gain在Meter、Qwen和不同输入中所需尺度不同；较大的lambda改善Meter部分输入，但个别Qwen相邻k的成功类不确定性仍较大。提出用同一个后验分布变化量来定义干预强度，而不是假设各模型raw logits可直接比较。
- p_lambda=softmax(z0+lambda*D)，D仍为对称区域证据或扣除首帧静态参考的时间区域证据。选择lambda∈[0,64]中满足KL(p0||p_lambda)≤delta的最大值；delta={0.25,0.5,1,2,4} nats。对D做p0加权中心化后，KL等于logsumexp(z0+lambda*D_centered)−logsumexp(z0)，在lambda≥0上单调，使用64步二分；局部曲率为Var_p0[D]。整个计算只读当前样本原生分数向量，不用标签、输出端点或额外success head。
- 全native bins保留；D为常量或delta0时精确返回baseline。最大gain64是预先固定的数值/噪声放大上限，不把近零方向无限放大来强行满足预算。每个结果保存实际KL、有效gain、是否触顶、方向方差。预算不一定能达到，此时必须如实报告。该方法可能仍放大无用方向，因此必须保留wrong/low对照，不能从数学约束直接断言有效。
- score_kl_development_v1全三模型×五输入冻结同60/24视频簇，k32/64、all/last、region/temporal。增加delta0数值对照，以及同KL预算、仅以baseline logits为方向的temperature-only对照。无新神经前向、无新训练；对原始完整源文件只读引用并核验SHA，输出在新目录。
- 10项组合/置信度/KL数值测试通过，包含KL独立定义复算达到目标、全bins保留、常量/零方向原样返回、score整体平移不改变后验、极小方向受gain上限限制、同KL温度保持argmax，以及旧temperature/temporal/置信度接口。当前尚未读取本候选的性能，确认集仍未执行。

补充负结果：RR interleaved时间对比全234仅k72的lambda2/3/4通过同k原bias改良；k24/32虽双类及总+10pp，但不如对应原bias，k64的MAE也差于原bias。不存在三个相邻k的通过范围，因此不计为稳定有效输入。原bias最优k24/32也优于k72候选，不能用挑选原方法较弱k来宣称总体改良。


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


### 2026-09-09 15:10：机制解释的三项限定

1. 视觉总量守恒以每次hook当前收到的Q/K/V为条件，逐head、逐query成立。前层干预仍会改变后层hidden states及Q/K，因此不能声称整网所有层的视觉总量都与另一次完全无干预forward相同。数值零干预与完整递归一致性另由实际审计证明。
2. 分布比表达式是softmax的代数恒等式；attention正负支路不是来自独立数据生成过程的校准似然，因此不能把该比值直接称为真实Bayes likelihood ratio。KL预算约束相对原生分布的偏离，不保证相对真值误差下降，且较大的预算可能给出很弱的约束。
3. 代码中的native_completion_confidence命名沿用实验接口。Qwen/RR的q实际是归一化评分期望，并非答案正确性的概率、最大类别概率或熵；Meter的q来自已训练success head，也未在这里另作概率校准。调节会保护原生高完成度判断，包括错误的高分判断。RR image_text相对同强度uniform主要改善suc、略损fail，与这一机制相符；不能据此宣称普适的不确定性估计。


## 2026-09-09：守恒方向与时间差分的理论边界

为明确“为什么考虑这种改良”，可把单个head/query的视觉token条件分布写成r_j=p_j/M，M为该query当前视觉总概率。令t_j在目标区域取+1、其它视觉token取−1，干预的视觉内条件分布为r_j(b)∝r_j(0)exp(b t_j)。局部归一化校正使M保持不变，因此在当前Q/K固定的这一调用内：

`d p_j(b)/db = p_j(b) [t_j − E_{r(b)}t]`（j为视觉token），非视觉概率导数为0。

若value向量v_j也固定，则该head输出的局部导数为`M Cov_{r(b)}(v_j,t_j)`。它衡量目标与其它视觉内容的value差异，并保持总视觉权重；不是简单让输出带上“更多视觉”。它不要求给某一奖励档指定正号，所以对全部原生档位做差分不会在公式中硬写成功或失败端点。真正的评分方向仍取决于后续网络和当前指令，符号可能错误。

对完整网络的输出logits Z(b)，仅在b足够小且局部平滑时，`[Z(b)−Z(−b)]/(2b)=Z′(0)+O(b²)`。本轮b=3应称对称有限干预对比，不能当作已经测得的精确梯度；b=1.5复验在RR video_text全234未达到门槛，也说明更接近局部方向不保证性能更好。λ与b是两个变量，较大的读出放大也会放大方向噪声及高阶项，必须通过邻域、同强度、wrong/low和温度对照评估。

`D_observed−D_static`是一项“视频变化×区域敏感性”的交互对比。只有当静态外观的干预响应在两个输入中近似共同、且重复首帧参考合理时，才可能抵消共有部分；它不会严格消除所有静态因素，更不等价于对真实物理世界成功状态的随机干预。参考视频仍保留文本、时间结构和像素对应的grounding，没有把首帧reward设成0。

这些推导说明了可检验的机制及其假设，不证明总体MAE或两类准确率必定改善。冻结498确认与完整846描述性结果、失败模块和任务异质性将决定本任务中的有效性结论。


### 2026-09-09 16:17：MAE定义与论文Overall聚合方式的核验

新读取[RoboReward论文v2](https://arxiv.org/html/2601.00675v2)：Section3明确`MAE=(1/N)Σ|r_hat−r|`，预测与标签均为1..5；本轮离散模型的逐样本MAE与此一致。Table1的Overall另外以RoboRewardBench各子集的MAE作group-wise聚合，不能把其23个公开benchmark子集与本数据的34个自定义task混为同一评测。正文、下载时间与SHA存于`RESEARCH/references/roboreward_mae_definition_check_20260909T1615`，来源账本已追加。

当前确认的主指标一直是按样本平均的原生1..5 MAE；连续模型另外报告`|1+4p−y|`，这项连续适配不能冒称论文原生离散输出。为完整呈现聚合敏感性，新增`task_macro_summary.py`从既有逐task统计计算等权task-macro MAE/准确率，不修改已冻结metrics模块、主门槛或参数选择。整task无有效预测时macro MAE记为undefined，不删除该task来制造完整结果；逐task有效数和原始完整性同时保留。

初次真实执行针对BASE/metrics_snapshot_20260909T124540.json，得到180项等权task摘要，文件`BASE/task_macro_snapshot_20260909T1617_v1.json`；其中历史未完成条件仍按原覆盖率标识。最终BASE、确认498和完整846结果完成后都应再生成对应macro补充，并与micro并列、明确各cohort实际出现的task数。这是描述性补充，不在读取确认结果后换用较有利的聚合方式来决定是否通过。

输入来源也需精确措辞：Meter text_image为checkpoint官方结构，所选三个Meter候选输入均为适配；RoboReward现有代码将text_video说明为公开HELM请求顺序，video_text为model card引用的视频推理例顺序，没有据此宣称存在已公开的独立官方benchmark evaluator。本轮固定8帧与attention计算预算，不能把输入顺序相同称为全套官方推理参数复现。


## 2026-09-09 18:16：确认后的新探索——加入原方法的完整评分分布参照

原九项冻结确认已经出现RR text_video的严格同k MAE失败；这些数据、参数、primary和验收不变，后续不会把新方案的结果回写为v1确认成功。按用户要求继续进行理论—实现—实验迭代，同时明确证据阶段：后续498只能叫重复使用验证集上的探索，不能叫新的独立确认。

提出并在计算本候选性能前保存`post_confirmation_anchor_development_v1/prospective_plan_v1.json`及源码SHA。令z_parent为原冻结区域/时间证据方案，z0为同读出无干预分布，z_original为相同k的原强bias分布，新增：

`z_anchor = z_parent + ρ (z_original − z0)`，`ρ∈{0,.25,.5,.75,1}`。

等价于原生分布与原方法分布在logit空间作凸组合，再施加原来的区域证据项；固定原η时，对全部评分档位有`p_anchor(j)∝p0(j)^(1−ρ) p_original(j)^ρ exp(ηD(j))`。该代数表达没有指定某个端点。对ρ的log概率导数是`(z_original−z0)_j−E_p(z_original−z0)`；它描述原方法提供的整体评分方向，不保证任何真值指标单调改善。原强bias可能保留更好的序数信息，也可能再次损害suc，因此两类、两baseline、原方法和整个k邻域仍需一起检验。分布代数可对照本轮已核实的ASCD/VCD机制背景，不能据此冒称独立校准的Bayes证据或引用文献证明本任务效果。

本轮仅在development234上运行：保留原九个模型/输入、每项三个k及中间primary，原radius/scope/gain/置信度/KL配置均不变；新增加的唯一变量是每case共用一个ρ。全部九项、所有五个ρ都保存，ρ0必须精确重现原父方案；先完成无标签评分组合，再join开发labels做评价。所有父sweep的SHA逐项核验，只读复用，不做新神经前向，不改冻结的21个推理文件和三个统计模块。

选型需所有三个k完整、对两baseline四向/+10pp、对同k原方法MAE/总准确率改良、全部邻居五档MAE下降，且primary胜开发预先选定的原方法。在合格ρ中先最大化最弱同k原方法MAE改良余量，再最大化最弱同k准确率余量，平局取较小ρ。**这是观察到v1确认失败后的开发决策，不是最初的预注册，也没有消除后来复用498的适应性偏差。** 当前计划已冻结并实际启动开发CPU组合，尚未宣称新方法成功。


## 2026-09-09：熵缩放与跨输入共享指数（本轮提出的后续假设）

在原确认及anchor失败后，本轮新增`z=z0+s h^β(z_parent−z0)+ρ(z_original−z0)`，其中h为原生全档位分布的归一化熵。h∈[0,1]且对档位置换对称；β>0时高确定性分布的区域证据增量减小，β0回到旧强度。它不指定1或5，不替代评分，不使用真值，但模型熵不是校准正确性；原父方案的成功置信度项仍可能不对称。以上是本轮理论与实验提案，不冒称所引ASCD/VCD等论文已经提出此具体式子或证明机器人任务有效。

64组s/β/ρ仅在234开发上选型，先生成全部评分再join labels。单独输入选型的RR text_video选中s4/β0却在重复498/full846损害suc，失败保留。随后引入同一RR模型三个输入共用β的结构约束；开发最弱协议余量选择β=.5，仍保持每输入原primary和三个k。最终846与重复498全部九项通过观测门槛，完整机制与数值见`SHARED_ENTROPY_RESULT_20260909T1932.md`。所有后续证据仍受适应性探索影响，没有新的独立确认。

有ρ时整个方法不保证视觉总量守恒，只有区域证据分支保持既有局部性质；额外缩放通常也不继承父KL预算。实际保留的Qwen KL输入是s1/β0/ρ0，所以恰好不变。wrong控制并未被所有正确区域指标支配，故不能据点验收宣称因果物理理解。


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
