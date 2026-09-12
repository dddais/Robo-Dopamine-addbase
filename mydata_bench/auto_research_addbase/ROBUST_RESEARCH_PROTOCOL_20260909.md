# 目标3续研：降低选型过拟合的前瞻协议

## Material Passport

- 时间：2026-09-09；模式：academic-research-suite / experiment-agent，内联执行。
- 状态：方法工程与开发实验；没有新的独立确认结果。
- 目标仓库：Robo-Dopamine-addbase。仅新增代码、配置、数据副本和结果；无git操作。
- 新结果根：`results/mydata_bench/experiments_v2_corssmodel/auto_research/session_20260909_robust_v1/`。

## 目标与不可改变的边界

继续完成原计划目标3：四模型中至少三模型，各至少三输入，至少三个相邻已测k，MAE下降、成功及失败准确率严格提高、总准确率至少+10个百分点，并改良原attention。五种输入全部尝试；不能通过只选弱original对照、删失败或端点映射过关。原数据集的目标不能被外部数据结果替代。

拒绝沿用确认失败后再用498选规则的共享熵/anchor最终配方。旧846的全部结果已被接触，即使重新切分、嵌套交叉验证或标签只在234选择，均不能恢复研究层面的独立性。本轮旧数据只作开发、故障分析与完整描述。没有方法可以凭公式保证不过拟合；有限自由度和新数据验证是证据要求。

最终方法必须在打开外部确认性能前冻结：实现、输入构造、head排序来源、全部数值参数、k邻域、primary、比较方法、分母/错误处理、指标及分析。确认失败后本候选判失败，后续方法需要另一批未用确认数据；不得反复使用同一确认数据宣称独立通过。

## 验证数据审计

旧full1213包含846个grounded样本及34条ranking样本。按视频字节SHA排除cohort/ranking所有视频后剩322条、108视频簇。它们缺少grounding，而且旧baseline已覆盖全1213，不能称完全未接触数据。本轮暂不读这些条目的标签或新方法表现，也不按可分性选择。

公开RoboReward数据集卡指出官方test为2831条人工验证的RoboRewardBench，包含1–5全部等级；已核验API和卡片，尚未读取逐例测试表现。需要冻结数据版本、同源/同视频隔离规则和grounding失败处理。外部测试必须保留中间等级以检查只迎合端点的行为；它是额外泛化证据，不能替代原指定数据集目标。RoboFAC亦核验了公开元数据，但尚未选定为确认源。

## 候选F1：逐帧注意力质量搬运

动机：旧全视觉守恒仍可能在帧之间转移注意力，原+6/-6则同时破坏视觉/文字与帧间比例。目标区域面积也影响统一logit bias实际搬运的概率质量。F1直接规定每帧搬运的非目标概率比例，使干预大小有一致解释。

对当前一层一个head/query，帧f内目标概率和为T，其它位置为N。固定a=1/2：

- 目标位置：`p'_j = p_j + a*N*p_j/T`。
- 同帧非目标位置：`p'_j = (1-a)*p_j`。
- T=0（例如因果上尚不可见）时该帧不变；非视觉位置不变。

每帧总概率、非视觉概率、因果mask在当前Q/K下保持；所有p'非负。这不声称与另一次整网无干预的后层Q/K相同。用原生SDPA输出加`(p'-p)V`实施，a=0精确回到原生前向。无需正负评分分支、静态参考、熵、success head或输出分数放大。

**首轮冻结**：Qwen/RoboReward/Robometer × 五输入，全部使用a=.5、全部帧、全部有效query、原无标签raw-mass head排序。k={32,48,64}，primary=48；不按输入或模型选择a。SOLE已有负结果仍保留，F1先验证可直接共享的三个模型，若需第四模型则执行完整递归并单独冻结，不能用终步近似替代。

首轮在旧234开发样本运行全部15case，不使用仅60条的小筛选，避免成功类100%的筛选天花板。每case包括原生输出、同读出baseline、三个k的原方法和F1、k48的zero/wrong/low，总11个条件。wrong几何失败不删除主样本；单独报告控制覆盖率。

这是根据已知旧方法问题提出的新候选，尚未看到其指标，不是研究开始前预注册，也不预告一定有效。首轮失败后可在旧开发数据提出新的有机制依据的候选；每次保存新协议、所有失败和研究选择。没有按外部测试反馈改方法的选优。

## 分析与验收

主指标沿用原计划：Meter连续ordinal MAE及两套端点阈值，离散模型原生1–5 MAE/准确率。主阈值.125/.875与敏感性.2/.8均预先固定。两套连续阈值下的成功/失败方向均作为本轮稳健性准入，不靠换阈值称通过。原生baseline与同读出baseline分别比较；同k原attention及在开发集选择的original最优k都报告。

完整报告全部模型/输入/k、每任务及macro、预测分布、同视频suc-fail配对、错误和覆盖率。稳定性不能只凭单点；k邻域必须采用相同其它参数。独立确认使用视频簇bootstrap和跨预选主case的多重性控制，并报告+10pp点门槛与置信区间的区别。旧数据的统计量只能描述探索结果，不提供独立确认p值。

最终完成必须同时有原目标范围的有效结果与上述抗过拟合验证证据；当前均未达到。

## 2026-09-09 23:38：F1首批负结果与候选F2

F1在完整234开发集上的前三个已分析case均失败，zero全部逐例精确。primary k48：Qwen text_video的MAE下降.19658，总准确率下降.8547pp、成功下降5.2632pp；RR text_video的MAE下降.41026，总准确率仅+1.2821pp、失败不变；RR image_text的MAE下降.12393，总准确率仅+2.5641pp、成功不变。尚未结束的F1其它case按原计划继续，不改参数或删失败。

这些结果不能说明该机制没有任何用途，但说明注意力守恒本身不足以达到目标，原生评分的视觉依赖仍需检验。F2把固定F1与[Visual Contrastive Decoding](https://arxiv.org/abs/2311.16922)的受损视觉对比分布机制结合：

`z_F2 = 2*z_F1(clean) - z_F1(corrupted)`。

在同一模型已经预处理的视觉tensor上加扩散噪声，保留全部非视觉token、输入长度、时间戳和grid；正负分支使用相同clean区域位置。固定1000步sigmoid beta schedule、step500、对比权重1、F1 fraction=.5、k32/48/64、primary48，三模型五输入全部相同。噪声seed由视频字节SHA、输入协议和pixel字段确定，不使用instruction、reward或数据类别。相同视频不同指令不获得不同随机噪声。

这不是原共享熵配方，也不使用其按case优化的scope、静态参考、lambda、rho、beta。它仍属于需要验证的对比分数方法：受损视觉输出不是真正无视觉或校准先验，减去它不保证误差下降。没有把简单公式当作不过拟合的证明。

官方噪声源码已逐行查看并固定到commit `d6568ff81b8fd306a49e630df44f2db5c2300191`，路径`references_v1/vcd_noise_master_pinned.py`。本轮改成确定性的按视频seed，并在float32做噪声后转回模型dtype。F2保留全部五档/十bins，不采用官方生成解码的词表plausibility截断，因此称VCD机制在reward读出上的适配，不能声称完整复现官方VCD。

F2复用内容SHA核验的F1原始clean分支，在两条预先选择的工程视频上重跑核对缓存与当前前向逐logit一致，只新增需要的noisy分支。额外控制包含VCD-only、每个k的原强bias加相同VCD、F1无VCD。要归因于新的attention方案，F2还应优于这些匹配的VCD控制；不能把VCD本身的收益全部归到F1。

F2全方案zero同时关闭概率搬运和VCD权重，精确回到clean baseline；仅关闭搬运会退化为VCD-only，不能称zero。主方法没有可按模型/输入选取的输出增益或类别门控。当前只在旧开发数据推进；完整2831官方test正在准备，未计算任何模型测试性能。

选型偏差参考：[Cawley & Talbot, JMLR 2010](https://www.jmlr.org/papers/v11/cawley10a.html)已核验原始网页。外部一次性确认、旧数据全部降为开发的安排用于处理研究选择偏差，而不是用普通开发bootstrap来抵消反复选优。

## 2026-09-10 00:12：F3直接检验head选择，外部数据完整性修复

F1已分析九项均未达标，包括全部RR五输入。F2继续按原冻结计划完成，不因中途失败改参数。另提出F3：只改变head排序，原+6/-6 bias、all_frames/all queries、原生全部评分bins保持。没有评分分布放大、熵调节、VCD或按case输出参数。

原raw_mass排序同时受目标面积和该head视觉总注意力影响。对原34个ranking视频，定义`excess = target_mass - (target_token_count/visual_token_count)*visual_mass`。同视频先平均，评分为`mean(excess) - 1.96*SE_video(excess)`；排除前8层，按固定排序取k32/48/64，primary48。这个波动惩罚用于优先选取较一致的目标关注，**不是被选head具有95%同时置信覆盖的声明**。它不保证head拥有因果完成度能力，wrong/low和真正效果验证仍必需。

34条ranking的全部原始目标/视觉质量已重算excess一致，三模型五输入各34个独立视频、896个候选heads，未读reward标签。排序规则和1.96系数跨模型/输入共享；各模型实际排序随其attention测量不同。两项CPU测试通过，包括面积混杂可反转raw_mass排序，以及同视频重复不能增加有效样本量。

F3使用`R/specificity_frozen_plan_v1.json`及15个`addbase_robust_specificity_v1_*.yaml`。推理前冻结全部head次序、源SHA、k、强度和对照。原F1完整clean baseline及same-k原attention按SHA复用，预定两条工程视频重跑baseline/原attention，要求逐logit/score一致，并验证新head集合bias0精确。F3没有逐帧质量守恒承诺。完整开发每case234×11，全部失败保留，不复用外部确认性能选规则。

外部2831条已全部下载/准备完成，媒体没有被替换或删除。原508条未ready中，502个视频是元数据帧数比真实解码帧数多1；OpenCV及FFprobe无错误且实际帧数一致，两次解码一致后按真实首尾重新取8帧。另6条是urllib SSL连接错误，保留原失败，使用已验证的requests+download=true CDN路径补齐；未改变数据revision。最新`inputs_without_labels_v4.jsonl`为2831/2831 ready、2551个视频字节SHA。

`perceptual_overlap_audit_v4.json`对全部2831与旧407视频检查：0字节/8帧精确重叠，0按预先定义三帧DCT距离规则的潜在重叠。该稀疏检查不能证明排除所有同源episode或模型训练污染，不据此声称完全无污染。原v1/v2/v3准备及每次失败/修复审计都保留。

独立数据仅在进行无标签的目标解析：1043个不同task，用既有Qwen3-4B parser，固定提示和greedy/max256/既有解析失败回退。推理字段只有task及opaque task hash；没有新模型奖励预测。后续grounding采用首帧单个明确目标实例、跟踪模型实际观察的八帧；无法适用/定位失败时在完整测试中回退baseline，不能删除难样本。外部测试的最终方法和评价仍须先冻结，当前没有任何测试性能，目标未完成。

## 2026-09-10 00:26：F4联合任务文字与目标区域注意力

F3最早完成的RR text_video k48使MAE下降.62821，却让成功准确率下降25pp、总准确率下降6.8376pp，未通过。不能用MAE单项改善掩盖任务成功判断受损。其它F3按原冻结计划继续。

原+6 ROI bias、-6其它视觉、文字0，会把目标相对任务词的注意力赔率提高exp(6)倍。F4假设这种做法可能使模型看目标时丢失具体命令，因而同时给**实际Task字段的token和ROI视觉token**加6，其它视觉减6，其它文字保持0。相同正偏置使当前Q/K下任务词与ROI之间的相对赔率保持原值；视觉总量不守恒，前层改变后的后层Q/K也不保证等于另一无干预网络。

实现从实际input_ids解码后的显式Task字段获取字符范围，再用fast tokenizer offset定位token，并要求截至task末尾的重编码IDs与实际输入逐个一致。Qwen/RR普通格式、interleaved与Meter分别按其真实模板定位；不选择评分rubric或答案文字。允许原子token边界含相邻标点，不能伪造不存在的sub-token。真实模型每case两条工程样本仍需验证，匹配失败直接保留工程错误，不猜位置。

F4所有模型/输入共用原bias6、原raw_mass排序、all_frames/all有效query、k32/48/64、primary48，**没有与F3重新排序混合**，没有输出后处理。底层采用与原方法相同的native residual kernel及非预加载索引路径，以减少无关数值差异。两个CPU/实际tokenizer测试验证任务与ROI赔率、因果未来位置零、未选head不变、零bias精确，以及三种模板的实际token字段定位。

配置`addbase_robust_joint_task_v1_*.yaml`和`R/joint_task_frozen_plan_v1.json`已在F4性能读取前冻结。每case234×12：native/readout baseline、same-k原visual-only三k、新joint三k、zero/wrong/low、task-only k48。任务文字始终来自当前输入的命令，不是视频真实执行命令、标签、同视频其它样本或元数据路径。外部验证结果仍未打开。


## 2026-09-10：可信度复核、外部episode分组及候选F5

旧方法的直接泄漏证据、适应性选型和阈值问题单独记录在[METHOD_VALIDITY_AUDIT_20260910.md](METHOD_VALIDITY_AUDIT_20260910.md)。没有发现直接端点赋值，并不能抵消反复使用确认反馈造成的研究选型偏差。旧报告“完成”不适用于当前续研目标。

外部2831条的下载来源文件名中含score后缀；当前模型媒体路径为哈希，不将来源文件名传给模型。新增`robust_external_episode_groups.py`只取source subset、original split及首个episode index，RoboArena取UUID，忽略后续全部后缀；再按相同视频字节/八帧像素SHA做连通分组。不读取reward文件或模型预测、不删任何样本。结果1172组，最大8条，无未解析来源，供将来的簇不确定性分析。这个保守分组可能合并不同后缀来源，不能证明不存在训练污染。两个测试覆盖后缀独立、数据split隔离、同字节跨来源合并的传递性和顺序不变性；第一次pytest只因测试的相对import无法收集，改为绝对import后2项通过，未修改已冻结数据处理。

F4最先完成的RR image_text/text_video/video_text仍未通过，primary成功准确率分别下降28.9474/26.3158/25个百分点。直接同时给task和ROI加6并未解决任务判断，不能把相对赔率不变当作有效性的证明。F1已分析13项均失败；F2/F3的RR五输入均无稳定通过，剩余模型照原冻结计划继续。

**F5：扣除移除任务语义后仍存在的attention响应。** 设A为原+6/-6、all_frames/all queries、原raw_mass排序的attention；Ø只把当前Task字段的输入embedding置零。固定：

`z_F5 = z_A(task) - [z_A(Ø) - z_0(Ø)]`。

等价于`z_0(task) + {[z_A(task)-z_0(task)]-[z_A(Ø)-z_0(Ø)]}`，用于检验attention改变中是否存在任务文字与视觉区域的交互信息。三个模型五输入共用单位系数、bias6、k32/48/64、primary48；不叠加旧熵、置信度、KL、anchor、静态视频或按case的幅度选择。全部原生bins保留。A=0时两条Ø支路相同，完整方法精确返回原始baseline。

Ø来自已经过F4真实tokenizer验证的当前Task字段；原始input_ids、序列长度、视觉像素、grid、timestamps不改变。embedding hook只改实际任务token，完整前向须命中且仅命中一次；不是把文本替换成另一个样本的正确任务。ROI仍来自真实当前指令，因此Ø分支并非完全无任务信息，任务长度/位置也仍存在。零embedding可能超出训练分布，该交互量不是物理因果成功效应；这些都是需要在外部数据检验的限制。

每case完整234，包含两个baseline、三k原attention/F5、whole zero、wrong/low全公式控制，另报`2*z_0(task)-z_0(Ø)`的无attention任务对比以检查归因。使用F1冻结clean结果作实际A分支缓存；工程样本真实复算baseline、原A和新公式，要求缓存逐logit一致，零干预的真实/Ø支路均精确。新方法设置在读取F5任何性能前冻结。当前仍仅开发数据，未开始外部奖励模型测试。

记录UTC：2026-09-09T16:50:09.052828+00:00


## 2026-09-10 01:22：F5负结果与F6外部监督选头路线

上一轮属于实质进展：补充可信度审计，完成外部episode分组，实现并冻结F5。当前F5的RR image_text/text_video/video_text完整234均未通过。primary相对baseline的MAE分别+1.09829/+0.37179/+1.07692，不能用成功率局部改善掩盖总体评分退化。F1–F5剩余输入继续按原配置运行。

外部测试解析1043/1043已完成，0解析runtime错误；实际SAM3定位/跟踪正常启动，已核验5条实际外部轨迹均覆盖8帧。到01:19观察1032条，691可定位、341按既定规则回退baseline，0runtime错误；这是中间覆盖率，不是最终覆盖率，也没有打开reward标签或奖励模型表现。

### F6的机制与训练边界

无监督raw-mass或面积调整排序只能估计“看哪里”，不直接估计“哪个head的干预能降低评分错误”。新增核验[Michel, Levy & Neubig, 2019, Are Sixteen Heads Really Better than One?](https://arxiv.org/abs/1905.10650)，正文4.1的head重要性为`E|dL/dξ_h|`，其中ξ控制整个head输出。原文强调绝对值用于保留敏感度。本候选**不是该剪枝方法的复现**：我们对reward query处的ROI log-bias求导，并保留符号，目标是选择一致降低监督损失的干预方向。原文HTML、正文和来源SHA在`R/references_v1/michel_2019_head_importance_html_v1.*`；初次本地提取因未安装bs4中断，随后用Python标准库解析，HTTP200并实际读取4.1公式和适用范围。

定义所有非前8层head的参数b_h，在真实评分query上给目标视觉key加b_h，其它视觉key减b_h；其它query/head保持原生输出。训练时冻结整个模型权重，在b=0测每条拟合样本的监督损失梯度。离散模型损失使用全部1–5原生选项的交叉熵；Meter用原生十bin、真实1–5等级在相邻进度bin上的线性软标签。保留所有中间等级，不把训练数据改成二分类。

同一个训练episode/视频内容连通组先平均梯度，再对组等权平均，得到μ_h与组间SE_h。排序规则固定为`abs(μ_h)-1.96*SE_h`，推理方向固定为`-sign(μ_h)`；1.96只是波动惩罚，不能声称选中head的同时95%置信保证。无目标ROI时拟采用baseline回退，相应的ROI参数梯度定义为0并留在其组内；实际梯度计算出错时整项拟合失败，不丢该条继续选择。

所有模型/输入统一推理幅度|b|=1、k32/48/64、primary48、all_frames、真实reward-query，全部原生档位和原始读出保留。这个固定有限步长不保证一阶损失下降在实际网络仍成立；验证会直接检验该限制。拟合只产生head排序和方向，无熵、输出分数放大、anchor、测试标签或按旧234优化的参数。**本路线使用公开训练标签，是监督辅助选头，不再称完全免训练。** 当前只是候选，没有性能承诺。

### 独立训练来源与验证约束

另下载同revision `469b9af76e8539e8d2ac553b081307738fd92ca5` 的官方train metadata（45072条，10168个来源episode）。仅用来源subset/split/episode字段排除保留test来源，0个来源episode交叉；没有打开test reward字段。训练集包含1–5全部等级；其标签允许拟合，保留test标签仍封存。官方RoboReward骨干可能已经用这些训练数据训练过，不称这些训练样本对骨干全新；独立性要求针对最终保留test和本轮拟合/验证划分。

在每个训练source subset按固定盐SHA选择最多20个episode，前floor(n/5)、至少1个留作训练验证，其余拟合，保留所选episode的所有行。未用标签分层或模型输出选样。共2509条、568组：拟合455组，验证113组。分组采样近似平衡来源，后续梯度是episode组等权，不能冒称精确source-macro风险。文件在`R/external_training_research_v1/`，所有选择在拟合性能读取前固定。

原视频下载、FFprobe与两次OpenCV真实帧数校验、取真实首尾8帧正在运行。随后按字节、八帧像素、固定三帧DCT检查与保留test/旧数据重叠；命中的整个训练来源/内容连通组隔离，跨训练fit/validation的精确重复组也隔离。所有2509条在审计文件保留，隔离只影响训练资格，不改变保留test或旧目标数据的分母。之后仅对合格训练数据做同一无标签parser/SAM3定位。

F6梯度工程已在RR image_text、Qwen text_video、Meter官方text_image各两条旧工程样本通过：实际zero和F1缓存逐logit一致，训练前向b=0亦无差异；896个候选head得到有限非零梯度，前8层梯度精确0，所有骨干权重冻结。工程使用明确合成的grade3探针，未读取这些旧样本真值，也未用其梯度拟合排序。峰值显存约19.4/18.4/14.3GiB，不是全部训练样本显存保证。两个kernel测试与两个组均值/方向测试通过。

`reward_gradient_fit.py`将只读冻结的fit IDs/fit labels，训练validation、旧234及保留test均不参与选头。`freeze_reward_gradient_fit.py`在无标签准备完成后、第一条实际拟合前保存完整输入/源码SHA和15项配置。后续还必须冻结实际学到的ranking哈希、推理实现和所有评价条件，完成独立训练验证、旧234/846描述性评价以及最终一次性外部确认，才可能认定目标3完成。拟追加控制为同幅度readout-query raw-mass选头、学习排序但统一正方向，以及完整zero/wrong/low；这些控制用于检验选头和方向归因，不取代原attention最强开发对照。

记录UTC：2026-09-09T17:24:05.297290+00:00


### 01:32 状态追加：训练媒体审计已完成

媒体重叠审计于UTC17:30:04正常结束，不能再将其列为运行中。21个匹配记录的原因计数为{'reserved_test_sparse_perceptual_flag': 21}；保留训练资格的条数为{'validation': 493, 'fit': 1942}，隔离条数为{'validation': 28, 'fit': 46}。逐条隔离原因计数为{'reserved_test_sparse_perceptual_flag': 61, 'media_unavailable': 13}（一条可有多个原因，不能直接相加当独立样本数）。全部2509条仍在审计文件中，保留test分母未变。无标签训练任务解析已启动，1072个不同任务；尚未选头或测试奖励表现。

记录UTC：2026-09-09T17:32:32.230933+00:00


## F6训练验证流程已实现并前瞻冻结（2026-09-10 01:57）

冻结文件为`R/f6_evaluation_policy_v1.json`，SHA `2e589f11d9ffeac573a82b177b9b268bd1bbfde65eee1c74aea8b1c40e40628e`，包含27份推理/评价依赖源码SHA与内容快照、493条验证ID/108组，以及仅从旧开发选择的强original对照。实际学到的ranking文件将在全部15项拟合完成后、任何F6验证/旧开发评分前冻结。训练和原有冻结源码未修改。该规则在F1–F5开发结果已知后制定，不冒称整个研究开始前注册；当前没有F6拟合或验证性能。

新增`reward_gradient_validation.py`只接收无标签输入、独立验证ID、learned/original ranking及条件；两个真实grounded验证样本和一个已有fallback样本（如存在）先做工程核验。旧工程也验证实际基线/原attention与F1缓存一致。无ROI的完整方法与每个ROI原方法对照回退typed baseline；所有验证条目保留。分数失败写error行，不重试/丢弃来改善结果。原生baseline与typed baseline各自单列。

`reward_validation_metrics.py`正确计算包含全部1–5等级的exact accuracy、原生ordinal MAE、Meter连续ordinal和标准五档MAE，成功/失败类在两套阈值下另外报告。不得把中间等级全部当错来计算“总准确率”。无效预测在完整分母计错误；MAE同时报告最大误差4惩罚的完整分母版本和valid-only诊断；必需预测有错误则准入失败。误差4是报告惩罚，不是模型输出赋值。按task、source、grade、grounding状态给分布，另列task/source macro与episode macro。

外部训练验证准入：三个已测k均对两baseline、same-k原方法和固定旧开发best-original实现MAE下降与全部等级准确率上升，对两baseline的端点准确率在两套Meter阈值均不下降，并且whole-zero精确。这是混合等级数据上的补充选型门槛；原旧234的总准确率+10pp、成功/失败严格提高、三k稳定与强original比较门槛保持。**必须相同的至少三种输入协议在两种数据上同时通过，不能各挑不同的三个输入。** 完整旧846描述和一次性保留test还须后续完成。

不确定性使用20000次配对来源/内容整组bootstrap，报告各k的边际95%区间；固定primary k48的MAE/全部等级准确率相对4个参照构成IUT，15个case做Holm。中心化bootstrap单侧p是近似零改善检验，不是+10pp置信下限，也不校正整个历史适应性研究。验证用于选型，不再称它是最终测试。

4项有意义的CPU检查通过：中间等级正确计分和失败分母、Meter标准分档/两端阈值、同来源复制不能伪造样本独立性、无ROI回退保留中间评分且不调用区域/模型。5个新工作模块语法解析通过。真实F6验证尚未开始，不能把这些工程检查写成评分效果验证。

`reward_gradient_evaluation_queue.py`已启动PID2464770，等待拟合队列PID2357092；全部15项拟合成功后先冻结旧开发和训练验证，然后在GPU0/3按空余显存运行工程→训练验证→旧234。全部15项完成后才关联验证标签并执行冻结分析。失败保留且不静默重试；该队列不会打开保留test。

保留test无标签定位已完成2831/2831：1788可用（63.16%）、1043回退（36.84%）、0runtime错误。回退原因314条超出单明确目标范围、607条首帧无合法目标、122条跟踪不完整。全量结构审计确认原输入字段不变、ID完整无重复、1788条轨迹覆盖全部实际观察帧，见`R/external_grounding_complete_structural_audit_20260910T0154_v1.json`；该审计没有重新解码像素，先前真实proxy工程另存。定位覆盖率不等于奖励模型性能。

记录UTC：2026-09-09T17:57:51.818978+00:00


## 原始846完整验收补齐（2026-09-10 02:12）

本次完成4个增量模块：`reward_gradient_full_old.py`、`freeze_reward_gradient_full_old.py`、`analyze_reward_gradient_full_old.py`、`reward_gradient_full_old_queue.py`。它们保留全部原846的ID/像素输入与评分条件，并在独立训练验证和旧234的同输入联合门槛通过后，扩展所有联合通过的case。不能只因外部数据改善就跳过原数据验收；也不会根据完整846结果更改head、bias1或k32/48/64。

旧234完整预测按SHA和配置/条件逐项核验后原样保留，包括失败行。新框架先用两条真实旧工程样本复算baseline、原attention、新F6和whole-zero，要求四者与旧234缓存精确一致；仅在工程通过后推理剩余样本与新增的固定best-original条件。全846逐条件/逐样本完整性和zero仍须重新全量检查。两项CPU测试已通过：必须每模型相同至少三输入联合达标；缓存失败不能被删除且条件变化被拒绝。真实full846尚未推理，测试不构成性能证明。

完整指标含全部k及对照、两套Meter阈值、MAE/连续ordinal MAE、suc/fail/总准确率、task/split预测分布、同视频suc−fail的负/0/1/2/3/4档配对、top8列表及top8/32/64跨case head索引重合度。head索引相同不代表跨模型功能相同。所有846此前已暴露，仅给描述性统计，不重新命名为独立确认。

条件队列PID2466415已启动并核验存活，等待F6独立验证队列2464770正常完成。若联合门槛失败，保存不符合准入的记录并停止此扩展；若通过才冻结完整846配置、按GPU0/3剩余显存运行并分析。四份新源码已由队列捕获SHA并另存内容快照；原拟合和27份评价依赖均未修改。**该队列不会打开保留外部test；最终一次性确认仍需单独实现和冻结。**

新增负结果：F2的Meter官方text_image完整234，primary MAE−0.308073、总准确率0pp、成功−22.3684pp、失败+10.7595pp，不通过；F3的Qwen video_text MAE+0.414530、总准确率−14.1026pp、成功−27.6316pp、失败−7.5949pp，不通过。这进一步说明无监督视觉区域敏感性不能直接等价于奖励判断能力；F6用独立训练损失选头只是待验证的后续假设，未获得实际效果证据。

记录UTC：2026-09-09T18:12:15.436641+00:00


## F1–F6完整结果与F7全query监督选头（2026-09-10 10:53）

以当前文件和完成记录为准：F1–F6的三模型×五输入均已完成旧234开发，共90个case，完整稳定通过数均为0。F6的15项独立1942条拟合也全部完成，1253条实际梯度、689条无ROI零梯度，每case无拟合运行错误。15项493条验证及旧234推理均结束。F6旧234的primary最高总准确率提升为Qwen video_text的3.8462pp，仍低于10pp；RR多输入的MAE反而增加。验证点门槛通过RR text_video/interleaved和Meter text_video三项，但与旧234联合通过数为0，全部primary Holm p为1。原846条件队列正常记录eligible=false并退出，未扩展、未打开奖励模型保留测试。

验证错误需要区分：wrong-region在离散video两种输入每case有100条、离散image/interleaved约73条、Meter video79条、Meter image45条无法构造等面积完全错位区域；这些不是主评分分支失败。Qwen interleaved另有2条native-generation格式失败。所有错误均保留，主分母不删。wrong控制无效时不能用它的惩罚后均值声称区域归因成功，后续应单列控制可构造范围与同范围配对结果。F6主效应本身不足以通过，不通过不能归咎于这些控制错误。

本次核验了完整2435条训练grounding：1565可定位、870回退；fit为1253/689，validation为312/181。全量轨迹的source indices覆盖一致，另实际重新解码前5条proxy的8帧，与原输入PNG逐像素精确一致。记录`R/training_grounding_complete_audit_20260910T1047_v1.json`。这不证明语义目标选择全都正确。旧数据用户给定grounding与新数据无标签自动定位的适用边界仍不同。

F6并非因排序全无信号而失效：按模型/输入有285–764个head的固定惩罚后score为正；但仅在最后评分query干预的实际分数变化小。新增F7将同一监督ROI-bias梯度、同一组均值/SE排序、同一正负方向规则扩展到**所有causal prefill query**，仍以原生评分位置的完整1–5/十bin损失拟合。它允许改变评分之前的视觉/任务表征，而不是只改最后聚合。所有query仍遵守原生因果遮罩，不能读取其未来key；整套steering使用完整输入任务的定位，因此不声称在线流式因果决策。局部未选head不变不等于后续层等于另一无干预网络。

F7同样排除前8层，保留骨干冻结、独立fit IDs/labels和无ROI零梯度；固定推理幅度1、k32/48/64、primary48、all_frames，无熵、anchor、输出放大或端点赋值。改变query范围会改变拟合梯度与head排名，因此重新在同一1942条fit数据测量。拟合仍不读旧234或493验证标签。493条已用于F6分析，从F7开始明确作为反复使用的模型选型数据，不称新的独立确认；保留2831条test表现仍未打开。

一个CPU核验通过：zero精确、真实all-query变化、未选head不变、未来视觉不可见时不变化、显式/隐式causal mask一致、梯度与有限差分一致。真实工程Qwen text_video、RR image_text、Meter官方text_image各两条旧工程样本均通过：零推理、缓存、梯度前向logits精确，896个非零有限梯度，前8层零，骨干无权重梯度。峰值显存约20.7/23.4/37.8GB（实际以MiB记录为准）；合成grade3仅用于工程，不是样本真值或性能测试。

F7源码与配置已冻结：`all_query_reward_gradient_attention.py`、`all_query_reward_gradient_engineering.py`、`all_query_reward_gradient_fit.py`、`freeze_all_query_reward_gradient_fit.py`及fit queue；`R/all_query_reward_gradient_fit_frozen_plan_v1.json`有全部15项。队列PID2553345、GPU0/3已启动真实拟合，当前进度以日志为准。**F7后续推理/验证/原846扩展尚待适配并在效果读取前冻结，当前队列仅拟合。**

本次另准备了F6保留确认的4个模块及2项CPU边界检查，但F6不具备准入资格，因此这些模块未激活、未生成confirmation policy/claim、未运行保留测试。它们不能作为已经完成最终确认的证据；以后适配新候选时仍须完整冻结，不能读取测试来调方法。

记录UTC：2026-09-10T02:53:57.554791+00:00


## 2026-09-10 11:18：方法完整性说明与F7评价冻结

此前共享熵方案是在观察原确认失败后继续开发得到的观测结果，原498已复用，不能作为新的独立确认；历史“目标完成”不作为当前稳健主线3的完成依据。F1–F6完整旧234稳定通过数均为0，当前F7尚未完成效果验证。详见[METHOD_INTEGRITY_EXPLANATION_20260910T111808.md](mydata_bench/auto_research_addbase/METHOD_INTEGRITY_EXPLANATION_20260910T111808.md)。

F7评价policy已在任何F7效果读取前冻结：64份源码快照、90份F6比较文件摘要、493条复用选型数据、固定幅度1与k32／48／64、primary48。新增相同学习头只在评分query干预的控制，错位区域不可构造时不据其惩罚分数归因。持久评价队列PID2567628等待全部15项拟合成功，然后冻结配置、跑工程与493／旧234并分析；此队列不运行保留测试。


## 2026-09-10 11:39：F7条件最终确认与训练诊断

F7已完成8项拟合，无拟合失败，效果评价仍待全部15项完成。最终确认policy已在保留测试前冻结72份源码；条件队列2571353只在原范围的联合门槛和完整846门槛通过后，才执行全部预选case的一次性2831条保留确认。保留claim与奖励模型测试输出当前不存在。3项最终确认范围/完整性测试通过。

7项已完成拟合的F6/F7零干预native logits及损失在每项1253条实际梯度样本逐条一致；训练数据有300个同视频多指令、多等级组。另记录8项零点方向导数和Qwen五输入梯度兼容性，均为训练诊断，不是F7效果结果，不改变已冻结方法。完整证据见[ROBUST_PROGRESS_20260910T113923.md](/mnt/public1/dais/workspace/Robo-Dopamine-addbase/mydata_bench/auto_research_addbase/ROBUST_PROGRESS_20260910T113923.md)。目标3仍未达成。


## 2026-09-10 12:06：F7拟合12/15与有限幅度训练诊断

F7完整拟合已完成12项，Qwen/Meter各五输入、RR两输入，无拟合失败。独立训练记录诊断已核对11项每项1253条实际梯度的F6/F7零bias前向逐例精确。GPU2/3空闲后，新增固定Qwen text_video与Meter官方text_image的完整1942条fit诊断：实际baseline、全query b1/k48、同头同方向readout-only。全部无ROI样本也实际运行baseline，不删样本；不搜索参数，不读取493/原234/保留test效果，不改F7验收。规则/源码/配置已冻结，两个进程在正常产生记录。详情见[ROBUST_PROGRESS_20260910T120611.md](/mnt/public1/dais/workspace/Robo-Dopamine-addbase/mydata_bench/auto_research_addbase/ROBUST_PROGRESS_20260910T120611.md)。
