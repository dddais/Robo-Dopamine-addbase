# 开发阶段统计与复现核验（2026-09-09 13:02）

## Material Passport

- 工作模式：academic-research-suite / experiment-agent validate，内联执行。
- 数据与代码边界：仅Robo-Dopamine-addbase；用户指定数据、grounding及模型权重只读。原始结果append-only，错误版本保留并有有效性注释。
- 材料：冻结60条/24视频簇初筛、234条完整开发集、34个ranking样本，以及全量1213/grounded846的原baseline描述。
- 验证状态：ANALYZED。列明的数值/缓存一致性检查已实际执行；算法总体有效性仍属探索，未完成独立确认。
- 确认状态：498条/179视频簇，和development234的完整视频内容hash不重叠；新方法确认性能未用于选型。部分确认像素已用于原baseline描述和少量无标签工程检查，不能声称像素完全未接触。
- 输出：confirmation_analysis_protocol_v1.json、candidate_inventory时间快照、paired_analysis簇统计，以及新增baseline汇总；所有路径位于本目标仓库results/mydata_bench。

## 11/11统计与方法偏差类型已检查

|类型|本任务的证据与处理|当前边界|
|---|---|---|
|1. 总体/分组方向反转|准确率同时报告suc、fail及逐task；门槛要求两大类均上升，不能用多数fail掩盖suc下降。配对统计保留per_task。|不要求每个task都上升；任务间异质性必须披露，不把总体改善推广到每个任务。|
|2. 群体推断个体|同视频多指令作为视频簇重采样，推理输出仍为逐样本。|有平均改善不代表每条指令改善；不把846条当846个独立视频。|
|3. 样本选择偏差|grounded846来自全1213的可定位cohort，用户明确接受其grounding；确认排除ranking同视频。|效果只能针对该任务分布与cohort；不能无依据推广到任意视频或未定位目标。|
|4. 条件筛选/碰撞偏差|wrong-region只在同帧、等面积、无重叠几何可行时比较；target全cohort不随wrong可行性删样本。|wrong比较是可行子集上的结果，必须列出被排除样本，不能混作全cohort控制。|
|5. 忽略基准发生率|全量407 suc/806 fail，cohort268/578，开发76/158；全部同时报告两类及总准确率。|总准确率受本数据比例影响，不代表部署环境中的阳性预测值。|
|6. 回归均值|60初筛较好的点需要完整234复核；RR video_text原对比在234仅+5.13pp的失败已经保留，未按初筛+10pp算成功。|234包含60，仍是开发选型；其区间不是筛选偏差已消除的确认。|
|7. 幸存者偏差|错误保留；准确率分母为所有expected，MAE注明valid数量。SOLE text_video原强biask64的37个格式失败明确报告。|有效样本MAE不宜独立于coverage解读；失败不会从reasoning中猜数补齐。|
|8. 多重搜索|多个机制、k、scope和gain确实经过探索；失败轮次、原始分支及前瞻grid均记录。|开发最优点不能称无偏发现；最终参数需先冻结，确认primary用跨case Holm/IUT并给效应量与簇区间。|
|9. 分析自由度|每轮在本轮性能计算前冻结配置；天花板初筛规则、SOLE格式修复排除、同强度和温度对照均留有时间记录。|这不是最初一次预注册的单一检验；追加机制及比较均承认是探索。|
|10. 相关与因果混淆|实际修改模型注意力并保留同输入baseline、零、原bias、区域/低head等干预对照。|可说在指定实现上干预改变输出；不能由attention质量、定位精度或一次性能改善宣称模型学会完整因果任务理解。|
|11. 反向因果/泄漏|推理代码不读取labels；head由独立ranking集合冻结；static参考、native置信度均由当前输入/模型输出计算。|不得用单条真值选gain、head、输出端点；不能用确认结果反调参数再称独立确认。|

## 统计解释与必要限定

主要不确定性单位是完整视频内容簇；paired_analysis使用20000次簇bootstrap与簇符号翻转，既有单元检查验证整簇复制不改变区间、零效应p=1、缺失样本不能过完整门槛、Holm单调。簇符号翻转依赖零假设下的成对可交换性，不假称随机抽取了真实部署任务。

最终门槛中的“+10%”按更明确的+10个百分点执行；p值检验改善方向，不检验区间下界超过10个百分点。suc/fail样本比例不均，必须给出各自绝对变化和区间。不同统计方法的95%区间与单侧p值不能机械要求互为等价判定；所有方法和单/双侧方向必须标明。

原始baseline与新分数读出必须并列；Meter官方整序列概率猜测与typed softmax的已知差异单列，SOLE完整native生成与201整数分布读出单列。原生置信度调节和更大gain的效应必须由同强度消融区分；只缩放原生logits的温度对照不计作attention方法成功。

原强bias除了同k比较，还在candidate_inventory中描述开发集全部已测试k的最小MAE/最高准确率包络；该包络可能由不同k贡献，是更严格的描述性对照，不能伪装成单个实际原方案。确认时原方案的选择必须在开发集完成并固定。


### 2026-09-09 15:10：机制解释的三项限定

1. 视觉总量守恒以每次hook当前收到的Q/K/V为条件，逐head、逐query成立。前层干预仍会改变后层hidden states及Q/K，因此不能声称整网所有层的视觉总量都与另一次完全无干预forward相同。数值零干预与完整递归一致性另由实际审计证明。
2. 分布比表达式是softmax的代数恒等式；attention正负支路不是来自独立数据生成过程的校准似然，因此不能把该比值直接称为真实Bayes likelihood ratio。KL预算约束相对原生分布的偏离，不保证相对真值误差下降，且较大的预算可能给出很弱的约束。
3. 代码中的native_completion_confidence命名沿用实验接口。Qwen/RR的q实际是归一化评分期望，并非答案正确性的概率、最大类别概率或熵；Meter的q来自已训练success head，也未在这里另作概率校准。调节会保护原生高完成度判断，包括错误的高分判断。RR image_text相对同强度uniform主要改善suc、略损fail，与这一机制相符；不能据此宣称普适的不确定性估计。


### 2026-09-09：样本、视频簇和配对分母的精确范围

`frozen_cohort_completion_v1/partition_pairing_coverage_audit_v1.json`核验：原始1213条为407视频簇，407 suc/806 fail；grounded846为299视频簇，268 suc/578 fail；开发234为86簇、76/158；确认498为179簇、158/340；ranking重叠114为34簇、34/80。划分时同一视频在eligible grounded cohort内的全部条目一起分配，没有跨开发/确认/ranking视频泄漏。

“完整视频簇”指eligible grounded条目在切分和bootstrap时整体保留，不能误写成原始1213中的该视频所有条目都有可用grounding。grounded中31个视频簇缺eligible suc，确认中为21簇。故grounded可配对fail是543/578，确认可配对fail是315/340；这些fail仍参与总体/分类准确率，只在要求对应suc也存在的same-video配对统计中没有配对。原始806个fail在全1213中均有对应suc。


### 2026-09-09 16:03：确认分析完整性补充

确认启动后、尚未读取新方法确认性能时的代码检查发现：manifest虽记录全部输入SHA，原分析入口尚未重新核验这些SHA。已给`analyze_confirmation`增加四个冻结输入文件与metrics/paired_analysis/analyze_development三个统计模块的SHA检查；给846汇总增加冻结输入与三个partition IDs的SHA检查。全部实际核验通过。该改动只在文件不匹配时拒绝分析，不改变任何模型参数、评分、指标、门槛、重采样次数/种子或case选择。新分析源码内容SHA和变更理由保存在`confirmation_v1/analysis_integrity_checks_supplement_v1.json`；新结果另记录实际分析源码SHA。原冻结manifest与旧源码快照均保留。

复核与复现入口集中于`mydata_bench/auto_research_addbase/REPRODUCTION_GUIDE_20260909.md`。完整实现补充清单含141个Python文件，原冻结21个推理相关文件仍全部相同；补充清单不冒称在确认开始前就已预注册。


### 2026-09-09 16:17：MAE定义与论文Overall聚合方式的核验

新读取[RoboReward论文v2](https://arxiv.org/html/2601.00675v2)：Section3明确`MAE=(1/N)Σ|r_hat−r|`，预测与标签均为1..5；本轮离散模型的逐样本MAE与此一致。Table1的Overall另外以RoboRewardBench各子集的MAE作group-wise聚合，不能把其23个公开benchmark子集与本数据的34个自定义task混为同一评测。正文、下载时间与SHA存于`RESEARCH/references/roboreward_mae_definition_check_20260909T1615`，来源账本已追加。

当前确认的主指标一直是按样本平均的原生1..5 MAE；连续模型另外报告`|1+4p−y|`，这项连续适配不能冒称论文原生离散输出。为完整呈现聚合敏感性，新增`task_macro_summary.py`从既有逐task统计计算等权task-macro MAE/准确率，不修改已冻结metrics模块、主门槛或参数选择。整task无有效预测时macro MAE记为undefined，不删除该task来制造完整结果；逐task有效数和原始完整性同时保留。

初次真实执行针对BASE/metrics_snapshot_20260909T124540.json，得到180项等权task摘要，文件`BASE/task_macro_snapshot_20260909T1617_v1.json`；其中历史未完成条件仍按原覆盖率标识。最终BASE、确认498和完整846结果完成后都应再生成对应macro补充，并与micro并列、明确各cohort实际出现的task数。这是描述性补充，不在读取确认结果后换用较有利的聚合方式来决定是否通过。

输入来源也需精确措辞：Meter text_image为checkpoint官方结构，所选三个Meter候选输入均为适配；RoboReward现有代码将text_video说明为公开HELM请求顺序，video_text为model card引用的视频推理例顺序，没有据此宣称存在已公开的独立官方benchmark evaluator。本轮固定8帧与attention计算预算，不能把输入顺序相同称为全套官方推理参数复现。


### 2026-09-09 16:23：全部Meter邻居的五档MAE补查

`confirmation_v1/development_all_neighbor_five_bin_mae_audit_v1.json`仅检查开发234，确认性能未读取。Meter三个输入×三个k的五档MAE均相对两套baseline下降：image_text约−1.1410/−1.1111/−1.0769；text_video约−0.6838/−0.6752/−0.6453；video_text约−0.7350/−0.7607/−0.6880。原连续ordinal MAE改善与此相容。最终确认/846总结需逐邻居同时核对并给出两种MAE，不能只写primary或只挑较好口径。既有确认门槛/参数不改；任何更严格的全邻居五档检查须与原冻结point gate分列。


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


## 2026-09-09 19:36：共享熵方案通过观测门槛，独立性限制不变

完整结果见`SHARED_ENTROPY_RESULT_20260909T1932.md`，唯一最终观测方案为`RESEARCH/shared_entropy_evaluation_v1/frozen_manifest_v1.json`。开发234、重复498、全846均有全部9项×3个k通过两baseline四向/+10pp、same-k原方法及primary开发预选原方法检查；另逐个summary用严格算术复核全部81点，见`direct_final_acceptance_audit_v1.json`。不再选择参数。

原v1独立确认8/9通过仍保持False；anchor v1及entropy单独选型v1的失败均保留。共享β是看到entropy v1失败后提出的开发规则，虽参数仍按234选取，重复498与full846也不能被称新独立确认。无新确认性p；原IUT/Holm不能转移给最终方案。11类偏差处理继续适用，尤其多重搜索、选择偏差、微/宏口径、总体与任务异质性、错误对照可行子集、关联与因果区分。

新whole controls全部9项target/zero精确，但RR image_text相对wrong的suc略低，RR text_video相对wrong的总/fail更低，不能宣称所有控制指标支配。完整846的最小suc改善2.2388pp、重复498为1.8987pp，需如实披露余量；114分区仍有失败且未剔除。原模型分布保留中间档，标签未进入推理/新评分生成，熵只改变全bin证据的标量强度。

两个SOLE mandatory全七步仍运行；全量baseline最终表/重复科学记录审计尚待完成。整体goal不能提前标complete。


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


## 2026-09-09 21:07：统计与完成状态最终更新

全部mandatory实验已结束，最终统计见`BASE/metrics_snapshot_20260909T205912.json`与`BASE/task_macro_final_v1.json`。最终研究总报告为`FINAL_RESEARCH_REPORT_20260909.md`，验证状态仍为ANALYZED：真实记录与确定性组合已审计，不冒称独立全神经复现或新的确认性统计。原独立确认整体False、新方案复用验证、两阈值差异、任务/配对及控制权衡维持20:10以来的完整说明。11类偏差检查继续适用。
