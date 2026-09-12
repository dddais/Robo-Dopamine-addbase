# F8 有序评分梯度：拟合前方案

## Material Passport
研究模式：代码实验；阶段：prospective fitting；实际目标目录：Robo-Dopamine-addbase；本文件记录时间以文件内 launch JSON 为准。拟合和验证数据分别为1942条/446组、493条/108组；493已经复用于方法选择。原846完全暴露；保留2831条奖励性能仍未打开。研究目标3未完成。

## 动机与方法
F7的两项完整拟合诊断中，固定 b=1、k=48 确实降低训练CE，但Qwen text_video的准确率下降0.1545个百分点、ordinal MAE增加0.071061。CE只奖励真等级的概率，不能区分同等概率下的近档错误和远档错误。该现象只是训练诊断，不能提前判定F7验证失败。

F8唯一的方法变化是拟合损失。对全部B个原生有序档位的softmax分布p、原生目标分布q，在每个相邻边界t计算预测CDF Fp(t)及目标CDF Fq(t)，最小化平均Bernoulli log loss：

`L = -mean_t[Fq(t) log Fp(t) + (1-Fq(t)) log(1-Fp(t))]`。

Qwen/RR使用全部5档onehot目标；Meter保留原grade到10 bins的相邻线性插值目标。两侧累积概率直接在log空间计算，避免CDF舍入后clamp。每个边界都是proper Bernoulli score，全部CDF唯一确定原生分布，因此其期望目标仍是原生评分分布；不声称这能保证有限干预下MAE或argmax准确率提高。

保留F7全causal prefill query区域干预、前8层排除、episode等权梯度均值、abs(mu)-1.96SE排序、-sign(mu)方向，以及b=1、k=32/48/64、primary48。每模型每输入分别拟合，三个模型各五种输入全部纳入。没有采用跨输入共同头的新假说。推理读出、档位与阈值均不改变；不使用评价端点阈值设计损失，无输出增益、anchor或端点赋值。

## 拟合及验证边界
新的GPU2队列先做三模型真实梯度工程检查（仅合成grade3，不读实际工程标签），通过后一次冻结全部15项配置，再运行完整1942拟合。689条无ROI的样本以零梯度纳入组均值；运行错误保存并使对应拟合不合格。三模型工程或冻结失败不启动拟合；没有自动重试。F7的GPU0/3队列、所有源码和参数保持不变。

本阶段只授权拟合，不启动验证／保留测试。后续F8评价规则须在读F8验证性能前单独冻结，继续采用原目标范围、两套Meter阈值、原方法强对照、全样本分母与共同k邻域。493只能叫复用选型集；全846只能作旧数据描述与原目标检查。F7有既定保留测试优先权；如果其打开并失败，不能把相同保留集当F8新确认。

## 实现与核验
native_cumulative_ordinal_loss.py：两项CPU测试通过，覆盖proper score、中间档、距离敏感性、极端logits及有限差分梯度。新增ordinal_reward_gradient_engineering.py、ordinal_reward_gradient_fit.py、freeze_ordinal_reward_gradient_fit.py、ordinal_reward_gradient_fit_queue.py。配置检查确认原配置未改写，训练数据、排名、k和推理参数保留，损失和元数据一起改变。

队列启动时存源码SHA与内容快照，后续每个阶段复核；工程输入及已有基线也记录SHA。所有结果新增保存。输出根目录R/ordinal_reward_gradient_v1；队列R/f8_fit_queue_plan_v1.json。

## 文献范围
Cao、Mirjalili、Raschka的CORAL论文（arXiv:1901.07884）标题、作者与摘要已在arXiv核实。它支持将有序目标写成二元子任务的动机；F8不是CORAL架构，使用原生softmax的单调CDF，不新增阈值分类头。截至本方案正文未核实，不借用其理论结果声称F8保证泛化。

本方法属于使用外部训练标签的监督式选头，不能称training-free。公开数据预训练污染不能排除，本地封存也不是第三方盲测。当前没有F8收益结果，不宣称完成目标。

R=`results/mydata_bench/experiments_v2_corssmodel/auto_research/session_20260909_robust_v1`。
