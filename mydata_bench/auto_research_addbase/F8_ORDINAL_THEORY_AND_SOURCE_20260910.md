# F8有序损失：理论说明与正文核验

## Material Passport
模式：方法推导／来源核验；阶段：F8拟合进行中；不包含F8验证结果，不改变已冻结源码、k或评价规则。

## 一个直观例子
真值都是3分。两种预测分别为[0.01,0.40,0.18,0.40,0.01]与[0.40,0.01,0.18,0.01,0.40]。正确档位概率都为0.18，因此原生CE都为1.714798，但第一种主要错在2/4，第二种主要错在1/5。有序累计log loss分别为0.268842和0.519229，会区分错误距离。这是数学示例，不是研究数据的效果。

## 推导及限制
将每个原生相邻边界t看作事件Y≤t。在真实CDF为Q_t时，该事件的预期Bernoulli log loss在P_t=Q_t处最小。全部边界的等正权和同时最小时，各CDF均匹配，差分恢复原生分布。因此F8损失是对原生目标分布的proper score；不是把Y转换成成功／失败二分类。Meter仍使用原grade对应的10-bin插值目标。

这种性质**不保证固定b1干预后MAE或准确率改善**。CE和累计log loss的理想分布最优解其实相同；本实验检验的是不同梯度几何在冻结backbone、稀疏选头和有限幅度约束下是否更合适。原生argmax也不等于绝对误差的Bayes中位数规则；本实验不改原生读出。不能把proper scoring或CDF单调性说成目标3达标定理。

## 正文核验
已下载并提取arXiv:1901.07884v7（2020-11-13，9页）官方PDF，阅读第3节相关公式、第4.3节训练选择说明和补充定理的适用条件。作者Wenzhi Cao、Vahid Mirjalili、Sebastian Raschka；标题Rank consistent ordinal regression for neural networks with application to age estimation。原件及SHA见R/ordinal_loss_literature_verification_v1/fulltext_verification_v1.json。

CORAL第3.2.2节Eq.(4)在K−1个二元任务上求weighted binary CE；其架构通过共享输出权重和独立bias构造sigmoid概率，Eq.(5)用0.5二元判定并累计得到rank。F8使用现有softmax各档位的CDF，CDF自然单调；没有实现CORAL架构或改变评分规则。只引用其有序二元分解的动机，不借用其架构定理保证本方法。

该版本第4.3节存在报告描述歧义：先写“the best model was selected via MAE performance on the validation set”，随后又写“reported the best test set performance within the 200 training epochs”。这两句对选型边界的描述不一致。未核查其训练代码和完整日志，不能断言实际执行了哪一种；因此不以该论文的实验数字证明独立泛化，也不沿用后一种表述作为本研究选择规则。F8仍遵守冻结规则和一次性最终确认边界。

## 实际拟合核验
首项qwen text_video完整1942条核验：1253有ROI样本的原生logits及定位与F7逐值相同；689无ROI按同样零梯度纳入。1253梯度数组全部改变。CPU从已保存原生logits重算F8损失，最大绝对差5.722e-6，满足事先设置atol/rtol均1e-5。top48与F7重合41个头且这些头方向相同，仅证明选头发生了有限变化，不证明改善。完整15项CPU审计observer等待拟合完成；不会读取验证或保留性能。

R=`results/mydata_bench/experiments_v2_corssmodel/auto_research/session_20260909_robust_v1`。目标3仍未完成。
