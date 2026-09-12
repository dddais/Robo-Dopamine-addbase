# 同视频配对监督：来源核验与后续假设

## Material Passport
来源核验／理论探索阶段，不是已实现或有效的新方法。仅基于已核实fit数据结构及F7训练诊断，不读取F7/F8部分验证效果，不改变冻结方法。

Christopher J. C. Burges，From RankNet to LambdaRank to LambdaMART: An Overview，Microsoft Research Technical Report MSR-TR-2010-82。微软官方PDF下载成功，共19页，SHA和原件见R/pairwise_ranking_literature_verification_v1/verification_v1.json。阅读标题、引言、第2/2.1节RankNet公式及第4节关于排序目标不匹配的讨论；未宣称读完所有boosting推导。该报告明确说没有呈现新的实验比较，不能引用为本机器人任务的效果证据。

RankNet在同一query内比较有不同相关等级的项目，用sigmoid(s_i−s_j)和成对binary CE学习相对顺序。第2.1节将每个样本涉及的成对梯度累计成lambda，再乘该样本score对参数的导数，可减少重复反传。此结构可以为“同视频、不同任务指令”的监督提供参考，但本研究不是网页检索，也没有实现LambdaRank或NDCG优化。

## 必须保留绝对评分约束
仅使用成对顺序不能确定绝对等级：例如真实等级[2,3,4]，预测[2,3,4]和[1,2,3]的所有分差一样，但后者MAE为1。RankNet的分差目标无法单独区分这种共同偏移；它也不直接优化原生argmax准确率。因此任何未来配对方案仍需原生所有等级的绝对监督，不能用排序改善替代目标3。

一种尚未实施的后续假设，是在F8原生有序损失之外，加同视频真实等级差的Huber误差：

`L_G = mean_(i in G) L_ordinal(i) + mean_((i,j) in I_G) Huber[(s_i-s_j)-(y_i-y_j)]`。

这里s是全部原生评分概率的期望等级，仅用作可微训练量；推理保留原生输出。I_G只包含相同实际显示视频、不同指令的训练对；当前fit有1038对、300个相同视频组，包含全部等级差1–4，不能将“不同指令”当成必须一个成功一个失败。配对与episode内样本相关，不能当作独立样本。无配对组保留绝对监督，无ROI保留冻结fallback的零干预导数与完整分母。

使用真实等级差可以避免单纯追求无限分离；在理想观测目标下，s_i=y_i同时满足绝对与差值目标，但这不是有噪声分布上的新proper-scoring定理，也不是有限b／原生准确率保证。Huber形式、权重与训练采样若进入实验证据，必须先另行固定，不得从已打开的最终测试选取。

链式法则可将配对梯度写成sum_i lambda_i * d s_i / d b。若复用现有先按episode求样本均值的聚合器，行梯度需含相应n_G因子，使聚合后的梯度恰好等于声明的完整目标，不能把线性化的行代理值冒称真实配对损失。未来实现必须用耦合autograd／有限差分验证，且当前缓存只有每条样本的一种标量损失梯度，不能凭空当作score Jacobian复用。

这些是待检验的候选设计约束。当前没有F9配置、GPU实验或验收结论，F7/F8全部固定流程继续。目标3未完成。

官方来源：https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf

R=`results/mydata_bench/experiments_v2_corssmodel/auto_research/session_20260909_robust_v1`。
