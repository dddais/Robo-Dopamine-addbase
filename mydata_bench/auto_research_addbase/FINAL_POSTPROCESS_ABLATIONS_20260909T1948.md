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
