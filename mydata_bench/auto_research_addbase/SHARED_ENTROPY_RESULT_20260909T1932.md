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
