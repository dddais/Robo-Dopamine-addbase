# 已关闭实验阶段的综合结果

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
