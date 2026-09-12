# 探索文件清理归档

目标仓库：`/mnt/public1/dais/workspace/Robo-Dopamine-addbase`。清理日期：2026-09-12。

| 类型 | 移出工作目录 | 内容 |
| --- | ---: | --- |
| YAML | 298 | 早期差分、熵与 F1–F6 等探索配置，以及 8 个 SOLE pilot 配置 |
| Python | 105 | 不属于保留入口依赖或冻结源码清单的旧实验、恢复、分析及测试脚本，含根目录的一次性配对分析脚本 |
| Markdown | 20 | 未被保留正文链接引用的重复 progress/checkpoint 文档 |
| 合计 | 423 | 压缩包按仓库相对路径保存原始字节 |

保留正式 baseline/attention、F7–F9 的 fit/development/validation 与训练诊断配置、公共模块、必要源码哈希依赖、两份简短总结及科学结论文档。数据、预测、失败记录、结果快照和保留测试均未改动。

- [exploration_files.tar.gz](exploration_files.tar.gz)：原文件归档。
- [manifest.json](manifest.json)：逐文件原路径、SHA-256、大小、权限、时间及归档原因；另含保留文件的清理前哈希。
- [archive.sha256](archive.sha256)：压缩包校验值，路径以仓库根目录为基准。
- [verification.json](verification.json)：归档、保留源码、冻结配置与相关检查结果。

清理前先验证归档内 423 个文件的原始字节，再移出工作目录；保留的既有文件不作改写。旧版本 `robust_external_overlap.py` 原有语法错误也原样归档，未修饰历史失败证据。

## 恢复

在目标仓库根目录运行以下命令。先校验压缩包；校验成功才恢复。`--keep-old-files` 防止覆盖清理后新建的同名文件，遇到冲突会报错。

```bash
cd /mnt/public1/dais/workspace/Robo-Dopamine-addbase
sha256sum -c archives/exploration_cleanup_20260912/archive.sha256 && \
  tar --keep-old-files -xzf archives/exploration_cleanup_20260912/exploration_files.tar.gz
```

只恢复单个文件时，在 `tar` 命令末尾加上 `manifest.json` 中的完整相对路径；复现旧路线通常需要同时恢复其脚本和配置。

归档恢复到本次清理前的原路径与原始字节。历史结果中记录了配置、旧队列和源码路径，旧冻结流程及旧复现命令可能需要先恢复；已有实验更早的源码版本仍应按结果中的 source snapshot 和原哈希查找。新实验另建配置和输出目录，避免写入历史结果。
