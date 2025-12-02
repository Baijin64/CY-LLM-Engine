归档说明
=================

此文件夹存放在迁移过程中备份/归档的文件，以便后续审计和回退使用。

此目录包含：
- `rename_occurrences_raw.old`：重命名搜索原始结果归档；作为迁移历史记录保存。
- `rename_occurrences_raw.txt`：占位或小型摘要版本。
- `.env.example.bak`：原始 `.env.example` 的备份副本。

策略：
- 需要时再执行恢复或删除。此目录会保留在仓库中供审计。
