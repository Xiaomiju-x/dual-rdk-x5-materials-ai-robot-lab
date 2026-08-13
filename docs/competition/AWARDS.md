# 竞赛与奖项

奖项信息只从 [`award_status.yaml`](award_status.yaml) 读取。该文件是唯一权威数据源；README 与本页中的文字是发布时生成的展示，不应单独手改。

<!-- AWARD_STATUS:START -->
| 阶段 | 当前状态 | 证据边界 |
| --- | --- | --- |
| 西南赛区 | 一等奖 | `team_confirmed`：队伍确认，官方获奖来源待补 |
| 全国总决赛 | 二等奖 | `team_confirmed`：队伍确认，组委会官方获奖来源待补 |
<!-- AWARD_STATUS:END -->

## 官方来源补齐规则

全国总决赛二等奖现由队伍确认，状态为 `team_confirmed`。取得组委会官方获奖页面或证书公开件后，只编辑 `award_status.yaml` 中 `national` 的 `status`、`source_url`、`evidence_path`、`evidence_sha256` 和 `announced_at`，把状态升级为 `official_verified`；不要直接改 README 的奖项文案。随后从仓库根目录运行 `python tools/publication/render_award_status.py` 生成三个展示块，并以 `python tools/publication/render_award_status.py --check` 核对。脚本会验证本地证据文件及其 SHA-256。公开结果必须与官方称谓逐字一致。

在 `source_url` 与可核查证据缺失时，只允许依据队伍确认写入 `team_confirmed`，并保持所有官方证据字段为空；不能伪称 `official_verified`，也不能继续预测更高等级。

西南赛区一等奖目前是队伍确认的事实。找到官方获奖页或证书公开件后，应在同一权威文件中把 `regional.status` 改为 `official_verified` 并补齐来源；在此之前不把公开晋级名单误写成获奖证明。
