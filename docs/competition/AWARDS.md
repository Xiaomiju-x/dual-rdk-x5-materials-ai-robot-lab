# 竞赛与奖项

奖项信息只从 [`award_status.yaml`](award_status.yaml) 读取。该文件是唯一权威数据源；README 与本页中的文字是发布时生成的展示，不应单独手改。

<!-- AWARD_STATUS:START -->
| 阶段 | 当前状态 | 证据边界 |
| --- | --- | --- |
| 西南赛区 | 第1名 | `team_confirmed`：队伍确认，官方排名来源待补 |
| 全国总决赛 | 奖项待组委会正式公布 | 待官方公布：不预测、不预填奖项 |
<!-- AWARD_STATUS:END -->

## 明日更新规则

组委会正式公布后，只编辑 `award_status.yaml` 中 `national` 的 `status`、`result`、`source_url`、`evidence_path`、`evidence_sha256` 和 `announced_at`；不要直接改 README 的奖项文案。随后从仓库根目录运行 `python tools/publication/render_award_status.py` 生成三个展示块，并以 `python tools/publication/render_award_status.py --check` 核对。脚本会验证本地证据文件及其 SHA-256。公开结果必须与官方称谓逐字一致。

在 `source_url` 与可核查证据缺失时，`national.result` 必须保持 `null`，不能写“预计”“大概率”“保底”或用奖牌图标暗示等级。

西南赛区第 1 名目前是队伍提供的事实。找到官方排名页或证书公开件后，应在同一权威文件中把 `regional.status` 改为 `official_verified` 并补齐来源；在此之前不把公开晋级名单误写成名次证明。
