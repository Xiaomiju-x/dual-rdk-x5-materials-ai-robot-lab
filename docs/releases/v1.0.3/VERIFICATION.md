# v1.0.3 发布验证记录

本页记录“基于双 RDK X5 异构协同的材料合成 AI 预测与多机具身实验助理机器人”`v1.0.3` 的发布门。它证明当前公开仓库形态、证书文件完整性、奖项一致性、安全静态检查和离线合同，不证明 Tier 3/4 真机安全、科学泛化或未测试环境。

## 复核命令

从仓库根目录运行：

```bash
python -B tools/publication/render_award_status.py --check
python -B tools/publication/check_markdown_links.py . --format text
python -B tools/publication/generate_sbom.py --check
python -B tools/publication/audit_release.py --root . --strict
python -B tools/publication/verify_media.py --root .
python -B -m unittest discover -s tests_public -p "test_*.py" -v
python -B examples/offline_demo/run_demo.py
gitleaks dir . --config .gitleaks.toml --redact=100 --no-banner
gitleaks git . --config .gitleaks.toml --redact=100 --no-banner
```

两套前端继续使用随仓锁文件复核：

```bash
cd workstation_frontend_public
npm ci
npm run build

cd embodied_brain/ros2_ws/src/my_robot_dashboard/frontend
corepack enable
corepack prepare pnpm@9.12.3 --activate
pnpm install --frozen-lockfile
pnpm run build
```

## 发布结果

| 门禁 | 结果 | 精确范围 |
| --- | --- | --- |
| 奖项展示与单一事实源 | `PASS` | 西南赛区一等奖、全国总决赛二等奖；两项均为 `certificate_verified`；证书路径、编号、颁发机构和 SHA-256 校验通过 |
| 仓内 Markdown 链接 | `PASS` | 证书、README、奖项页、画廊与证据索引链接均纳入检查，0 findings |
| 确定性 SPDX SBOM | `PASS` | SPDX 2.3；1197 packages / 6 files / 2433 relationships；顶层包版本 `1.0.3`；确定性重建一致 |
| 严格公开边界审计 | `PASS` | 证书元数据字段、文件存在性和 SHA-256 纳入 fail-closed 审计，0 findings |
| 媒体完整性与隐私元数据 | `PASS` | 22 个清单条目；新增 2 张证书 PNG；0 findings |
| 零硬件单元测试 | `PASS` | Windows：94 passed / 1 skipped（账户无创建 symlink 权限）；包含证书哈希绑定、篡改拒绝、README 可见性与奖项状态回归测试 |
| 确定性离线 demo | `PASS` | `OFFLINE_SYNTHETIC_NO_ACTUATION`；无网络、相机、串口、机器人 SDK 或执行器访问 |
| 工作树与历史秘密扫描 | `PASS` | 发布前工作树与可达 Git 历史均执行脱敏扫描 |
| GitHub CI / CodeQL | `PASS` | 推送后的必需检查全部成功；CodeQL 开放告警 0 |
| `main` 分支保护 | `PASS` | 禁止强推/删除，保留必需检查、线性历史与对话解决门 |

## 证书专项检查

- 区域证书逐字写明“西南赛区一等奖”，不是“第一名”。
- 全国证书逐字写明“全国总决赛二等奖”。
- 两张证书的学校、作品名、参赛队员和指导教师一致，作品名使用完整正式名称。
- 两张证书公开文件与收到的 PNG 逐字节一致；原文件自身仅含 `IHDR`、`IDAT` 与 `IEND` PNG 数据块。
- 仓库不声称存在尚未取得的独立组委会 HTTPS 成绩页；`certificate_verified` 与 `official_verified` 保持明确区分。

## 边界

PASS 仅说明这些自动化门在记录时退出 0。获奖证书证明证书上记载的竞赛成绩；它不扩展任何模型、机器人或科学性能主张。
