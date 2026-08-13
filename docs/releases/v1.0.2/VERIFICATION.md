# v1.0.2 发布验证记录

本页记录“基于双 RDK X5 异构协同的材料合成 AI 预测与多机具身实验助理机器人”`v1.0.2` 的发布门。它只证明当前公开仓库形态、奖项一致性、安全静态检查和离线合同，不证明 Tier 3/4 真机安全、科学泛化或未测试环境。

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

两套前端均使用随仓锁文件复核：

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

全部工作线合并后，本表填写最终精确文件、字节、测试和模块计数；发布附件、SBOM 与公开媒体的校验值见 [`SHA256SUMS.txt`](SHA256SUMS.txt)。

| 门禁 | 结果 | 精确范围 |
| --- | --- | --- |
| 奖项展示与单一事实源 | `PASS` | 全国总决赛二等奖；`team_confirmed`；renderer check 退出 0 |
| 仓内 Markdown 链接 | `PASS` | 108 个 Markdown 文件、303 个链接、0 findings |
| 确定性 SPDX SBOM | `PASS` | SPDX 2.3；579 packages / 6 files / 1108 relationships；顶层包版本 `1.0.2` |
| 严格公开边界审计 | `PASS` | 2686 files / 211,441,947 bytes / 0 findings |
| 媒体完整性与隐私元数据 | `PASS` | 20 个清单条目；6 张最新照片、3 段 MP4、3 个 GIF；0 findings |
| 零硬件单元测试 | `PASS` | 79 passed / 1 skipped；跳过项仅为 Windows 账户无创建 symlink 权限 |
| Python 源码语法门 | `PASS` | 712 个仓内 Python 文件全部 AST 解析通过 |
| 确定性离线 demo | `PASS` | `OFFLINE_SYNTHETIC_NO_ACTUATION`；网络、相机、串口、机器人 SDK 与写入均为 `false` |
| 工作站前端构建 | 待最终云端复核 | npm 锁文件安装、`vue-tsc -b` 与 Vite production build |
| 具身仪表盘前端构建 | `PASS` | pnpm 11.19.0 离线 frozen install：554/554 本地复用、0 下载；793 modules；Vite/PWA production build |
| 工作树与可达历史秘密扫描 | 待最终云端复核 | Gitleaks 目录与 Git 历史扫描 |
| GitHub CodeQL | 待最终云端复核 | Python 与 JavaScript/TypeScript |
| `main` 分支保护 | `PASS` | 6 个必需检查；禁止强推/删除；线性历史；对话解决 |

## 事实专项检查

- 正式项目名不得退化为仓库 slug 或旧英文简称。
- 竞赛名称、赛道与赛题为“2026 全国大学生嵌入式芯片与系统设计竞赛·芯片应用赛道·地瓜机器人赛题”。
- 区域奖项为“西南赛区一等奖”，全国奖项为“全国总决赛二等奖”；两项目前均是 `team_confirmed`。
- 三段 MP4 均有可点击动态预览、说明和 [`MEDIA_PROVENANCE.yml`](../../../assets/media/MEDIA_PROVENANCE.yml) 记录。
- 最新六张照片全部直接显示在中英文 README，包括保留二维码的项目海报和现场联调合影。

## 边界

PASS 仅说明这些自动化门在记录时退出 0。媒体记录的是特定设备、工装和演示时刻；奖项生成器保证展示与 YAML 一致，但不会代替组委会官方来源。
