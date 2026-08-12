# 安全离线快速开始

这条路径用于在没有 RDK X5、机器人、串口、相机、凭据和私有模型权重的情况下理解项目。它不会触发网络访问或物理运动。

## 复现分层

| Tier | 内容 | 设备/副作用 | 当前入口 |
| ---: | --- | --- | --- |
| 0 | 阅读文档、图片、报告与静态站点快照 | 无安装、无设备 | 本文 |
| 1 | mock、fixture、静态合同或离线回放 | PC 本地；不得访问硬件 | 各模块中明确标记的离线测试 |
| 2 | 获得许可的数据/模型评估 | 需要单独下载、版本与哈希核验 | 模型/数据目录的专用说明 |
| 3 | RDK X5 固定输入推理，无运动 | 需要板卡环境和人工核验 | 板端 runbook 与验收回执 |
| 4 | 真实传感器、执行器和机器人 | 有物理风险；现场授权 | 不提供通用一键命令 |

Tier 0/1 是普通贡献和代码评审的默认范围。不要因为源码存在就越级到 Tier 3/4。

## Tier 0：十分钟审阅

1. 从 [项目首页](../../README.md) 阅读状态表和真实性边界。
2. 查看 [最新硬件 Hero](../../assets/media/hero/project-hardware-hero.webp)、[实机与视频画廊](../gallery.md)和[系统架构图](../../assets/images/system/fig_xrd_architecture_html.png)。
3. 在浏览器中直接打开 `public_site_static/index.html`，检查归档的只读证据站快照。该快照不连接真实设备，也不代表当前在线服务状态。
4. 查看 [`schemas/`](../../schemas/) 中的只读状态合同与示例响应。
5. 用 [证据索引](../evidence/EVIDENCE_INDEX.md) 交叉核对模型会计、板端后端和候选状态。
6. 阅读 [已知限制](../evaluation/KNOWN_LIMITATIONS.md)，确认没有把固定输入、回放或仿真结果扩大解释。

预期结果：静态页面与图片可离线浏览；没有服务启动、设备连接、模型加载或动作输出。

## Tier 1：安全代码审阅

优先选择以下不依赖真实设备的内容：

- `schemas/` 的 JSON 合同与示例；
- `edge_public/` 和 `workstation_public/` 的 mock、接口与回放边界；
- 具身脑、双臂后继候选中明确声明为 offline/replay 的测试；
- `evidence/ai_brain/` 的机器可读 acceptance 与 board overlay。

### 已验证的根级零硬件检查

在仓库根目录使用 Python 3.11 运行：

```bash
python -B tools/publication/audit_release.py --root . --strict
python -B tools/publication/check_markdown_links.py . --format text
python -B tools/publication/render_award_status.py --check
python -B tools/publication/generate_sbom.py --check
python -B tools/publication/verify_media.py --root .
python -B -m unittest discover -s tests_public -p "test_*.py" -v
python -B examples/offline_demo/run_demo.py
```

这六条命令只依赖 Python 标准库；`-B` 禁止生成 `__pycache__`。它们不打开网络、相机、串口、GPIO、机器人 SDK 或执行器，也不写入仓库。离线 demo 将 JSON 回执输出到标准输出，并明确报告 `OFFLINE_SYNTHETIC_NO_ACTUATION`、50 个注册模型、50 个 release-ready 合同、24 个 PC BPU 工具链编译记录，以及全部为 `false` 的副作用字段。

2026-08-13 发布验收结果：

| 检查 | 结果 |
| --- | --- |
| 公开边界审计 | `PASS`：0 findings |
| 仓内 Markdown 链接 | `PASS`：0 findings |
| 奖项展示一致性 | `PASS`；唯一事实源未漂移 |
| 确定性 SPDX SBOM | `PASS`；生成结果与随仓清单一致 |
| 无硬件单元测试 | `PASS` |
| 确定性离线 demo | `PASS` |
| 工作站前端 | `npm ci` 与 `npm run build` 通过 |

公开边界审计会检查凭据、私钥、私网信息、个人路径、模型权重扩展名、大文件、图片元数据、JSON、禁止目录和奖项单一事实源。退出码 0 才表示扫描树通过；它不证明科学模型效果或真机安全。

### 前端构建复核

工作站前端不是根级零依赖检查；它需要 Node.js 与 pnpm。需要复核 UI 构建时：

```bash
cd workstation_frontend_public
pnpm install
pnpm exec vue-tsc -b
pnpm exec vite build
```

依赖安装会访问包注册表并写入本地依赖目录，因此不属于“零网络、只读”的六条根级检查。前端构建仍不连接机器人硬件。

## 配置原则

- 从 [`.env.example`](../../.env.example) 了解变量名称，在本地副本中填写；不要提交真实 `.env`。
- 保持硬件、动作和采集变量为 `disabled`。示例变量只是公开配置约定，不是已经验证的运行时强制门。
- 不把模块旧文档中的现场路径、设备标识或历史网络参数复制到 issue、日志和新配置。
- 需要 Tier 3/4 时，先阅读 [物理安全规范](../safety/PHYSICAL_SAFETY.md)，再由现场负责人使用目标硬件的最新 runbook。

## 常见误区

- PC 编译回执不等于 actual X5 BPU 性能。
- 固定张量差分不等于真实相机准确率或导航成功率。
- 回放学习指标不等于真实机械臂策略成功率。
- 静态证据站不是机器人远程控制台。
