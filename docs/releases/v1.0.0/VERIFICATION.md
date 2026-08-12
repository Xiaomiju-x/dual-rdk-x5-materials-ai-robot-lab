# v1.0.0 发布验证记录

本页记录 `v1.0.0` 的可重复工程门。它只证明公开仓库形态与离线合同，不证明 Tier 3/4 真机安全、科学泛化或未测试环境。

## 干净检出门

在全新 clone、Python 3.11 环境中从仓库根目录运行：

```bash
python -B tools/publication/audit_release.py --root . --strict
python -B tools/publication/check_markdown_links.py . --format text
python -B tools/publication/render_award_status.py --check
python -B tools/publication/generate_sbom.py --check
python -B -m unittest discover -s tests_public -p "test_*.py" -v
python -B examples/offline_demo/run_demo.py
```

验收要求是六条命令全部退出 0，审计与链接检查均为 0 findings，奖项展示与唯一事实源一致，SBOM 可逐字节重建，测试全部通过，离线 demo 的网络、相机、串口、GPIO、机器人 SDK 与执行器副作用字段全部为 `false`。

## 工作站前端门

在 Node.js 20 环境运行：

```bash
cd workstation_frontend_public
npm ci
npm run build
```

`npm ci` 必须严格使用随仓 `package-lock.json`；类型检查和 Vite 构建必须同时通过。生成的 `node_modules/` 与 `dist/` 不进入 Git。

## 供应链与历史门

- [`sbom.spdx.json`](../../../sbom.spdx.json) 是确定性 SPDX 2.3 清单；范围与精度边界见 [SBOM 说明](../../safety/SBOM.md)。
- 随仓浏览器组件的来源和许可证见 [`THIRD_PARTY_NOTICES.md`](../../../THIRD_PARTY_NOTICES.md) 与 [`third_party/licenses/`](../../../third_party/licenses/)。
- 不可确认再分发权的晶体缓存不在发布树或 `main` 可达历史中；公开仓只保留来源标识和获取说明。
- 原始手机媒体不进入仓库；发布派生文件的裁切、时间码、隐私复核和 SHA-256 见 [`MEDIA_PROVENANCE.yml`](../../../assets/media/MEDIA_PROVENANCE.yml)。

## 远端复核

[GitHub Actions CI](https://github.com/Xiaomiju-x/dual-rdk-x5-materials-ai-robot-lab/actions/workflows/ci.yml) 会在每次 `main` 推送和拉取请求上重复上述发布、测试与前端门。CodeQL 和 Dependabot 分别提供持续静态分析与依赖更新入口。
