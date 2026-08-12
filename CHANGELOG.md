# 变更记录

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 的组织方式，并使用语义化版本号。

## [Unreleased]

## [1.0.0] - 2026-08-13

### Added

- 完整公开的 AI 脑、具身脑、双机械臂、STM32F407、指挥中心与安全审计源码树。
- 双 X5 候选板端验收回执与模型会计证据。
- 顶层中英文项目说明、文档中心、复现分层、证据矩阵、安全与社区治理文档。
- 竞赛奖项单一事实源，区分队伍确认的区域成绩与待官方公布的全国奖项。
- 经隐私、元数据与真实性复核的设备照片、静音演示短片、归档站截图和媒体 provenance。
- 可离线确定性重建的 SPDX 2.3 SBOM、仓内链接检查器与发布边界审计器。

### Changed

- 项目定位从“仅供评审的非核心公开边界”升级为完整工程源码发布；仍排除凭据、个人信息、受限数据与不可再分发资产。
- 统一使用 `live`、`shadow`、`replay`、`sim-only`、`offline`、`experimental` 和 `rejected` 标记事实边界。
- 补齐工作站 Vue 前端入口、路由、样式、锁文件、类型检查与可复现构建。
- 删除不能确认再分发权的晶体缓存，改为来源标识、获取方法与许可边界说明。

### Security

- 公开树与可达 Git 历史通过秘密、私网设备身份、个人路径、禁止权重格式、大文件与媒体元数据门禁。
- 真实硬件路径保持 Tier 4 现场人工授权；公开指挥中心保持只读，不反向控制设备。
- 受限晶体缓存从 `main` 可达历史中清除；发布前保留离线恢复 bundle，不将其重新分发。

[Unreleased]: https://github.com/Xiaomiju-x/xrd/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/Xiaomiju-x/xrd/releases/tag/v1.0.0
