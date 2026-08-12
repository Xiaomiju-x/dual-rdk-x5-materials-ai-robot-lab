# 复赛第 4 部分：公网展示交接

> 日期：2026-07-20
> 状态：Site32 v1.13 已完成备份、首页单渲染器热修复、原子发布和线上只读验收。
> 适用范围：复赛现场第 4 部分公网展示。

## 1. 当前结论

- 第 1 部分具身脑、第 2 部分 AI 脑和第 3 部分双机械臂均已完成并冻结。本轮没有访问、修改、重启或调用两台 X5、两台 Pi、机械臂、F407、Dashboard 或四线服务。
- 第 4 部分只展示公开只读证据，不提供机器人、底盘、机械臂或实验设备控制入口。
- 前三部分公开口径已统一：Lab-FSD 仍为 `shadow/assist`；双臂已真机完成 arm01 单臂视觉冗余、投袋和与 arm02 并发四周期研磨；袋状态以 X5 CPU/OpenCV 为权威，BPU 仅作辅助语义和真实执行证据。
- X5/Pi 未上电时，网站继续使用 `mirror/replay/offline/unknown` 等来源标签，不冒充实时设备状态。

## 2. 生产权威基线

| 项目 | 权威值 |
|---|---|
| release | `site32-global-commercial-v1.13-20260720` |
| manifest digest | `612e4eb069e7cbd3f807908538a62c2cf6c46a2b6a396909d6f75b6e1c7cbd44` |
| 生产部署后 manifest artifact SHA-256 | `0ca31df571d67af46f551a411bc78f4eb654e8613e6e1518b7e37576127d1eae` |
| 本机 immutable candidate manifest artifact SHA-256 | `1d1ad3a45cac2cc46c53a4b893fd23bc924ddcf5838e653090af284d9e5a08a9` |
| Service Worker cache | `cmdcenter-shell-v96-site32-global-commercial-v1.13` |
| 公网入口 | `https://xiaomiju.xyz` |
| 生产回滚快照 | `/home/rdk/cmdcenter/_releases/site32-global-commercial-v1.13-20260720-predeploy-20260720-082952` |
| 本机发布前备份 | `backups/site32_home_scene_flicker_prechange_20260720_081005/` |
| 详细验收 | `docs/upgrade2026/site32_v113_home_scene_hotfix_20260720.md` |
| GitHub public | `main@9b701e4` |

`site31_*` 工具和 API 名称只属于兼容层，不代表当前生产版本仍是 Site31。

## 3. 三种手动模式

1. **完整模式，默认**：完整动态液态玻璃背景、26 个桌面导航、全部页面和功能、Three.js 数字孪生均保留。
2. **答辩模式，现场推荐**：只冻结背景液态元素动画并使用平衡渲染；导航、页面、功能、状态来源标签和 3D 场景全部保留。
3. **极简模式，第三档降级**：手动减少背景和导航表现，用于笔记本性能明显不足时的最后降级。

三种模式只允许用户手动切换，不做自动降级。现场进入后应显式选择答辩模式，不能依赖浏览器上一次保存的模式。

## 4. 现场唯一浏览路径

1. 打开 `https://xiaomiju.xyz`，使用既有评审账号登录。
2. 点击右上角三点菜单，在视觉模式中显式选择“答辩模式”，然后关闭菜单。
3. 按顶部 GUI 固定顺序浏览：`总览 -> 亮点 -> 答辩防御 -> 全球对标 -> FSD 世界模型 -> 机群 -> 资产`。
4. 只讲当前页面已经显示的来源、限制和公开证据，不进入 `Command`，不展示内部运维路径，不尝试从公网下发动作。

如答辩笔记本仍出现不可接受卡顿，再从同一菜单手动选择“极简模式”；功能性展示顺序不变。

## 5. 验收结果

- Python：共执行 `156` 项，`155 passed, 1 skipped`。跳过项仅为 Windows 可选符号链接测试。
- 静态浏览器合同：`46 PASS / 0 FAIL`。
- 认证安全烟测：`66/66`。
- Site32 preflight 和 deployed gate：均为 `pass`，关键失败为 0，内部 readiness 为 `97.1/100`。
- 浏览器视口：1280x720、1366x768、1440x900、1536x864、1920x1080，以及 960x540 的 200% 缩放等效视口均无横向溢出。
- 26 个桌面导航可见；三点菜单可展开；完整、答辩、极简三模式可往返切换；3D 场景非空。答辩模式连续 8 秒 40 次采样完全一致：真实 `#c3d` 始终可见、`#scene2d` 始终隐藏、legacy renderer UI 始终不存在。
- 生产 `xrd-cmdcenter`、`xrd-auth`、`caddy` 均为 active；origin 仅监听 loopback；UFW active 且默认拒绝入站；Caddy 配置验证通过。
- 未登录首页、状态 API 和 Service Worker 均返回 401；伪造 `X-User/X-Role` 仍返回 401，符合现有 SSO 边界。
- 线上浏览器未登录时正确进入登录页，控制台无错误。本轮未绕过 SSO，也未读取或修改账号会话。

Cloudflare WAF/rate limit、第三方渗透测试、真实用户 CWV p75、NVDA/VoiceOver 和 WCAG-EM 仍为 `manual-check`。上述门禁不是安全认证、WCAG 认证或全球排名。

## 6. 发布与回滚

- 发布只通过候选内的 `tools/deploy_staged.sh` 完成，没有在 VPS 生产目录手工编辑。
- v1.13 候选压缩包 SHA-256 为 `3cea7e4feeb23150e2f9c87fa2e345c360aa03cd34776aede185186afae905ad`；仅上传到 `_staging`，再由候选内 `deploy_staged.sh` 原子发布。
- 当前 `.prev` 已指向上表的 v1.13 发布前快照；该快照保存已验证的 v1.12 生产树和 manifest。
- 后续若必须改站，仍须先备份、生成新 release/cache、重跑自动门禁和多视口浏览器验收，再使用全新 immutable candidate 原子发布。不得原地覆盖生产目录。

## 7. 不可突破的边界

1. 不访问或修改前三部分设备和冻结代码，不修改 PC Wi-Fi、TUN、VPN、代理、路由或 ARP。
2. 公网保持只读，不公开私钥、密码、token、内网地址、设备命令、串口/GPIO/PWM 或未脱敏数据。
3. 不修改 Cloudflare、Caddy、SSO 或 DNS 来绕过登录；未登录 401/403 是预期行为。
4. 不把 `shadow`、`mirror/replay`、BPU 辅助或人工确认升级描述成未发生的自主物理闭环。
5. 不宣称“绝对安全”“已通过渗透认证”“全球第一”或其他没有第三方证据的结论。
