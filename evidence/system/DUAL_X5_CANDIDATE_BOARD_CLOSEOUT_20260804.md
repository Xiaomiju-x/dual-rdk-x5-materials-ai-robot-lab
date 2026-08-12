# 双 X5 决赛候选上板收口

日期：2026-08-04

## 总结

本轮完成 AI 脑模型库与具身脑 v5r1 的隔离上板验收。全过程未改变复赛冻结演示入口、F407 固件、
Dashboard、五端口、既有 CPU/BPU 模型槽、相机所有权、Wi-Fi、VPN/TUN、路由或 ARP。

| 模块 | 最终状态 | 真机结论 |
|---|---|---|
| AI 脑 X5-ICMat Foundry | `COMPLETE_WITH_REJECTIONS_AND_EXPERIMENTALS` | 新增 38 个候选：31 validated / 4 experimental / 3 rejected |
| AI 脑 BPU | `24/24 ACTUAL_X5_BPU_EXECUTED` | 含三套分段 LLM 的 6 个 bin；不是同时常驻 |
| 具身脑 v5r1 | `X5_BPU_FIXED_CONTRACT_ACCEPTED_SHADOW_ONLY` | TinyOccFlow/CamSemLite 实际 BPU、200 次延迟、30 次恢复通过 |
| 具身脑 Cortex | `MONITOR_OFFLINE` | PC 基础通过；真实候选 ROS 图、完整传感器会话和 ION 证据仍缺 |
| 复赛生产链 | `PRESERVED` | 两台板均无候选 current 链接或候选常驻进程 |

## AI 脑结果

- 38 个新增候选全部有板端最终状态；31 个通过固定任务合同。
- 24 个 BPU-primary 均在 Bayes-e actual backend 执行；14 个 CPU-primary 中 11 个实际运行。
- `F-KNW-01/02/04` 因 staging 缺 `config.json`/`tokenizer.json` 被拒绝。
- `F-KNW-03` 的 INT8 固定标量门未通过，保留实验态。
- `F-LLM-03/04/05` 的 6 个分段 bin 均实际执行且 part1/part2 内容绑定正确，但固定 next-token 不一致，保留实验态，不宣称通用生成。
- 最终生产非干扰 `33/33 PASS`；板端 FleetAudit `59/59 PASS`，退出后为 `DEPLOYED_OFF`。

权威回执：

- `icmat_foundry/finals_50model/evidence/x5_board_20260804/final_acceptance_v1/X5_BOARD_PHASE_FINAL_RECEIPT_20260804.md`
- `icmat_foundry/finals_50model/evidence/x5_board_20260804/final_acceptance_v1/x5_board_phase_acceptance.v1.json`
- FleetAudit SHA-256：`1a757d570141bb75bc949fd87b6637cc9d5ea3d2df5ef2155d31ec080185d2a7`

## 具身脑结果

- v5r1 使用板端 `hbm_runtime.HB_HBMRuntime`，修复了 `pyeasy_dnn` 对 featuremap 输入/输出布局处理不可靠的问题。
- TinyOccFlow p95 `5.193 ms`，CamSemLite p95 `6.417 ms`，均低于 10 ms 设计门。
- 六个固定输出的 PC FP32 / X5 INT8 最低余弦相似度 `0.995193`。
- 30 次加载/退出失败 `0`，CMA 漂移 `0 KiB`。
- 该结果是 shadow 固定任务验收，不是定点导航或真实相机准确率验收，不获得运动控制权。

权威回执：

- `embodied_brain/finals_successor/docs/X5_BOARD_ACCEPTANCE_20260804.md`

## 当前板端资源

- AI 脑：`MemAvailable 3,986,168 KiB`，`CmaFree 26,292 KiB`，BPU 利用率 `0%`，BPU 温度约 `45.2 C`。
  CMA 较低来自既有生产 BPU 槽常驻，不是候选残留；五端口和四个生产 llama 进程正常。
- 具身脑：`MemAvailable 4,028,788 KiB`，`CmaFree 317,224 KiB`，BPU 利用率 `0%`，BPU 温度约 `46.2 C`。
- 两台板桌面均未关闭。

## 后续入口

决赛现场继续只使用冻结演示：

```bash
bash ~/tools/finals_lift_nav_demo.sh
```

决赛候选默认关闭。若后续要做真实 shadow 会话，必须单独确认零运动/物理安全，先恢复 `/imu` 与 `/odom`
数据，再以手动候选命名空间启动；失败只允许回到 `MONITOR_OFFLINE`，不得阻断冻结演示。
