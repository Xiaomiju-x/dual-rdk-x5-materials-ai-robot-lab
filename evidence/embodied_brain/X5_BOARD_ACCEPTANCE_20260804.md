# X5-TriBEV-Flow v5r1 板端验收回执

状态：`X5_BPU_FIXED_CONTRACT_ACCEPTED_SHADOW_ONLY`

日期：2026-08-04

## 结论

- 候选以内容寻址目录隔离部署到 `embodied-x5`，没有覆盖 `~/tools`、`~/ros2_ws`、F407、Nav2 或复赛入口。
- `TinyOccFlow v5r1` 与 `CamSemLite` 均在 RDK X5 Bayes-e BPU 上由 `hbm_runtime.HB_HBMRuntime` 实际执行。
- 200 次固定输入测试通过延迟门；30 次新进程交替加载/退出无失败，CMA 前后漂移为 0 KiB。
- PC FP32 与 X5 INT8 的六个输出固定张量余弦相似度最低为 `0.995193`。
- 候选未注册服务、未创建 `current` 链接、未打开 F407 串口、未发布 `/cmd_vel` 或 TF，也未产生物理运动。
- 冻结入口仍为 `bash ~/tools/finals_lift_nav_demo.sh`，本候选不能替代或阻断它。

## 内容身份

- 最终 deploy ZIP：`bpu/packages/x5-tribev-flow-deploy-29998bfe8a183d99463b9ccb.zip`
- ZIP SHA-256：`803129ec2c8f9742463a7f571d52d405a0587a563ae0d661d71f8590cce9ae84`
- 内容 SHA-256：`29998bfe8a183d99463b9ccb4b721f1c2526cd479b6542fb6af38b26b694826d`
- 板端目录：`~/xrd_candidates/finals_successor/29998bfe8a183d99463b9ccb4b721f1c2526cd479b6542fb6af38b26b694826d/`
- TinyOccFlow SHA-256：`90e01859991c2eabaf71147de299123c656569ef1115df049d03e25f1471fdf9`
- CamSemLite SHA-256：`cb582808a90ae93c46dbffdce2ddd676ceacfeab3d865a67f755c322968c6f7c`

## 真机指标

| 模型 | 固定输入次数 | p50 | p95 | p99 | 最大值 |
|---|---:|---:|---:|---:|---:|
| TinyOccFlow v5r1 | 200 | 4.261 ms | 5.193 ms | 7.128 ms | 8.710 ms |
| CamSemLite | 200 | 2.346 ms | 6.417 ms | 8.857 ms | 11.274 ms |

两个模型顺序加载后的候选进程 PSS 为 `57,592 KiB`；`CmaFree` 保持 `317,220 KiB`。板端温度约为
DDR `48.7 C`、CPU `48.5 C`。这些是固定合成/标定张量的单进程实测，不是持续导航负载或 TOPS 利用率。

`hrt_model_exec perf` 的独立随机输入吞吐测试还记录了：TinyOccFlow 平均 `2.444 ms`，CamSemLite 平均
`0.851 ms`。该工具吞吐数字与 Python 端到端调用延迟口径不同，不应混为同一指标。

## INT8 固定任务差分

| 输出 | PC FP32 / X5 INT8 余弦相似度 |
|---|---:|
| Cam quality logits | 0.999820 |
| Cam semantic logits | 0.995193 |
| Dynamic / uncertainty | 0.999763 |
| Flow | 0.997647 |
| Future occupancy | 0.999616 |
| Trajectory logits | 0.998587 |

边界：这些结果仅证明固定张量合同，不证明真实相机语义准确率、真实导航成功率或自动驾驶控制能力。

## 恢复与非干扰

- 30 次 TinyOccFlow/CamSemLite 交替单帧加载退出，失败 `0`。
- 每次 `CmaFree` 均为 `317,192 KiB -> 317,192 KiB`。
- 最终板端 `successor current` 与 `cortex current` 均为空，BPU 利用率为 `0`。
- `embodied_brain.service` 保持 active；冻结脚本 SHA-256 为
  `3598de623e84e69781aeb75856ca1fd280aada9ec25562169557281cf487145e`。

## 尚未晋级的内容

- 本轮只完成 BPU 固定输入和资源恢复验收，没有启动候选 ROS shadow 节点。
- 只读观测时 `/scan`、深度和 Lab-FSD 诊断有数据，但 `/imu`、`/odom` 未形成可用消息；因此不能声称完整实时融合已经验收。
- `finals_cortex` 仍是 PC 基础设施与离线/回放算法，缺少候选专属 ROS 图、完整实时传感器会话和可读 ION 分配证据，状态维持 `MONITOR_OFFLINE`。

## 权威证据

- `evidence/x5_board_20260804/fixed_probe_calibration/x5_hbm_runtime_20260804T091505Z/x5_probe.json`
- `evidence/x5_board_20260804/fixed_probe_calibration/int8_differential_hbm_runtime.json`
- `evidence/x5_board_20260804/x5_cycles_20260804T085501Z/RESULT.txt`
- `evidence/x5_board_20260804/x5_models_20260804T085237Z/`
- `bpu/compatibility/compatibility_record.json`
