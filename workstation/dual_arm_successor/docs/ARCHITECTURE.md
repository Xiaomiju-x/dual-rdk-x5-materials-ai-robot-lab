# DualArm-ShadowVLA / X5-BiSkill Shadow 架构合同

状态：候选设计
模式：`POST_RUN_REPLAY`
日期：2026-07-30

## 1. 目标

`DualArm-ShadowVLA` 是决赛双机械臂的学习算法候选总称；部署到 RDK X5
的轻量学生侧称为 `X5-BiSkill Shadow`。它读取复赛冻结 v3 已经完成的
双臂演示副本，离线预测当前阶段、后续动作块、双臂同步程度、任务结果和
OOD 状态，用于算法证据与答辩可视化。

候选不替换、不包装、不门控复赛冻结 v3。冻结 v3 始终是唯一运动权威。

```text
Frozen v3 (sole motion authority)
  -> completes run and writes immutable evidence
  -> CLOSED_LOOP_DONE
  -> read-only evidence copy
  -> episode adapter
  -> teacher / student shadow inference
  -> prediction + model receipt
  -> evidence UI

Shadow output -X-> robot SDK / serial / GPIO / G23 / motion process
```

`-X->` 表示物理上和软件合同上均禁止的路径。

## 2. 不可变权限

所有 episode、prediction 和 model receipt 必须同时满足：

```json
{
  "motion_authority": false,
  "execution_allowed": false,
  "actuator_commands_issued": 0
}
```

任何一个字段缺失或值不同，产物即为无效候选证据。候选进程不得：

- 导入或调用机器人 SDK、`pymycobot`、串口、GPIO、PWM 或 G23 控制接口。
- 打开 arm01/arm02 的运动串口或相机设备节点。
- 发布动作、服务、ROS command、TF 或机械臂控制消息。
- 修改冻结 v3 的代码、参数、点位、速度、G23、视觉门、并发时序或入口。
- 在冻结演示执行期间持有相机、串口、日志或结果文件的写锁。
- 因影子算法失败而停止、延迟、重启或改变冻结演示。

候选失败的唯一外部效果是生成 `SHADOW_OFFLINE` 或无影子结果。冻结 v3
继续独立完成动作。

## 3. 首阶段唯一运行模式

首阶段只允许 `POST_RUN_REPLAY`：

1. 冻结 v3 独立运行。
2. 冻结 v3 输出 `CLOSED_LOOP_DONE`，两臂回 START。
3. 复制所需证据到候选输入区；候选只读取副本。
4. 对副本执行 episode 适配、VLA/ACT 推理、世界模型推理和 X5 学生推理。
5. 写入候选自己的 prediction、receipt 和可视化证据。

以下模式不在本合同授权范围：

- `LIVE_SHADOW`
- `ASSIST`
- `ENFORCE`
- `CLOSED_LOOP_CONTROL`
- `ONLINE_LEARNING`

未来即使新增模式，也必须另立合同、重新验收并取得用户明确授权；不得
修改本合同来追认既有运行。

## 4. 数据平面

### 4.1 双臂 13D

每个状态和动作向量固定为 13 维，顺序不可改变：

| 索引 | 字段 | 单位 |
|---:|---|---|
| 0..5 | `arm01_j1` ... `arm01_j6` | degree |
| 6 | `arm01_g23` | raw PWM endpoint/value |
| 7..12 | `arm02_j1` ... `arm02_j6` | degree |

向量简称：

```text
[arm01_j1, arm01_j2, arm01_j3, arm01_j4, arm01_j5, arm01_j6,
 arm01_g23,
 arm02_j1, arm02_j2, arm02_j3, arm02_j4, arm02_j5, arm02_j6]
```

13D 只是离线记录和预测表示。预测值不得转换为任何执行命令。

### 4.2 两路相机

固定相机逻辑身份：

- `arm01_wrist`：arm01 腕部相机；对应单臂冗余 AprilTag 观察。
- `arm02_overhead`：arm02 工位相机；对应空皿/有袋状态和双臂研磨观察。

候选读取的是冻结结果引用或复制帧，不直接打开 `/dev/video*`。每帧必须带
时间戳、序号、SHA-256、尺寸和来源路径。相机帧与 13D 状态通过单调时间戳
对齐。

### 4.3 阶段

标准阶段枚举：

1. `PRE_RUN`
2. `SINGLE_ARM_OBSERVE_OUTBOUND`
3. `APRILTAG_OBSERVE`
4. `SINGLE_ARM_RETURN_START`
5. `DUAL_ARM_PICK_APPROACH`
6. `DUAL_ARM_PICK_GRASP`
7. `DUAL_ARM_BAG_TRANSPORT`
8. `DUAL_ARM_DISH_DROP`
9. `ARM01_CLEAR_TOP`
10. `DUAL_ARM_GRIND_CONCURRENT`
11. `DUAL_ARM_RETURN_START`
12. `COMPLETED`
13. `UNKNOWN`

阶段标签必须来源于冻结日志、人工审计标签或明确的派生规则，并记录
`label_source`。模型预测不得覆写事实标签。

### 4.4 动作块

动作块是长度为 `H` 的 13D 预测序列：

```text
action_chunk.shape = [H, 13]
```

每步包含相对时间、13D 建议动作和不确定性。动作块只用于：

- 与冻结 v3 已执行轨迹做离线误差比较。
- 展示 ACT/VLA 的时序预测能力。
- 蒸馏 X5 轻量学生模型。

动作块必须标注 `ADVISORY_ONLY`，不得有 command topic、device path、
robot address 或执行回调。

## 5. 模型分层

### 5.1 离线教师层

教师候选可包含：

- SmolVLA：主要轻量 VLA 候选。
- Compact ACT / Tiny-ACT：动作分块基线。
- OpenVLA-OFT：重型双臂教师和方法对标。
- Xiaomi-Robotics-0：开放 VLA 架构与后训练参考。
- Xiaomi-Robotics-1：方法与规模化数据参考，不假定代码或权重可用。
- Xiaomi-Robotics-U0：世界模型和数据增强参考，不作为动作策略。

教师在 GPU 工作站或租用 GPU 上离线运行，不在 X5 上声明完整模型部署。

### 5.2 X5 学生层

`X5-BiSkill Shadow` 是可量化的轻量模型，建议由视觉骨干与时序头组成：

- 输入：两路相机特征、13D 状态窗口、任务文本或固定任务 token。
- 输出：阶段分布、下一技能、短动作块、双臂同步分数、成功概率、
  OOD/不确定性和世界模型结果。
- BPU：执行已转换的轻量视觉/时序前向。
- CPU：证据读取、时间对齐、规则一致性、OOD/Conformal 计算和落盘。

X5 结果仍为影子建议，`motion_authority=false`。

## 6. 世界模型结果

世界模型只预测，不生成控制。每个预测时域至少包含：

- `horizon_ms`
- `predicted_stage`
- `bag_state`: `EMPTY` / `PRESENT` / `UNKNOWN`
- `dual_arm_sync_score`
- `success_probability`
- `uncertainty`

可选预测时域建议为 `+400 ms`、`+800 ms`、`+1200 ms`。若输入 OOD、
时间不同步或相机缺帧，必须输出 `UNKNOWN` 或降低置信度，不得补造确定结果。

## 7. OOD 与降级

OOD 合同必须记录：

- `score`
- `threshold`
- `is_ood`
- `method`
- `reason_codes`

典型原因包括 `CAMERA_MISSING`、`STATE_MISSING`、`CLOCK_SKEW`、
`VISUAL_SHIFT`、`STATE_RANGE`、`MODEL_UNCERTAIN` 和 `UNKNOWN_TASK`。

出现 OOD 时：

- 允许继续生成带明确不确定性的离线诊断。
- 禁止把结果表述为可靠动作建议。
- 不影响冻结 v3 的成功判定或动作。

## 8. 产物合同

三类权威候选产物：

- `shadow_episode_v1.schema.json`：只读回放 episode。
- `shadow_prediction_v1.schema.json`：模型影子预测。
- `model_receipt_v1.schema.json`：模型来源、训练、转换和验收回执。

所有产物必须具备内容哈希、创建时间和不可变权限三元组。模型回执必须
区分 PC、GPU、X5 编译和 X5 真机状态，禁止把编译器估算写成板端实测。

## 9. 非干扰验收

候选进入任何现场展示前至少满足：

1. 冻结 v3 文件与入口哈希未变化。
2. 候选源码静态扫描无机器人 SDK、串口、GPIO、PWM、G23 写入。
3. 只接受 `CLOSED_LOOP_DONE` 的证据副本。
4. 候选关闭、崩溃或被强制结束时，冻结 v3 行为不变。
5. 影子开/关配对回放结果可复现。
6. 所有 Schema 验证通过。
7. UI 明示 `MOTION AUTHORITY = FROZEN V3`、
   `SHADOW ACTUATOR COMMANDS = 0`。

本架构不授权真机运动测试。
