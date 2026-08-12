# 复赛第 3 部分：双机械臂现场交接

日期：2026-07-20
状态：第 1、2、3 部分均已完成真机复验并冻结；第 3 部分于 2026-07-20 完成一键整链验收。
最终收口保留已真机通过的 PC 编排作为降级入口，并增加具身脑 X5 的隔离一键入口；同时修正 arm02 迟到有效回读判定、arm01 袋爪有界保持和两路视觉展示时序。没有修改任何标定点、运动速度、G23 端点值、研磨范围、双臂并发时序、AI 脑生产代码、`embodied_brain/` 代码或现场常驻服务。

## 1. 交接结论

- 第 1 部分具身脑已真机整链通过，冻结入口仍为 `bash ~/tools/finals_lift_nav_demo.sh`。
- 第 2 部分 AI 脑已由用户在平板完成现场流程彩排，现有 Dashboard、XRD 视觉链和合成预测保持原样。
- 第 3 部分使用已经真机验收的双臂答辩 v3：arm01 投袋后竖直离开研磨皿，随后 arm01 返回 START 与 arm02 四周期研磨并发执行，最终两臂都回 START。
- 第 3 部分首选从具身脑 X5 `.85` 的桌面图标“复赛双机械臂演示”启动，终端等价命令为 `bash ~/tools/finals_part3_demo.sh`；PC 桌面 `C:\Users\YOUR_USER\Desktop\run_finals_part3.cmd` 原样保留为降级入口。两个入口都先执行 arm01 单臂视觉冗余，再无缝衔接冻结双臂 v3，最后完成 arm02 皿内视觉和 X5 BPU 辅助证据。
- 两个入口最终调用相同的两台 Pi 动作脚本、相同点位和速度参数，机械臂关节运动速度一致；SSH、抓帧、传图和推理造成的整场墙钟时间允许有小幅差异。
- X5 显示顺序已冻结为：AprilTag `id=2` 标注图全屏 5 秒 -> 空研磨皿等待图 -> 投袋后、右臂研磨期间立即出现的 `BAG_PRESENT` 标注图；图像推理和展示没有机械臂运动权限。
- AI X5 在第 3 部分只提供既有轻量视觉推理算力。第 1、2 部分已结束，因此没有计划内的前台推理竞争；无需关闭、重启或重配任何 AI 脑服务。
- X5-RB-VoE 只讲技术栈，不参与现场控制链；不得运行 runner/collector/probe，不得注册或启用相关服务。

## 2. 摄像头与算力边界

| 能力 | 物理摄像头 | 推理位置 | 权威判定 |
|---|---|---|---|
| AI 脑第 2 部分 4K 识图 | AI X5 的 IMX415 | AI X5 | 第 2 部分现有 XRD 视觉链 |
| arm01 视觉冗余 | arm01 腕部 USB 相机 | AI X5 CPU/OpenCV | `DICT_APRILTAG_36h11 id=2` 精确命中 |
| arm02 皿内状态 | arm02 自有 `/dev/video0` 工位相机 | AI X5 CPU/OpenCV + Bayes-e BPU 辅助 | CPU/OpenCV 空皿/有袋门控 |

三台摄像头是不同物理设备。双臂视觉不会打开或占用 AI 脑 IMX415。AI X5 只接收双臂相机帧并执行推理，因此第 2、3 部分不存在相机冲突。

Bayes-e BPU 上的 `mobilenetv2_224x224_nv12.bin` 只作为通用语义辅助和 BPU 真机执行证据，不得表述为袋子二分类器。袋子状态仍以 CPU/OpenCV 门控为权威。

arm02 工位相机的临时采集必须在 v3 物理运动前正常停止并释放 arm02 自己的 `/dev/video0`。这是第 3 部分内部的 owner 门禁，与 AI IMX415 无关。

## 3. 网络合同

- 设备间固定入口保持：AI X5 `192.0.2.103`、arm01 `192.0.2.64`、arm02 `192.0.2.136`。
- 具身脑 X5 `192.0.2.85` 只承担第 3 部分总编排；视觉仍在 AI X5 `.103` 推理，机械动作仍在两台 Pi 本地执行。
- 第 2 部分平板使用的当次 K70 DHCP `10.*` 地址只用于浏览器进入 AI Dashboard，不得写入双臂脚本、SSH 合同或证据。
- 不扫描网络，不运行旧 discovery/route-fix 工具，不修改 PC Wi-Fi、TUN、VPN、代理、路由或 ARP。
- PC 直达机械臂固定 overlay 异常时，只按根 `AGENTS.md` 的已验证合同使用 AI X5 `.103` 固定跳板；不得回退历史 K70 `10.*` 设备地址。

## 4. 已冻结事实

- v3 真机 predecessor 编排器 SHA-256：`2c40e81f5fe47ca0036f2ab53ce646ab23f59d2c88223c256862cd25b4202b62`。
- PC 一键入口：`workstation/dual_arm/run_finals_part3.ps1`，SHA-256=`44625a05bea759fef078385f5b394137107c322a279b17392b8e63fadb1cdde6`；无参数默认为 `PlanOnly`。PC 桌面保底 `run_finals_part3.cmd` SHA-256=`ee618293e08479cc6f88e177a53fc784b270a35c15080ee17376ed82a613141a`，保持不变。
- 跨平台一键编排器：`workstation/dual_arm/finals_part3_orchestrator.py`，SHA-256=`d78c03c1c309e4c475cdb0581989e18dbae37910e022819bdb6d5efd6cc92a2d`。
- Windows v3 wrapper：`workstation/dual_arm/run_dual_arm_bag_grind.ps1`，SHA-256=`1e7b1fd9a61c3c612296141300df93a021e52c918a66ae4f4927be6abdaf7a39`。
- Linux v3 wrapper：`workstation/dual_arm/run_dual_arm_bag_grind_edge.py`，SHA-256=`95f30e2a19e18ffaea4b9b33956035ed37f654b04c4b1eba6e7627e0722b68e9`。
- 具身脑启动 shell：`workstation/dual_arm/run_finals_part3_edge.sh`，SHA-256=`0cee0d888a80cc73f33ca59276defee80a1ebb0bdb7b10a5ee459e535d5dfcc0`；桌面文件 SHA-256=`593db7498cfeec1601b3e41df7142ffa7cd7b41ec8319d30e9e599fe4688c83e`；SSH config SHA-256=`1a399f551c91c1ae2df60a1ccedc085a241086a14efcbc33b95e4e72627b0b72`。
- arm01 动作脚本本地/远端 SHA-256：`36d15a3181b2143b3b19a71f6e150a080eaf9094958848288537ae5b74a76176`；START 以 `G23=17` 张开，PICK 只下发一次 `G23=9` 并持续保持到投袋点，随后立即以 `G23=17` 松袋并释放 PWM；不改变 `17/9` 端点值。
- arm02 动作脚本本地/远端 SHA-256：`c070db7c87455723dd43b3d4727f7968343fa0200483c68b68cd9e4ccb518619`；只修正关节回读可能跨过旧 deadline 后被误判超时的问题，点位、速度、J6 范围和四周期均未改。
- arm01 当前视觉标记是 `DICT_APRILTAG_36h11 id=2`。所有仍绑定旧重相机 `id=8` 的历史入口不得用于当前答辩。
- arm01 `xrd-workcockpit.service` 必须保持 `disabled/inactive`；它会占用 `/dev/ttyAMA0`。
- arm02 不得启动旧 `workstation/web/arm02_service.py`；它会打开机器人串口并调用 `power_on()`。仅可在视觉阶段使用纯相机链路。
- 双臂 v3 的点位、速度、G23 `17/9` 端点、四周期研磨范围和并发时序均已验收，不再重教点；袋爪 PWM 只在 PICK 到投袋点之间有界保持，投袋后不得继续闭合保持。

## 5. 第 3 部分现场顺序

1. 读取根 `AGENTS.md` 和本文件，确认第 1、2 部分冻结。
2. 只读核验 AI X5、arm01、arm02 的固定身份、主机键、服务状态、串口 owner、相机 owner 和冻结文件哈希。
3. 复用既有 arm01 腕部相机链，确认 X5 CPU/OpenCV 精确识别 `DICT_APRILTAG_36h11 id=2`；不得使用宽松 dark-square fallback 作为通过依据。
4. 复用 arm02 纯工位相机链，确认 AI X5 可获得 CPU/OpenCV 皿内状态；可显示 Bayes-e BPU 辅助推理，但不得改变权威判定口径。
5. 正常结束双臂视觉采集，确认 arm02 `/dev/video0` 无 owner、两台 Pi `/dev/ttyAMA0` 无冲突 owner。
6. 在具身脑 X5 运行第 3 部分整链无运动核验：

   ```bash
   bash ~/tools/finals_part3_demo.sh --validate-only
   ```

7. 只有用户完成现场清场、工具/袋/研磨皿检查并明确授权物理运动后，才执行：

   ```bash
   bash ~/tools/finals_part3_demo.sh
   ```

8. 具身入口不可用时，才在 PC 双击 `C:\Users\YOUR_USER\Desktop\run_finals_part3.cmd`；不要同时启动两个入口。
9. 结束后确认整链输出 `CLOSED_LOOP_DONE`、arm01 与 arm02 都回 START、CPU 权威门为 `BAG_PRESENT`、BPU forward 已执行且 X5 最终标注图保持打开。

## 6. 成功判据与停止条件

成功判据：

- 三台固定身份、ED25519 主机键和冻结脚本哈希全部通过。
- arm01 当前 `id=2` 视觉严格命中；arm02 空皿/有袋结果来源清楚，AI X5 推理可用。
- 视觉阶段没有向机器人 SDK、串口、GPIO 或 G23 发送动作。
- 进入运动前，相机与串口 owner 满足冻结编排器的 fail-closed 门禁。
- v3 发出 `completed_dish_clear_top` 后，皿内抓帧和 CPU 判定立即与双臂研磨并发，最终 `CLOSED_LOOP_DONE` 且两臂回 START。
- X5 显示按 AprilTag -> 空皿等待 -> `BAG_PRESENT` 顺序切换；CPU/OpenCV 是袋状态权威，BPU 仅作辅助证据。

立即停止且不得绕过：

- 身份、主机键、哈希、服务状态或 owner 任一不符。
- arm01 只出现旧 `id=8`、宽松 fallback 或错误 AprilTag 字典。
- arm02 相机进程未退出，或任一 `/dev/ttyAMA0` 被未知进程占用。
- 工具、袋、研磨皿、机械臂 START 姿态或现场净空与验收状态不同。
- 任何 agent 尝试启用 RB-VoE、恢复厂家 `automatic-ager`、启动旧 arm02 Web 服务或修改 PC 网络。

## 7. 权威证据

- `workstation/dual_arm/evidence/commissioning_20260718/dual_arm_answer_profile_v3_overlap_20260718.json`
- `workstation/dual_arm/evidence/commissioning_20260718/dual_arm_answer_profile_v3_overlap_live_20260718.json`
- `workstation/dual_arm/evidence/vision_20260718/single_arm_redundancy_live_20260718_150603/RUN_RESULT.md`
- `workstation/dual_arm/evidence/vision_20260718/overhead_bag_state_live_20260718_154437/RUN_RESULT.md`
- `workstation/dual_arm/evidence/finals_part3_execute_20260720_052630_4956/result.json`（当前最终 PC 真机整链，`CLOSED_LOOP_DONE`）
- `workstation/dual_arm/evidence/finals_part3_execute_20260720_052630_4956/apriltag_live/exact_gate_summary.json`（20 px 白边只作防贴边预处理，严格 `id=2` 4/4，无 fallback）
- `workstation/dual_arm/evidence/finals_part3_execute_20260720_052630_4956/overhead_live/cpu_result.json`（投袋后并行判定 `BAG_PRESENT`，6/6，支持袋色或相对空皿基线变化）
- `workstation/dual_arm/evidence/finals_part3_execute_20260720_052630_4956/overhead_live/bpu_result.json`（BPU forward 13 次）
- `workstation/dual_arm/evidence/edge_launcher_setup_20260720_053900/`（具身脑隔离入口部署与完整无动作 `ValidateOnly` 证据；尚未单独执行物理运动）
