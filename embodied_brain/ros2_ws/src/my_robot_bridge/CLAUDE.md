# my_robot_bridge

> AI 脑 (198.51.100.103) ↔ 具身脑 (198.51.100.85) 跨网通信桥. **Phase 4 完工 2026-04-26 (X5 实测 mock AI 脑全链路通, 13s 7-stage 任务)**. Round 4 BPU Sprint 不影响这层 — bridge 只搬运 task / telemetry / report 不碰 BPU.

## 是什么

按 ADR-EB-7, 两脑跨网双路径冗余:
1. **HTTP** (主路径): 周期拉 AI 脑 dashboard:8888 task → 转 ROS2 DispatchTask action; 上报 telemetry / 报警 / Qwen-VL 复核
2. **ROS2 DDS** (副路径): cyclonedds peers 静态 IP, 待真跨网联调启用

## 节点 / 文件

```
my_robot_bridge/
├── package.xml + setup.py
├── my_robot_bridge/
│   ├── __init__.py
│   └── ai_brain_bridge.py      ← Phase 4 主桥 (✅ 实测)
├── launch/
│   └── bridge.launch.py        ← 一键拉 dispatch_server + ai_brain_bridge + telemetry
├── config/
│   └── cyclonedds_peers.xml    ← 跨网 DDS 静态 peers (Phase 9 启用)
└── scripts/
    ├── mock_ai_brain.py        ← 测试用假 AI 脑 (Python http.server)
    └── test_bridge_integration.sh  ← 集成测试脚本
```

## ai_brain_bridge 行为

### 上行 (具身脑 → AI 脑)

| 触发 | 端点 | payload |
|---|---|---|
| 周期 5s (param: report_interval_s) | POST `/api/embodied/report` | telemetry: cpu/ram/bpu/位姿/SLAM/Nav2 状态 |
| /alarm 来一条 | POST `/api/embodied/alarm` | source/level/title (snapshot 不发, 太大) |
| /furnace_reading 含 needs_vl_recheck | POST `/api/qwen_vl_check` | snapshot_b64 + OCR 给的初值 |
| dispatch_server 完成 task | POST `/api/embodied/report` | task_id + success + message + elapsed |

### 下行 (AI 脑 → 具身脑)

| 周期 | 拉 | 处理 |
|---|---|---|
| 2s (param: poll_interval_s) | GET `/api/embodied/dispatch_queue` | 解析 task list → DispatchTask action goal → dispatch_server |
| (Qwen-VL 同步返回) | POST 应答 | TODO Phase 5: 把复核结果回灌 /furnace_reading |

## AI 脑端期望端点 (按 ADR 留口子)

AI 脑收尾冻结, 这些端点等以后一次性加 (代码桩可用 mock_ai_brain.py 跑通逻辑):

| Method | Path | 用途 |
|---|---|---|
| GET | `/api/embodied/dispatch_queue` | 返 [task1, task2, ...] (待执行), 已分发的不重复返 |
| POST | `/api/embodied/report` | 收 telemetry / task 结果, 写入 dashboard |
| POST | `/api/embodied/alarm` | 收报警, 写入 dashboard 警报中心 |
| POST | `/api/qwen_vl_check` | snapshot → Qwen-VL 复核 PV/SV/MV |
| POST | `/api/say` | TTS, 走 M260C 麦克阵列 |

## X5 实测结果 (2026-04-26)

✅ **集成测试** 通过 (用 mock_ai_brain.py 在 X5 本地 :8889):
- ai_brain_bridge 启动后 ~2s 拉到 mock 的 fetch_sample task
- DispatchTask action goal 投递成功
- dispatch_server 7 stages stub 模拟执行 13s 完成
- bridge 收到 task DONE 回报后 POST /report 上报最终结果
- telemetry_publisher 同步 3s 周期推 cpu/ram/slam/nav 状态

**通信日志**:
```
GET /api/embodied/dispatch_queue       (poll 2s, ~10 次/30s)
POST /api/embodied/report              (telemetry + task result, 共 ~7 条)
```

## 部署 cyclonedds peers (Phase 9 联调时启用)

车载脑 ~/.bashrc:
```bash
export CYCLONEDDS_URI=file:///opt/ros/humble/share/my_robot_bridge/config/cyclonedds_peers.xml
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```
AI 脑端 (198.51.100.103) 同款配置, peer 列表交换 IP. 现在还没启用 (HTTP 路径足够 Phase 4 验证).

## 已知坑

- `python3-requests` 出厂没装 (Ubuntu 22.04 server 自带 minimal Python). `apt install python3-requests` 一行解决. install_third_party.sh 已加.
- `requests.Session()` 复用连接, 别每次 new (TIME_WAIT 累积爆 socket)
- AI 脑收到具身脑请求后处理 ≥ 200ms (Flask + 9 LLM 排队), timeout 设 ≥ 5s
- HTTP 桥用 `threading.Thread(daemon=True)` fire-and-forget, 别在 ROS2 callback 里 block (会卡 spin)
- 跨网带宽低 (手机热点 ~10Mbps), 别在 DDS 上传图像 / 大点云
- mock_ai_brain.py `dispatch_queue` 端点要做"已发过的不重复返", 否则 bridge 会无限重发同一 task

## 下一步 (Phase 5)

- 写 command_interpreter 接口 (rule / local LLM / remote AI 脑 三档抽象层), 让 bridge 可以直接接 "中文人话指令"
- AI 脑收尾期一次性加上面 5 个端点
- 真跨网时启 cyclonedds peers, 测对比 HTTP vs DDS 延迟
