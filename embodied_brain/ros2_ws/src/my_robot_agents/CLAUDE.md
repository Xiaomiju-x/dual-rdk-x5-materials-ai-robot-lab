# my_robot_agents

> Python 任务 Agent 集合. **Phase 3 完工 2026-04-26 + Round 4 Phase 4 完工 2026-04-30 (X5 全部实测通过)**

## 是什么

ROS2 + rclpy 写的 Python 节点, 充当具身脑的"小脑". 跟 my_robot_drivers (硬件层) 和 my_robot_perception (BPU 感知层) 解耦, 只通过 ROS2 topic/service/action 通信.

## 节点清单

| Agent | 入口 | Phase | 状态 |
|---|---|---|---|
| **fake_odom** | `fake_odom_node:main` | 2 | ✅ X5 实测 50Hz, 给 SLAM 测试用 |
| **furnace_ocr** | `furnace_ocr_node:main` | 3 | ✅ X5 实测, OCR 1350°C 准 |
| **furnace_monitor** | `furnace_monitor_agent:main` | 3 | ✅ X5 实测, I1 报警链路通 |
| **alert_dispatcher** | `alert_dispatcher:main` | 3 | ✅ X5 实测, log 通道 OK; TTS/Email/微信 待环境变量配 |
| **voice_input** | `voice_input_node:main` | Round 4 B1 | ✅ SenseVoice ASR INT8 RTF 0.44 |
| **voice_output** | `voice_output_node:main` | Round 4 B2 | ✅ Piper VITS 中文 + 化学式归一化字典 50 项 |
| **smolvlm** | `smolvlm_node:main` | Round 4 C1 | ✅ Day 8-11 hybrid 模式 33s/query 语义稳; full_bpu 编通但 INT8 PTQ 精度需 QAT |
| **vlm_voice_relay** | `vlm_voice_relay:main` | Round 4 C1 Day 12 | ✅ ASR 触发词 → /vlm_query → TTS, 9/9 关键词测试通, X5 端到端 27.5s |
| dispatch_server (扩) | (见 dispatch_server.py) | Round 4 C1 Day 13 | ✅ 加 'observe' task type: Nav2 + VlmQuery 集成, X5 实测 stage 流 + answer 回归 (48s) |
| command_interpreter (扩) | (见 command_interpreter.py) | Round 4 C1 Day 13 | ✅ Rule 模式加 8 条 observe 中文模式 (看一下/描述/N 号炉怎么样) |
| **bottle_ocr_bpu** | `bottle_ocr_bpu:main` | Round 4 A4 | ✅ X5 实测: BPU det 2.7MB/6ms + PaddleOCR 2.8.1 rec CPU, `bottle_ocr_bpu_node ready` |
| **rnnoise** | `rnnoise:main` | Round 4 E3 | ✅ X5 实测: backend=noisereduce spectral gating (librnnoise0 apt 源不可用时的正确降级) |
| **odas** | `odas:main` | Round 4 E1/E2 | ✅ X5 实测: delay-and-sum DOA lat≈4.4ms, `odas_node ready\|mode=delay-and-sum` |
| pickup_agent | `pickup_agent:main` | 6 | ⏸ Phase 6 硬件到位后 |
| telemetry_publisher | `telemetry_publisher:main` | 4 | ⏸ |
| command_interpreter | `command_interpreter:main` | 5 | ⏸ |

## 文件树

```
my_robot_agents/
├── package.xml + setup.py + setup.cfg
├── my_robot_agents/
│   ├── __init__.py
│   ├── fake_odom_node.py             ← Phase 2 (✅ 实测)
│   ├── furnace_ocr.py                ← Phase 3 (✅ OCR 核心, 5/5 单元测试过)
│   ├── furnace_ocr_node.py           ← Phase 3 (✅ ROS2 wrap)
│   ├── furnace_monitor_agent.py      ← Phase 3 (✅ 4 类报警逻辑)
│   ├── alert_dispatcher.py           ← Phase 3 (✅ TTS/Email/微信/Log 4 通道)
│   ├── bottle_ocr_bpu_node.py        ← Round 4 A4 (✅ BPU det 6ms + PaddleOCR rec; PaddleOCR==2.8.1 兼容)
│   ├── rnnoise_node.py               ← Round 4 E3 (✅ 3-backend: pip rnnoise→ctypes→noisereduce)
│   └── odas_node.py                  ← Round 4 E1/E2 (✅ delay-and-sum DOA + odaslive bridge)
├── launch/
│   └── furnace_monitor.launch.py     一键拉起 OCR + monitor + dispatcher
├── config/
│   └── furnace_roi.yaml              OCR ROI 标定 (默认 4K@1080p)
├── scripts/
│   └── test_furnace_integration.sh   集成测试 (用合成图)
└── test/
    └── test_furnace_ocr.py           OCR 单元测试 (5 case)
```

## Phase 3 详解

### 烧结炉 OCR Pipeline (按 ADR-EB-4 H5 方案)

```
图像源 (Phase 7 小米云台 / Phase 8 K3 备选)
   ↓ /pt_camera/image_raw
furnace_ocr_node.py
   ├─ 切 panel ROI (从 yaml 配的位置)
   ├─ 二值化 + 7-段位检测 (~5ms)
   ├─ Power Indicator HSV 红色检测
   ├─ 火焰 HSV 检测
   ├─ 烟雾 HSV + 帧间运动差分
   └─ 置信度 < 0.7 时 needs_vl_recheck=True
   ↓ /furnace_reading (1Hz)
furnace_monitor_agent.py
   ├─ I1: PV 超 [-10, 1600]°C → CRITICAL
   ├─ I2: |PV-SV| > 100°C 持 30s → WARNING/CRITICAL
   ├─ I4: Power 灯灭 > 5s → CRITICAL
   ├─ I6: fire/smoke flag → CRITICAL
   └─ 每个 source 60s cooldown 防轰炸
   ↓ /alarm (按需)
alert_dispatcher.py
   ├─ CHAN_LOG    → /tmp/embodied_brain_alarms.log
   ├─ CHAN_TTS    → POST AI 脑 dashboard:8888/api/say (Phase 4 接通) + 本地 espeak fallback
   ├─ CHAN_EMAIL  → smtplib SSL 直发张丹老师 (env: EB_SMTP_*)
   └─ CHAN_WECHAT → 企业微信 webhook (env: EB_WECHAT_WEBHOOK)
```

### 通道开关 + 等级映射

```python
# furnace_monitor_agent 内部:
LEVEL_INFO     → CHAN_LOG only
LEVEL_WARNING  → CHAN_TTS + CHAN_LOG
LEVEL_CRITICAL → CHAN_TTS + CHAN_EMAIL + CHAN_WECHAT + CHAN_LOG  (全开)
```

### 环境变量配置 (X5 ~/.bashrc)

```bash
# 邮件 (用 QQ 邮箱授权码)
export EB_SMTP_HOST=smtp.qq.com
export EB_SMTP_PORT=465
export EB_SMTP_USER=your@qq.com
export EB_SMTP_PASS=qq_app_pass        # QQ 邮箱授权码, 不是密码
export EB_SMTP_TO=zhang_dan@todo.todo  # 张丹老师邮箱

# 企业微信群机器人 webhook
export EB_WECHAT_WEBHOOK=https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxx

# AI 脑 (TTS 用)
export EB_AI_BRAIN_URL=http://198.51.100.103:8888
export EB_TTS_ENDPOINT=/api/say
```

## X5 实测结果 (2026-04-26)

✅ **OCR 单元测试**: 5/5 通过 (单数字 0-9 / 1350°C 完整画面 / power off / 边界 case)

✅ **集成测试**:
- 场景 1 (PV=1350 SV=1350 正常): 不报警 ✓
- 场景 2 (PV=1750 超阈值): 触发 I1 CRITICAL, log 写入: `[CRITICAL] [TEMPERATURE_OUT_OF_RANGE] 烧结炉温度超出合理范围 :: 实测 PV = 1750°C 超出 [-10.0, 1600.0]°C 安全区间`

## command_interpreter 接口设计 (Phase 5 重点)

```python
class CommandInterpreter(ABC):
    @abstractmethod
    def parse(self, utterance: str, context: Optional[Dict] = None) -> Dict:
        ...

class RuleInterpreter(CommandInterpreter):     # 默认: 正则匹配 5-10 条指令
class LocalLLMInterpreter(CommandInterpreter): # 本地 llama.cpp Qwen3-0.6B/Gemma 4 1B
class RemoteLLMInterpreter(CommandInterpreter):# 调 AI 脑 dashboard:8888
```

ROS2 param `~interpreter_backend: rule | local | remote` 切换. 默认 rule.

## 已知坑 (Phase 3 踩过的)

- `panel_h=150` 不够装 3 行数字 (PV/SV/MV 各 50 像素), 需 ≥190. 默认值改成 200.
- `screen_visible` 用 mean 不行: LED 数码管黑底亮字, mean 低. 改成 max + 高亮像素比.
- `colcon build --symlink-install` 之后修改 .py 不需要重 build (symlink 即时生效); 但 .msg 改了要重 build my_robot_msgs (需要 ~1.5min 因为 rosidl 重生成).
- `bash heredoc 嵌套引号` 跟 ssh 一起用很容易引号炸. 写脚本到 .sh 文件再 scp + bash 跑稳得多.
- `timeout 3 ros2 topic echo` 之后再 echo 同一 topic, "rcl node's context invalid" 错误是 rclpy 资源回收 race, 不影响功能, ignore.

## Round 4 Phase 4 踩坑 (2026-04-30 X5 首次部署)

- **PaddleOCR 3.5.0 API 完全变了**: `show_log`/`use_gpu`/`det` 全改. 用 `pip3 install "paddleocr==2.8.1" --no-deps` + 补 `scikit-image lmdb python-docx rapidfuzz cython fire imgaug`
- **rnnoise pip/apt 在 ARM64 X5 均不可用**: `librnnoise0` apt 源没有; pip 也没 ARM64 wheel. 正确降级到 `noisereduce` (spectral gating). 节点日志 `backend=noisereduce spectral gating` ≠ 坏事
- **audio_pipeline 测试前要先 `fuser /dev/snd/*` 检查谁占着 ALSA card2**: 多次测试后 PCM capture 进程堆积, 导致 "Device or resource busy"
- **colcon build 必须带 `--merge-install`**: 旧的 install/ 是 merged layout, 不加参数直接报错

## 下一步 (Phase 4 跨网)

- 把 alert_dispatcher 的 TTS 通道接通到 AI 脑 dashboard:8888 (现在 fallback espeak)
- 把 OCR needs_vl_recheck=True 的截图通过 my_robot_bridge 送给 AI 脑 Qwen-VL 复核, 拿回准确 PV/SV/MV 重发 /furnace_reading
- 接 telemetry_publisher 周期上报 SystemTelemetry 给 AI 脑

## 下一步 (Phase 7 云台拉流后)

- 把 image_topic 从默认 /pt_camera/image_raw 改成实际拉流出来的 topic
- 用真实摄像头 + 真烧结炉调 ROI yaml (panel_x/y/w/h, power_led 位置)
- 写 calibrate_furnace_roi.py 交互标定 (鼠标拖框)
