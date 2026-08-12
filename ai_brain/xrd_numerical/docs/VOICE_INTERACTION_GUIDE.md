# 语音交互功能移植指南 — 数值线 (web_demo.py)

> 本文档供数值线 Claude Code 参考，描述视觉线已实现的语音交互系统，以便数值线也接入语音交互功能。

---

## 1. 视觉线语音交互架构总览

视觉线 (`deploy_xrd_system.py`) 已在 v3.0 实现完整语音交互闭环:

```
M260C麦克风(plughw:2,0)
    │ arecord持续录音 (16kHz, S16_LE, 单声道)
    ▼
VAD语音活动检测 (RMS能量阈值)
    │
    ├─ 无分析结果 → 触发新分析 → LLM分析 → TTS播报摘要
    │
    └─ 已有分析结果 → 百度ASR语音识别 → LLM跟进提问 → TTS播报
                                                    │
                                                    ▼
                                          M260C扬声器(plughw:1,0)
                                          百度TTS(主) / espeak-ng(备)
```

---

## 2. 硬件配置

| 硬件 | 说明 | ALSA设备 |
|------|------|----------|
| M260C智能音箱 | USB串口连接, 板载麦克风+扬声器 | - |
| 麦克风 | M260C板载, 全向拾音 | `plughw:2,0` |
| 扬声器 | M260C板载 | `plughw:1,0` |

**注意**: ALSA设备号可能因USB插入顺序变化，可用 `arecord -l` 和 `aplay -l` 确认。

---

## 3. 依赖安装

```bash
# RDK X5 上
pip3 install baidu-aip        # 百度语音SDK (ASR + TTS)
pip3 install pyserial          # M260C串口通信 (可选, 仅用于唤醒词)
sudo apt install espeak-ng     # 离线TTS备用
```

---

## 4. 百度语音API配置 (ASR + TTS 共用同一个AipSpeech客户端)

```python
from aip import AipSpeech

BAIDU_TTS_APP_ID     = "<REMOVED_FROM_HISTORY>"
BAIDU_TTS_API_KEY    = "<REMOVED_FROM_HISTORY>"
BAIDU_TTS_SECRET_KEY = "<REMOVED_FROM_HISTORY>"

client = AipSpeech(BAIDU_TTS_APP_ID, BAIDU_TTS_API_KEY, BAIDU_TTS_SECRET_KEY)
```

**ASR调用** (语音→文字):
```python
# audio_bytes: PCM原始音频 (16kHz, 16bit, 单声道)
result = client.asr(audio_bytes, 'pcm', 16000, {'dev_pid': 1537})
# dev_pid=1537 普通话(纯中文输入)
# 成功: {'err_no': 0, 'result': ['识别的文字']}
# 失败: {'err_no': 3301, 'err_msg': '...'}
text = result['result'][0] if result.get('err_no') == 0 else ""
```

**TTS调用** (文字→语音):
```python
# 返回音频bytes(WAV格式), 失败返回dict
audio = client.synthesis(text, 'zh', 1, {
    'per': 4,    # 发音人 (4=度丫丫, 0=度小美, 1=度小宇)
    'spd': 5,    # 语速 (0-15)
    'pit': 5,    # 音调 (0-15)
    'vol': 10,   # 音量 (0-15)
    'aue': 6,    # 音频格式 (6=WAV)
})
if not isinstance(audio, dict):  # 成功
    # 用aplay播放
    proc = subprocess.Popen(
        ['aplay', '-D', 'plughw:1,0', '-q'],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    proc.communicate(input=audio, timeout=30)
```

---

## 5. 核心模块代码 (可直接复用)

### 5a. VAD语音活动检测

```python
import struct
import subprocess
import threading
import time

M260C_MIC_DEV    = "plughw:2,0"
M260C_SPK_DEV    = "plughw:1,0"
M260C_VAD_THRESH = 800       # RMS能量阈值 (环境噪音一般<300)
M260C_VAD_HOLD   = 1.0       # 语音结束后等待秒数
M260C_TTS_MAX    = 100       # TTS最大字符数

def vad_thread(state):
    """daemon线程: 麦克风持续录音, 检测语音活动"""
    CHUNK_MS = 100                          # 每次100ms
    RATE = 16000
    CHUNK_SAMPLES = RATE * CHUNK_MS // 1000  # 1600
    CHUNK_BYTES = CHUNK_SAMPLES * 2          # 3200 bytes
    COOLDOWN = 10.0                          # 冷却秒数

    cmd = ["arecord", "-D", M260C_MIC_DEV, "-f", "S16_LE",
           "-r", str(RATE), "-c", "1", "-t", "raw", "-q"]

    while state.running:
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.DEVNULL)
            voiced_chunks = 0
            silent_chunks = 0
            triggered = False
            audio_buffer = bytearray()
            MAX_AUDIO_BUF = 960000  # 30秒上限

            while state.running:
                data = proc.stdout.read(CHUNK_BYTES)
                if len(data) < CHUNK_BYTES:
                    break

                # TTS播放时跳过 (防自触发!)
                with state.lock:
                    if state.tts_playing:
                        voiced_chunks = 0
                        audio_buffer = bytearray()
                        continue

                # 计算RMS能量
                samples = struct.unpack(f'<{CHUNK_SAMPLES}h', data)
                rms = (sum(s * s for s in samples) / CHUNK_SAMPLES) ** 0.5

                if rms > M260C_VAD_THRESH:
                    voiced_chunks += 1
                    silent_chunks = 0
                    if len(audio_buffer) < MAX_AUDIO_BUF:
                        audio_buffer.extend(data)
                else:
                    silent_chunks += 1
                    if silent_chunks > int(M260C_VAD_HOLD * 1000 / CHUNK_MS):
                        if voiced_chunks >= 5 and not triggered:
                            # 有效语音 (>=500ms)
                            now = time.time()
                            with state.lock:
                                last = state.voice_last_time
                            if now - last > COOLDOWN:
                                with state.lock:
                                    state.voice_last_time = now

                                # ★ 这里是数值线需要自定义的逻辑:
                                # 视觉线: 无结果→分析, 有结果→ASR跟进
                                # 数值线: ASR识别 → 对当前分析结果跟进提问
                                handle_voice_input(state, bytes(audio_buffer))

                                triggered = True
                        voiced_chunks = 0
                        audio_buffer = bytearray()
                        triggered = False

            proc.terminate()
        except Exception as e:
            print(f"[VAD] 麦克风错误: {e}, 3s后重试")
            time.sleep(3)
```

### 5b. TTS播报引擎

```python
import shutil

HAS_TTS = shutil.which("espeak-ng") is not None

def tts_speak(text, client):
    """百度TTS(优先) → espeak-ng(备用)"""
    # 百度在线TTS
    if client is not None:
        try:
            result = client.synthesis(
                text, 'zh', 1,
                {'per': 4, 'spd': 5, 'pit': 5, 'vol': 10, 'aue': 6})
            if not isinstance(result, dict):
                proc = subprocess.Popen(
                    ['aplay', '-D', M260C_SPK_DEV, '-q'],
                    stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
                proc.communicate(input=result, timeout=30)
                return
        except Exception as e:
            print(f"[TTS] 百度失败: {e}, 回退espeak")

    # espeak-ng离线备用
    if HAS_TTS:
        try:
            p1 = subprocess.Popen(
                ['espeak-ng', '-v', 'zh', text, '--stdout'],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            p2 = subprocess.Popen(
                ['aplay', '-D', M260C_SPK_DEV, '-q'],
                stdin=p1.stdout, stderr=subprocess.DEVNULL)
            p2.communicate(timeout=30)
        except Exception as e:
            print(f"[TTS] espeak失败: {e}")


def tts_worker(state, client):
    """daemon线程: 消费TTS队列"""
    while state.running:
        text = None
        with state.lock:
            if state.tts_queue:
                text = state.tts_queue.pop(0)
                state.tts_playing = True
        if text:
            tts_speak(text, client)
            with state.lock:
                state.tts_playing = False
        else:
            time.sleep(0.3)


def enqueue_tts(state, text):
    """将文本加入TTS播报队列"""
    with state.lock:
        if len(state.tts_queue) < 3:
            state.tts_queue.append(text[:M260C_TTS_MAX])
```

### 5c. 提示音生成 (纯Python, 无依赖)

```python
def play_feedback_tone(freq=800, duration_ms=200):
    """播放短提示音 (正弦波)"""
    import struct as _st, math as _m
    rate = 16000
    n = int(rate * duration_ms / 1000)
    samples = b""
    for i in range(n):
        env = min(1.0, i / 200, (n - i) / 200)  # 淡入淡出防爆音
        val = int(32767 * env * _m.sin(2 * _m.pi * freq * i / rate))
        samples += _st.pack('<h', max(-32768, min(32767, val)))
    hdr = _st.pack('<4sI4s4sIHHIIHH4sI',
                   b'RIFF', 36 + len(samples), b'WAVE', b'fmt ', 16,
                   1, 1, rate, rate * 2, 2, 16, b'data', len(samples))
    with state.lock:
        state.tts_playing = True
    try:
        proc = subprocess.Popen(['aplay', '-D', M260C_SPK_DEV, '-q'],
                                stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        proc.communicate(input=hdr + samples, timeout=5)
    except Exception:
        pass
    finally:
        with state.lock:
            state.tts_playing = False
```

### 5d. TTS摘要提取 (从LLM回复中提取前2句)

```python
import re

def extract_tts_summary(response):
    """从回复中提取前2句话作为播报摘要"""
    sentences = re.split(r'[。\n；]', response)
    summary = ""
    count = 0
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        # 跳过纯Markdown格式行
        if s.startswith('**') and s.endswith(':'):
            continue
        if s.startswith('**') and '**:' in s:
            s = re.sub(r'\*\*(.*?)\*\*[:：]?\s*', '', s).strip()
            if not s:
                continue
        if len(summary) + len(s) > 150:
            break
        summary += s + "。"
        count += 1
        if count >= 2:
            break
    return summary or response[:M260C_TTS_MAX]
```

---

## 6. 数值线接入语音交互的建议

### 6a. State类需要新增的字段

```python
# 在web_demo.py的全局或State类中添加:
self.lock = threading.Lock()
self.running = True
self.voice_last_time = 0
self.tts_queue = []
self.tts_playing = False
self.tts_enabled = True
# 数值线特有:
self.current_report = ""          # 当前分析报告文本
self.current_filename = ""        # 当前分析的文件名
```

### 6b. 数值线语音交互的逻辑差异

视觉线是实时摄像头场景, 数值线是选文件分析场景, 逻辑有所不同:

| 场景 | 视觉线做法 | 数值线建议 |
|------|-----------|-----------|
| 语音触发分析 | 截取当前摄像头帧→千问VL | 播报最近一次分析结果摘要 |
| 语音跟进提问 | ASR→千问VL(带图)跟进 | ASR→DeepSeek(带上下文)跟进 |
| 无结果时语音 | "未检测到图谱" | "请先选择.raw文件进行分析" |

### 6c. 数值线的`handle_voice_input()`实现建议

```python
def handle_voice_input(state, audio_bytes):
    """数值线语音输入处理"""
    with state.lock:
        has_report = bool(state.current_report)
        report = state.current_report
        filename = state.current_filename

    if not has_report:
        enqueue_tts(state, "请先选择样品文件进行分析")
        return

    # ASR识别
    play_feedback_tone(freq=600, duration_ms=150)
    text = do_asr(audio_bytes)  # 复用百度ASR

    if not text or len(text.strip()) <= 1:
        # ASR失败 → 播报当前结果摘要
        summary = extract_tts_summary(report)
        enqueue_tts(state, summary)
        return

    print(f"[ASR] 识别: {text}")
    enqueue_tts(state, "收到，正在查询")

    # 调用DeepSeek跟进提问
    try:
        followup_prompt = f"""之前对{filename}的XRD分析结果:
{report}

用户追问: {text}

请针对追问详细解答，控制在200字以内。"""

        followup_result = call_deepseek(followup_prompt)

        with state.lock:
            state.current_report = followup_result

        summary = extract_tts_summary(followup_result)
        enqueue_tts(state, summary)
    except Exception as e:
        print(f"[Voice] 跟进失败: {e}")
        enqueue_tts(state, "抱歉，查询失败")
```

### 6d. 启动语音线程

```python
# 在 web_demo.py 的 main() 中, Flask启动前:
from aip import AipSpeech
client = AipSpeech(BAIDU_TTS_APP_ID, BAIDU_TTS_API_KEY, BAIDU_TTS_SECRET_KEY)

threading.Thread(target=tts_worker, args=(state, client), daemon=True).start()
threading.Thread(target=vad_thread, args=(state,), daemon=True).start()
```

### 6e. 命令行参数

```python
parser.add_argument("--no-voice", action="store_true", help="禁用语音交互")
```

---

## 7. 关键注意事项

1. **防自触发是最重要的**: TTS播放时`tts_playing=True`, VAD必须跳过。否则扬声器声音会被麦克风拾取, 形成无限循环。
2. **COOLDOWN冷却**: 触发后至少等10秒, 避免TTS尾音触发二次识别。
3. **线程安全**: 所有共享状态通过`state.lock`保护, 但不要在lock内做IO操作。
4. **arecord/aplay**: 都是ALSA命令行工具, RDK X5默认已安装。通过subprocess调用。
5. **ASR音频格式**: 必须是PCM原始格式(16kHz, 16bit, 单声道), 不要传WAV头。
6. **TTS队列上限**: 最多缓存3条, 防止队列膨胀。
7. **espeak-ng中文**: 质量较差(机械音), 仅作百度TTS网络不可用时的备用。
8. **M260C串口**: 数值线不需要串口功能(那是用于唤醒词的), 只需要麦克风和扬声器的ALSA设备。

---

## 8. 视觉线源码参考位置

以下是 `deploy_xrd_system.py` 中语音相关代码的行号(v3.0, 1914行):

| 功能 | 行号范围 | 说明 |
|------|----------|------|
| M260C配置常量 | 89-100 | BAUD, TTS_MAX, MIC_DEV, SPK_DEV等 |
| State类(语音字段) | 315-328 | voice_active, tts_queue, asr_text等 |
| find_m260c_port() | 542-563 | 自动检测M260C串口 |
| parse_m260c_binary() | 566-590 | 解析二进制唤醒帧 |
| play_feedback_tone() | 593-617 | 提示音生成 |
| trigger_voice_analyze() | 620-636 | 语音触发分析 |
| vad_thread() | 639-748 | VAD主循环 |
| do_asr() | 751-764 | 百度ASR调用 |
| do_followup_async() | 767-786 | 语音跟进提问 |
| m260c_thread() | 789-889 | M260C串口通信 |
| enqueue_tts() | 894-898 | TTS入队 |
| extract_tts_summary() | 901-924 | 摘要提取 |
| tts_speak() | 927-956 | TTS引擎(百度+espeak) |
| tts_worker() | 959-979 | TTS消费线程 |
| 启动语音线程 | 1869-1885 | 初始化+启动daemon线程 |
