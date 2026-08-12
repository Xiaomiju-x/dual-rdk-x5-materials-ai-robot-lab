"""voice_backend.py — 4 条线共享的语音后端: ALSA 探测 + 百度 TTS + VAD + 百度 ASR.

抽自 xrd_vision/visual_line/deploy_xrd_system.py 的 ALSA / TTS / VAD / ASR 实现.
4 条线 (xrd_vision, xrd_num, spec_vision, spec_num) 共用 M260C 麦克风/扬声器,
同一时刻只能有一条线开"语音输入" (mic 锁通过 shared_locks.acquire_mic_lock).

典型用法:
    voice = VoiceState(line_name="xrd_vision")
    voice.start()                                 # ALSA 探测 + 百度 client + TTS worker
    voice.enqueue_tts("欢迎使用")
    ok, info = voice.enable_voice_input(on_speech=on_asr_text)  # 抢 mic 锁 + 启 VAD
    if not ok:
        print(f"麦被 {info.get('holder_name')} 占用")
    voice.disable_voice_input()
"""
from __future__ import annotations

import math
import os
import re
import shutil
import struct
import subprocess
import threading
import time
from typing import Callable, Optional

try:
    import shared_locks
except ImportError:
    shared_locks = None  # type: ignore

# 百度 TTS/ASR (同一个 client 对象, 共享凭据)
try:
    from aip import AipSpeech
    HAS_BAIDU_SDK = True
except ImportError:
    AipSpeech = None  # type: ignore
    HAS_BAIDU_SDK = False

HAS_ESPEAK = shutil.which("espeak-ng") is not None

# 凭据只从运行环境读取；开源代码永不提供非空 fallback。
DEFAULT_BAIDU_APP_ID = os.environ.get("BAIDU_APP_ID", "").strip()
DEFAULT_BAIDU_API_KEY = os.environ.get("BAIDU_API_KEY", "").strip()
DEFAULT_BAIDU_SECRET_KEY = os.environ.get("BAIDU_SECRET_KEY", "").strip()

# ============ VAD 参数 ============
M260C_VAD_THRESH = 800        # 能量阈值
M260C_VAD_HOLD = 1.0          # 静音保持秒
M260C_TTS_MAX = 100           # TTS 单次最大字符数
ASR_COOLDOWN = 10.0           # 触发后冷却秒


# ============ ALSA 探测 ============
def setup_alsa() -> tuple[str, str]:
    """自动检测 M260C 扬声器和麦克风的 ALSA 设备号.

    Returns (mic_dev, spk_dev), 如 ("plughw:2,0", "plughw:1,0").
    """
    mic, spk = "plughw:2,0", "plughw:1,0"
    try:
        out = subprocess.check_output(
            ["arecord", "-l"], stderr=subprocess.DEVNULL, timeout=5
        ).decode("utf-8", "ignore")
        for line in out.split("\n"):
            if "XFMDPV" in line or "XFM-DP" in line:
                m = re.search(r"card\s+(\d+):", line)
                if m:
                    mic = f"plughw:{m.group(1)},0"
                    break
        out = subprocess.check_output(
            ["aplay", "-l"], stderr=subprocess.DEVNULL, timeout=5
        ).decode("utf-8", "ignore")
        for line in out.split("\n"):
            if "USB Audio" in line and "Camera" not in line and "ES8326" not in line:
                m = re.search(r"card\s+(\d+):", line)
                if m:
                    spk = f"plughw:{m.group(1)},0"
                    break
    except Exception as e:
        print(f"[ALSA] 探测失败 {e}, 用默认 mic={mic} spk={spk}")
    print(f"[ALSA] 选择 扬声器={spk} 麦克风={mic}")
    return mic, spk


# ============ 工具: 清理 LLM 输出里的 DSML / function_calls / 工具调用残留 ============
# DeepSeek-R1 在工具调用流程里, content 字段偶尔会混入 XML 风格的工具协议标记
# (e.g. <|DSML|function_calls>, <invoke>, <parameter>...). 这些不能给最终用户看.
import re as _re_clean
_DSML_PATTERNS = [
    _re_clean.compile(r"<[\s|]*DSML[\s|]*function_calls[\s|]*>.*?<[\s|]*/[\s|]*DSML[\s|]*function_calls[\s|]*>",
                      _re_clean.DOTALL | _re_clean.IGNORECASE),
    _re_clean.compile(r"<[^>]*DSML[^>]*>", _re_clean.IGNORECASE),
    _re_clean.compile(r"<[^>]*function_calls[^>]*>", _re_clean.IGNORECASE),
    _re_clean.compile(r"</?\s*invoke[^>]*>", _re_clean.IGNORECASE),
    _re_clean.compile(r"</?\s*parameter[^>]*>", _re_clean.IGNORECASE),
    _re_clean.compile(r"<\|[^|>]+\|>"),                       # <|im_start|> 之类
    _re_clean.compile(r"\n{3,}"),
]


def clean_llm_output(text: str) -> str:
    """剥掉 LLM 返回里偶尔混入的工具协议标记 / 控制 token, 给前端用户面板展示用.

    在 4 条线 R1 / Qwen-VL 调用返回 content 后立刻调用这个再写入 state.
    """
    if not text:
        return ""
    out = text
    for pat in _DSML_PATTERNS[:-1]:
        out = pat.sub("", out)
    out = _DSML_PATTERNS[-1].sub("\n\n", out).strip()
    return out


# ============ 工具: 提取 TTS 摘要 ============
def extract_tts_summary(text: str) -> str:
    """取前 2 句 (≤150 字), 跳过 Markdown 标题行."""
    if not text:
        return ""
    sentences = re.split(r"[。\n；]", text)
    summary = ""
    count = 0
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if s.startswith("**") and s.endswith(":"):
            continue
        if s.startswith("**") and "**:" in s:
            s = re.sub(r"\*\*(.*?)\*\*[:：]?\s*", "", s).strip()
            if not s:
                continue
        if len(summary) + len(s) > 150:
            break
        summary += s + "。"
        count += 1
        if count >= 2:
            break
    return summary or text[:M260C_TTS_MAX]


# ============ 工具: 语音命令匹配 ============
VOICE_COMMANDS = {
    "export":    ["保存", "导出", "生成报告"],
    "reanalyze": ["重新分析", "再分析", "再看一次"],
    "reset":     ["重置", "清除", "清空"],
    "compare":   ["对比", "上一次", "比较"],
}


def match_voice_command(text: str) -> str:
    for cmd, kws in VOICE_COMMANDS.items():
        if any(kw in text for kw in kws):
            return cmd
    return ""


# ============ VoiceState: 每条线一个实例 ============
class VoiceState:
    """单条线的语音状态 + 后台 worker 控制."""

    def __init__(self,
                 line_name: str,
                 baidu_app_id: str = DEFAULT_BAIDU_APP_ID,
                 baidu_api_key: str = DEFAULT_BAIDU_API_KEY,
                 baidu_secret_key: str = DEFAULT_BAIDU_SECRET_KEY):
        self.line_name = line_name

        # UI/契约状态
        self.tts_enabled: bool = True
        self.voice_input_enabled: bool = False  # 默认关 (要抢 mic 锁)
        self.teach_mode: bool = False

        # 内部状态
        self.tts_queue: list[str] = []
        self.tts_playing: bool = False
        self.mic_ok: bool = False
        self.voice_active: bool = False         # VAD 是否正在听到声音
        self.voice_energy: float = 0.0
        self.voice_last_time: float = 0.0
        self.asr_status: str = "idle"           # idle/recognizing/done/error
        self.asr_text: str = ""

        self.lock = threading.RLock()
        self._running = True

        # 设备 + 客户端 (lazy init via start())
        self.mic_dev = ""
        self.spk_dev = ""
        self._baidu: Optional[object] = None
        self._tts_thread: Optional[threading.Thread] = None
        self._vad_thread: Optional[threading.Thread] = None
        self._on_speech: Optional[Callable[[str], None]] = None

        self._baidu_app_id = baidu_app_id
        self._baidu_api_key = baidu_api_key
        self._baidu_secret_key = baidu_secret_key

    # ============ 启动 / 停止 ============
    def start(self):
        """ALSA 探测 + 百度 client 初始化 + TTS worker 启动."""
        self.mic_dev, self.spk_dev = setup_alsa()
        if HAS_BAIDU_SDK and all(
            (self._baidu_app_id, self._baidu_api_key, self._baidu_secret_key)
        ):
            try:
                self._baidu = AipSpeech(self._baidu_app_id,
                                        self._baidu_api_key,
                                        self._baidu_secret_key)
                print(f"[{self.line_name}] 百度 AipSpeech 客户端就绪")
            except Exception as e:
                print(f"[{self.line_name}] 百度 client 初始化失败 {e}")
                self._baidu = None
        if self._tts_thread is None:
            self._tts_thread = threading.Thread(
                target=self._tts_worker, daemon=True, name=f"{self.line_name}-tts")
            self._tts_thread.start()

    def stop(self):
        self._running = False
        self.disable_voice_input()

    # ============ TTS 队列 ============
    def enqueue_tts(self, text: str):
        if not text:
            return
        with self.lock:
            if self.tts_enabled and len(self.tts_queue) < 3:
                self.tts_queue.append(text[:M260C_TTS_MAX])

    def _tts_worker(self):
        if self._baidu is None and not HAS_ESPEAK:
            print(f"[{self.line_name}][TTS] 无可用引擎 (百度+espeak 都缺)")
            return
        engine = "百度+espeak" if self._baidu is not None else "espeak"
        print(f"[{self.line_name}][TTS] worker 启动 engine={engine} spk={self.spk_dev}")
        while self._running:
            text = None
            with self.lock:
                if self.tts_queue:
                    text = self.tts_queue.pop(0)
                    self.tts_playing = True
            if text:
                try:
                    self._tts_speak(text)
                finally:
                    with self.lock:
                        self.tts_playing = False
            else:
                time.sleep(0.3)

    def _tts_speak(self, text: str):
        """百度 TTS → espeak-ng 兜底, 都 aplay 到 self.spk_dev."""
        if self._baidu is not None:
            try:
                res = self._baidu.synthesis(  # type: ignore
                    text, "zh", 1,
                    {"per": 4, "spd": 5, "pit": 5, "vol": 10, "aue": 6})
                if not isinstance(res, dict):
                    proc = subprocess.Popen(
                        ["aplay", "-D", self.spk_dev, "-q"],
                        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
                    proc.communicate(input=res, timeout=30)
                    return
                print(f"[{self.line_name}][TTS] 百度返回错误 {res.get('err_msg', '')}, 回退 espeak")
            except Exception as e:
                print(f"[{self.line_name}][TTS] 百度异常 {e}, 回退 espeak")
        if HAS_ESPEAK:
            try:
                p1 = subprocess.Popen(
                    ["espeak-ng", "-v", "zh", text, "--stdout"],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                p2 = subprocess.Popen(
                    ["aplay", "-D", self.spk_dev, "-q"],
                    stdin=p1.stdout, stderr=subprocess.DEVNULL)
                p2.communicate(timeout=30)
            except Exception as e:
                print(f"[{self.line_name}][TTS] espeak 失败 {e}")

    # ============ ASR ============
    def asr_baidu(self, audio_bytes: bytes) -> str:
        """PCM 16k mono → 百度 ASR → 文字."""
        if self._baidu is None or len(audio_bytes) < 3200:
            return ""
        try:
            result = self._baidu.asr(  # type: ignore
                audio_bytes, "pcm", 16000, {"dev_pid": 1537})
            if result and result.get("err_no") == 0:
                texts = result.get("result", [])
                return texts[0] if texts else ""
            print(f"[{self.line_name}][ASR] 错误 {result.get('err_msg', 'unknown')}")
            return ""
        except Exception as e:
            print(f"[{self.line_name}][ASR] 调用失败 {e}")
            return ""

    # ============ 语音输入开关 (拿 mic 锁) ============
    def enable_voice_input(self, on_speech: Callable[[str], None]) -> tuple[bool, dict]:
        """请求开启语音输入. 返回 (成功?, info).

        - 抢 mic 锁失败 → (False, {holder_name, holder_pid})
        - 已开过 → (True, {already})
        """
        if self.voice_input_enabled:
            return True, {"already": True}

        if shared_locks is not None:
            ok, info = shared_locks.acquire_mic_lock(self.line_name)
            if not ok:
                return False, info

        self._on_speech = on_speech
        with self.lock:
            self.voice_input_enabled = True
        if self._vad_thread is None or not self._vad_thread.is_alive():
            self._vad_thread = threading.Thread(
                target=self._vad_loop, daemon=True, name=f"{self.line_name}-vad")
            self._vad_thread.start()
        return True, {"started": True}

    def disable_voice_input(self):
        with self.lock:
            self.voice_input_enabled = False
        if shared_locks is not None:
            shared_locks.release_mic_lock()
        # _vad_loop 自己会观察 voice_input_enabled 后停; 不强杀线程

    # ============ VAD 录音线程 ============
    def _vad_loop(self):
        """监听 self.mic_dev, RMS>阈值触发录音, 静音超时后调 on_speech(asr_text)."""
        CHUNK_MS = 100
        RATE = 16000
        CHUNK_SAMPLES = RATE * CHUNK_MS // 1000
        CHUNK_BYTES = CHUNK_SAMPLES * 2
        cmd = ["arecord", "-D", self.mic_dev, "-f", "S16_LE",
               "-r", str(RATE), "-c", "1", "-t", "raw", "-q"]
        print(f"[{self.line_name}][VAD] 启动 mic={self.mic_dev}")

        proc = None
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.DEVNULL)
            with self.lock:
                self.mic_ok = True
            voiced = 0
            silent = 0
            audio_buf = bytearray()
            MAX_BUF = 960000  # 30 秒上限

            while self._running:
                with self.lock:
                    if not self.voice_input_enabled:
                        # 关掉了 → 清空, 但继续 read 防 buffer 堆积
                        voiced = 0
                        audio_buf.clear()
                        # 读丢一块, 不 sleep
                    listening = self.voice_input_enabled
                    tts_now = self.tts_playing

                data = proc.stdout.read(CHUNK_BYTES)
                if not data or len(data) < CHUNK_BYTES:
                    break

                if not listening or tts_now:
                    voiced = 0
                    audio_buf.clear()
                    continue

                samples = struct.unpack(f"<{CHUNK_SAMPLES}h", data)
                rms = (sum(s * s for s in samples) / CHUNK_SAMPLES) ** 0.5
                with self.lock:
                    self.voice_energy = rms

                if rms > M260C_VAD_THRESH:
                    voiced += 1
                    silent = 0
                    if len(audio_buf) < MAX_BUF:
                        audio_buf.extend(data)
                    with self.lock:
                        self.voice_active = True
                else:
                    silent += 1
                    if silent > int(M260C_VAD_HOLD * 1000 / CHUNK_MS):
                        # 静音超阈值, 处理累积音频
                        if voiced >= 5:   # ≥500ms 才算有效语音
                            now = time.time()
                            with self.lock:
                                last = self.voice_last_time
                            if now - last > ASR_COOLDOWN:
                                with self.lock:
                                    self.voice_last_time = now
                                    self.asr_status = "recognizing"
                                play_feedback_tone(self.spk_dev, freq=600,
                                                   duration_ms=150)
                                text = self.asr_baidu(bytes(audio_buf))
                                with self.lock:
                                    if text:
                                        self.asr_text = text
                                        self.asr_status = "done"
                                    else:
                                        self.asr_status = "error"
                                if text and self._on_speech:
                                    try:
                                        self._on_speech(text)
                                    except Exception as e:
                                        print(f"[{self.line_name}][VAD] on_speech 异常 {e}")
                        voiced = 0
                        audio_buf.clear()
                        with self.lock:
                            self.voice_active = False
        except Exception as e:
            print(f"[{self.line_name}][VAD] 线程异常 {e}")
        finally:
            with self.lock:
                self.mic_ok = False
            if proc is not None:
                try:
                    proc.terminate()
                except Exception:
                    pass

    # ============ /api/voice_config 配套 snapshot ============
    def snapshot(self) -> dict:
        with self.lock:
            return {
                "tts_enabled": self.tts_enabled,
                "voice_input_enabled": self.voice_input_enabled,
                "teach_mode": self.teach_mode,
                "mic_ok": self.mic_ok,
                "tts_playing": self.tts_playing,
                "voice_active": self.voice_active,
                "voice_energy": round(self.voice_energy, 1),
                "asr_status": self.asr_status,
                "asr_text": self.asr_text,
                "spk_dev": self.spk_dev,
                "mic_dev": self.mic_dev,
                "engine": "baidu+espeak" if self._baidu is not None
                          else ("espeak" if HAS_ESPEAK else "none"),
            }


# ============ 反馈短音 (与 VoiceState 解耦, 便于其他场合直接调) ============
def play_feedback_tone(spk_dev: str, freq: int = 800, duration_ms: int = 200):
    rate = 16000
    n = int(rate * duration_ms / 1000)
    samples = bytearray()
    for i in range(n):
        env = min(1.0, i / 200, (n - i) / 200)
        val = int(32767 * env * math.sin(2 * math.pi * freq * i / rate))
        samples += struct.pack("<h", max(-32768, min(32767, val)))
    hdr = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + len(samples), b"WAVE", b"fmt ", 16,
        1, 1, rate, rate * 2, 2, 16, b"data", len(samples))
    try:
        proc = subprocess.Popen(
            ["aplay", "-D", spk_dev, "-q"],
            stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        proc.communicate(input=bytes(hdr + samples), timeout=5)
    except Exception:
        pass
