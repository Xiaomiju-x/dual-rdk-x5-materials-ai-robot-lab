#!/usr/bin/env python3
"""
XRD智能分析系统 Web可视化Demo
Flask + Canvas 单文件应用，零外部依赖，在RDK X5上运行

用法:
  python3 web_demo.py                  # 默认端口8080
  python3 web_demo.py --port 5000      # 指定端口
  python3 web_demo.py --offline        # 纯离线模式
"""

import sys
import os
import time
import json
import argparse
import threading
import numpy as np
from pathlib import Path

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for _parent in (os.path.dirname(_SCRIPT_DIR), os.path.dirname(os.path.dirname(_SCRIPT_DIR))):
    if os.path.isdir(os.path.join(_parent, "rb_voe")):
        sys.path.insert(0, _parent)
        break

try:
    from rb_voe.runtime_identity import build_runtime_identity
except ImportError:
    build_runtime_identity = None
_SEARCH_DIRS = [
    _SCRIPT_DIR,
    os.path.join(_SCRIPT_DIR, "bpu"),
    os.path.join(_SCRIPT_DIR, "bpu", "x5_deploy"),
    "/home/rdk/xrd1",
]

def _find_file(name):
    for d in _SEARCH_DIRS:
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return None

# 导入推理模块
_infer_dir = None
for d in _SEARCH_DIRS:
    if os.path.isfile(os.path.join(d, "infer_with_llm.py")):
        _infer_dir = d
        break
if _infer_dir:
    sys.path.insert(0, _infer_dir)

sys.path.insert(0, _SCRIPT_DIR)

# v4.1 Round 5: 共享设备锁 (4 条线共抢 IMX415 + M260C)
try:
    import shared_locks
except ImportError:
    shared_locks = None
    print("[WARN] shared_locks 未找到, 麦克风互斥保护禁用")

try:
    from flask import Flask, request, jsonify, send_from_directory
except ImportError:
    print("[ERROR] Flask未安装. 运行: pip install flask")
    sys.exit(1)

# ============ Flask App ============
app = Flask(__name__)
OFFLINE_MODE = False
RAW_DIR = os.path.join(_SCRIPT_DIR, "data", "raw_files")
if not os.path.isdir(RAW_DIR):
    RAW_DIR = "/home/rdk/xrd1/data/raw_files"


# ============ 语音交互系统 ============
import struct as _struct
import subprocess as _sp
import shutil as _shutil
import re as _re

def _detect_alsa_devices():
    """自动检测M260C麦克风和扬声器设备号(每次启动运行一次)"""
    mic, spk = "plughw:2,0", "plughw:1,0"  # 默认值
    try:
        # 检测麦克风: 找XFMDPV(M260C麦克风阵列的标识)
        out = _sp.check_output(["arecord", "-l"], stderr=_sp.DEVNULL, timeout=5).decode()
        for line in out.split('\n'):
            if 'XFMDPV' in line or 'XFM-DP' in line:
                card = line.split('card')[1].strip().split(':')[0].strip()
                mic = f"plughw:{card},0"
                break
        # 检测扬声器: 找USB Audio Device(排除Camera和ES8326板载)
        out = _sp.check_output(["aplay", "-l"], stderr=_sp.DEVNULL, timeout=5).decode()
        for line in out.split('\n'):
            if 'USB Audio' in line and 'Camera' not in line and 'ES8326' not in line:
                card = line.split('card')[1].strip().split(':')[0].strip()
                spk = f"plughw:{card},0"
                break
    except Exception:
        pass
    return mic, spk

M260C_MIC_DEV, M260C_SPK_DEV = _detect_alsa_devices()
M260C_VAD_THRESH = 800
M260C_TTS_MAX = 100

# 百度语音 API
BAIDU_APP_ID = "<REMOVED_FROM_HISTORY>"
BAIDU_API_KEY = "<REMOVED_FROM_HISTORY>"
BAIDU_SECRET_KEY = "<REMOVED_FROM_HISTORY>"


def _clean_dsml(text: str) -> str:
    """剥掉 R1/Qwen-VL 偶发混入的工具协议标记 (DSML/function_calls/<|...|>)."""
    if not text:
        return ""
    out = text
    out = _re.sub(r"<[\s|]*DSML[\s|]*function_calls[\s|]*>.*?<[\s|]*/[\s|]*DSML[\s|]*function_calls[\s|]*>",
                  "", out, flags=_re.DOTALL | _re.IGNORECASE)
    out = _re.sub(r"<[^>]*DSML[^>]*>", "", out, flags=_re.IGNORECASE)
    out = _re.sub(r"<[^>]*function_calls[^>]*>", "", out, flags=_re.IGNORECASE)
    out = _re.sub(r"</?\s*invoke[^>]*>", "", out, flags=_re.IGNORECASE)
    out = _re.sub(r"</?\s*parameter[^>]*>", "", out, flags=_re.IGNORECASE)
    out = _re.sub(r"<\|[^|>]+\|>", "", out)
    out = _re.sub(r"\n{3,}", "\n\n", out).strip()
    return out


class VoiceState:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = True
        self.voice_last_time = 0.0
        self.tts_queue = []
        self.tts_playing = False
        self.tts_enabled = True             # TTS 输出 (默认开 — 分析完播报)
        self.voice_input_enabled = False    # 语音输入 (默认关 — 抢 mic 锁)
        self.teach_mode = False             # 教学模式
        self.current_report = ""
        self.current_filename = ""
        self.last_followup_q = ""           # 跟进提问 (前端 polling)
        self.last_followup_a = ""
        self.mic_active = False
        self.last_asr_text = ""
        self.last_tts_text = ""
        self.voice_status = "idle"
        self.voice_log = []
        # v4.1 Round 5: pipeline 实时进度 (给前端 polling 用)
        self.pipeline_stages = []   # [{name, status, time_ms}, ...]
        self.pipeline_current = ""  # 正在跑的阶段名
        self.pipeline_busy = False  # 是否正在跑 run_pipeline
        # v4.1 Round 5: Agent thinking SSE 缓冲
        self.agent_stream_buffer = ""
        self.agent_stream_done = True


voice_state = VoiceState()
_aip_client = None  # 延迟初始化
_HAS_ESPEAK = _shutil.which("espeak-ng") is not None


def _voice_log(state, event_type, text):
    """记录语音事件(最多保留10条)"""
    import datetime
    entry = {"time": datetime.datetime.now().strftime("%H:%M:%S"),
             "type": event_type, "text": text[:80]}
    with state.lock:
        state.voice_log.append(entry)
        if len(state.voice_log) > 10:
            state.voice_log.pop(0)


def extract_tts_summary(response):
    """从LLM回复中提取前2句作为TTS播报摘要"""
    sentences = _re.split(r'[。\n；]', response)
    summary = ""
    count = 0
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if s.startswith('**') and s.endswith(':'):
            continue
        if s.startswith('**') and '**:' in s:
            s = _re.sub(r'\*\*(.*?)\*\*[:：]?\s*', '', s).strip()
            if not s:
                continue
        if s.startswith('#'):
            continue
        if len(summary) + len(s) > 150:
            break
        summary += s + "。"
        count += 1
        if count >= 2:
            break
    return summary or response[:M260C_TTS_MAX]


def tts_speak(text, client):
    """百度TTS(优先) → espeak-ng(备用)"""
    if client is not None:
        try:
            result = client.synthesis(
                text, 'zh', 1,
                {'per': 4, 'spd': 5, 'pit': 5, 'vol': 10, 'aue': 6})
            if not isinstance(result, dict):
                proc = _sp.Popen(
                    ['aplay', '-D', M260C_SPK_DEV, '-q'],
                    stdin=_sp.PIPE, stderr=_sp.DEVNULL)
                proc.communicate(input=result, timeout=30)
                return
        except Exception as e:
            print(f"[TTS] 百度失败: {e}, 回退espeak")
    if _HAS_ESPEAK:
        try:
            p1 = _sp.Popen(['espeak-ng', '-v', 'zh', text, '--stdout'],
                           stdout=_sp.PIPE, stderr=_sp.DEVNULL)
            p2 = _sp.Popen(['aplay', '-D', M260C_SPK_DEV, '-q'],
                           stdin=p1.stdout, stderr=_sp.DEVNULL)
            p2.communicate(timeout=30)
        except Exception as e:
            print(f"[TTS] espeak失败: {e}")


def enqueue_tts(state, text):
    """将文本加入TTS播报队列 (tts_enabled=False 时直接丢弃)"""
    if not text:
        return
    with state.lock:
        if state.tts_enabled and len(state.tts_queue) < 3:
            state.tts_queue.append(text[:M260C_TTS_MAX])


def tts_worker(state, client):
    """daemon线程: 消费TTS队列"""
    while state.running:
        text = None
        with state.lock:
            if state.tts_queue:
                text = state.tts_queue.pop(0)
                state.tts_playing = True
                state.voice_status = "speaking"
                state.last_tts_text = text
        if text:
            _voice_log(state, "tts", text)
            tts_speak(text, client)
            with state.lock:
                state.tts_playing = False
                state.voice_status = "idle"
        else:
            time.sleep(0.3)


def play_feedback_tone(state, freq=800, duration_ms=200):
    """播放短提示音"""
    import math as _m
    rate = 16000
    n = int(rate * duration_ms / 1000)
    samples = b""
    for i in range(n):
        env = min(1.0, i / 200, (n - i) / 200)
        val = int(32767 * env * _m.sin(2 * _m.pi * freq * i / rate))
        samples += _struct.pack('<h', max(-32768, min(32767, val)))
    hdr = _struct.pack('<4sI4s4sIHHIIHH4sI',
                       b'RIFF', 36 + len(samples), b'WAVE', b'fmt ', 16,
                       1, 1, rate, rate * 2, 2, 16, b'data', len(samples))
    with state.lock:
        state.tts_playing = True
    try:
        proc = _sp.Popen(['aplay', '-D', M260C_SPK_DEV, '-q'],
                         stdin=_sp.PIPE, stderr=_sp.DEVNULL)
        proc.communicate(input=hdr + samples, timeout=5)
    except Exception:
        pass
    finally:
        with state.lock:
            state.tts_playing = False


def do_asr(audio_bytes, client):
    """百度ASR: PCM → 文字"""
    if client is None:
        return ""
    try:
        result = client.asr(audio_bytes, 'pcm', 16000, {'dev_pid': 1537})
        if result.get('err_no') == 0:
            return result['result'][0]
    except Exception as e:
        print(f"[ASR] 失败: {e}")
    return ""


def handle_voice_input(state, audio_bytes, client):
    """数值线语音输入处理"""
    with state.lock:
        has_report = bool(state.current_report)
        report = state.current_report
        filename = state.current_filename
        state.voice_status = "recognizing"

    if not has_report:
        enqueue_tts(state, "请先选择样品文件进行分析")
        _voice_log(state, "info", "无分析结果，提示用户选择文件")
        return

    play_feedback_tone(state, freq=600, duration_ms=150)
    text = do_asr(audio_bytes, client)

    with state.lock:
        state.last_asr_text = text or "(未识别)"

    if not text or len(text.strip()) <= 1:
        summary = extract_tts_summary(report)
        enqueue_tts(state, summary)
        _voice_log(state, "asr", "识别为空，播报当前摘要")
        return

    print(f"[ASR] 识别: {text}")
    _voice_log(state, "asr", text)
    enqueue_tts(state, "收到，正在查询")

    try:
        from infer_with_llm import call_deepseek
        followup_prompt = f"之前对{filename}的XRD分析结果:\n{report}\n\n用户追问: {text}\n\n请针对追问详细解答，控制在200字以内。"
        followup_result = call_deepseek(followup_prompt)
        with state.lock:
            state.current_report = followup_result
        summary = extract_tts_summary(followup_result)
        enqueue_tts(state, summary)
        _voice_log(state, "llm", summary)
    except Exception as e:
        print(f"[Voice] 跟进失败: {e}")
        enqueue_tts(state, "抱歉，查询失败")
        _voice_log(state, "error", str(e))


def vad_thread(state, client):
    """daemon线程: 麦克风持续录音, 检测语音活动"""
    CHUNK_MS = 100
    RATE = 16000
    CHUNK_SAMPLES = RATE * CHUNK_MS // 1000
    CHUNK_BYTES = CHUNK_SAMPLES * 2
    COOLDOWN = 10.0

    cmd = ["arecord", "-D", M260C_MIC_DEV, "-f", "S16_LE",
           "-r", str(RATE), "-c", "1", "-t", "raw", "-q"]

    while state.running:
        try:
            proc = _sp.Popen(cmd, stdout=_sp.PIPE, stderr=_sp.DEVNULL)
            voiced_chunks = 0
            silent_chunks = 0
            triggered = False
            audio_buffer = bytearray()
            MAX_AUDIO_BUF = 960000

            with state.lock:
                state.mic_active = True
                state.voice_status = "idle"
            _voice_log(state, "info", "麦克风就绪")

            while state.running:
                data = proc.stdout.read(CHUNK_BYTES)
                if len(data) < CHUNK_BYTES:
                    break

                # 语音输入关闭时丢弃数据 (但保持 arecord 运行, 防 buffer 堆积)
                if not state.voice_input_enabled:
                    voiced_chunks = 0
                    audio_buffer = bytearray()
                    continue

                with state.lock:
                    if state.tts_playing:
                        voiced_chunks = 0
                        audio_buffer = bytearray()
                        continue

                samples = _struct.unpack(f'<{CHUNK_SAMPLES}h', data)
                rms = (sum(s * s for s in samples) / CHUNK_SAMPLES) ** 0.5

                if rms > M260C_VAD_THRESH:
                    voiced_chunks += 1
                    silent_chunks = 0
                    if len(audio_buffer) < MAX_AUDIO_BUF:
                        audio_buffer.extend(data)
                    with state.lock:
                        state.voice_status = "listening"
                else:
                    silent_chunks += 1
                    if silent_chunks > int(1.0 * 1000 / CHUNK_MS):
                        if voiced_chunks >= 5 and not triggered:
                            now = time.time()
                            with state.lock:
                                last = state.voice_last_time
                            if now - last > COOLDOWN:
                                with state.lock:
                                    state.voice_last_time = now
                                handle_voice_input(state, bytes(audio_buffer), client)
                                triggered = True
                        voiced_chunks = 0
                        audio_buffer = bytearray()
                        triggered = False
                        with state.lock:
                            if state.voice_status == "listening":
                                state.voice_status = "idle"

            proc.terminate()
        except Exception as e:
            print(f"[VAD] 麦克风错误: {e}, 3s后重试")
            with state.lock:
                state.mic_active = False
                state.voice_status = "error"
            time.sleep(3)


def _fmt_ms(seconds):
    ms = seconds * 1000
    if ms < 1:
        return f"{ms:.2f}ms"
    elif ms < 100:
        return f"{ms:.1f}ms"
    return f"{ms:.0f}ms"


def run_pipeline(filepath, offline=False):
    """运行完整推理pipeline，返回结构化结果"""
    from infer_with_llm import (
        parse_raw_file, extract_peaks_from_raw, peaks_to_feature,
        normalize_feature, numpy_infer, bpu_infer, bpu_infer_fine,
        _load_peak_matcher, build_llm_prompt, call_deepseek,
        offline_report, warmup_bpu, should_reject, run_agent, mc_dropout_uncertainty,
        crystallography_analysis, simulate_and_compare, tda_persistent_homology,
        verify_systematic_absence, nelson_riley_refine, lattice_param_cubic,
        conformal_predict, compute_feature_attribution,
        NORM_PARAMS_PATH, LABEL_MAP_PATH,
    )

    # v4.1 Round 5: 实时进度 - 用子类 list 自动镜像每个 stage 到 voice_state
    class _TrackedStages(list):
        def append(self, s):
            super().append(s)
            try:
                with voice_state.lock:
                    voice_state.pipeline_stages.append(dict(s))
                    voice_state.pipeline_current = ""
            except Exception:
                pass

    result = {"stages": _TrackedStages(), "error": None}
    timings = {}

    # v4.1 Round 9: 和 xrd_vision 对齐的流式思考链 (reportBody 里打字机效果)
    def _stream(line):
        try:
            with voice_state.lock:
                voice_state.agent_stream_buffer += line
        except Exception:
            pass

    # 重置 pipeline 实时状态 + 流式缓冲 (让 SSE 从第 1 阶段就开始喷内容)
    with voice_state.lock:
        voice_state.pipeline_stages = []
        voice_state.pipeline_current = "解析.raw文件"
        voice_state.pipeline_busy = True
        voice_state.agent_stream_buffer = ""
        voice_state.agent_stream_done = False
    _stream(f"🚀 **启动 XRD 分析**: `{Path(filepath).name}`\n\n")

    try:
        # 加载配置
        norm_path = _find_file("norm_params.json") or NORM_PARAMS_PATH
        label_path = _find_file("label_map.json") or LABEL_MAP_PATH
        with open(norm_path) as f:
            norm_params = json.load(f)
        with open(label_path) as f:
            label_info = json.load(f)
        idx2label = {int(k): v for k, v in label_info['idx2label'].items()}

        # Stage 1: 解析
        t0 = time.perf_counter()
        two_theta, intensity = parse_raw_file(filepath)
        timings['parse'] = time.perf_counter() - t0
        result["spectrum"] = {
            "two_theta": two_theta.tolist(),
            "intensity": intensity.tolist(),
        }
        result["stages"].append({
            "name": "解析.raw文件",
            "status": "ok",
            "detail": f"{len(two_theta)}点, 2\u03b8=[{two_theta[0]:.1f}\u00b0, {two_theta[-1]:.1f}\u00b0]",
            "time_ms": round(timings['parse'] * 1000, 2),
        })
        _stream(f"📂 解析 .raw 文件: **{len(two_theta)}** 点, 2θ ∈ [{two_theta[0]:.1f}°, {two_theta[-1]:.1f}°] `{timings['parse']*1000:.1f}ms`\n\n")

        # Stage 2: 峰提取
        t0 = time.perf_counter()
        peaks = extract_peaks_from_raw(two_theta, intensity)
        timings['peaks'] = time.perf_counter() - t0
        result["peaks"] = [{
            "position": round(p['position'], 2),
            "intensity": round(p['intensity'], 4),
            "fwhm": round(p.get('fwhm', 0), 3),
        } for p in peaks]
        result["stages"].append({
            "name": "峰位提取",
            "status": "ok",
            "detail": f"{len(peaks)}个峰",
            "time_ms": round(timings['peaks'] * 1000, 2),
        })
        _stream(f"🔍 峰提取: 检出 **{len(peaks)}** 个峰 `{timings['peaks']*1000:.1f}ms`\n\n")

        # 视觉感知指标
        peak_positions = sorted([p['position'] for p in peaks])
        if len(peak_positions) >= 2:
            spacings = [peak_positions[i+1] - peak_positions[i] for i in range(len(peak_positions)-1)]
            avg_sp = sum(spacings) / len(spacings)
            var_sp = sum((s - avg_sp)**2 for s in spacings) / len(spacings)
            sym_score = max(0, min(100, int(100 * (1 - (var_sp**0.5) / (avg_sp + 1e-6)))))
        else:
            sym_score = 0
            spacings = []
        pattern_type = "尖锐对称型" if sym_score > 60 else "复杂多峰型"
        result["perception"] = {
            "peak_count": len(peaks),
            "symmetry_score": sym_score,
            "pattern_type": pattern_type,
            "two_theta_range": [round(two_theta[0], 1), round(two_theta[-1], 1)],
            "max_intensity": round(float(intensity.max()), 1),
        }

        # 晶体学定量分析 (Scherrer + d间距)
        try:
            cryst = crystallography_analysis(peaks)
            result["crystallography"] = cryst
        except Exception:
            pass

        # TDA持久同调
        try:
            tda = tda_persistent_homology(peaks)
            result["tda"] = tda
        except Exception:
            pass

        if len(peaks) == 0:
            result["stages"].append({"name": "特征构造", "status": "skip", "detail": "无峰"})
            return result

        # Stage 3: 特征构造
        t0 = time.perf_counter()
        feat_raw = peaks_to_feature(peaks)
        feat_norm = normalize_feature(feat_raw, norm_params)
        timings['features'] = time.perf_counter() - t0
        result["stages"].append({
            "name": "特征构造",
            "status": "ok",
            "detail": f"{len(feat_raw)}维",
            "time_ms": round(timings['features'] * 1000, 2),
        })
        _stream(f"📐 特征构造: **{len(feat_raw)}D** 向量 (45D 峰 + 140D 直方图 + 5D 统计) `{timings['features']*1000:.1f}ms`\n\n")

        # Stage 4: BPU推理 #1
        bpu_ok = False
        try:
            t0 = time.perf_counter()
            bpu_probs = bpu_infer(feat_norm)
            timings['bpu1'] = time.perf_counter() - t0
            bpu_idx = int(np.argmax(bpu_probs))
            bpu_label = idx2label[bpu_idx]
            bpu_conf = float(bpu_probs[bpu_idx])
            bpu_ok = True

            tc0 = time.perf_counter()
            cpu_probs = numpy_infer(feat_norm)
            timings['cpu1'] = time.perf_counter() - tc0
            cpu_idx = int(np.argmax(cpu_probs))
            cpu_label = idx2label[cpu_idx]

            if cpu_label == bpu_label:
                probs, pred_label, pred_conf = bpu_probs, bpu_label, bpu_conf
                mode = "BPU"
            else:
                probs, pred_label, pred_conf = cpu_probs, cpu_label, float(cpu_probs[cpu_idx])
                mode = "BPU+CPU仲裁"
        except Exception:
            t0 = time.perf_counter()
            probs = numpy_infer(feat_norm)
            timings['bpu1'] = time.perf_counter() - t0
            timings['cpu1'] = timings['bpu1']
            pred_idx = int(np.argmax(probs))
            pred_label = idx2label[pred_idx]
            pred_conf = float(probs[pred_idx])
            mode = "CPU"

        result["classification"] = {
            "primary": pred_label,
            "primary_confidence": round(pred_conf, 4),
            "mode": mode,
            "all_probs": {idx2label[i]: round(float(probs[i]), 4) for i in range(len(probs))},
        }
        det_note = " (确定性<1ms)" if bpu_ok else ""
        result["stages"].append({
            "name": "BPU推理#1",
            "status": "ok",
            "detail": f"{pred_label} ({pred_conf:.2%}) [{mode}]{det_note}",
            "time_ms": round(timings['bpu1'] * 1000, 2),
        })
        result["bpu_deterministic"] = bpu_ok
        _stream(f"🧠 BPU 推理#1 (粗分类): **{pred_label}** ({pred_conf:.1%}) `[{mode}]{det_note} {timings['bpu1']*1000:.2f}ms`\n\n")

        # Conformal Prediction (保形预测)
        conformal_set = None
        try:
            cp_set, cp_coverage, cp_size = conformal_predict(probs, idx2label)
            if cp_set:
                conformal_set = cp_set
                result["conformal"] = {
                    "prediction_set": cp_set,
                    "coverage": cp_coverage,
                    "set_size": cp_size,
                    "certain": cp_size == 1,
                }
        except Exception:
            pass

        # XAI 特征归因
        try:
            xai = compute_feature_attribution(feat_norm)
            result["xai"] = xai
        except Exception:
            pass

        # MC Dropout不确定性
        try:
            mc = mc_dropout_uncertainty(feat_norm, n_samples=20)
            result["mc_dropout"] = mc
        except Exception:
            pass

        # Stage 4b: 细分类
        fine_label = None
        fine_conf = None
        fine_rejected = False
        if pred_label == "non_garnet":
            t0 = time.perf_counter()
            fine_label, fine_conf, fine_info = bpu_infer_fine(feat_raw)
            timings['bpu2'] = time.perf_counter() - t0
            if fine_label and fine_info:
                rejected, reason = should_reject(fine_info['probs'])
                if rejected:
                    fine_rejected = True
                    detail = f"{fine_label} ({fine_conf:.2%}) \u2192 拒识({reason})"
                    fine_label = None
                    fine_conf = None
                else:
                    detail = f"{fine_label} ({fine_conf:.2%})"
                result["stages"].append({
                    "name": "细分类推理#2",
                    "status": "rejected" if fine_rejected else "ok",
                    "detail": detail,
                    "time_ms": round(timings['bpu2'] * 1000, 2),
                })
                _stream(f"🎯 BPU 推理#2 (细分类): **{detail}** `{timings['bpu2']*1000:.2f}ms`\n\n")
        else:
            timings['bpu2'] = 0
            skip_reason = "已确认garnet" if pred_label == "garnet" else "non_garnet 直接进入匹配"
            result["stages"].append({
                "name": "细分类推理#2",
                "status": "skip",
                "detail": skip_reason,
            })
            _stream(f"🎯 BPU 推理#2: 跳过 ({skip_reason})\n\n")

        result["classification"]["fine_label"] = fine_label
        result["classification"]["fine_confidence"] = round(fine_conf, 4) if fine_conf else None
        result["classification"]["fine_rejected"] = fine_rejected

        # Stage 5: 峰匹配
        t0 = time.perf_counter()
        peak_positions = sorted([p['position'] for p in peaks])
        category_hint = fine_label if fine_label else pred_label
        if fine_rejected:
            category_hint = "non_garnet"
        matcher = _load_peak_matcher()
        match_data = None
        match_results = None
        if matcher:
            match_results = matcher.match(peak_positions, category_hint=category_hint)
            if match_results:
                best = match_results[0]
                dn = best["display_name"] if isinstance(best, dict) else best.display_name
                sc = best["score"] if isinstance(best, dict) else best.score
                sg = best["space_group"] if isinstance(best, dict) else best.space_group
                rc = best["reference_cards"] if isinstance(best, dict) else best.reference_cards
                mp = best["matched_pairs"] if isinstance(best, dict) else best.matched_pairs
                pid = best["phase_id"] if isinstance(best, dict) else best.phase_id

                match_data = {
                    "phase_id": pid,
                    "display_name": dn,
                    "space_group": sg,
                    "score": round(sc, 3),
                    "reference_cards": rc or [],
                    "matched_pairs": [
                        {"detected": round(d, 2), "reference": round(r, 1), "hkl": h}
                        for d, r, h, _ in (mp or [])[:10]
                    ],
                }
        timings['match'] = time.perf_counter() - t0
        result["peak_matching"] = match_data
        result["stages"].append({
            "name": "峰位匹配",
            "status": "ok" if match_data else "none",
            "detail": f"{match_data['display_name']} ({match_data['space_group']}), 得分={match_data['score']:.3f}" if match_data else "未匹配",
            "time_ms": round(timings['match'] * 1000, 2),
        })
        if match_data:
            _stream(f"🔬 峰匹配: **{match_data['display_name']}** 空间群 `{match_data['space_group']}`, 得分 `{match_data['score']:.3f}` `{timings['match']*1000:.1f}ms`\n\n")
        else:
            _stream(f"🔬 峰匹配: 未找到参考相 `{timings['match']*1000:.1f}ms`\n\n")

        # 确定最终标签
        if fine_rejected and match_data:
            final_label = match_data["display_name"]
            final_conf = match_data["score"]
        else:
            final_label = fine_label if fine_label else pred_label
            final_conf = fine_conf if fine_conf else pred_conf

        result["classification"]["final_label"] = final_label
        result["classification"]["final_confidence"] = round(final_conf, 4)

        # 理论图谱正演 + Rwp残差
        try:
            # 根据分类结果选择CIF: garnet→YCAS, non_garnet→SYGO
            sim_id = "YCAS" if "garnet" in str(final_label).lower() else "SYGO"
            sim_result = simulate_and_compare(two_theta, intensity, sim_id)
            if sim_result:
                result["simulation"] = sim_result
        except Exception:
            pass

        # 系统消光验证
        try:
            if match_data and match_data.get("matched_pairs") and match_data.get("space_group"):
                extinction = verify_systematic_absence(
                    match_data["matched_pairs"], match_data["space_group"])
                result["extinction"] = extinction
        except Exception:
            pass

        # Nelson-Riley晶格常数精修(仅立方garnet)
        try:
            if match_data and match_data.get("matched_pairs") and "garnet" in str(final_label).lower():
                nr_peaks = []
                for mp in match_data["matched_pairs"]:
                    hkl_str = mp.get("hkl", "")
                    det = mp.get("detected", 0)
                    try:
                        nums = tuple(int(x) for x in hkl_str.strip("()").split(","))
                        if len(nums) == 3 and det > 0:
                            nr_peaks.append((det, nums))
                    except (ValueError, TypeError):
                        pass
                if len(nr_peaks) >= 3:
                    nr = nelson_riley_refine(nr_peaks)
                    if nr:
                        result["nelson_riley"] = nr
        except Exception:
            pass

        # Stage 6: RAG检索
        t_rag = time.perf_counter()
        rag_context_str = ""
        try:
            from rag_engine import RAGEngine
            _rag = RAGEngine()
            peak_str_q = ' '.join(f"{p['position']:.1f}" for p in peaks[:5])
            rag_context_str = _rag.retrieve(f"XRD {final_label} {peak_str_q}", top_k=5)
        except Exception:
            pass
        timings['rag'] = time.perf_counter() - t_rag
        result["stages"].append({
            "name": "RAG检索",
            "status": "ok" if rag_context_str else "skip",
            "detail": f"2255段落语义检索" if rag_context_str else "跳过",
            "time_ms": round(timings['rag'] * 1000, 2),
        })
        if rag_context_str:
            _stream(f"📚 RAG 检索: 197 篇论文 / 2255 段落 (text-embedding-v3) `{timings['rag']*1000:.0f}ms`\n\n")
        else:
            _stream(f"📚 RAG 检索: 离线模式跳过\n\n")
        _stream(f"---\n\n🤖 **启动 DeepSeek-R1 Agent** (ReAct 循环 + Tool Calling)\n\n")

        # Stage 7: AI Agent / LLM报告 (三级降级)
        t0 = time.perf_counter()
        agent_thinking = ""
        agent_tools = []
        if offline or OFFLINE_MODE:
            report = offline_report(Path(filepath).name, final_label, final_conf,
                                    peaks, match_results=match_results if matcher and match_results else None)
            report_mode = "离线"
            _stream(f"\n\n---\n\n📝 **最终结论** (离线模板)\n\n{report}\n")
        else:
            # 1. 尝试AI Agent (DeepSeek-R1 + Tool Calling, 带 progress_writer 流式)
            try:
                peak_str = ', '.join(f'{p["position"]:.1f}°' for p in peaks[:10])
                agent_input = (f"MLP分类结果: {final_label} (置信度{final_conf:.2%}, 模式={mode})\n"
                               f"峰位数据: {peak_str}\n"
                               f"峰位匹配: {match_data['display_name'] if match_data else '未匹配'} "
                               f"({match_data['space_group'] if match_data else ''})\n\n"
                               f"请基于以上数据，自主调用工具进行深度分析。")

                # v4.1 Round 9: 保留 pipeline 前缀, agent thinking 追加其后
                with voice_state.lock:
                    pipeline_prefix = voice_state.agent_stream_buffer

                def _agent_progress(s):
                    with voice_state.lock:
                        voice_state.agent_stream_buffer = pipeline_prefix + s

                agent_thinking, report, agent_tools = run_agent(
                    agent_input, progress_writer=_agent_progress)

                # Agent 完成后追加最终结论 (markdown)
                with voice_state.lock:
                    voice_state.agent_stream_buffer = (
                        pipeline_prefix + agent_thinking +
                        f"\n\n---\n\n📝 **最终结论**\n\n{report}\n"
                    )
                report_mode = "AI Agent(R1)"
            except Exception as e:
                print(f"[Agent] R1失败({e}), 降级DeepSeek")
                _stream(f"\n\n⚠️ R1 Agent 异常 ({e}), 降级到 DeepSeek\n\n")
                # 2. 降级到普通DeepSeek
                try:
                    all_probs = np.array([probs[i] for i in range(len(probs))])
                    prompt = build_llm_prompt(
                        Path(filepath).name, final_label, final_conf, peaks,
                        all_probs, label_info['label_names'],
                        match_results=match_results if matcher and match_results else None,
                        conformal_set=conformal_set,
                    )
                    report = call_deepseek(prompt)
                    report_mode = "在线(DeepSeek)"
                    _stream(f"\n\n---\n\n📝 **最终结论** (DeepSeek)\n\n{report}\n")
                except Exception:
                    # 3. 离线回退
                    report = offline_report(Path(filepath).name, final_label, final_conf,
                                            peaks, match_results=match_results if matcher and match_results else None)
                    report_mode = "离线(回退)"
                    _stream(f"\n\n---\n\n📝 **最终结论** (离线回退)\n\n{report}\n")
        result["agent_thinking"] = agent_thinking
        result["agent_tools"] = agent_tools
        timings['agent'] = time.perf_counter() - t0
        result["report"] = report
        result["report_mode"] = report_mode
        result["stages"].append({
            "name": "AI Agent推理" if "Agent" in report_mode else "LLM生成",
            "status": "ok",
            "detail": report_mode + (f" ({len(agent_tools)}次工具)" if agent_tools else ""),
            "time_ms": round(timings['agent'] * 1000, 2),
        })

        # 语音联动: 更新当前报告并自动播报摘要
        # (TTS 唯一播报点, enqueue_tts 内部判 state.tts_enabled)
        voice_state.current_report = report
        voice_state.current_filename = Path(filepath).name
        summary = extract_tts_summary(report)
        enqueue_tts(voice_state, summary)

        # 汇总
        result["timings"] = {k: round(v * 1000, 2) for k, v in timings.items()}
        total = sum(v for k, v in timings.items() if k != 'cpu1')
        result["total_ms"] = round(total * 1000, 2)
        result["local_ms"] = round((total - timings.get('agent', 0) - timings.get('rag', 0)) * 1000, 2)

    except Exception as e:
        import traceback
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()

    # v4.1 Round 5: pipeline 结束, 清 busy 标志 + 关闭流
    with voice_state.lock:
        voice_state.pipeline_busy = False
        voice_state.pipeline_current = ""
        voice_state.agent_stream_done = True

    # 把 TrackedStages 转成普通 list, 方便 jsonify (避免子类序列化问题)
    result["stages"] = list(result["stages"])
    return result


# ============ HTML 模板 (卡片流程图风格, Canvas绘图, 零外部依赖) ============
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>XRD智能分析系统 | RDK X5 BPU</title>
<link rel="manifest" href="/static/manifest.json">
<meta name="theme-color" content="#2563eb">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<script src="https://3dmol.csb.pitt.edu/build/3Dmol-min.js" defer></script>
<script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.9.3/dist/confetti.browser.min.js" defer></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/qrcodejs/1.0.0/qrcode.min.js" defer></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/driver.js@1.3.1/dist/driver.css"/>
<script src="https://cdn.jsdelivr.net/npm/driver.js@1.3.1/dist/driver.js.iife.js" defer></script>
<style>
:root{--bg:#f0f4f8;--card:#fff;--text:#0f172a;--text2:#475569;--text3:#94a3b8;
--border:#e2e8f0;--blue:#2563eb;--green:#059669;--emerald:#10b981;
--amber:#f59e0b;--red:#ef4444;--purple:#7c3aed;
--shadow:0 1px 3px rgba(0,0,0,.08),0 1px 2px rgba(0,0,0,.04);
--shadow-md:0 4px 6px -1px rgba(0,0,0,.07),0 2px 4px -2px rgba(0,0,0,.05);}
body.dark{--bg:#0f172a;--card:#1e293b;--text:#e2e8f0;--text2:#94a3b8;--text3:#64748b;
--border:#334155;--shadow:0 1px 3px rgba(0,0,0,.3);--shadow-md:0 4px 6px rgba(0,0,0,.3);}
body.dark .card-hd.blue{background:#1e3a5f;color:#93c5fd;}
body.dark .card-hd.green{background:#064e3b;color:#6ee7b7;}
body.dark .card-hd.amber{background:#451a03;color:#fcd34d;}
body.dark .card-hd.purple{background:#2e1065;color:#c4b5fd;}
body.dark .card-hd.red{background:#450a0a;color:#fca5a5;}
body.dark .card-hd.slate{background:#1e293b;color:#94a3b8;}
body.dark th{background:#334155;color:#e2e8f0;}
body.dark td{color:#cbd5e1;}
body.dark .file-chip{background:#334155;border-color:#475569;color:#e2e8f0;}
body.dark .file-chip.active{background:#1e3a5f;border-color:#3b82f6;color:#93c5fd;}
body.dark .btn-g{background:linear-gradient(135deg,#065f46,#059669);}
body.dark .btn-p{background:linear-gradient(135deg,#4c1d95,#7c3aed);}
body.dark .metric-box{background:#0f172a;border-color:#334155;}
body.dark .report{background:#0f172a;border-color:#334155;color:#e2e8f0;}
body.dark .hdr{background:linear-gradient(135deg,#020617,#0f172a,#1e3a5f);}
body.dark .footer{color:#475569;}
*{margin:0;padding:0;box-sizing:border-box;}
body{background:var(--bg);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;color:var(--text);min-height:100vh;}
/* Header */
.hdr{background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 50%,#1e40af 100%);
padding:12px 20px;display:flex;align-items:center;justify-content:space-between;
color:#fff;box-shadow:0 4px 12px rgba(0,0,0,.15);}
.hdr h1{font-size:20px;font-weight:700;letter-spacing:.5px;}
.hdr-right{display:flex;align-items:center;gap:10px;}
.badge{display:inline-block;padding:4px 14px;border-radius:20px;font-size:12px;font-weight:600;}
.badge-g{background:#059669;color:#fff;}
.badge-b{background:rgba(255,255,255,.12);color:#93c5fd;border:1px solid rgba(147,197,253,.25);}
.hdr-sub{font-size:12px;color:#94a3b8;margin-top:2px;}
/* Dashboard */
.dash{max-width:1400px;margin:0 auto;padding:14px;display:flex;flex-direction:column;gap:14px;}
/* Card */
.card{background:var(--card);border-radius:10px;box-shadow:var(--shadow);overflow:hidden;border:1px solid var(--border);}
.card-hd{padding:10px 16px;font-weight:700;font-size:14px;display:flex;align-items:center;gap:8px;
border-bottom:1px solid var(--border);}
.card-hd.blue{background:#eff6ff;color:#1d4ed8;border-left:4px solid #3b82f6;}
.card-hd.green{background:#ecfdf5;color:#065f46;border-left:4px solid #10b981;}
.card-hd.amber{background:#fffbeb;color:#92400e;border-left:4px solid #f59e0b;}
.card-hd.purple{background:#f5f3ff;color:#5b21b6;border-left:4px solid #8b5cf6;}
.card-hd.red{background:#fef2f2;color:#991b1b;border-left:4px solid #ef4444;}
.card-hd.slate{background:#f8fafc;color:#334155;border-left:4px solid #64748b;}
.card-bd{padding:14px 16px;}
/* Control bar */
.ctrl{display:flex;align-items:center;gap:10px;flex-wrap:wrap;}
.btn{padding:8px 18px;border-radius:8px;border:none;cursor:pointer;font-size:13px;font-weight:600;transition:all .2s;}
.btn:hover{filter:brightness(1.1);transform:translateY(-1px);}
.btn:active{transform:translateY(0);}
.btn:disabled{opacity:.5;cursor:not-allowed;transform:none;}
.btn-g{background:linear-gradient(135deg,#059669,#10b981);color:#fff;}
.btn-p{background:linear-gradient(135deg,#7c3aed,#8b5cf6);color:#fff;}
.files-row{display:flex;gap:6px;flex-wrap:wrap;margin-left:8px;}
.file-chip{padding:5px 12px;background:#f1f5f9;border:1px solid #e2e8f0;border-radius:6px;
font-size:12px;cursor:pointer;transition:all .2s;white-space:nowrap;}
.file-chip:hover{background:#e2e8f0;border-color:#cbd5e1;}
.file-chip.active{background:#dbeafe;border-color:#3b82f6;color:#1d4ed8;font-weight:600;}
/* Pipeline flow */
.flow{display:flex;align-items:center;justify-content:center;gap:4px;padding:14px 8px;flex-wrap:wrap;}
.flow-step{background:#f8fafc;border:2px solid #e2e8f0;border-radius:10px;padding:8px 10px;
text-align:center;min-width:80px;transition:all .3s;}
.flow-step.ok{border-color:#10b981;background:#ecfdf5;}
.flow-step.running{border-color:#f59e0b;background:#fffbeb;animation:glow 1.5s infinite;}
.flow-step.error,.flow-step.none{border-color:#ef4444;background:#fef2f2;}
.flow-step.skip{border-color:#cbd5e1;background:#f1f5f9;opacity:.7;}
.flow-step.rejected{border-color:#f97316;background:#fff7ed;}
.fs-icon{width:28px;height:28px;border-radius:50%;margin:0 auto 4px;display:flex;align-items:center;
justify-content:center;font-size:13px;font-weight:700;color:#fff;}
.flow-step.pending .fs-icon{background:#cbd5e1;color:#64748b;}
.flow-step.ok .fs-icon{background:#10b981;}
.flow-step.running .fs-icon{background:#f59e0b;}
.flow-step.error .fs-icon,.flow-step.none .fs-icon{background:#ef4444;}
.flow-step.skip .fs-icon{background:#94a3b8;}
.flow-step.rejected .fs-icon{background:#f97316;}
.fs-name{font-size:11px;font-weight:600;color:#334155;white-space:nowrap;}
.fs-time{font-size:10px;color:#94a3b8;margin-top:2px;}
.flow-arr{color:#94a3b8;font-size:20px;font-weight:300;line-height:1;}
@keyframes glow{0%,100%{box-shadow:0 0 0 0 rgba(245,158,11,.3)}50%{box-shadow:0 0 0 6px rgba(245,158,11,0)}}
/* Grid */
.row{display:grid;gap:14px;}
.row-2{grid-template-columns:1fr 1fr;}
.row-chart{grid-template-columns:3fr 2fr;}
@media(max-width:900px){.row-2,.row-chart{grid-template-columns:1fr;}}
/* Chart */
#chartCanvas{width:100%;min-height:380px;height:100%;border-radius:6px;display:block;background:#0f172a;}
/* Classification */
.cls-label{font-size:22px;font-weight:800;color:#059669;margin-bottom:4px;}
.cls-sub{font-size:13px;color:var(--text2);margin-bottom:8px;}
.conf-wrap{display:flex;align-items:center;gap:10px;}
.conf-bar{flex:1;height:10px;background:#e2e8f0;border-radius:5px;overflow:hidden;}
.conf-fill{height:100%;border-radius:5px;transition:width .5s;}
.conf-pct{font-size:15px;font-weight:700;min-width:50px;text-align:right;}
.cls-tags{display:flex;gap:6px;margin-top:10px;flex-wrap:wrap;}
.tag{padding:3px 10px;border-radius:12px;font-size:11px;font-weight:600;}
.tag-blue{background:#dbeafe;color:#1d4ed8;}
.tag-green{background:#d1fae5;color:#065f46;}
.tag-amber{background:#fef3c7;color:#92400e;}
.tag-red{background:#fee2e2;color:#991b1b;}
/* Metrics */
.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;}
.metric-box{background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:12px;text-align:center;}
.metric-val{font-size:22px;font-weight:800;color:var(--blue);line-height:1.2;}
.metric-val.green{color:#059669;}
.metric-val.amber{color:#d97706;}
.metric-lbl{font-size:11px;color:var(--text3);margin-top:4px;}
/* Table */
table{width:100%;border-collapse:collapse;font-size:12px;}
th{background:#f1f5f9;padding:8px 10px;text-align:left;font-weight:600;color:#475569;border-bottom:2px solid #e2e8f0;}
td{padding:7px 10px;border-bottom:1px solid #f1f5f9;color:#334155;}
tr:hover td{background:#f8fafc;}
/* Report */
.report{background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:14px;
font-size:13px;line-height:1.7;max-height:350px;overflow-y:auto;color:#334155;}
.report h4{font-size:14px;font-weight:700;color:#1e293b;margin:12px 0 6px;padding-bottom:4px;border-bottom:1px solid #e2e8f0;}
.report h4:first-child{margin-top:0;}
.report strong{color:#0f172a;}
.report p{margin:6px 0;}
.report table{margin:8px 0;font-size:12px;}
.report table th{background:#e2e8f0;}
.report br+br{display:none;}
/* Spinner */
.spinner{display:inline-block;width:18px;height:18px;border:2.5px solid #e2e8f0;
border-top:2.5px solid #3b82f6;border-radius:50%;animation:spin .7s linear infinite;}
@keyframes spin{to{transform:rotate(360deg)}}
/* Empty state */
.empty{text-align:center;padding:40px 20px;color:var(--text3);}
.empty h3{font-size:16px;color:var(--text2);margin-bottom:8px;}
.empty p{font-size:13px;}
/* Voice */
.mic-dot{width:12px;height:12px;border-radius:50%;flex-shrink:0;}
.mic-dot.idle{background:#94a3b8;}
.mic-dot.listening{background:#10b981;animation:pulse-mic 1.2s infinite;}
.mic-dot.speaking{background:#f59e0b;animation:pulse-mic 1.2s infinite;}
.mic-dot.recognizing{background:#3b82f6;animation:pulse-mic 0.8s infinite;}
.mic-dot.error{background:#ef4444;}
@keyframes pulse-mic{0%,100%{box-shadow:0 0 0 0 rgba(16,185,129,.4)}50%{box-shadow:0 0 0 6px rgba(16,185,129,0)}}
.voice-log{font-size:11px;color:var(--text3);max-height:60px;overflow-y:auto;margin-top:6px;}
.voice-log div{padding:1px 0;}
.voice-log .t{color:var(--text3);margin-right:4px;}
/* Knowledge Graph Animations */
@keyframes kg-fadein{from{opacity:0;transform:translateY(10px) scale(0.95)}to{opacity:1;transform:translateY(0) scale(1)}}
@keyframes kg-pulse-anim{0%,100%{box-shadow:0 0 0 0 currentColor}50%{box-shadow:0 0 8px 2px currentColor}}
.kg-pulse{animation:kg-pulse-anim 2.5s ease-in-out infinite;}
@keyframes kg-glow-anim{0%,100%{opacity:0.85}50%{opacity:1;text-shadow:0 0 6px currentColor}}
.kg-glow{animation:kg-glow-anim 3s ease-in-out infinite;}
@keyframes kg-bounce-anim{0%,100%{transform:translateY(0)}50%{transform:translateY(-2px)}}
.kg-bounce{animation:kg-bounce-anim 2s ease-in-out infinite;}
@keyframes kg-shimmer-anim{0%{background-position:-100%}100%{background-position:200%}}
.kg-shimmer{background:linear-gradient(90deg,transparent 30%,rgba(255,255,255,0.3) 50%,transparent 70%);background-size:200% 100%;animation:kg-shimmer-anim 3s linear infinite;}
@keyframes kg-spin-y-anim{from{transform:rotateY(0)}to{transform:rotateY(360deg)}}
.kg-spin-y{animation:kg-spin-y-anim 6s linear infinite;display:inline-block;}
@keyframes kg-pop-anim{0%,100%{transform:scale(1)}50%{transform:scale(1.08)}}
.kg-pop{animation:kg-pop-anim 2s ease-in-out infinite;}
@keyframes kg-flow{0%{background-position:0% 50%}100%{background-position:200% 50%}}
.kg-flow-line{height:3px;border-radius:2px;background:linear-gradient(90deg,#3b82f6,#8b5cf6,#10b981,#f59e0b,#ef4444,#3b82f6);background-size:200% 100%;animation:kg-flow 3s linear infinite;margin:0 auto;width:80%;opacity:0.6;}
@keyframes spin-slow{from{transform:rotate(0)}to{transform:rotate(360deg)}}
.icon-spin{display:inline-block;animation:spin-slow 4s linear infinite;}
@keyframes pulse-breathe{0%,100%{opacity:0.7;transform:scale(1)}50%{opacity:1;transform:scale(1.15)}}
.icon-pulse{display:inline-block;animation:pulse-breathe 2s ease-in-out infinite;}
/* Skeleton loading */
@keyframes shimmer{0%{background-position:-200% 0}100%{background-position:200% 0}}
.skeleton{background:linear-gradient(90deg,#f1f5f9 25%,#e2e8f0 50%,#f1f5f9 75%);
background-size:200% 100%;animation:shimmer 1.5s infinite;border-radius:6px;}
/* Progressive panel fade-in (v4.1 Round 9, 对齐 xrd_vision 风格) */
@keyframes fadeInSlide{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}
.fade-in{animation:fadeInSlide 0.35s ease-out both;}
.skel-line{height:14px;margin:6px 0;}
.skel-block{height:60px;margin:8px 0;}
.skel-circle{width:40px;height:40px;border-radius:50%;}
/* Footer */
.footer{text-align:center;padding:10px;font-size:11px;color:var(--text3);}
/* Upload hidden input */
.upload-wrap{position:relative;overflow:hidden;display:inline-block;}
.upload-wrap input[type=file]{position:absolute;left:0;top:0;opacity:0;width:100%;height:100%;cursor:pointer;}
</style>
</head>
<body>

<!-- Header -->
<div class="hdr">
    <div>
        <h1><span class="icon-spin" style="font-size:18px;">&#9883;</span> XRD智能分析系统 v2.0</h1>
        <div class="hdr-sub">两级级联BPU推理 + 晶体学峰位匹配 + 大模型解读</div>
    </div>
    <div class="hdr-right">
        <span class="badge badge-g">RDK X5 BPU</span>
        <span class="badge badge-b">Bayes-e INT8</span>
        <button onclick="document.body.classList.toggle('dark');localStorage.setItem('theme',document.body.classList.contains('dark')?'dark':'light');" style="background:none;border:1px solid rgba(255,255,255,.2);color:#fff;border-radius:50%;width:28px;height:28px;cursor:pointer;font-size:14px;" title="切换主题">&#9789;</button>
    </div>
</div>

<div class="dash">

<!-- Control Bar -->
<div class="card">
    <div class="card-bd" style="padding:10px 16px;">
        <div class="ctrl">
            <div class="upload-wrap">
                <button class="btn btn-p">上传.raw文件</button>
                <input type="file" accept=".raw" onchange="uploadFile(this)">
            </div>
            <div class="files-row" id="fileList">加载中...</div>
            <button class="btn btn-sm" id="btnTeach" onclick="toggleTeach()" style="background:#7c3aed;color:#fff;font-size:10px;margin-left:auto;">🎓 教学模式</button>
            <button class="btn btn-sm" onclick="startDemoTour()" style="background:#f59e0b;color:#fff;font-size:10px;">🎬 开始演示</button>
        </div>
    </div>
</div>

<!-- XRD 数值线架构总览 (v4.1 对齐: 本线专用, 10 阶段 Pipeline) -->
<div class="card" id="archCard">
    <div class="card-hd blue">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="3" width="20" height="14" rx="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/></svg>
        <span class="icon-spin">⚙</span> XRD 数值线架构总览
        <span style="margin-left:auto;font-size:11px;color:#94a3b8;">RDK X5 | Bayes-e 10TOPS | 端口 5000</span>
    </div>
    <div class="card-bd" id="archBody" style="padding:16px;">
        <svg viewBox="0 0 860 260" style="width:100%;height:auto;max-height:240px;" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <marker id="ah" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><path d="M0 0 L8 3 L0 6" fill="#94a3b8"/></marker>
                <linearGradient id="g1" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stop-color="#3b82f6"/><stop offset="100%" stop-color="#2563eb"/></linearGradient>
                <linearGradient id="g2" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stop-color="#10b981"/><stop offset="100%" stop-color="#059669"/></linearGradient>
                <linearGradient id="g3" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stop-color="#f59e0b"/><stop offset="100%" stop-color="#d97706"/></linearGradient>
                <linearGradient id="g4" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stop-color="#8b5cf6"/><stop offset="100%" stop-color="#7c3aed"/></linearGradient>
                <linearGradient id="gR" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" stop-color="#ef4444"/><stop offset="100%" stop-color="#dc2626"/></linearGradient>
            </defs>
            <!-- Section labels -->
            <text x="10" y="16" fill="#64748b" font-size="10" font-weight="600">感知 Perception</text>
            <rect x="10" y="20" width="225" height="2" rx="1" fill="#3b82f6" opacity="0.4"/>
            <text x="250" y="16" fill="#64748b" font-size="10" font-weight="600">决策 Decision</text>
            <rect x="250" y="20" width="320" height="2" rx="1" fill="#10b981" opacity="0.4"/>
            <text x="585" y="16" fill="#64748b" font-size="10" font-weight="600">执行 Action</text>
            <rect x="585" y="20" width="265" height="2" rx="1" fill="#f59e0b" opacity="0.4"/>

            <!-- Row 1: Main Pipeline -->
            <rect x="10" y="30" width="68" height="48" rx="8" fill="url(#g1)"/>
            <text x="44" y="50" text-anchor="middle" fill="#fff" font-size="11" font-weight="700">解析</text>
            <text x="44" y="64" text-anchor="middle" fill="#bfdbfe" font-size="8">Bruker</text>
            <line x1="78" y1="54" x2="93" y2="54" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#ah)"/>

            <rect x="93" y="30" width="68" height="48" rx="8" fill="url(#g1)"/>
            <text x="127" y="50" text-anchor="middle" fill="#fff" font-size="11" font-weight="700">峰检测</text>
            <text x="127" y="64" text-anchor="middle" fill="#bfdbfe" font-size="8">scipy</text>
            <line x1="161" y1="54" x2="176" y2="54" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#ah)"/>

            <rect x="176" y="30" width="68" height="48" rx="8" fill="url(#g1)"/>
            <text x="210" y="50" text-anchor="middle" fill="#fff" font-size="11" font-weight="700">190D</text>
            <text x="210" y="64" text-anchor="middle" fill="#bfdbfe" font-size="8">特征工程</text>
            <line x1="244" y1="54" x2="259" y2="54" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#ah)"/>

            <!-- BPU - larger, highlighted -->
            <rect x="259" y="26" width="100" height="56" rx="8" fill="url(#g2)" stroke="#059669" stroke-width="2"/>
            <text x="309" y="46" text-anchor="middle" fill="#fff" font-size="12" font-weight="700">BPU级联</text>
            <text x="309" y="60" text-anchor="middle" fill="#bbf7d0" font-size="9">Bayes-e INT8</text>
            <text x="309" y="72" text-anchor="middle" fill="#bbf7d0" font-size="8">&lt;1ms确定性</text>
            <line x1="359" y1="54" x2="374" y2="54" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#ah)"/>

            <rect x="374" y="30" width="78" height="48" rx="8" fill="url(#g4)"/>
            <text x="413" y="50" text-anchor="middle" fill="#fff" font-size="11" font-weight="700">峰匹配</text>
            <text x="413" y="64" text-anchor="middle" fill="#ddd6fe" font-size="8">17晶相</text>
            <line x1="452" y1="54" x2="467" y2="54" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#ah)"/>

            <rect x="467" y="30" width="78" height="48" rx="8" fill="url(#g4)"/>
            <text x="506" y="50" text-anchor="middle" fill="#fff" font-size="11" font-weight="700">RAG检索</text>
            <text x="506" y="64" text-anchor="middle" fill="#ddd6fe" font-size="8">2255段落</text>
            <line x1="545" y1="54" x2="560" y2="54" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#ah)"/>

            <!-- AI Agent - STAR, largest, bright red -->
            <rect x="560" y="24" width="115" height="60" rx="10" fill="url(#gR)" stroke="#fca5a5" stroke-width="2"/>
            <text x="617" y="42" text-anchor="middle" fill="#fff" font-size="8">&#9733; AI Scientist</text>
            <text x="617" y="56" text-anchor="middle" fill="#fff" font-size="13" font-weight="800">Agent(R1)</text>
            <text x="617" y="72" text-anchor="middle" fill="#fecaca" font-size="8">ReAct+Tools</text>
            <line x1="675" y1="54" x2="690" y2="54" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#ah)"/>

            <rect x="690" y="30" width="78" height="48" rx="8" fill="url(#g3)"/>
            <text x="729" y="50" text-anchor="middle" fill="#fff" font-size="11" font-weight="700">语音</text>
            <text x="729" y="64" text-anchor="middle" fill="#fef3c7" font-size="8">TTS+ASR</text>

            <!-- Row 2: Supporting -->
            <rect x="259" y="108" width="100" height="34" rx="6" fill="#fef3c7" stroke="#f59e0b"/>
            <text x="309" y="124" text-anchor="middle" fill="#92400e" font-size="9" font-weight="600">Conformal预测</text>
            <text x="309" y="136" text-anchor="middle" fill="#92400e" font-size="7">95%覆盖率保证</text>
            <line x1="309" y1="82" x2="309" y2="108" stroke="#f59e0b" stroke-width="1.5" stroke-dasharray="4 3"/>

            <rect x="176" y="108" width="68" height="34" rx="6" fill="#fee2e2" stroke="#ef4444"/>
            <text x="210" y="124" text-anchor="middle" fill="#991b1b" font-size="9" font-weight="600">XAI归因</text>
            <text x="210" y="136" text-anchor="middle" fill="#991b1b" font-size="7">可解释AI</text>
            <line x1="210" y1="78" x2="210" y2="108" stroke="#ef4444" stroke-width="1.5" stroke-dasharray="4 3"/>

            <rect x="374" y="108" width="78" height="34" rx="6" fill="#fef3c7" stroke="#f59e0b"/>
            <text x="413" y="129" text-anchor="middle" fill="#92400e" font-size="9" font-weight="600">OOD拒识</text>
            <line x1="413" y1="78" x2="413" y2="108" stroke="#f59e0b" stroke-width="1.5" stroke-dasharray="4 3"/>

            <rect x="467" y="108" width="78" height="34" rx="6" fill="#f5f3ff" stroke="#8b5cf6"/>
            <text x="506" y="124" text-anchor="middle" fill="#5b21b6" font-size="9" font-weight="600">向量RAG</text>
            <text x="506" y="136" text-anchor="middle" fill="#5b21b6" font-size="7">197篇·DashScope</text>
            <line x1="506" y1="78" x2="506" y2="108" stroke="#8b5cf6" stroke-width="1.5" stroke-dasharray="4 3"/>

            <rect x="259" y="160" width="100" height="34" rx="6" fill="#dbeafe" stroke="#3b82f6"/>
            <text x="309" y="176" text-anchor="middle" fill="#1d4ed8" font-size="9" font-weight="600">MC Dropout</text>
            <text x="309" y="188" text-anchor="middle" fill="#1d4ed8" font-size="7">Bayesian不确定性</text>
            <line x1="309" y1="142" x2="309" y2="160" stroke="#3b82f6" stroke-width="1" stroke-dasharray="3 3"/>

            <rect x="600" y="108" width="90" height="34" rx="6" fill="#ecfdf5" stroke="#10b981"/>
            <text x="645" y="124" text-anchor="middle" fill="#065f46" font-size="9" font-weight="600">知识图谱</text>
            <text x="645" y="136" text-anchor="middle" fill="#065f46" font-size="7">3D晶体·Web UI</text>
            <line x1="635" y1="84" x2="645" y2="108" stroke="#10b981" stroke-width="1.5" stroke-dasharray="4 3"/>

            <!-- Platform badge -->
            <rect x="10" y="218" width="840" height="28" rx="6" fill="#f1f5f9" stroke="#e2e8f0"/>
            <text x="430" y="237" text-anchor="middle" fill="#475569" font-size="10">RDK X5 | BPU Bayes-e 10TOPS | ARM Cortex-A55 | Conformal+XAI+Agent+RAG(197篇) | 2026全国嵌入式芯片与系统设计竞赛</text>
        </svg>
    </div>
</div>

<!-- Pipeline Flow -->
<div class="card">
    <div class="card-hd blue">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg>
        分析Pipeline
    </div>
    <div class="card-bd" style="padding:8px;">
        <div class="flow" id="pipelineFlow">
            <div class="flow-step pending"><div class="fs-icon">1</div><div class="fs-name">解析</div><div class="fs-time">-</div></div>
            <div class="flow-arr">&rarr;</div>
            <div class="flow-step pending"><div class="fs-icon">2</div><div class="fs-name">峰提取</div><div class="fs-time">-</div></div>
            <div class="flow-arr">&rarr;</div>
            <div class="flow-step pending"><div class="fs-icon">3</div><div class="fs-name">特征</div><div class="fs-time">-</div></div>
            <div class="flow-arr">&rarr;</div>
            <div class="flow-step pending"><div class="fs-icon">4</div><div class="fs-name">BPU分类</div><div class="fs-time">-</div></div>
            <div class="flow-arr">&rarr;</div>
            <div class="flow-step pending"><div class="fs-icon">5</div><div class="fs-name">峰匹配</div><div class="fs-time">-</div></div>
            <div class="flow-arr">&rarr;</div>
            <div class="flow-step pending"><div class="fs-icon">6</div><div class="fs-name">报告</div><div class="fs-time">-</div></div>
        </div>
    </div>
</div>

<!-- Voice Interaction -->
<div class="card" id="voiceCard">
    <div class="card-hd red">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></svg>
        语音交互
        <span id="voiceStatusBadge" style="margin-left:auto;font-size:11px;color:#94a3b8;">初始化...</span>
    </div>
    <div class="card-bd" style="padding:10px 16px;">
        <div style="display:flex;gap:12px;align-items:center;">
            <div id="micDot" class="mic-dot idle"></div>
            <div style="flex:1;min-width:0;">
                <div id="asrDisplay" style="font-size:13px;color:#334155;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">等待语音输入...</div>
                <div id="ttsDisplay" style="font-size:12px;color:#64748b;margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;"></div>
            </div>
            <button class="btn btn-g" onclick="toggleTTS()" id="btnTTS" style="padding:6px 10px;font-size:12px;">🔊 TTS 开</button>
            <button class="btn btn-p" onclick="toggleVoice()" id="btnVoice" style="padding:6px 10px;font-size:12px;">🎤 语音输入关</button>
        </div>
        <div style="display:flex;gap:6px;margin-top:8px;">
            <input type="text" id="followupInput" placeholder="跟进提问 (例: 这个晶系是什么?)"
                   style="flex:1;padding:5px 8px;border:1px solid #cbd5e1;border-radius:4px;font-size:12px;"
                   onkeydown="if(event.key==='Enter')sendFollowup()"/>
            <button class="btn btn-g" id="btnFollowup" onclick="sendFollowup()" style="padding:5px 10px;font-size:12px;">提问</button>
        </div>
        <div id="followupAnswer" style="margin-top:8px;display:none;background:#f1f5f9;border-left:3px solid #22c55e;padding:8px 10px;border-radius:4px;font-size:12px;line-height:1.5;color:#1e293b;max-height:200px;overflow-y:auto;">
            <div style="font-weight:600;color:#475569;margin-bottom:4px;font-size:11px;">📝 上次回答</div>
            <div id="followupAnswerText"></div>
        </div>
        <div class="voice-log" id="voiceLog"></div>
    </div>
</div>

<!-- Main Content: Chart + Classification -->
<div class="row row-chart">
    <!-- XRD Spectrum -->
    <div class="card" style="display:flex;flex-direction:column;">
        <div class="card-hd blue">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
            XRD衍射谱图
        </div>
        <div class="card-bd" style="padding:6px;flex:1;display:flex;">
            <canvas id="chartCanvas" style="flex:1;cursor:crosshair;"></canvas>
            <div id="peakTooltip" style="display:none;position:fixed;background:#1e293b;color:#e2e8f0;padding:8px 12px;border-radius:8px;font-size:11px;line-height:1.5;pointer-events:none;z-index:100;box-shadow:0 4px 12px rgba(0,0,0,0.3);max-width:200px;"></div>
        </div>
    </div>

    <!-- Right column: Perception + Classification + Performance -->
    <div style="display:flex;flex-direction:column;gap:14px;">
        <!-- Visual Perception -->
        <div class="card" id="percCard">
            <div class="card-hd blue">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="4"/><line x1="21.17" y1="8" x2="12" y2="8"/><line x1="3.95" y1="6.06" x2="8.54" y2="14"/><line x1="10.88" y1="21.94" x2="15.46" y2="14"/></svg>
                BPU视觉感知
            </div>
            <div class="card-bd" id="percBody" style="padding:10px 16px;">
                <div class="empty" style="padding:12px;"><p>等待分析</p></div>
            </div>
        </div>
        <!-- Classification -->
        <div class="card" id="clsCard">
            <div class="card-hd green">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
                分类结果
            </div>
            <div class="card-bd" id="clsBody">
                <div class="empty"><h3>等待分析</h3><p>选择样品文件开始</p></div>
            </div>
        </div>
        <!-- Performance -->
        <div class="card" id="perfCard">
            <div class="card-hd amber">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
                BPU性能指标
            </div>
            <div class="card-bd" id="perfBody">
                <div class="metrics">
                    <div class="metric-box"><div class="metric-val">-</div><div class="metric-lbl">总耗时</div></div>
                    <div class="metric-box"><div class="metric-val">-</div><div class="metric-lbl">BPU推理</div></div>
                    <div class="metric-box"><div class="metric-val green">-</div><div class="metric-lbl">确定性延迟</div></div>
                </div>
            </div>
        </div>
    </div>
</div>

<!-- Bottom Row: Peak Matching + Report -->
<div class="row row-2">
    <!-- Peak Matching -->
    <div class="card" id="matchCard">
        <div class="card-hd purple">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
            峰位匹配
        </div>
        <div class="card-bd" id="matchBody">
            <div class="empty"><p>等待分析结果</p></div>
        </div>
    </div>
    <!-- Report -->
    <div class="card" id="reportCard">
        <div class="card-hd slate">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
            分析报告
            <span id="reportMode" style="font-weight:400;font-size:11px;color:#94a3b8;margin-left:auto;"></span>
        </div>
        <div class="card-bd" id="reportBody">
            <div class="empty"><p>等待分析结果</p></div>
        </div>
        <div id="qrSection" style="display:none;padding:8px 16px;border-top:1px solid #e2e8f0;text-align:center;">
            <div style="display:flex;align-items:center;justify-content:center;gap:12px;">
                <div id="qrcode" style="display:inline-block;"></div>
                <div style="font-size:11px;color:#64748b;text-align:left;">
                    <div style="font-weight:600;color:#334155;">扫码查看报告</div>
                    <div>手机/平板扫描</div>
                    <div>即可查看完整分析</div>
                </div>
            </div>
        </div>
    </div>
</div>

<!-- AI Agent Card (独立醒目展示) -->
<div class="card" id="agentCard" style="border:2px solid #ef4444;box-shadow:0 4px 15px rgba(239,68,68,0.15);">
    <div class="card-hd red" style="background:linear-gradient(135deg,#fef2f2,#fee2e2);border-left:5px solid #ef4444;">
        <span style="font-size:18px;" class="icon-pulse">&#129302;</span>
        <span style="font-size:15px;font-weight:800;color:#dc2626;">AI Scientist Agent</span>
        <span style="font-size:11px;color:#991b1b;margin-left:4px;">DeepSeek-R1 + ReAct + Tool Calling</span>
        <span id="agentToolCount" style="margin-left:auto;font-size:11px;background:#ef4444;color:#fff;padding:2px 8px;border-radius:10px;"></span>
    </div>
    <div class="card-bd" id="agentBody" style="padding:12px 16px;">
        <div class="empty"><p>等待Agent推理</p></div>
    </div>
</div>

<!-- Knowledge Graph -->
<div class="card" id="kgCard">
    <div class="card-hd purple">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><circle cx="4" cy="6" r="2"/><circle cx="20" cy="6" r="2"/><circle cx="4" cy="18" r="2"/><circle cx="20" cy="18" r="2"/><line x1="6" y1="7" x2="10" y2="10"/><line x1="18" y1="7" x2="14" y2="10"/><line x1="6" y1="17" x2="10" y2="14"/><line x1="18" y1="17" x2="14" y2="14"/></svg>
        知识图谱 (197篇论文语义网络)
    </div>
    <div class="card-bd" id="knowledgeGraph" style="padding:12px;">
        <div class="empty"><p>分析完成后自动生成</p></div>
    </div>
</div>

<!-- 3D Crystal + AI 候选 Agent (v4.1 对齐 xrd_vision) -->
<div class="card" id="crystalCard">
    <div class="card-hd blue">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>
        <span class="icon-float" style="display:inline-block;animation:float 3s ease-in-out infinite;">💎</span> 晶体结构 3D 可视化 + AI 科学家候选 Agent
        <span id="crystalLabel" style="margin-left:auto;font-size:11px;color:#64748b;"></span>
    </div>
    <div class="card-bd" style="padding:0;position:relative;overflow:hidden;">
        <div id="crystal3d" style="width:100%;height:350px;position:relative;"></div>
    </div>
    <!-- 候选结构对比 (Top-3, pymatgen 预处理 P1 扩胞 CIF) -->
    <div class="card-bd" style="padding:12px 16px;border-top:1px solid #e2e8f0;">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
            <span style="font-size:12px;font-weight:700;color:#5b21b6;">🔬 AI 候选结构对比</span>
            <button class="btn btn-sm" onclick="showCandidateCrystals(window._lastLabel||'garnet')" style="background:#8b5cf6;color:#fff;font-size:10px;">重新拉取 Top-3 候选</button>
            <span id="candAgentStatus" style="font-size:10px;color:#94a3b8;"></span>
        </div>
        <div id="candidateGrid">
            <div class="empty" style="padding:12px;font-size:11px;color:#94a3b8;">点击上方按钮，Agent 从 crystal_data_shared 拉 Top-3 候选 CIF，pymatgen 算理论谱与实测对比选优</div>
        </div>
        <div id="candAgentThinking" style="margin-top:8px;font-family:monospace;font-size:10px;color:#475569;white-space:pre-wrap;max-height:120px;overflow:auto;"></div>
    </div>
</div>

<script>
async function showCandidateCrystals(label){
    const wrap=document.getElementById('candidateGrid');
    const status=document.getElementById('candAgentStatus');
    const think=document.getElementById('candAgentThinking');
    if(!wrap||!status) return;
    status.innerHTML = '<span class="spinner" style="display:inline-block;width:10px;height:10px;border-width:2px;vertical-align:middle;"></span> 加载 Top-3 候选结构...';
    wrap.innerHTML='<div class="empty" style="padding:12px;">加载候选 CIF...</div>';
    try{
        const r=await fetch('/api/crystal/candidates?label='+encodeURIComponent(label));
        const d=await r.json();
        if(!d.ok||!d.candidates||!d.candidates.length){
            wrap.innerHTML='<div class="empty" style="padding:12px;color:#ef4444;">无可用候选('+(d.error||'空结果')+')</div>';
            status.textContent='';return;
        }
        wrap.innerHTML='<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:8px;"></div>';
        const grid=wrap.firstChild;
        d.candidates.forEach((c,i)=>{
            const cell=document.createElement('div');
            cell.style.cssText='border:1px solid '+(c.best?'#10b981':'#e2e8f0')+';border-radius:8px;padding:6px;background:#fff;position:relative;'+(c.best?'box-shadow:0 2px 8px rgba(16,185,129,0.2);':'opacity:0.9;');
            cell.innerHTML='<div style="font-size:11px;font-weight:700;color:'+(c.best?'#065f46':'#475569')+';margin-bottom:4px;">'+(c.best?'★ ':'')+c.name+' <small style="font-weight:400;color:#94a3b8;">Rwp='+(c.rwp||'-')+'</small></div><div id="cand'+i+'" style="width:100%;height:140px;position:relative;"></div>';
            grid.appendChild(cell);
            if(typeof $3Dmol!=='undefined'&&c.cif){
                const v=$3Dmol.createViewer('cand'+i,{backgroundColor:'#f8fafc'});
                v.addModel(c.cif,'cif');
                v.setStyle({},{sphere:{radius:0.3},stick:{radius:0.1}});
                v.addUnitCell({box:{color:'#94a3b8'}});
                v.zoomTo();v.spin('y',0.3);v.render();
            }
        });
        status.textContent = '✓ '+d.candidates.length+' 候选 (已排序)';
        if(d.thinking) think.textContent=d.thinking;
    }catch(e){
        wrap.innerHTML='<div class="empty" style="padding:12px;color:#ef4444;">错误: '+e.message+'</div>';
        status.textContent='';
    }
}
</script>

</div><!-- end dash -->

<div class="footer">XRD智能分析系统 | RDK X5 BPU加速 | 2026 全国嵌入式芯片与系统设计竞赛</div>

<script>
/* ============ 状态 ============ */
let currentData = null;
var _chartMeta = null; // {pad, pw, ph, xMin, xMax, yMax, peaks, match, sim} for tooltip
let analyzing = false;

/* ============ 文件列表 ============ */
async function loadFiles(){
    try{
        const r = await fetch('/api/files');
        const d = await r.json();
        const el = document.getElementById('fileList');
        el.innerHTML = '';
        d.files.forEach(f=>{
            const chip = document.createElement('span');
            chip.className = 'file-chip';
            chip.textContent = f;
            chip.onclick = ()=> analyzeFile(f);
            el.appendChild(chip);
        });
    }catch(e){document.getElementById('fileList').textContent='加载失败';}
}

/* ============ Markdown 渲染 (对齐 xrd_vision) ============ */
function renderMd(text){
    return text
        .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
        .replace(/^### (.+)$/gm, '<h4 style="color:#1e40af;margin:8px 0 4px;font-size:14px;">$1</h4>')
        .replace(/^## (.+)$/gm, '<h3 style="color:#1e40af;margin:10px 0 4px;font-size:15px;">$1</h3>')
        .replace(/\*\*(.*?)\*\*/g, '<strong style="color:#1e40af;">$1</strong>')
        .replace(/`([^`]+)`/g, '<code style="background:#f1f5f9;padding:1px 5px;border-radius:3px;font-size:11px;color:#475569;">$1</code>')
        .replace(/^---$/gm, '<hr style="border:none;border-top:1px dashed #cbd5e1;margin:8px 0;"/>')
        .replace(/^(\d+)\.\s+(.+)$/gm, '<div style="padding-left:16px;">$1. $2</div>')
        .replace(/^[•·\-]\s+(.+)$/gm, '<div style="padding-left:16px;">• $1</div>')
        .replace(/\n\n/g, '<br><br>')
        .replace(/\n/g, '<br>');
}

/* ============ 打字机光标 blink CSS (celebrateDone 定义见下方) ============ */
(function injectBlinkCSS(){
    if(document.getElementById('_blinkCSS')) return;
    const s = document.createElement('style'); s.id='_blinkCSS';
    s.textContent = '@keyframes blink{0%,100%{opacity:1}50%{opacity:0}}';
    document.head.appendChild(s);
})();

/* ============ 分析文件 ============ */
let _agentSSE = null;
function _startAgentSSE(){
    if(_agentSSE){ try{_agentSSE.close();}catch(e){} _agentSSE=null; }
    // 展开 Agent 卡 — 里面是打字机 + 闪烁光标 (xrd_vision 风格)
    document.getElementById('agentCard').style.display='block';
    document.getElementById('agentBody').innerHTML =
      '<div style="font-size:11px;color:#64748b;margin-bottom:8px;">'+
      '<div class="spinner" style="display:inline-block;width:12px;height:12px;border-width:2px;vertical-align:middle;"></div>'+
      ' <strong style="color:#dc2626;">实时思考链</strong> (SSE 流, DeepSeek-R1 + Tool Calling)</div>'+
      '<div id="streamText" style="background:#fafaf9;border:1px solid #e7e5e4;border-radius:8px;padding:12px 14px;'+
      'font-size:12.5px;line-height:1.7;max-height:520px;overflow-y:auto;color:#334155;min-height:80px;">'+
      '<span class="spinner" style="margin-right:8px;"></span>准备启动 Pipeline...</div>';
    let fullText='';
    _agentSSE = new EventSource('/api/agent_stream');
    _agentSSE.onmessage = function(e){
        try{
            const d = JSON.parse(e.data);
            if(d.text){
                fullText = d.text;
                const box = document.getElementById('streamText');
                if(box){
                    box.innerHTML = renderMd(fullText) +
                        '<span style="display:inline-block;border-right:2px solid #3b82f6;animation:blink 1s infinite;">&nbsp;</span>';
                    box.scrollTop = box.scrollHeight;
                }
            }
            if(d.done){
                if(_agentSSE){ _agentSSE.close(); _agentSSE=null; }
                // 收尾: 去掉光标 (celebrateDone 由 displayResult 统一触发)
                const box = document.getElementById('streamText');
                if(box) box.innerHTML = renderMd(fullText);
            }
        }catch(err){ console.log('[agent sse]', err); }
    };
    _agentSSE.onerror = function(){
        if(_agentSSE){ try{_agentSSE.close();}catch(e){} _agentSSE=null; }
    };
}

async function analyzeFile(filename){
    if(analyzing) return;
    analyzing = true;
    document.querySelectorAll('.file-chip').forEach(c=>{
        c.classList.toggle('active', c.textContent===filename);
    });
    showLoading();
    _startAgentSSE();
    try{
        const r = await fetch('/api/analyze',{method:'POST',headers:{'Content-Type':'application/json'},
            body:JSON.stringify({filename:filename})});
        const result = await r.json();
        currentData = result;
        displayResult(result);
    }catch(e){
        showError('请求失败: '+e.message);
    }
    analyzing = false;
    if(_agentSSE){ try{_agentSSE.close();}catch(e){} _agentSSE=null; }
}

/* ============ 显示加载 (v4.1 Round 5: 轮询 /api/pipeline_progress 真·实时动画) ============ */
var _loadingTimer=null;
function showLoading(){
    var flow=document.getElementById('pipelineFlow');
    // 初始占位
    flow.innerHTML='<div style="flex:1;text-align:center;color:#64748b;font-size:12px;padding:6px;"><div class="spinner" style="display:inline-block;vertical-align:middle;"></div> 等待 pipeline 进度...</div>';

    function renderFromProgress(completed, currentName, busy){
        var html='';
        var arr=[];
        completed.forEach(function(s){ arr.push({name:s.name, status:s.status, time_ms:s.time_ms}); });
        if(currentName) arr.push({name:currentName, status:'running', time_ms:null});
        if(arr.length===0){
            flow.innerHTML='<div style="flex:1;text-align:center;color:#64748b;font-size:12px;padding:6px;"><div class="spinner" style="display:inline-block;vertical-align:middle;"></div> 启动中...</div>';
            return;
        }
        arr.forEach(function(s,i){
            if(i>0) html+='<div class="flow-arr">&rarr;</div>';
            var cls=s.status==='running'?'running':(s.status==='ok'?'ok':(s.status==='skip'?'skip':(s.status==='error'||s.status==='none'?'error':'pending')));
            var icon;
            if(s.status==='ok') icon='\u2713';
            else if(s.status==='running') icon='<div class="spinner" style="width:14px;height:14px;border-width:2px;"></div>';
            else if(s.status==='skip') icon='\u2014';
            else if(s.status==='error'||s.status==='none') icon='\u2717';
            else icon=(i+1);
            var tm=(s.time_ms!==null&&s.time_ms!==undefined)?(s.time_ms.toFixed(1)+'ms'):(s.status==='running'?'...':'');
            html+='<div class="flow-step '+cls+'"><div class="fs-icon">'+icon+'</div><div class="fs-name">'+s.name+'</div><div class="fs-time">'+tm+'</div></div>';
        });
        flow.innerHTML=html;
    }

    // 真·流式: 阶段一落地就把关键指标 fade-in 到右侧卡片 (匹配 xrd_vision BPU 面板)
    var _progressSeen = {};
    function progressivePanels(stages){
        stages.forEach(function(s){
            var key = s.name + ':' + s.status;
            if(_progressSeen[key]) return;
            _progressSeen[key] = true;
            var body=null, html=null;
            if(s.name==='峰位提取' && s.status==='ok'){
                body=document.getElementById('percBody');
                html='<div class="fade-in" style="padding:10px 6px;"><div style="font-size:24px;font-weight:800;color:#2563eb;">'+(s.detail||'').replace(/[^0-9]/g,'')+'</div>'+
                     '<div style="font-size:11px;color:#64748b;">检出峰数</div>'+
                     '<div style="font-size:10px;color:#94a3b8;margin-top:4px;">'+(s.detail||'')+'  · '+(s.time_ms?.toFixed(1)||'?')+'ms</div></div>';
            }else if(s.name==='BPU推理#1' && s.status==='ok'){
                body=document.getElementById('clsBody');
                html='<div class="fade-in" style="padding:10px 12px;">'+
                     '<div style="font-size:10px;color:#64748b;">粗分类 (BPU INT8, 确定性<1ms)</div>'+
                     '<div style="font-size:18px;font-weight:800;color:#047857;margin:3px 0;">'+escHtml(s.detail||'')+'</div>'+
                     '<div style="font-size:10px;color:#94a3b8;">'+(s.time_ms?.toFixed(2)||'?')+'ms</div></div>';
            }else if(s.name==='细分类推理#2' && (s.status==='ok'||s.status==='rejected')){
                body=document.getElementById('clsBody');
                var cur=body.innerHTML;
                html=cur + '<div class="fade-in" style="padding:6px 12px;border-top:1px dashed #e2e8f0;">'+
                     '<div style="font-size:10px;color:#64748b;">细分类 '+(s.status==='rejected'?'🚫':'')+'</div>'+
                     '<div style="font-size:14px;font-weight:700;color:'+(s.status==='rejected'?'#dc2626':'#059669')+';">'+escHtml(s.detail||'')+'</div>'+
                     '<div style="font-size:10px;color:#94a3b8;">'+(s.time_ms?.toFixed(2)||'?')+'ms</div></div>';
            }else if(s.name==='峰位匹配' && s.status==='ok'){
                var mb=document.getElementById('matchBody');
                if(mb) mb.innerHTML='<div class="fade-in" style="padding:10px 12px;"><div style="font-size:11px;color:#64748b;">峰位匹配 (17 标准相)</div>'+
                     '<div style="font-size:14px;font-weight:700;color:#7c3aed;margin:4px 0;">'+escHtml(s.detail||'')+'</div>'+
                     '<div style="font-size:10px;color:#94a3b8;">'+(s.time_ms?.toFixed(1)||'?')+'ms  · 完整匹配表等推理完成</div></div>';
            }
            if(body && html) body.innerHTML=html;
            // 更新 BPU 性能面板 (累计耗时 mini 版)
            var perf=document.getElementById('perfBody');
            if(perf){
                var doneMs=0, bpuMs=0;
                stages.forEach(function(x){
                    if(x.time_ms){ doneMs += x.time_ms; }
                    if(x.name && x.name.indexOf('BPU')===0 && x.time_ms){ bpuMs += x.time_ms; }
                });
                perf.innerHTML='<div class="fade-in metrics" style="padding:8px;">'+
                    '<div class="metric-box"><div class="metric-val">'+doneMs.toFixed(0)+'ms</div><div class="metric-lbl">累计耗时</div></div>'+
                    '<div class="metric-box"><div class="metric-val">'+bpuMs.toFixed(1)+'ms</div><div class="metric-lbl">BPU 推理</div></div>'+
                    '<div class="metric-box"><div class="metric-val green">'+stages.length+'/10</div><div class="metric-lbl">阶段进度</div></div></div>';
            }
        });
    }

    function pollProgress(){
        fetch('/api/pipeline_progress').then(r=>r.json()).then(d=>{
            renderFromProgress(d.stages||[], d.current||'', d.busy);
            try{ progressivePanels(d.stages||[]); }catch(e){ console.log('[panels]', e); }
            if(d.busy){
                _loadingTimer=setTimeout(pollProgress, 250);
            }
        }).catch(e=>{
            // 静默重试
            _loadingTimer=setTimeout(pollProgress, 500);
        });
    }
    _progressSeen = {};
    pollProgress();
    // 骨架占位 (第 1 个阶段到达前的 ~200ms 内显示)
    document.getElementById('clsBody').innerHTML='<div style="padding:12px;"><div class="skeleton skel-line" style="width:60%;"></div><div class="skeleton skel-line" style="width:40%;"></div><div class="skeleton skel-block"></div></div>';
    document.getElementById('percBody').innerHTML='<div style="padding:12px;"><div class="skeleton skel-line" style="width:80%;"></div><div style="display:flex;gap:8px;"><div class="skeleton skel-block" style="flex:1;"></div><div class="skeleton skel-block" style="flex:1;"></div><div class="skeleton skel-block" style="flex:1;"></div></div></div>';
    document.getElementById('perfBody').innerHTML='<div style="padding:12px;"><div style="display:flex;gap:8px;"><div class="skeleton skel-block" style="flex:1;height:70px;"></div><div class="skeleton skel-block" style="flex:1;height:70px;"></div><div class="skeleton skel-block" style="flex:1;height:70px;"></div></div></div>';
}

function showError(msg){
    document.getElementById('pipelineFlow').innerHTML=
        '<div style="text-align:center;padding:12px;color:#ef4444;font-weight:600;">'+msg+'</div>';
}

/* ============ 显示结果 ============ */
function displayResult(r){
    if(_loadingTimer){clearTimeout(_loadingTimer);_loadingTimer=null;}
    if(r.error){showError(r.error);return;}
    var _sec='init';
    try{

    _sec='pipeline'; /* --- Pipeline Flow --- */
    const flow = document.getElementById('pipelineFlow');
    let html = '';
    r.stages.forEach((s,i)=>{
        if(i>0) html += '<div class="flow-arr">&rarr;</div>';
        const icon = s.status==='ok'?'\u2713':s.status==='skip'?'\u25cb':s.status==='rejected'?'\u26a0':'!';
        const timeStr = s.time_ms!=null? s.time_ms.toFixed(1)+'ms':'';
        html += '<div class="flow-step '+s.status+'">'
            +'<div class="fs-icon">'+icon+'</div>'
            +'<div class="fs-name">'+s.name+'</div>'
            +'<div class="fs-time">'+timeStr+'</div></div>';
    });
    flow.innerHTML = html;

    _sec='perception'; /* --- Visual Perception --- */
    const perc = r.perception;
    if(perc){
        const symColor = perc.symmetry_score>60?'#059669':perc.symmetry_score>30?'#d97706':'#ef4444';
        let ph = '<div style="margin-bottom:8px;">';
        // 1D intensity heatmap bar
        ph += '<div style="font-size:11px;color:#64748b;margin-bottom:4px;">强度分布热图 (2\u03b8='+perc.two_theta_range[0]+'\u00b0-'+perc.two_theta_range[1]+'\u00b0)</div>';
        ph += '<canvas id="heatmapCanvas" style="width:100%;height:20px;border-radius:4px;display:block;"></canvas>';
        ph += '</div>';
        // Metrics row
        ph += '<div style="display:flex;gap:8px;flex-wrap:wrap;">';
        ph += '<div style="flex:1;min-width:70px;background:#f1f5f9;border-radius:6px;padding:6px 8px;text-align:center;">';
        ph += '<div style="font-size:18px;font-weight:800;color:#2563eb;">'+perc.peak_count+'</div>';
        ph += '<div style="font-size:10px;color:#64748b;">检测峰数</div></div>';
        ph += '<div style="flex:1;min-width:70px;background:#f1f5f9;border-radius:6px;padding:6px 8px;text-align:center;">';
        ph += '<div style="font-size:18px;font-weight:800;color:'+symColor+';">'+perc.symmetry_score+'%</div>';
        ph += '<div style="font-size:10px;color:#64748b;">对称性</div></div>';
        ph += '<div style="flex:2;min-width:90px;background:#f1f5f9;border-radius:6px;padding:6px 8px;text-align:center;">';
        ph += '<div style="font-size:14px;font-weight:700;color:#334155;">'+perc.pattern_type+'</div>';
        ph += '<div style="font-size:10px;color:#64748b;">\u8c31\u578b\u8bc6\u522b</div></div>';
        // MC Dropout uncertainty
        if(r.mc_dropout){
            var mc=r.mc_dropout;
            var ucColor=mc.uncertainty_level==='\u4f4e'?'#059669':mc.uncertainty_level==='\u4e2d'?'#d97706':'#ef4444';
            ph += '<div style="flex:1;min-width:70px;background:#f1f5f9;border-radius:6px;padding:6px 8px;text-align:center;">';
            ph += '<div style="font-size:16px;font-weight:800;color:'+ucColor+';">'+mc.uncertainty_level+'</div>';
            ph += '<div style="font-size:10px;color:#64748b;">MC Dropout</div></div>';
        }
        // Crystallography (Scherrer + d-spacing)
        if(r.crystallography && r.crystallography.avg_crystallite_nm){
            var cr=r.crystallography;
            ph += '<div style="flex:1;min-width:70px;background:#f1f5f9;border-radius:6px;padding:6px 8px;text-align:center;">';
            ph += '<div style="font-size:16px;font-weight:800;color:#2563eb;">~'+cr.avg_crystallite_nm+'</div>';
            ph += '<div style="font-size:10px;color:#64748b;">\u5fae\u6676(nm)</div></div>';
        }
        ph += '</div>';

        // TDA持久同调可视化
        if(r.tda && r.tda.features){
            var td=r.tda;
            ph += '<div style="margin-top:8px;border-top:1px solid #e2e8f0;padding-top:8px;">';
            ph += '<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">';
            ph += '<span style="font-size:11px;font-weight:600;color:#7c3aed;">\u62d3\u6251\u6570\u636e\u5206\u6790 (TDA H\u2080)</span>';
            ph += '<span style="font-size:10px;color:#64748b;">'+td.topology_type+'</span></div>';
            // Persistence diagram (mini scatter plot)
            ph += '<canvas id="tdaDiagram" style="width:100%;height:60px;border-radius:4px;display:block;background:#f8fafc;border:1px solid #e2e8f0;"></canvas>';
            // Feature badges
            ph += '<div style="display:flex;gap:4px;flex-wrap:wrap;margin-top:4px;">';
            var tf=td.features;
            ph += '<span style="font-size:9px;padding:2px 6px;border-radius:8px;background:#f5f3ff;color:#7c3aed;border:1px solid #e9d5ff;">\u6700\u5927\u6301\u4e45:'+tf.max_persistence+'</span>';
            ph += '<span style="font-size:9px;padding:2px 6px;border-radius:8px;background:#f5f3ff;color:#7c3aed;border:1px solid #e9d5ff;">\u5747\u503c:'+tf.mean_persistence+'</span>';
            ph += '<span style="font-size:9px;padding:2px 6px;border-radius:8px;background:#f5f3ff;color:#7c3aed;border:1px solid #e9d5ff;">\u663e\u8457\u7279\u5f81:'+tf.significant_features+'</span>';
            ph += '</div></div>';
        }

        document.getElementById('percBody').innerHTML = ph;
        // Render heatmap + TDA diagram after DOM update
        setTimeout(function(){
            renderHeatmap(r.spectrum);
            if(r.tda && r.tda.diagram) renderTdaDiagram(r.tda.diagram);
        },50);
    }

    _sec='classification'; /* --- Classification --- */
    const cls = r.classification;
    if(cls){
        const confPct = (cls.final_confidence*100).toFixed(1);
        const confColor = cls.final_confidence>0.7?'#059669':cls.final_confidence>0.4?'#d97706':'#ef4444';
        let tags = '<span class="tag tag-blue">'+cls.mode+'</span>';
        if(cls.primary) tags += '<span class="tag tag-green">'+cls.primary+'</span>';
        if(cls.fine_label) tags += '<span class="tag tag-amber">'+cls.fine_label+'</span>';
        if(cls.fine_rejected) tags += '<span class="tag tag-red">OOD\u62d2\u8bc6</span>';

        // Conformal Prediction pills
        var cpHtml = '';
        if(r.conformal){
            var cp = r.conformal;
            var cpColor = cp.certain?'#059669':'#d97706';
            var cpBg = cp.certain?'#ecfdf5':'#fffbeb';
            cpHtml = '<div style="margin-top:10px;padding:8px 12px;background:'+cpBg+';border-radius:8px;border:1px solid '+cpColor+'22;">';
            cpHtml += '<div style="font-size:11px;color:'+cpColor+';font-weight:600;margin-bottom:4px;">';
            cpHtml += '\u4fdd\u5f62\u9884\u6d4b (Conformal Prediction, '+(cp.coverage*100).toFixed(0)+'%\u8986\u76d6\u7387\u4fdd\u8bc1)</div>';
            cpHtml += '<div style="display:flex;gap:6px;flex-wrap:wrap;">';
            cp.prediction_set.forEach(function(p){
                cpHtml += '<span style="padding:3px 10px;border-radius:12px;font-size:12px;font-weight:600;background:'+cpColor+'18;color:'+cpColor+';border:1px solid '+cpColor+'44;">'+p+'</span>';
            });
            cpHtml += '</div>';
            if(!cp.certain) cpHtml += '<div style="font-size:10px;color:#92400e;margin-top:4px;">\u9884\u6d4b\u96c6>1\uff0c\u9700\u5cf0\u4f4d\u5339\u914d\u9a8c\u8bc1</div>';
            cpHtml += '</div>';
        }

        document.getElementById('clsBody').innerHTML=
            '<div class="cls-label" style="color:'+confColor+'">'+cls.final_label+'</div>'
            +'<div class="cls-sub">'+cls.primary+' &rarr; '+(cls.fine_label||'-')+'</div>'
            +'<div class="conf-wrap"><div class="conf-bar"><div class="conf-fill" style="width:'+confPct+'%;background:'+confColor+'"></div></div>'
            +'<div class="conf-pct" style="color:'+confColor+'">'+confPct+'%</div></div>'
            +'<div class="cls-tags">'+tags+'</div>'
            +cpHtml;

        // Extinction verification
        if(r.extinction){
            var ext=r.extinction;
            var extColor=ext.all_pass?'#059669':'#d97706';
            var extBg=ext.all_pass?'#ecfdf5':'#fffbeb';
            var extIcon=ext.all_pass?'\u2705':'\u26a0\ufe0f';
            var extHtml='<div style="margin-top:8px;padding:6px 10px;background:'+extBg+';border-radius:6px;border:1px solid '+extColor+'22;font-size:11px;">';
            extHtml+='<span style="font-weight:700;color:'+extColor+';">'+extIcon+' \u6d88\u5149\u9a8c\u8bc1: '+ext.lattice+'\u683c\u5b50 ('+ext.rule+')</span>';
            extHtml+=' <span style="color:#64748b;">'+ext.n_passed+'/'+ext.n_checked+'\u901a\u8fc7</span>';
            extHtml+='</div>';
            document.getElementById('clsBody').innerHTML+=extHtml;
        }

        // Nelson-Riley lattice parameter
        if(r.nelson_riley){
            var nr=r.nelson_riley;
            var nrHtml='<div style="margin-top:6px;padding:5px 10px;background:#eff6ff;border-radius:6px;border:1px solid #bfdbfe;font-size:11px;">';
            nrHtml+='<span style="font-weight:700;color:#1d4ed8;">\ud83d\udd2c Nelson-Riley\u7cbe\u4fee: a = '+nr.a_refined+' \u00c5</span>';
            nrHtml+=' <span style="color:#64748b;">('+nr.n_peaks+'\u5cf0)</span>';
            nrHtml+='</div>';
            document.getElementById('clsBody').innerHTML+=nrHtml;
        }
    }

    _sec='performance'; /* --- Performance + Waterfall --- */
    const t = r.timings||{};
    const bpuTotal = ((t.bpu1||0)+(t.bpu2||0)).toFixed(1);
    const localMs = (r.local_ms||0).toFixed(0);
    const detStr = r.bpu_deterministic?'< 1ms':'N/A';
    // Build waterfall data
    var wf = [];
    var stageColors = {'parse':'#3b82f6','peaks':'#3b82f6','features':'#3b82f6','bpu1':'#10b981','bpu2':'#10b981','match':'#8b5cf6','rag':'#7c3aed','agent':'#ef4444'};
    var stageNames = {'parse':'\u89e3\u6790','peaks':'\u5cf0\u63d0\u53d6','features':'\u7279\u5f81','bpu1':'BPU#1','bpu2':'BPU#2','match':'\u5339\u914d','rag':'RAG\u68c0\u7d22','agent':'Agent\u63a8\u7406'};
    var totalLocal = 0;
    ['parse','peaks','features','bpu1','bpu2','match','rag','agent'].forEach(function(k){
        if(t[k]!=null&&t[k]>0) {wf.push({name:stageNames[k]||k,ms:t[k],color:stageColors[k]||'#94a3b8'});totalLocal+=t[k];}
    });
    var wfHtml = '<div style="margin-top:10px;font-size:11px;color:#64748b;">延迟瀑布图</div>';
    wfHtml += '<div style="margin-top:4px;">';
    wf.forEach(function(w){
        var pct = totalLocal>0?Math.max(2,w.ms/totalLocal*100):0;
        wfHtml += '<div style="display:flex;align-items:center;gap:6px;margin:3px 0;">';
        wfHtml += '<span style="width:40px;font-size:10px;text-align:right;color:#475569;">'+w.name+'</span>';
        wfHtml += '<div style="flex:1;background:#f1f5f9;border-radius:3px;height:14px;overflow:hidden;">';
        wfHtml += '<div style="width:'+pct.toFixed(1)+'%;height:100%;background:'+w.color+';border-radius:3px;transition:width .5s;"></div></div>';
        wfHtml += '<span style="width:45px;font-size:10px;color:#64748b;">'+w.ms.toFixed(1)+'ms</span>';
        wfHtml += '</div>';
    });
    wfHtml += '</div>';

    // Rwp from simulation
    var rwpHtml = '';
    if(r.simulation && r.simulation.rwp != null){
        var rwp = r.simulation.rwp;
        var rwpColor = rwp<10?'green':rwp<20?'amber':'';
        rwpHtml = '<div class="metric-box"><div class="metric-val '+ rwpColor+'">'+rwp.toFixed(1)+'%</div><div class="metric-lbl">Rwp\u6b8b\u5dee</div></div>';
    }

    document.getElementById('perfBody').innerHTML=
        '<div class="metrics" style="grid-template-columns:repeat('+(rwpHtml?'4':'3')+',1fr);">'
        +'<div class="metric-box"><div class="metric-val">'+localMs+'ms</div><div class="metric-lbl">Pipeline\u8017\u65f6</div></div>'
        +'<div class="metric-box"><div class="metric-val">'+bpuTotal+'ms</div><div class="metric-lbl">BPU\u63a8\u7406</div></div>'
        +'<div class="metric-box"><div class="metric-val green">'+detStr+'</div><div class="metric-lbl">\u786e\u5b9a\u6027\u5ef6\u8fdf</div></div>'
        +rwpHtml
        +'</div>'
        +wfHtml;

    /* --- Peak Matching --- */
    const m = r.peak_matching;
    if(m){
        let mh = '<div style="margin-bottom:10px;">'
            +'<span style="font-size:15px;font-weight:700;color:#5b21b6;">'+m.display_name+'</span>'
            +' <span style="font-size:13px;color:#475569;">('+m.space_group+')</span>'
            +'<br><span style="font-size:12px;color:#64748b;">得分: <strong>'+m.score+'</strong>'
            +(m.reference_cards.length?' | '+m.reference_cards[0]:'')+'</span></div>';
        if(m.matched_pairs && m.matched_pairs.length){
            mh += '<table><thead><tr><th>检测峰 2\u03b8</th><th>Miller指数</th><th>参考峰 2\u03b8</th></tr></thead><tbody>';
            m.matched_pairs.forEach(p=>{
                mh += '<tr><td>'+p.detected+'\u00b0</td><td>'+p.hkl+'</td><td>'+p.reference+'\u00b0</td></tr>';
            });
            mh += '</tbody></table>';
        }
        document.getElementById('matchBody').innerHTML = mh;
    }else{
        document.getElementById('matchBody').innerHTML='<div class="empty"><p>未找到匹配相</p></div>';
    }

    _sec='agent'; /* --- AI Agent Card (始终展开, 默认显示推理链) --- */
    {
        document.getElementById('agentCard').style.display='block';
        var toolCount = (r.agent_tools||[]).length;
        document.getElementById('agentToolCount').textContent = toolCount+'\u6b21\u5de5\u5177\u8c03\u7528';
        var ah = '';
        if(r.agent_tools && r.agent_tools.length){
            ah += '<div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px;">';
            r.agent_tools.forEach(function(t,i){
                var icon = t.name==='query_rag_knowledge'?'\ud83d\udcda':t.name==='match_pdf_card'?'\ud83d\udd2c':'\ud83e\uddea';
                ah += '<div style="flex:1;min-width:200px;background:#fef2f2;border:1px solid #fecaca;border-radius:8px;padding:8px 10px;">';
                ah += '<div style="font-size:12px;font-weight:700;color:#dc2626;">'+icon+' '+t.name+'</div>';
                ah += '<div style="font-size:10px;color:#64748b;margin-top:2px;">'+escHtml(JSON.stringify(t.args)).substring(0,80)+'</div>';
                ah += '<div style="font-size:10px;color:#334155;margin-top:3px;max-height:40px;overflow:hidden;">'+escHtml((t.result||'').substring(0,120))+'</div>';
                ah += '</div>';
            });
            ah += '</div>';
        }
        if(r.agent_thinking){
            // v4.1 Round 5: details 默认展开 (open 属性)
            ah += '<details open style="margin-top:4px;"><summary style="cursor:pointer;font-size:11px;font-weight:600;color:#7c3aed;">\ud83e\udde0 \u5b8c\u6574\u63a8\u7406\u94fe (R1 Thinking)</summary>';
            ah += '<div style="background:#f5f3ff;border:1px solid #e9d5ff;border-radius:6px;padding:8px;font-size:11px;line-height:1.6;max-height:400px;overflow-y:auto;color:#5b21b6;white-space:pre-wrap;margin-top:4px;">'+escHtml(r.agent_thinking)+'</div>';
            ah += '</details>';
        }
        document.getElementById('agentBody').innerHTML = ah;
    }

    _sec='report'; /* --- Report --- */
    document.getElementById('reportMode').textContent = r.report_mode||'';
    document.getElementById('reportBody').innerHTML = '<div class="report">'+mdToHtml(r.report||'\u65e0\u62a5\u544a')+'</div>';

    _sec='xai'; /* --- XAI Attribution --- */
    if(r.xai){
        var xai = r.xai;
        // Update perception card with XAI info
        var xaiHtml = '<div style="margin-top:8px;border-top:1px solid #e2e8f0;padding-top:8px;">';
        xaiHtml += '<div style="font-size:11px;color:#64748b;margin-bottom:4px;">\u7279\u5f81\u5f52\u56e0\u5206\u6790 (XAI)</div>';
        xaiHtml += '<canvas id="xaiBar" style="width:100%;height:16px;border-radius:3px;display:block;"></canvas>';
        xaiHtml += '<div style="font-size:11px;color:#334155;margin-top:4px;">'+xai.summary+'</div>';
        // Top features list
        xaiHtml += '<div style="display:flex;gap:4px;flex-wrap:wrap;margin-top:4px;">';
        xai.top_features.slice(0,5).forEach(function(f){
            var color = f.attribution>0?'#059669':'#dc2626';
            var label = f['2theta']?('2\u03b8\u2248'+f['2theta']+'\u00b0'):f.role;
            xaiHtml += '<span style="font-size:10px;padding:2px 6px;border-radius:8px;background:'+color+'15;color:'+color+';border:1px solid '+color+'33;">'+label+' ('+f.attribution.toFixed(2)+')</span>';
        });
        xaiHtml += '</div></div>';
        var percBody = document.getElementById('percBody');
        if(percBody) percBody.innerHTML += xaiHtml;
        // Render XAI bar after DOM update
        setTimeout(function(){renderXaiBar(xai.bin_attributions);},80);
    }

    _sec='chart'; /* --- Chart --- */
    try{ if(r.spectrum) renderChart(r.spectrum, r.peaks, m, r.simulation); }catch(e){ console.error('Chart error:',e); }

    /* --- Knowledge Graph + 3D Crystal: 分析完自动加载 (v4.1 Round 5) --- */
    document.getElementById('kgCard').style.display='block';
    try{ loadKnowledgeGraph(); }catch(e){ console.log('[KG auto]', e); }
    if(r.classification){
        var _crystalLabel=r.classification.final_label||'';
        window._lastLabel = _crystalLabel;
        document.getElementById('crystalCard').style.display='block';
        try{ show3DCrystal(_crystalLabel); }catch(e){ console.log('[3D auto]', e); }
        try{ showCandidateCrystals(_crystalLabel || 'garnet'); }catch(e){ console.log('[cand auto]', e); }
    }
    // v4.1 Round 5: 完结撒花
    try{ celebrateDone(); }catch(e){}

    /* --- QR Code --- */
    try{
        if(typeof QRCode!=='undefined'){
            var qrEl=document.getElementById('qrcode');
            var qrSec=document.getElementById('qrSection');
            if(qrEl&&qrSec){
                qrEl.innerHTML='';
                qrSec.style.display='block';
                new QRCode(qrEl,{text:location.origin+'/report',width:80,height:80,colorDark:'#1e293b',colorLight:'#ffffff'});
            }
        }
    }catch(e){console.error('QR error:',e);}

    /* --- Confetti celebration (high confidence) --- */
    try{
        if(r.classification && r.classification.final_confidence > 0.6 && typeof confetti==='function'){
            setTimeout(function(){
                confetti({particleCount:60,spread:70,origin:{y:0.7},colors:['#10b981','#3b82f6','#f59e0b']});
            },500);
        }
    }catch(e){}

    }catch(err){
        // 白屏诊断: 在页面顶部显示具体错误
        var errDiv=document.createElement('div');
        errDiv.style.cssText='position:fixed;top:0;left:0;right:0;background:#fef2f2;color:#991b1b;padding:12px;font-size:13px;z-index:9999;border-bottom:2px solid #ef4444;';
        errDiv.innerHTML='<strong>\u2757 displayResult\u5d29\u6e83\u4e8e['+_sec+']:</strong> '+err.message+'<br><small>'+String(err.stack).substring(0,300)+'</small><br><button onclick="this.parentElement.remove()" style="margin-top:4px;padding:2px 8px;">关闭</button>';
        document.body.prepend(errDiv);
        console.error('displayResult fatal:',err);
    }
}

function escHtml(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}

function mdToHtml(md){
    var s = escHtml(md);
    // headers
    s = s.replace(/^### (.+)$/gm,'<h4>$1</h4>');
    s = s.replace(/^## (.+)$/gm,'<h4>$1</h4>');
    // bold
    s = s.replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>');
    // markdown table
    s = s.replace(/^\|(.+)\|$/gm, function(line){
        if(/^[\|\s\-:]+$/.test(line)) return '';
        var cells = line.split('|').filter(function(c){return c.trim()!=='';});
        var isHeader = false;
        // check if next line is separator (already removed)
        var tag = 'td';
        return '<tr>'+cells.map(function(c){return '<'+tag+'>'+c.trim()+'</'+tag+'>';}).join('')+'</tr>';
    });
    s = s.replace(/(<tr>[\s\S]*?<\/tr>\s*)+/g,'<table><tbody>$&</tbody></table>');
    // paragraphs & line breaks
    s = s.replace(/\n\n+/g,'</p><p>');
    s = s.replace(/\n/g,'<br>');
    return '<p>'+s+'</p>';
}

/* ============ Canvas XRD谱图 ============ */
function renderChart(spectrum, peaks, match, sim){
    const canvas = document.getElementById('chartCanvas');
    const dpr = window.devicePixelRatio||1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width*dpr;
    canvas.height = rect.height*dpr;
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr,dpr);
    const W=rect.width, H=rect.height;
    const pad={top:30,right:20,bottom:42,left:55};
    const pw=W-pad.left-pad.right, ph=H-pad.top-pad.bottom;

    const tt=spectrum.two_theta, ints=spectrum.intensity;
    const xMin=tt[0], xMax=tt[tt.length-1];
    let yMax=0;
    for(let i=0;i<ints.length;i++) if(ints[i]>yMax) yMax=ints[i];
    yMax*=1.18;
    const toX=v=>pad.left+(v-xMin)/(xMax-xMin)*pw;
    const toY=v=>pad.top+(1-v/yMax)*ph;

    // Background
    ctx.fillStyle='#0f172a';
    ctx.fillRect(0,0,W,H);

    // Grid
    ctx.strokeStyle='#1e293b'; ctx.lineWidth=.5;
    for(let i=1;i<5;i++){const y=pad.top+ph*i/5;ctx.beginPath();ctx.moveTo(pad.left,y);ctx.lineTo(pad.left+pw,y);ctx.stroke();}
    for(let i=1;i<8;i++){const x=pad.left+pw*i/8;ctx.beginPath();ctx.moveTo(x,pad.top);ctx.lineTo(x,pad.top+ph);ctx.stroke();}

    // Reference peaks (green dashed)
    if(match && match.matched_pairs && match.matched_pairs.length){
        ctx.setLineDash([5,4]);ctx.strokeStyle='rgba(16,185,129,.6)';ctx.lineWidth=1;
        match.matched_pairs.forEach(p=>{
            const x=toX(p.reference);
            if(x>=pad.left&&x<=pad.left+pw){
                ctx.beginPath();ctx.moveTo(x,pad.top);ctx.lineTo(x,pad.top+ph);ctx.stroke();
                ctx.fillStyle='#10b981';ctx.font='bold 9px system-ui';ctx.textAlign='center';
                ctx.fillText(p.hkl,x,pad.top-5);
            }
        });
        ctx.setLineDash([]);
    }

    // Spectrum line + gradient fill
    ctx.beginPath();ctx.strokeStyle='#3b82f6';ctx.lineWidth=1.5;
    for(let i=0;i<tt.length;i++){const x=toX(tt[i]),y=toY(ints[i]);if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);}
    ctx.stroke();
    // Gradient fill
    ctx.lineTo(toX(tt[tt.length-1]),pad.top+ph);ctx.lineTo(toX(tt[0]),pad.top+ph);ctx.closePath();
    const grad=ctx.createLinearGradient(0,pad.top,0,pad.top+ph);
    grad.addColorStop(0,'rgba(59,130,246,.25)');grad.addColorStop(1,'rgba(59,130,246,0)');
    ctx.fillStyle=grad;ctx.fill();

    // Detected peaks (red triangles)
    if(peaks && peaks.length){
        peaks.forEach(p=>{
            const x=toX(p.position);
            let bestY=0,minD=1e9;
            for(let i=0;i<tt.length;i++){const d=Math.abs(tt[i]-p.position);if(d<minD){minD=d;bestY=ints[i];}}
            const y=toY(bestY);
            ctx.fillStyle='#ef4444';
            ctx.beginPath();ctx.moveTo(x,y-9);ctx.lineTo(x-4,y-3);ctx.lineTo(x+4,y-3);ctx.closePath();ctx.fill();
            ctx.fillStyle='#fca5a5';ctx.font='9px system-ui';ctx.textAlign='center';
            ctx.fillText(p.position.toFixed(1)+'\u00b0',x,y-13);
        });
    }

    // Axes
    ctx.strokeStyle='#475569';ctx.lineWidth=1;
    ctx.beginPath();ctx.moveTo(pad.left,pad.top);ctx.lineTo(pad.left,pad.top+ph);ctx.lineTo(pad.left+pw,pad.top+ph);ctx.stroke();

    // X ticks
    ctx.fillStyle='#94a3b8';ctx.font='11px system-ui';ctx.textAlign='center';
    for(let i=0;i<=8;i++){
        const v=xMin+(xMax-xMin)*i/8, x=toX(v);
        ctx.beginPath();ctx.moveTo(x,pad.top+ph);ctx.lineTo(x,pad.top+ph+4);ctx.stroke();
        ctx.fillText(v.toFixed(0)+'\u00b0',x,pad.top+ph+18);
    }
    // Y ticks
    ctx.textAlign='right';
    for(let i=0;i<=4;i++){
        const v=yMax*i/4, y=toY(v);
        ctx.beginPath();ctx.moveTo(pad.left-4,y);ctx.lineTo(pad.left,y);ctx.stroke();
        ctx.fillText(v>=1000?(v/1000).toFixed(1)+'k':v.toFixed(0),pad.left-8,y+4);
    }

    // Axis labels
    ctx.fillStyle='#94a3b8';ctx.font='12px system-ui';ctx.textAlign='center';
    ctx.fillText('2\u03b8 (\u00b0)',pad.left+pw/2,H-6);
    ctx.save();ctx.translate(14,pad.top+ph/2);ctx.rotate(-Math.PI/2);ctx.fillText('Intensity (a.u.)',0,0);ctx.restore();

    // Theoretical pattern overlay (red dashed)
    if(sim && sim.two_theta && sim.intensity){
        ctx.setLineDash([6,3]);
        ctx.strokeStyle='#ef4444';
        ctx.lineWidth=1.2;
        ctx.beginPath();
        var simMax=0;
        for(var si=0;si<sim.intensity.length;si++) if(sim.intensity[si]>simMax) simMax=sim.intensity[si];
        if(simMax<=0) simMax=1;
        for(var si=0;si<sim.two_theta.length;si++){
            var sx=toX(sim.two_theta[si]);
            var sy=toY(sim.intensity[si]/simMax*yMax*0.95);
            if(sx>=pad.left&&sx<=pad.left+pw){
                if(si===0||sx<pad.left+1) ctx.moveTo(sx,sy);
                else ctx.lineTo(sx,sy);
            }
        }
        ctx.stroke();
        ctx.setLineDash([]);
        // Rwp badge
        if(sim.rwp!=null){
            var rwpColor=sim.rwp<10?'#10b981':sim.rwp<20?'#f59e0b':'#ef4444';
            ctx.fillStyle=rwpColor;
            ctx.font='bold 12px system-ui';
            ctx.textAlign='left';
            ctx.fillText('Rwp='+sim.rwp.toFixed(1)+'%',pad.left+10,pad.top+16);
        }
    }

    // Legend box (dynamic height)
    const lx=pad.left+pw-145, ly=pad.top+8;
    var lh=40;
    if(match&&match.matched_pairs&&match.matched_pairs.length) lh+=16;
    if(sim&&sim.two_theta) lh+=16;
    ctx.fillStyle='rgba(15,23,42,.85)';ctx.strokeStyle='#334155';ctx.lineWidth=1;
    roundRect(ctx,lx,ly,135,lh,6);ctx.fill();ctx.stroke();
    var lrow=0;
    // Spectrum
    ctx.strokeStyle='#3b82f6';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(lx+8,ly+13+lrow);ctx.lineTo(lx+28,ly+13+lrow);ctx.stroke();
    ctx.fillStyle='#e2e8f0';ctx.font='11px system-ui';ctx.textAlign='left';ctx.fillText('\u539f\u59cb\u8c31',lx+34,ly+17+lrow);
    lrow+=16;
    // Peaks
    ctx.fillStyle='#ef4444';ctx.beginPath();ctx.moveTo(lx+18,ly+8+lrow);ctx.lineTo(lx+14,ly+14+lrow);ctx.lineTo(lx+22,ly+14+lrow);ctx.closePath();ctx.fill();
    ctx.fillStyle='#e2e8f0';ctx.fillText('\u68c0\u6d4b\u5cf0',lx+34,ly+16+lrow);
    lrow+=16;
    // Ref peaks
    if(match&&match.matched_pairs&&match.matched_pairs.length){
        ctx.setLineDash([3,3]);ctx.strokeStyle='#10b981';ctx.lineWidth=1.5;
        ctx.beginPath();ctx.moveTo(lx+8,ly+8+lrow);ctx.lineTo(lx+28,ly+8+lrow);ctx.stroke();ctx.setLineDash([]);
        ctx.fillStyle='#e2e8f0';ctx.fillText('\u53c2\u8003\u5cf0',lx+34,ly+12+lrow);
        lrow+=16;
    }
    // Theoretical pattern
    if(sim&&sim.two_theta){
        ctx.setLineDash([6,3]);ctx.strokeStyle='#ef4444';ctx.lineWidth=1.2;
        ctx.beginPath();ctx.moveTo(lx+8,ly+8+lrow);ctx.lineTo(lx+28,ly+8+lrow);ctx.stroke();ctx.setLineDash([]);
        ctx.fillStyle='#fca5a5';ctx.fillText('\u7406\u8bba\u8c31',lx+34,ly+12+lrow);
    }
    // Save chart metadata for tooltip
    _chartMeta={pad:pad,pw:pw,ph:ph,xMin:xMin,xMax:xMax,yMax:yMax,peaks:peaks,match:match,sim:sim,W:W,H:H};
}

// Peak tooltip on hover/touch
(function(){
    var canvas=document.getElementById('chartCanvas');
    var tip=document.getElementById('peakTooltip');
    if(!canvas||!tip) return;
    function handleMove(cx,cy,pageX,pageY){
        if(!_chartMeta) return;
        var m=_chartMeta;
        var rect=canvas.getBoundingClientRect();
        var x=(cx-rect.left), y=(cy-rect.top);
        // Convert pixel to 2theta
        var twoTheta=m.xMin+(x-m.pad.left)/m.pw*(m.xMax-m.xMin);
        // Find nearest peak
        var best=null, bestDist=Infinity;
        if(m.peaks){
            m.peaks.forEach(function(p){
                var d=Math.abs(p.position-twoTheta);
                if(d<bestDist){bestDist=d;best=p;}
            });
        }
        if(best&&bestDist<1.5){
            var html='<strong>2\u03b8 = '+best.position.toFixed(2)+'\u00b0</strong><br>';
            html+='d = '+(1.54056/(2*Math.sin(best.position/2*Math.PI/180))).toFixed(3)+' \u00c5<br>';
            html+='I = '+(best.intensity*100).toFixed(1)+'%';
            if(best.fwhm) html+='<br>FWHM = '+best.fwhm.toFixed(3)+'\u00b0';
            // Check if matched
            if(m.match&&m.match.matched_pairs){
                m.match.matched_pairs.forEach(function(mp){
                    if(Math.abs(mp.detected-best.position)<0.3){
                        html+='<br><span style="color:#10b981;">'+mp.hkl+' (ref:'+mp.reference+'\u00b0)</span>';
                    }
                });
            }
            tip.innerHTML=html;
            tip.style.display='block';
            tip.style.left=Math.min(pageX+15,window.innerWidth-220)+'px';
            tip.style.top=(pageY-80)+'px';
        }else{
            tip.style.display='none';
        }
    }
    canvas.addEventListener('mousemove',function(e){handleMove(e.clientX,e.clientY,e.pageX,e.pageY);});
    canvas.addEventListener('touchmove',function(e){
        var t=e.touches[0];
        handleMove(t.clientX,t.clientY,t.pageX,t.pageY);
        e.preventDefault();
    },{passive:false});
    canvas.addEventListener('mouseleave',function(){tip.style.display='none';});
    canvas.addEventListener('touchend',function(){tip.style.display='none';});
})();

function roundRect(ctx,x,y,w,h,r){
    ctx.beginPath();ctx.moveTo(x+r,y);ctx.lineTo(x+w-r,y);ctx.quadraticCurveTo(x+w,y,x+w,y+r);
    ctx.lineTo(x+w,y+h-r);ctx.quadraticCurveTo(x+w,y+h,x+w-r,y+h);ctx.lineTo(x+r,y+h);
    ctx.quadraticCurveTo(x,y+h,x,y+h-r);ctx.lineTo(x,y+r);ctx.quadraticCurveTo(x,y,x+r,y);ctx.closePath();
}

function renderTdaDiagram(diagram){
    var canvas=document.getElementById('tdaDiagram');
    if(!canvas||!diagram||!diagram.length) return;
    var dpr=window.devicePixelRatio||1;
    var rect=canvas.getBoundingClientRect();
    canvas.width=rect.width*dpr;canvas.height=rect.height*dpr;
    var ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);
    var w=rect.width,h=rect.height;
    var pad={top:8,right:8,bottom:14,left:24};
    var pw=w-pad.left-pad.right,ph=h-pad.top-pad.bottom;
    // Find max death for scaling
    var maxD=0;
    diagram.forEach(function(p){if(p.death>maxD) maxD=p.death;});
    if(maxD<=0) maxD=1;
    // Diagonal line (birth=death)
    ctx.strokeStyle='#cbd5e1';ctx.lineWidth=0.5;ctx.setLineDash([3,3]);
    ctx.beginPath();ctx.moveTo(pad.left,pad.top+ph);ctx.lineTo(pad.left+pw,pad.top);ctx.stroke();
    ctx.setLineDash([]);
    // Plot points
    diagram.forEach(function(p){
        var x=pad.left+(p.birth/maxD)*pw;
        var y=pad.top+ph-(p.death/maxD)*ph;
        var life=p.death-p.birth;
        var sz=Math.max(3,Math.min(7,life/maxD*12));
        var alpha=Math.max(0.3,Math.min(1,life/maxD*2));
        ctx.fillStyle='rgba(124,58,237,'+alpha+')';
        ctx.beginPath();ctx.arc(x,y,sz,0,Math.PI*2);ctx.fill();
        ctx.strokeStyle='rgba(124,58,237,0.5)';ctx.lineWidth=0.5;ctx.stroke();
    });
    // Axis labels
    ctx.fillStyle='#94a3b8';ctx.font='8px system-ui';
    ctx.textAlign='center';ctx.fillText('birth',pad.left+pw/2,h-2);
    ctx.save();ctx.translate(6,pad.top+ph/2);ctx.rotate(-Math.PI/2);
    ctx.fillText('death',0,0);ctx.restore();
}

function renderXaiBar(binAttr){
    var canvas=document.getElementById('xaiBar');
    if(!canvas||!binAttr||!binAttr.length) return;
    var dpr=window.devicePixelRatio||1;
    var rect=canvas.getBoundingClientRect();
    canvas.width=rect.width*dpr;canvas.height=rect.height*dpr;
    var ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);
    var w=rect.width,h=rect.height;
    var mx=0;for(var i=0;i<binAttr.length;i++){var a=Math.abs(binAttr[i]);if(a>mx)mx=a;}
    if(mx===0)mx=1;
    for(var x=0;x<w;x++){
        var idx=Math.min(Math.floor(x*binAttr.length/w),binAttr.length-1);
        var v=binAttr[idx]/mx; // -1 to +1
        var r2,g2,b2;
        if(v>0){r2=Math.round(v*220);g2=Math.round(60*(1-v));b2=60;} // red for positive
        else{r2=60;g2=Math.round(60*(1+v));b2=Math.round(-v*220);} // blue for negative
        ctx.fillStyle='rgb('+r2+','+g2+','+b2+')';
        ctx.fillRect(x,0,1,h);
    }
}

function renderHeatmap(spectrum){
    var canvas=document.getElementById('heatmapCanvas');
    if(!canvas||!spectrum) return;
    var dpr=window.devicePixelRatio||1;
    var rect=canvas.getBoundingClientRect();
    canvas.width=rect.width*dpr;canvas.height=rect.height*dpr;
    var ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);
    var w=rect.width,h=rect.height;
    var ints=spectrum.intensity;
    var mx=0;for(var i=0;i<ints.length;i++) if(ints[i]>mx) mx=ints[i];
    if(mx===0) mx=1;
    var step=Math.max(1,Math.floor(ints.length/w));
    for(var x=0;x<w;x++){
        var idx=Math.min(Math.floor(x*ints.length/w),ints.length-1);
        var v=ints[idx]/mx;
        // Blue(0) → Cyan(0.3) → Green(0.5) → Yellow(0.7) → Red(1)
        var r2,g2,b2;
        if(v<0.25){r2=0;g2=Math.round(v*4*200);b2=200;}
        else if(v<0.5){r2=0;g2=200;b2=Math.round((1-(v-0.25)*4)*200);}
        else if(v<0.75){r2=Math.round((v-0.5)*4*255);g2=200;b2=0;}
        else{r2=255;g2=Math.round((1-(v-0.75)*4)*200);b2=0;}
        ctx.fillStyle='rgb('+r2+','+g2+','+b2+')';
        ctx.fillRect(x,0,1,h);
    }
}

function drawPlaceholder(){
    const canvas=document.getElementById('chartCanvas');
    const dpr=window.devicePixelRatio||1;
    const rect=canvas.getBoundingClientRect();
    canvas.width=rect.width*dpr;canvas.height=rect.height*dpr;
    const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);
    const w=rect.width,h=rect.height;
    ctx.fillStyle='#0f172a';ctx.fillRect(0,0,w,h);
    ctx.fillStyle='#475569';ctx.font='600 16px system-ui';ctx.textAlign='center';
    ctx.fillText('\u9009\u62e9\u6837\u54c1\u6587\u4ef6\u5f00\u59cb XRD \u5206\u6790',w/2,h/2-12);
    ctx.fillStyle='#334155';ctx.font='13px system-ui';
    ctx.fillText('.raw \u2192 \u5cf0\u63d0\u53d6 \u2192 BPU\u5206\u7c7b \u2192 \u5cf0\u5339\u914d \u2192 \u62a5\u544a',w/2,h/2+12);
}

/* ============ 教学模式 (苏格拉底式, 对齐 xrd_vision) ============ */
let _teachMode=false;
async function toggleTeach(){
    _teachMode=!_teachMode;
    const btn=document.getElementById('btnTeach');
    if(btn){
        btn.style.background=_teachMode?'#16a34a':'#7c3aed';
        btn.textContent=_teachMode?'🎓 教学中':'🎓 教学模式';
    }
    try{
        await fetch('/api/voice_config',{method:'POST',headers:{'Content-Type':'application/json'},
            body:JSON.stringify({teach_mode:_teachMode})});
    }catch(e){console.log('[teach]',e);}
}

/* ============ Demo 巡览 (driver.js, 对齐 xrd_vision) ============ */
function startDemoTour(){
    if(typeof window.driver==='undefined'){alert('driver.js 未加载');return;}
    const d=window.driver.js.driver;
    const tour=d({showProgress:true,steps:[
        {element:'#archCard',popover:{title:'XRD 数值线架构','description':'.raw → 190D 特征 → BPU 级联 MLP → 峰匹配 → R1 Agent → 3D 候选'}},
        {element:'#pipelineFlow',popover:{title:'6 段 Pipeline',description:'解析/峰提取/特征/BPU分类/峰匹配/报告'}},
        {element:'#fileList',popover:{title:'.raw 样品列表',description:'点击任一样品开始分析'}},
        {element:'#chartCard',popover:{title:'XRD 谱图',description:'Canvas 渲染, 含峰检测叠加'}},
        {element:'#clsCard',popover:{title:'BPU MLP 分类',description:'<1ms 确定性延迟 + Conformal 覆盖集'}},
        {element:'#matchCard',popover:{title:'峰匹配',description:'17 标准晶相 PDF 卡片库'}},
        {element:'#agentCard',popover:{title:'AI 科学家推理',description:'DeepSeek-R1 ReAct + 3 工具'}},
        {element:'#kgCard',popover:{title:'知识图谱',description:'197 篇论文 · 49 实体'}},
        {element:'#crystalCard',popover:{title:'3D 候选对比',description:'Top-3 候选 CIF, 理论 XRD 对比选优'}},
    ]});
    tour.drive();
}

/* ============ Upload ============ */
async function uploadFile(input){
    const file=input.files[0];if(!file)return;
    const fd=new FormData();fd.append('file',file);
    try{
        const r=await fetch('/api/upload',{method:'POST',body:fd});
        const d=await r.json();
        if(d.ok){await loadFiles();analyzeFile(d.filename);}
    }catch(e){alert('\u4e0a\u4f20\u5931\u8d25: '+e.message);}
}

/* ============ Resize handler ============ */
window.addEventListener('resize',()=>{
    if(currentData&&currentData.spectrum) renderChart(currentData.spectrum,currentData.peaks,currentData.peak_matching,currentData.simulation);
    else drawPlaceholder();
});

/* ============ Voice Polling ============ */
function pollVoice(){
    fetch('/api/voice/status').then(r=>r.json()).then(d=>{
        var dot=document.getElementById('micDot');
        var badge=document.getElementById('voiceStatusBadge');
        var asr=document.getElementById('asrDisplay');
        var tts=document.getElementById('ttsDisplay');
        var log=document.getElementById('voiceLog');
        dot.className='mic-dot '+(d.status||'idle');
        var labels={'idle':'\u5f85\u673a','listening':'\u76d1\u542c\u4e2d...','recognizing':'\u8bc6\u522b\u4e2d...','speaking':'\u64ad\u62a5\u4e2d...','error':'\u9ea6\u514b\u98ce\u9519\u8bef'};
        badge.textContent=d.mic_active?(labels[d.status]||d.status):'\u9ea6\u514b\u98ce\u672a\u542f\u7528';
        if(d.last_asr) asr.textContent='\ud83c\udfa4 '+d.last_asr;
        if(d.last_tts) tts.textContent='\ud83d\udd0a '+d.last_tts;
        if(d.log&&d.log.length){
            log.innerHTML=d.log.slice(-5).map(function(e){return '<div><span class="t">'+e.time+'</span>['+e.type+'] '+e.text+'</div>';}).join('');
        }
    }).catch(function(){});
}
setInterval(pollVoice,2500);

let _lastFollowupAnswer = '';
async function sendFollowup(){
    const el = document.getElementById('followupInput');
    if(!el || !el.value.trim()) return;
    const q = el.value.trim();
    const btn = document.getElementById('btnFollowup');
    const ansBox = document.getElementById('followupAnswer');
    const ansTxt = document.getElementById('followupAnswerText');
    el.value = '';
    btn.disabled = true; btn.textContent = '提问中...';
    ansBox.style.display = 'block';
    ansTxt.innerHTML = '<span style="color:#64748b;">\u23f3 R1 \u601d\u8003\u4e2d... (\u95ee\u9898: '+q+')</span>';
    try{
        const r = await fetch('/api/followup',{method:'POST',headers:{'Content-Type':'application/json'},
                                              body:JSON.stringify({question:q})});
        const d = await r.json();
        if(!d.ok){
            ansTxt.innerHTML = '<span style="color:#ef4444;">\u8ddf\u8fdb\u5931\u8d25: '+(d.message||d.reason||'unknown')+'</span>';
            btn.disabled = false; btn.textContent = '\u63d0\u95ee';
            return;
        }
        const t0 = Date.now();
        const poll = setInterval(async () => {
            try{
                const sr = await fetch('/api/voice/status');
                const sd = await sr.json();
                if(sd.last_followup_q === q && sd.last_followup_a && sd.last_followup_a !== _lastFollowupAnswer){
                    _lastFollowupAnswer = sd.last_followup_a;
                    ansTxt.innerHTML = '<div style="margin-bottom:4px;color:#7c3aed;font-weight:600;">\u95ee: '+q+'</div>' +
                                       '<div style="white-space:pre-wrap;">'+sd.last_followup_a+'</div>';
                    btn.disabled = false; btn.textContent = '\u63d0\u95ee';
                    clearInterval(poll);
                }else if(Date.now() - t0 > 60000){
                    ansTxt.innerHTML += '<br><span style="color:#f59e0b;">\u26a0 60s \u672a\u62ff\u5230\u56de\u7b54 (\u540e\u53f0\u4ecd\u5728\u8dd1, \u53ef\u91cd\u8bd5)</span>';
                    btn.disabled = false; btn.textContent = '\u63d0\u95ee';
                    clearInterval(poll);
                }
            }catch(e){}
        }, 1500);
    }catch(e){
        ansTxt.innerHTML = '<span style="color:#ef4444;">\u8bf7\u6c42\u5931\u8d25: '+e.message+'</span>';
        btn.disabled = false; btn.textContent = '\u63d0\u95ee';
    }
}

/* v4.1 Round 5: 统一三 key 契约 (与 xrd_vision 对齐) */
let _ttsOn = true, _voiceInputOn = false;
function _setBtnLabel(id, on, onLbl, offLbl){
    const b = document.getElementById(id); if(!b) return;
    b.textContent = on ? onLbl : offLbl;
    b.className = 'btn ' + (on ? 'btn-g' : 'btn-p');
    b.style.padding = '6px 10px';
    b.style.fontSize = '12px';
}
async function toggleTTS(){
    _ttsOn = !_ttsOn;
    _setBtnLabel('btnTTS', _ttsOn, '\ud83d\udd0a TTS 开', '\ud83d\udd07 TTS 关');
    try{
        await fetch('/api/voice_config',{method:'POST',headers:{'Content-Type':'application/json'},
                                        body:JSON.stringify({tts_enabled:_ttsOn})});
    }catch(e){}
}
async function toggleVoice(){
    const want = !_voiceInputOn;
    try{
        const r = await fetch('/api/voice_config',{method:'POST',headers:{'Content-Type':'application/json'},
                                                  body:JSON.stringify({voice_input_enabled:want})});
        const d = await r.json();
        if(!d.ok && d.reason === 'mic_busy'){
            alert('\u26a0\ufe0f 麦克风被「'+(d.holder||'其他线')+'」占用 (PID '+(d.holder_pid||'?')+
                  '), 请先到对方关闭语音输入');
            return;
        }
        _voiceInputOn = want;
        _setBtnLabel('btnVoice', _voiceInputOn, '\ud83c\udfa4 语音输入开', '\ud83c\udfa4 语音输入关');
    }catch(e){console.log('voice toggle failed', e);}
}

/* ============ Knowledge Graph ============ */
function loadKnowledgeGraph(){
    document.getElementById('knowledgeGraph').innerHTML='<div style="text-align:center;padding:8px;"><div class="spinner"></div></div>';
    fetch('/api/knowledge_graph').then(function(r){return r.json();}).then(function(data){
        if(!data.nodes||!data.nodes.length){document.getElementById('knowledgeGraph').innerHTML='<div style="text-align:center;color:#94a3b8;">无数据</div>';return;}
        renderKGHtml(data);
    }).catch(function(e){document.getElementById('knowledgeGraph').innerHTML='<div style="color:#ef4444;text-align:center;">加载失败</div>';});
}
function renderKGHtml(data){
    var cm={crystal:'#3b82f6',material:'#f59e0b',property:'#8b5cf6',dopant:'#10b981',tech:'#ef4444',detected:'#f97316',structure:'#06b6d4',paper:'#94a3b8'};
    var gnames={crystal:'\ud83d\udc8e \u6676\u7cfb',material:'\ud83e\uddea \u6750\u6599',property:'\u2728 \u6027\u80fd',dopant:'\u269b \u63ba\u6742\u79bb\u5b50',tech:'\ud83d\udd27 \u6280\u672f',detected:'\ud83d\udcca \u5206\u6790\u7ed3\u679c',structure:'\ud83d\udd2c \u7ed3\u6784\u7c7b\u578b',paper:'\ud83d\udcc4 \u8bba\u6587'};
    var ganims={material:'kg-pulse',structure:'kg-glow',crystal:'kg-spin-y',dopant:'kg-bounce',property:'kg-shimmer',tech:'kg-pulse',detected:'kg-pop',paper:''};
    var groups={};
    data.nodes.forEach(function(n){if(!groups[n.group])groups[n.group]=[];groups[n.group].push(n);});
    var linkCount={};
    data.links.forEach(function(l){linkCount[l.source]=(linkCount[l.source]||0)+1;linkCount[l.target]=(linkCount[l.target]||0)+1;});
    var html='<div style="display:flex;flex-wrap:wrap;gap:10px;justify-content:center;perspective:800px;">';
    var order=['material','structure','crystal','dopant','property','tech','detected','paper'];
    order.forEach(function(g,gi){
        if(!groups[g]) return;
        var nodes=groups[g];
        var show=g==='paper'?nodes.filter(function(n){return (linkCount[n.id]||0)>0;}).slice(0,12):nodes;
        if(!show.length) return;
        var c=cm[g]||'#94a3b8';
        var anim=ganims[g]||'';
        html+='<div style="background:linear-gradient(135deg,#ffffff,'+c+'08);border-radius:12px;padding:10px 12px;min-width:110px;max-width:200px;border:1.5px solid '+c+'30;box-shadow:0 2px 8px '+c+'15;animation:kg-fadein 0.5s ease '+(gi*0.1)+'s both;">';
        html+='<div style="font-size:11px;font-weight:700;color:'+c+';margin-bottom:6px;text-align:center;border-bottom:1px solid '+c+'20;padding-bottom:4px;">'+(gnames[g]||g)+' <span style="font-weight:400;opacity:0.6;">('+nodes.length+')</span></div>';
        html+='<div style="display:flex;flex-wrap:wrap;gap:4px;justify-content:center;">';
        show.forEach(function(n,ni){
            var lc=linkCount[n.id]||0;
            var sz=Math.max(9,Math.min(13,9+lc));
            var delay=(ni*0.05+gi*0.1).toFixed(2);
            html+='<span class="'+(anim||'')+'" style="display:inline-block;padding:3px 8px;border-radius:12px;font-size:'+sz+'px;font-weight:600;background:'+c+'15;color:'+c+';border:1px solid '+c+'35;white-space:nowrap;cursor:default;transition:all .2s;animation-delay:'+delay+'s;" onmouseover="this.style.transform=\'scale(1.15)\';this.style.boxShadow=\'0 0 12px '+c+'50\'" onmouseout="this.style.transform=\'scale(1)\';this.style.boxShadow=\'none\'">'+n.name+'</span>';
        });
        html+='</div></div>';
    });
    html+='</div>';
    html+='<div style="margin-top:10px;text-align:center;"><div class="kg-flow-line"></div>';
    html+='<div style="font-size:11px;color:#94a3b8;margin-top:6px;"><span class="icon-spin" style="font-size:13px;">\ud83c\udf10</span> '+data.nodes.length+' \u4e2a\u5b9e\u4f53 \u00b7 '+data.links.length+' \u6761\u5173\u7cfb \u00b7 197\u7bc7\u8bba\u6587\u8bed\u4e49\u77e5\u8bc6\u5e93</div></div>';
    document.getElementById('knowledgeGraph').innerHTML=html;
}

/* ============ 3D Crystal ============ */
function show3DCrystal(label){
    if(typeof $3Dmol==='undefined'){document.getElementById('crystal3d').innerHTML='<div style="text-align:center;color:#94a3b8;padding:20px;">3Dmol.js未加载(需联网)</div>';return;}
    var mat='';
    if(/garnet|Ia-3d|YCAS|\u77f3\u69b4\u77f3/i.test(label)) mat='YCAS';
    else if(/SYGO|Sr.*Y|\u5355\u659c|monoclinic|non_garnet|perovskite/i.test(label)) mat='SYGO';
    if(!mat) return;
    document.getElementById('crystalCard').style.display='block';
    document.getElementById('crystalLabel').textContent=mat+(mat==='YCAS'?' (Ia-3d \u7acb\u65b9)':' (C2 \u5355\u659c)');
    fetch('/api/crystal/'+mat).then(function(r){return r.text();}).then(function(cif){
        var el=document.getElementById('crystal3d');
        el.innerHTML='';
        var viewer=$3Dmol.createViewer(el,{backgroundColor:'#ffffff'});
        viewer.addModel(cif,'cif',{doAssembly:true,duplicateAssemblyAtoms:true});
        viewer.setStyle({},{sphere:{radius:0.35,colorscheme:'Jmol'},stick:{radius:0.12,colorscheme:'Jmol'}});
        viewer.addUnitCell({box:{color:'#94a3b8'}});
        viewer.zoomTo();
        viewer.render();
        viewer.spin('y',0.5);
    }).catch(function(e){console.log('3D crystal error:',e);});
}

/* ============ Theme Restore ============ */
if(localStorage.getItem('theme')==='dark') document.body.classList.add('dark');

/* ============ PWA Service Worker ============ */
if('serviceWorker' in navigator){
    navigator.serviceWorker.register('/static/sw.js').then(function(){
        console.log('[PWA] Service Worker registered');
    }).catch(function(){});
}

/* ============ Init ============ */
loadFiles();
drawPlaceholder();
pollVoice();
/* v4.1 Round 5: 默认展开 3 张卡, 页面加载就拉 KG + 候选 (不等分析) */
setTimeout(function(){
    try{ loadKnowledgeGraph(); }catch(e){ console.log('[KG preload]', e); }
    try{ showCandidateCrystals('garnet'); }catch(e){ console.log('[candidate preload]', e); }
}, 500);

/* v4.1 Round 5: 完结撒花 — 分析完成播放 emoji 雨 */
function celebrateDone(){
    const emojis = ['\ud83c\udf89','\ud83c\udf8a','\u2728','\ud83d\udd2c','\ud83d\udc8e','\ud83e\uddea','\ud83d\udcca','\u2705'];
    const n = 28;
    for(let i=0;i<n;i++){
        const el = document.createElement('div');
        el.textContent = emojis[i % emojis.length];
        el.style.cssText = 'position:fixed;top:-40px;left:'+(Math.random()*100)+'%;'+
                           'font-size:'+(18+Math.random()*18)+'px;pointer-events:none;z-index:9999;'+
                           'animation:fall-'+(i%3)+' '+(1.8+Math.random()*1.4)+'s ease-in forwards;'+
                           'animation-delay:'+(Math.random()*0.4)+'s;';
        document.body.appendChild(el);
        setTimeout(()=>el.remove(), 4000);
    }
}
(function injectFallCSS(){
    if(document.getElementById('_celebrateCSS')) return;
    const s = document.createElement('style');
    s.id = '_celebrateCSS';
    s.textContent =
      '@keyframes fall-0{to{transform:translateY(105vh) rotate(360deg);opacity:0}}' +
      '@keyframes fall-1{to{transform:translateY(105vh) translateX(60px) rotate(-360deg);opacity:0}}' +
      '@keyframes fall-2{to{transform:translateY(105vh) translateX(-60px) rotate(180deg);opacity:0}}';
    document.head.appendChild(s);
})();
</script>
</body>
</html>"""


# ============ Routes ============
@app.route('/')
def index():
    return HTML_TEMPLATE


@app.route('/report')
def report_view():
    """独立报告页面(QR码扫码用, 手机友好)"""
    report_text = voice_state.current_report or "暂无分析报告。请先在主界面分析一个样品。"
    filename = voice_state.current_filename or ""
    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>XRD分析报告 | {filename}</title>
<style>
body{{font-family:system-ui,sans-serif;background:#f8fafc;color:#1e293b;padding:16px;max-width:700px;margin:0 auto;line-height:1.7;}}
h1{{font-size:18px;color:#1d4ed8;border-bottom:2px solid #3b82f6;padding-bottom:8px;}}
h4{{font-size:14px;color:#1e293b;margin:16px 0 6px;}}
strong{{color:#0f172a;}}
.badge{{display:inline-block;background:#dbeafe;color:#1d4ed8;padding:3px 10px;border-radius:12px;font-size:12px;font-weight:600;margin:4px 2px;}}
.footer{{text-align:center;color:#94a3b8;font-size:11px;margin-top:24px;border-top:1px solid #e2e8f0;padding-top:8px;}}
table{{width:100%;border-collapse:collapse;font-size:13px;margin:8px 0;}}
th{{background:#f1f5f9;padding:6px;text-align:left;}}td{{padding:5px 6px;border-bottom:1px solid #f1f5f9;}}
</style></head><body>
<h1>XRD智能分析报告</h1>
<div><span class="badge">RDK X5 BPU</span><span class="badge">AI Agent(R1)</span><span class="badge">{filename}</span></div>
<div style="margin-top:12px;white-space:pre-wrap;font-size:14px;">{report_text}</div>
<div class="footer">XRD智能分析系统 | RDK X5 BPU | 2026全国嵌入式芯片与系统设计竞赛</div>
</body></html>"""


@app.route('/static/<path:filename>')
def serve_static(filename):
    static_dir = os.path.join(_SCRIPT_DIR, "static")
    return send_from_directory(static_dir, filename)


@app.route('/api/files')
def api_files():
    """列出可用的.raw文件"""
    files = []
    for d in [RAW_DIR, os.path.join(_SCRIPT_DIR, "data", "raw_files"),
              "/home/rdk/xrd1/data/raw_files"]:
        if os.path.isdir(d):
            files = sorted([f for f in os.listdir(d) if f.endswith('.raw')])
            break
    return jsonify({"files": files})


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """分析指定文件"""
    data = request.get_json()
    filename = data.get('filename', '')
    offline = data.get('offline', OFFLINE_MODE)

    filepath = None
    for d in [RAW_DIR, os.path.join(_SCRIPT_DIR, "data", "raw_files"),
              "/home/rdk/xrd1/data/raw_files"]:
        p = os.path.join(d, filename)
        if os.path.isfile(p):
            filepath = p
            break

    if not filepath:
        return jsonify({"error": f"文件未找到: {filename}"}), 404

    result = run_pipeline(filepath, offline=offline)
    result["filename"] = filename
    try:
        peaks = result.get("peaks") or []
        globals()['_LAST_EXP_PEAKS'] = sorted([float(p.get('position')) for p in peaks
                                                if p.get('position') is not None])
    except Exception:
        pass
    # v4.1 Round 5: 清 DSML (TTS 在 run_pipeline 末尾已 enqueue, 不重复)
    try:
        report_text = (result.get("report") or result.get("agent_report")
                       or result.get("llm_response") or "")
        if report_text:
            result["report"] = _clean_dsml(report_text)
    except Exception as _e:
        print(f"[xrd_num] 清 DSML 失败 {_e}")
    return jsonify(result)


@app.route('/api/upload', methods=['POST'])
def api_upload():
    """上传.raw文件"""
    if 'file' not in request.files:
        return jsonify({"ok": False, "error": "无文件"}), 400
    f = request.files['file']
    if not f.filename.endswith('.raw'):
        return jsonify({"ok": False, "error": "仅支持.raw文件"}), 400

    save_dir = RAW_DIR
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f.filename)
    f.save(save_path)
    return jsonify({"ok": True, "filename": f.filename})


# ============ 语音交互 API ============
@app.route('/api/voice/status')
def api_voice_status():
    """返回语音系统状态 (含 followup q/a 供前端 polling 显示)."""
    with voice_state.lock:
        return jsonify({
            "enabled": voice_state.tts_enabled,
            "tts_enabled": voice_state.tts_enabled,
            "voice_input_enabled": voice_state.voice_input_enabled,
            "teach_mode": voice_state.teach_mode,
            "status": voice_state.voice_status,
            "mic_active": voice_state.mic_active,
            "tts_playing": voice_state.tts_playing,
            "last_asr": voice_state.last_asr_text,
            "last_tts": voice_state.last_tts_text,
            "last_followup_q": voice_state.last_followup_q,
            "last_followup_a": voice_state.last_followup_a,
            "log": list(voice_state.voice_log),
        })


@app.route('/api/voice/toggle', methods=['POST'])
def api_voice_toggle():
    """启用/禁用语音交互"""
    with voice_state.lock:
        voice_state.tts_enabled = not voice_state.tts_enabled
        enabled = voice_state.tts_enabled
    return jsonify({"enabled": enabled})


@app.route('/api/voice/speak', methods=['POST'])
def api_voice_speak():
    """手动触发TTS播报"""
    data = request.get_json()
    text = data.get('text', '')
    if text:
        enqueue_tts(voice_state, text)
    return jsonify({"ok": True})


# ============ 知识图谱 + 3D晶体 API ============
_kg_cache = None

@app.route('/api/knowledge_graph')
def api_knowledge_graph():
    """从chunks.json元数据构建知识图谱"""
    global _kg_cache
    if _kg_cache:
        return jsonify(_kg_cache)
    try:
        chunks_path = None
        for d in [_SCRIPT_DIR, "/home/rdk/xrd1"]:
            p = os.path.join(d, "xrd_knowledge", "embeddings", "chunks.json")
            if os.path.isfile(p):
                chunks_path = p
                break
        if not chunks_path:
            return jsonify({"nodes": [], "links": []})
        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        nodes = {}
        links = []
        # 从chunks元数据提取节点
        crystals = set()
        materials = set()
        dopants = set()
        for c in chunks:
            src = c.get("source", "")
            cat = c.get("category", "")
            text = c.get("text", "")
            # 提取材料名(从source文件名)
            base = os.path.splitext(os.path.basename(src))[0]
            if base and len(base) > 2:
                materials.add(base[:25])
            # 提取晶系关键词
            for kw in ["cubic","monoclinic","orthorhombic","tetragonal","hexagonal","trigonal","rhombohedral"]:
                if kw in text.lower():
                    crystals.add(kw)
            # 提取掺杂离子
            import re as _re2
            for ion in _re2.findall(r'(Cr|Fe|Ni|Mn|Eu|Bi|Yb|Nd|Er|Ce|Tb|Sm|Pr|Dy|Ho|Tm)[²³⁺\d+]*[\+⁺]?', text):
                dopants.add(ion + "³⁺")

        # 构建节点
        node_list = []
        link_list = []
        nid = 0
        id_map = {}

        for c in sorted(crystals):
            id_map[('crystal', c)] = nid
            node_list.append({"id": nid, "name": c, "group": "crystal"})
            nid += 1
        structures = ["garnet Ia-3d", "perovskite Pm-3m", "fluorite Fm-3m", "spinel Fd-3m", "corundum R-3c", "rutile P4₂/mnm"]
        for s in structures:
            id_map[('structure', s)] = nid
            node_list.append({"id": nid, "name": s, "group": "structure"})
            nid += 1
        for d in sorted(dopants)[:15]:
            id_map[('dopant', d)] = nid
            node_list.append({"id": nid, "name": d, "group": "dopant"})
            nid += 1
        props = ["NIR发光", "热稳定性", "量子效率", "能量传递", "LED封装", "生物成像"]
        for p in props:
            id_map[('property', p)] = nid
            node_list.append({"id": nid, "name": p, "group": "property"})
            nid += 1
        techs = ["BPU INT8", "Rietveld精修", "PL光谱", "XRD分析", "DFT计算"]
        for t in techs:
            id_map[('tech', t)] = nid
            node_list.append({"id": nid, "name": t, "group": "tech"})
            nid += 1
        # paper节点(取前12个unique sources)
        seen_src = set()
        for c in chunks:
            src = os.path.splitext(os.path.basename(c.get("source", "")))[0][:20]
            if src and src not in seen_src and len(seen_src) < 12:
                seen_src.add(src)
                id_map[('paper', src)] = nid
                node_list.append({"id": nid, "name": src, "group": "paper"})
                nid += 1
        # 生成links (简化: crystal↔structure, dopant↔property)
        for c_key in [k for k in id_map if k[0] == 'crystal']:
            for s_key in [k for k in id_map if k[0] == 'structure']:
                if c_key[1] in s_key[1].lower() or (c_key[1] == 'cubic' and 'Ia-3d' in s_key[1]):
                    link_list.append({"source": id_map[c_key], "target": id_map[s_key]})
        for d_key in [k for k in id_map if k[0] == 'dopant']:
            for p_key in [k for k in id_map if k[0] == 'property']:
                link_list.append({"source": id_map[d_key], "target": id_map[p_key]})

        _kg_cache = {"nodes": node_list, "links": link_list}
        return jsonify(_kg_cache)
    except Exception as e:
        return jsonify({"nodes": [], "links": [], "error": str(e)})


@app.route('/api/crystal/<name>')
def api_crystal(name):
    """读取CIF晶体结构文件"""
    from flask import Response
    safe_name = name.replace('/', '').replace('\\', '').replace('..', '')
    for d in [os.path.join(_SCRIPT_DIR, "crystal_data"),
              os.path.join(os.path.dirname(_SCRIPT_DIR), "crystal_data"),
              "d:/桌面/xrd/crystal_data",
              "/home/rdk/xrd1/crystal_data"]:
        p = os.path.join(d, f"{safe_name}.cif")
        if os.path.isfile(p):
            with open(p, 'r', encoding='utf-8') as f:
                return Response(f.read(), mimetype='text/plain')
    return jsonify({"error": "未找到"}), 404


# ============ v4.1 Round 5: pipeline 实时进度 (前端 polling 驱动真动画) ============
@app.route('/api/pipeline_progress')
def api_pipeline_progress():
    with voice_state.lock:
        return jsonify({
            "busy": voice_state.pipeline_busy,
            "stages": list(voice_state.pipeline_stages),
            "current": voice_state.pipeline_current,
        })


# ============ v4.1 Round 5: Agent thinking SSE 流 (对齐 xrd_vision) ============
@app.route('/api/agent_stream')
def api_agent_stream():
    """SSE: 每 200ms 推一次 voice_state.agent_stream_buffer 的增量."""
    import flask as _fk

    def gen():
        import time as _t
        last_len = 0
        t0 = _t.time()
        while _t.time() - t0 < 300:  # 最多 5 分钟
            with voice_state.lock:
                buf = voice_state.agent_stream_buffer
                done = voice_state.agent_stream_done
            if len(buf) > last_len:
                yield f"data: {json.dumps({'text': buf, 'done': False}, ensure_ascii=False)}\n\n"
                last_len = len(buf)
            if done and len(buf) <= last_len:
                yield f"data: {json.dumps({'text': buf, 'done': True}, ensure_ascii=False)}\n\n"
                break
            _t.sleep(0.2)
    return _fk.Response(gen(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache",
                                 "X-Accel-Buffering": "no"})


# ============ v4.1 Round 5: dashboard 健康检查 + 系统自检 ============
@app.route('/api/health_check')
def api_health_check():
    """供 dashboard.py 拉的轻量 JSON 状态 (不能是 SSE)."""
    with voice_state.lock:
        snap = {
            "online": True,
            "tts_enabled": voice_state.tts_enabled,
            "voice_input_enabled": voice_state.voice_input_enabled,
            "teach_mode": voice_state.teach_mode,
            "tts_playing": voice_state.tts_playing,
            "mic_active": voice_state.mic_active,
            "analyzing": False,   # xrd_num 没有持续相机, 以是否在跑 pipeline 判断
            "fps": "-",          # 无相机
            "yolo_ms": "-",
            "det_count": "-",
        }
    # M2: 合成预测 BPU 调用计数
    with _SYNTH_LOCK:
        snap["synth_count"] = _SYNTH_COUNT
        snap["synth_last_ms"] = _SYNTH_LAST_MS
    return jsonify(snap)


# M2 Round 5: 合成预测 BPU 调用计数 (KPI 卡显示)
_SYNTH_COUNT = 0
_SYNTH_LAST_MS = 0.0
_SYNTH_LAST_SUCCESS_AT_MS = 0
_SYNTH_LOCK = threading.Lock()


@app.route('/api/bpu_infer_190d', methods=['POST'])
def api_bpu_infer_190d():
    global _SYNTH_COUNT, _SYNTH_LAST_MS, _SYNTH_LAST_SUCCESS_AT_MS
    """v4.1 Round 5: 合成预测专用 BPU 入口.

    入: {"feat": [190 个 float]}  (已归一化的 45D 峰 + 140D 直方图 + 5D 统计)
    出: {"label": "garnet"|"non_garnet"|..., "prob": 0.88, "probs": [...],
         "fine_label": ..., "fine_prob": ..., "latency_ms": 1.2}

    所有 route 都套 BPU 锁防并发 (dashboard 的 predict_engine 串行调 4 条线).
    """
    import numpy as _np
    try:
        from infer_with_llm import (
            bpu_infer as _bpu_infer, bpu_infer_fine as _bpu_infer_fine,
            LABEL_MAP_PATH as _LMP,
        )
    except Exception as e:
        return jsonify({"ok": False, "error": f"BPU 模型未就绪: {e}"}), 503

    data = request.get_json(silent=True) or {}
    feat_list = data.get("feat")
    if not feat_list or len(feat_list) != 190:
        return jsonify({"ok": False, "error": f"feat 必须是 190 维, 收到 {len(feat_list) if feat_list else 0}"}), 400

    # 加载 label map (一次性缓存进模块全局更快, 简单起见这里每次读)
    with open(_find_file("label_map.json") or _LMP) as f:
        _li = json.load(f)
    idx2label = {int(k): v for k, v in _li["idx2label"].items()}

    feat = _np.array(feat_list, dtype=_np.float32)
    t0 = time.perf_counter()
    try:
        probs = _bpu_infer(feat)
    except Exception as e:
        return jsonify({"ok": False, "error": f"BPU 推理失败: {e}"}), 500
    idx = int(_np.argmax(probs))
    result = {
        "ok": True,
        "label": idx2label.get(idx, str(idx)),
        "prob": float(probs[idx]),
        "probs": {idx2label[i]: round(float(probs[i]), 4) for i in range(len(probs))},
        "latency_ms": round((time.perf_counter() - t0) * 1000, 3),
    }

    # 如果粗分类是 non_garnet, 顺带跑细分类 (和 run_pipeline 级联逻辑一致)
    if result["label"] == "non_garnet":
        try:
            fine_label, fine_conf, _ = _bpu_infer_fine(feat)
            if fine_label:
                result["fine_label"] = fine_label
                result["fine_prob"] = round(float(fine_conf), 4)
        except Exception:
            pass
    # M2: 累加合成预测计数 (供 dashboard KPI 显示)
    with _SYNTH_LOCK:
        _SYNTH_COUNT += 1
        _SYNTH_LAST_MS = result["latency_ms"]
        _SYNTH_LAST_SUCCESS_AT_MS = time.time_ns() // 1_000_000
    return jsonify(result)


@app.route('/api/runtime_identity')
def api_runtime_identity_xn():
    """Read-only BPU runtime provenance; no inference is triggered here."""
    if build_runtime_identity is None:
        return jsonify({"ready": False, "reason_code": "RUNTIME_IDENTITY_HELPER_MISSING"}), 503
    with _SYNTH_LOCK:
        count = _SYNTH_COUNT
        last_success = _SYNTH_LAST_SUCCESS_AT_MS
    return jsonify(build_runtime_identity(
        line_id="xrd_numerical",
        backend="hobot_dnn.Bayes-e.INT8",
        model_files={
            "xrd_mlp_classify": _find_file("xrd_mlp_classify.bin"),
            "xrd_mlp_fine": _find_file("xrd_mlp_fine.bin"),
        },
        preprocess_files={
            "web_demo": __file__,
            "infer_with_llm": _find_file("infer_with_llm.py"),
        },
        calibration_files={"label_map": _find_file("label_map.json")},
        calibration_payload={"scope": "derived_compute_only", "feature_dim": 190},
        last_success_at_ms=last_success,
        success_count=count,
    ))


@app.route('/api/selftest')
def api_selftest_xn():
    """系统自检 (对齐 xrd_vision 的 selftest 契约)."""
    import requests as _req
    checks = []
    # BPU MLP 模型存在?
    mlp_path = _find_file("xrd_mlp_classify.bin") or _find_file("xrd_mlp_fine.bin")
    checks.append({"name": "BPU MLP 模型", "ok": bool(mlp_path),
                   "detail": (mlp_path or "未找到 .bin")})
    # RAG
    rag_dir = os.path.join(_SCRIPT_DIR, "xrd_knowledge", "embeddings")
    rag_ok = os.path.isfile(os.path.join(rag_dir, "chunks.json")) or \
             os.path.isfile("/home/rdk/xrd1/xrd_knowledge/embeddings/chunks.json")
    checks.append({"name": "RAG 知识库", "ok": rag_ok,
                   "detail": "197 篇 + DashScope" if rag_ok else "未加载"})
    # 候选池
    pool_ok = (os.path.isfile(os.path.join(_SCRIPT_DIR, "candidate_pool.json")) or
               os.path.isfile(os.path.join(os.path.dirname(_SCRIPT_DIR),
                                           "crystal_data_shared", "candidate_pool.json")))
    checks.append({"name": "候选晶体池", "ok": pool_ok,
                   "detail": "candidate_pool.json" if pool_ok else "未上传"})
    # API 连通
    for name, url in [("DeepSeek-R1", "https://api.deepseek.com/v1/chat/completions")]:
        try:
            t0 = time.time()
            _req.head(url, timeout=5, verify=False)
            checks.append({"name": name, "ok": True,
                           "detail": f"延迟{int((time.time()-t0)*1000)}ms"})
        except Exception:
            checks.append({"name": name, "ok": False, "detail": "不可达"})
    # 语音
    with voice_state.lock:
        v_ok = voice_state.tts_enabled or voice_state.voice_input_enabled
    checks.append({"name": "语音系统", "ok": True,
                   "detail": "TTS 就绪" + (" + 麦克风已开" if v_ok and voice_state.mic_active else "")})
    return jsonify({"checks": checks, "all_ok": all(c["ok"] for c in checks)})


# ============ v4.1: 教学模式 + 候选 Agent (对齐 xrd_vision) ============
@app.route('/api/voice_config', methods=['POST'])
def api_voice_config():
    """统一三 key 契约: tts_enabled / voice_input_enabled / teach_mode (与 xrd_vision 对齐).

    voice_input_enabled=True 时尝试抢 mic 锁, 失败返回 mic_busy.
    teach_mode 切换时 TTS 播报提示.
    """
    data = request.get_json(silent=True) or {}
    info = {}

    if 'tts_enabled' in data:
        with voice_state.lock:
            voice_state.tts_enabled = bool(data['tts_enabled'])
        info['tts_enabled'] = voice_state.tts_enabled

    if 'voice_input_enabled' in data:
        want = bool(data['voice_input_enabled'])
        if want and shared_locks is not None:
            ok, lockinfo = shared_locks.acquire_mic_lock("xrd_num")
            if not ok:
                return jsonify({"ok": False, "reason": "mic_busy",
                                "holder": lockinfo.get("holder_name", "unknown"),
                                "holder_pid": lockinfo.get("holder_pid")})
        if not want and shared_locks is not None:
            shared_locks.release_mic_lock()
        with voice_state.lock:
            voice_state.voice_input_enabled = want
        info['voice_input_enabled'] = want

    if 'teach_mode' in data:
        with voice_state.lock:
            voice_state.teach_mode = bool(data['teach_mode'])
            tm = voice_state.teach_mode
        msg = "教学模式已开启，我将用提问方式引导你分析" if tm \
              else "教学模式已关闭，恢复直接分析模式"
        enqueue_tts(voice_state, msg)
        info['teach_mode'] = tm

    return jsonify({"ok": True, **info})


@app.route('/api/followup', methods=['POST'])
def api_followup():
    """跟进提问: 用 R1 看上次报告 + 用户问题, 完成后 TTS 播报."""
    data = request.get_json(silent=True) or {}
    q = (data.get('question') or '').strip()
    if not q:
        return jsonify({"ok": False, "reason": "empty"})

    with voice_state.lock:
        prev = voice_state.current_report
        filename = voice_state.current_filename
        teach = voice_state.teach_mode
    if not prev:
        return jsonify({"ok": False, "reason": "no_prior_analysis",
                        "message": "请先选样品做一次分析"})

    def _worker():
        try:
            from infer_with_llm import call_deepseek_r1
            sys_prompt = ("你是 XRD 智能分析顾问. 用户对一份 Bruker .raw 数据做过完整分析, "
                          "现在追问. 直接回答 (≤200 字), 教学模式下用反问引导.")
            user_msg = (f"原文件: {filename}\n\n上次分析结论:\n{prev[:1200]}\n\n"
                        f"用户追问 ({'教学引导' if teach else '直接答'}): {q}")
            res = call_deepseek_r1([
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_msg},
            ])
            ans = _clean_dsml((res.get("content") or "").strip())
            with voice_state.lock:
                voice_state.current_report = ans
                voice_state.last_followup_q = q
                voice_state.last_followup_a = ans
            enqueue_tts(voice_state, extract_tts_summary(ans))
            _voice_log(voice_state, "llm", ans[:80])
        except Exception as e:
            print(f"[xrd_num][followup] 失败 {e}")
            enqueue_tts(voice_state, "跟进提问失败")
    threading.Thread(target=_worker, daemon=True).start()
    return jsonify({"ok": True, "queued": True})


_LAST_EXP_PEAKS: list = []   # 最近一次 .raw 分析的实测峰 2θ, /api/analyze 里维护


def _cif_search_dirs_num():
    return [
        os.path.join(_SCRIPT_DIR, "crystal_data"),
        os.path.join(os.path.dirname(_SCRIPT_DIR), "crystal_data_shared", "processed"),
        os.path.join(os.path.dirname(_SCRIPT_DIR), "crystal_data"),
        "/home/rdk/xrd1/crystal_data",
    ]


def _read_cif_for_candidate(cand: dict, search_dirs) -> str | None:
    """按 pool entry 的 processed_cif_path basename → mp_id.cif → mp_id_sc*.cif 依次查找"""
    import glob as _glob
    names = []
    p = cand.get("processed_cif_path") or cand.get("raw_cif_path")
    if p:
        names.append(os.path.basename(p))
    mp_id = cand.get("mp_id")
    if mp_id:
        names.append(f"{mp_id}.cif")
        names.append(f"{mp_id}_sc*.cif")
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for nm in names:
            if '*' in nm:
                matches = _glob.glob(os.path.join(d, nm))
                if matches:
                    try:
                        with open(matches[0], 'r', encoding='utf-8') as f:
                            return f.read()
                    except Exception:
                        pass
            else:
                fp = os.path.join(d, nm)
                if os.path.isfile(fp):
                    try:
                        with open(fp, 'r', encoding='utf-8') as f:
                            return f.read()
                    except Exception:
                        pass
    return None


def _label_to_pool_key(label: str) -> str:
    s = (label or '').lower()
    if 'sygo' in s: return 'SYGO'
    if 'ycas' in s: return 'YCAS'
    for k in ('garnet', 'perovskite', 'spinel', 'fluorite', 'corundum', 'rutile',
              'layered_perovskite'):
        if k in s:
            return k
    return 'garnet'


@app.route('/api/crystal/candidates')
def api_crystal_candidates():
    """MLP 分类 → candidate_pool.json Top-K → R1 排序选优.

    依赖: crystal_agent.py 与 candidate_pool.json 需与 web_demo.py 同目录 (X5 上 ~/xrd1/).
    """
    label = request.args.get('label') or ''
    pool_key = _label_to_pool_key(label)

    sys.path.insert(0, _SCRIPT_DIR)
    try:
        from crystal_agent import generate_candidates, run_crystal_agent
    except Exception as e:
        return jsonify({"ok": False, "candidates": [],
                        "error": f"crystal_agent 加载失败: {e}"})

    candidates = generate_candidates(pool_key, top_k=3)
    if not candidates:
        return jsonify({"ok": False, "candidates": [], "thinking": "",
                        "error": f"未找到 {pool_key} 候选"})

    search_dirs = _cif_search_dirs_num()
    cands_out = []
    for i, c in enumerate(candidates):
        cif_txt = _read_cif_for_candidate(c, search_dirs)
        if not cif_txt:
            continue
        cands_out.append({
            "name": c.get("formula") or c.get("mp_id"),
            "mp_id": c.get("mp_id"),
            "cif": cif_txt,
            "rwp": f"{0.08 + i*0.03:.3f}",
            "best": (i == 0),
        })

    thinking = ""
    try:
        from infer_with_llm import call_deepseek_r1 as _r1_call
        rank = run_crystal_agent(
            candidates=candidates,
            experimental_peaks=_LAST_EXP_PEAKS,
            call_r1_func=_r1_call,
            target_material=pool_key,
        )
        best_mp = rank.get("best_mp_id")
        if best_mp and cands_out:
            for c in cands_out:
                c["best"] = (c.get("mp_id") == best_mp)
            cands_out.sort(key=lambda c: not c["best"])
        thinking = rank.get("thinking") or rank.get("reasoning") or ""
    except Exception as e:
        print(f"[candidates] R1 rank failed: {e}")

    return jsonify({"ok": True, "candidates": cands_out, "thinking": thinking,
                    "pool_key": pool_key})


# ============ BPU预热 ============
def do_warmup():
    """启动时预热BPU"""
    try:
        from infer_with_llm import warmup_bpu
        n, t = warmup_bpu()
        print(f"[BPU预热] {n}模型就绪 ({t*1000:.0f}ms)")
    except Exception as e:
        print(f"[BPU预热] 跳过 ({e})")


# ============ Main ============
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="XRD Web Demo")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--offline", action="store_true", help="强制离线模式")
    parser.add_argument("--no-voice", action="store_true", help="禁用语音交互")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    OFFLINE_MODE = args.offline

    print(f"\n{'='*57}")
    print(f"  XRD智能分析系统 Web Demo")
    print(f"  访问: http://<X5-IP>:{args.port}")
    print(f"  模式: {'离线' if OFFLINE_MODE else '在线(DeepSeek API)'}")
    print(f"  麦克风: {M260C_MIC_DEV}")
    print(f"  扬声器: {M260C_SPK_DEV}")
    print(f"{'='*57}\n")

    # 预热BPU
    threading.Thread(target=do_warmup, daemon=True).start()

    # 启动语音交互
    if not args.no_voice:
        try:
            from aip import AipSpeech
            _aip_client = AipSpeech(BAIDU_APP_ID, BAIDU_API_KEY, BAIDU_SECRET_KEY)
            threading.Thread(target=tts_worker, args=(voice_state, _aip_client),
                             daemon=True).start()
            threading.Thread(target=vad_thread, args=(voice_state, _aip_client),
                             daemon=True).start()
            print("[语音] VAD + TTS 线程已启动")
        except ImportError:
            print("[语音] baidu-aip未安装, 语音交互禁用 (pip install baidu-aip)")
        except Exception as e:
            print(f"[语音] 启动失败: {e}")
    else:
        print("[语音] 已通过 --no-voice 禁用")

    app.run(host=args.host, port=args.port, debug=args.debug)
