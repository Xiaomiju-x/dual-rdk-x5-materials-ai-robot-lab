"""command_interpreter — "中文人话指令" → DispatchTask goal 字段.

按 ADR-EB-8, 三档实现可一行切换:
    RuleInterpreter      Python 正则匹配 5-10 条固定指令模板, 离线可用 (默认)
    LocalLLMInterpreter  调本地 llama-server (Qwen3-0.6B / Gemma 4 1B INT4 GGUF)
    RemoteLLMInterpreter 调 AI 脑 dashboard:8888 现成 LLM (Qwen2.5-1.5B SFT v2)

接口签名跟 OpenAI Chat Completion 兼容, 后期换底层模型一行 import 改完事.

输入: utterance (str, 中文人话) + context (dict, 可选)
输出: TaskResult dataclass (字段名跟 InterpretCommand.srv 一致)
"""
from __future__ import annotations

import json
import os
import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

# ============== 数据类 ==============


@dataclass
class TaskResult:
    success: bool = False
    reason: str = ''
    backend_used: str = 'rule'
    task_id: str = ''
    task_type: str = ''           # fetch_sample / deliver_to_furnace / monitor_furnace / observe / patrol / home
    bottle_id: str = ''           # observe 模式下复用为 VLM prompt (英文)
    from_location: str = ''
    to_location: str = ''
    priority: int = 2             # 1=low 2=normal 3=high
    timeout_s: float = 0.0
    raw_response: str = ''


# ============== 抽象基类 ==============


class CommandInterpreter(ABC):
    """所有 backend 的统一接口."""

    @abstractmethod
    def parse(self, utterance: str, context: Optional[Dict[str, Any]] = None) -> TaskResult:
        ...

    @staticmethod
    def _new_task_id() -> str:
        return f'cmd-{uuid.uuid4().hex[:8]}'


# ============== Rule (默认) ==============


class RuleInterpreter(CommandInterpreter):
    """正则 + 关键词匹配 5-10 条固定指令. 不依赖网络/模型.

    支持的模式 (中文):
        "去 X 号(柜|架|架子) 取 Y 瓶"               → fetch_sample
        "把 Y 送到 X 号(炉|炉子)"                    → deliver_to_furnace
        "监控 X 号(炉|炉子)"                         → monitor_furnace
        "巡更" / "巡视一圈"                          → patrol
        "回(去|来|工位|home|原点)"                   → home
        "停(下|住)" / "急停"                         → home (优先级高)
    """

    # 中文数字 → 阿拉伯
    CN_NUM = {
        '零': 0, '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
        '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
    }

    @classmethod
    def _to_int(cls, s: str) -> Optional[int]:
        if not s:
            return None
        if s.isdigit():
            return int(s)
        if s in cls.CN_NUM:
            return cls.CN_NUM[s]
        return None

    def parse(self, utterance: str, context: Optional[Dict[str, Any]] = None) -> TaskResult:
        u = utterance.strip()
        r = TaskResult(backend_used='rule', task_id=self._new_task_id(), raw_response=u)

        if not u:
            r.reason = 'empty utterance'
            return r

        # 急停 / 回工位
        if re.search(r'(急停|停下|停住|强制停|停车)', u):
            r.success = True
            r.task_type = 'home'
            r.priority = 3  # high
            r.raw_response = 'matched: 急停'
            return r
        if re.search(r'回\s*(去|来|工位|home|原点|home位)', u, re.IGNORECASE):
            r.success = True
            r.task_type = 'home'
            r.priority = 2
            r.raw_response = 'matched: home'
            return r

        # 巡更
        if re.search(r'(巡更|巡视|巡逻|巡一圈|走一圈)', u):
            r.success = True
            r.task_type = 'patrol'
            r.priority = 1
            r.raw_response = 'matched: patrol'
            return r

        # 监控炉子
        m = re.search(r'监控\s*(\d+|[零一二三四五六七八九十])\s*号?\s*(炉子?|烧结炉)', u)
        if m:
            n = self._to_int(m.group(1))
            r.success = True
            r.task_type = 'monitor_furnace'
            r.to_location = f'furnace_{n}'
            r.priority = 2
            r.raw_response = f'matched: monitor furnace {n}'
            return r

        # observe (Day 13 C1) — 走到目标点 + VLM 描述
        # "看一下 N 号(炉|炉子|架|柜|试剂柜)" / "看看 N 号炉" / "去看 N 号炉" / "N 号炉怎么样"
        m = re.search(
            r'(?:看一?下|看看|描述|去看|瞧瞧)\s*'
            r'(\d+|[零一二三四五六七八九十])\s*号?\s*'
            r'(炉子?|烧结炉|架子?|试剂柜|柜子?)',
            u,
        )
        if m:
            n = self._to_int(m.group(1))
            kind = m.group(2)
            r.success = True
            r.task_type = 'observe'
            if kind in ('炉', '炉子', '烧结炉'):
                r.to_location = f'furnace_{n}'
                r.bottle_id = 'What is the temperature shown on the LCD?'
            else:
                r.to_location = f'shelf_{n}'
                r.bottle_id = 'Describe what bottles are on this shelf.'
            r.priority = 2
            r.raw_response = f'matched: observe {r.to_location} prompt={r.bottle_id!r}'
            return r
        # 兼容: "N 号炉怎么样" / "N 号炉烧到几度"
        m = re.search(
            r'(\d+|[零一二三四五六七八九十])\s*号?\s*(炉子?|烧结炉)\s*(?:怎么样|几度|多少度|烧到|温度)',
            u,
        )
        if m:
            n = self._to_int(m.group(1))
            r.success = True
            r.task_type = 'observe'
            r.to_location = f'furnace_{n}'
            r.bottle_id = 'What is the temperature shown on the LCD?'
            r.priority = 2
            r.raw_response = f'matched: observe furnace_{n} (temp)'
            return r

        # 取料: "去 X 号柜 取 Y" / "去取 Y" / "拿 Y" / "去 X 号 取 Y"
        m = re.search(
            r'(?:去|到)?\s*(?:(\d+|[零一二三四五六七八九十])\s*号?\s*'
            r'(?:试剂柜|柜子|架子|架))?\s*(?:取|拿)\s*'
            r'([A-Za-z][A-Za-z0-9_-]*\s*\d*\s*号?瓶?|[\d]+\s*号?瓶?)',
            u
        )
        if m:
            shelf_n = self._to_int(m.group(1) or '')
            bottle_raw = m.group(2).replace(' ', '').replace('号', '').replace('瓶', '')
            r.success = True
            r.task_type = 'fetch_sample'
            r.bottle_id = bottle_raw
            if shelf_n is not None:
                r.from_location = f'shelf_{shelf_n}'
            r.priority = 2
            r.raw_response = f'matched: fetch {bottle_raw} from shelf={shelf_n}'
            return r

        # 送货: "把 X 送到 Y 号炉" / "送 X 到 Y 号炉"
        m = re.search(
            r'(?:把|送)\s*([A-Za-z][A-Za-z0-9_-]*\s*\d*\s*号?瓶?|[\d]+\s*号?瓶?)\s*'
            r'(?:送)?(?:到|去)\s*(\d+|[零一二三四五六七八九十])\s*号?\s*(?:炉子?|烧结炉)',
            u
        )
        if m:
            bottle_raw = m.group(1).replace(' ', '').replace('号', '').replace('瓶', '')
            n = self._to_int(m.group(2))
            r.success = True
            r.task_type = 'deliver_to_furnace'
            r.bottle_id = bottle_raw
            r.to_location = f'furnace_{n}'
            r.priority = 2
            r.raw_response = f'matched: deliver {bottle_raw} → furnace {n}'
            return r

        r.reason = 'no rule pattern matched'
        return r


# ============== Local LLM (本地 llama-server) ==============


class LocalLLMInterpreter(CommandInterpreter):
    """调本地 llama-server (Qwen3-0.6B / Gemma 4 1B INT4 GGUF) 解析.

    假设 llama-server 在 X5 :9100 (或参数 endpoint), 兼容 OpenAI Chat Completion API:
        POST /v1/chat/completions
        {model: "qwen3-0.6b", messages: [...], temperature: 0.0}

    Phase 5 留接口, Phase 6 真训完小模型再启用.
    """

    SYSTEM_PROMPT = """你是一个中文指令解析器, 把用户口语化指令转换成 JSON.

输出格式严格:
{
  "task_type": "fetch_sample | deliver_to_furnace | monitor_furnace | patrol | home",
  "bottle_id": "字符串, 没有则空",
  "from_location": "shelf_N 或空",
  "to_location": "furnace_N 或空",
  "priority": "low | normal | high"
}

只输出 JSON, 不要任何解释或 markdown 代码块标记."""

    def __init__(self, endpoint: str = 'http://127.0.0.1:9100/v1/chat/completions',
                 model: str = 'qwen3-0.6b'):
        self.endpoint = endpoint
        self.model = model

    def parse(self, utterance: str, context: Optional[Dict[str, Any]] = None) -> TaskResult:
        r = TaskResult(backend_used='local', task_id=self._new_task_id(), raw_response='')
        try:
            import requests
        except ImportError:
            r.reason = 'requests not installed'
            return r

        try:
            resp = requests.post(
                self.endpoint,
                json={
                    'model': self.model,
                    'messages': [
                        {'role': 'system', 'content': self.SYSTEM_PROMPT},
                        {'role': 'user', 'content': utterance},
                    ],
                    'temperature': 0.0,
                    'max_tokens': 128,
                },
                timeout=10,
            )
            if resp.status_code != 200:
                r.reason = f'local llm http {resp.status_code}'
                return r
            j = resp.json()
            text = j['choices'][0]['message']['content']
            r.raw_response = text
            return _parse_json_to_task(text, r)
        except Exception as e:
            r.reason = f'local llm error: {e}'
            return r


# ============== Remote (调 AI 脑) ==============


class RemoteLLMInterpreter(CommandInterpreter):
    """调 AI 脑 dashboard:8888 现成 9 LLM (默认 Qwen2.5-1.5B SFT v2).

    AI 脑端期望端点 (跟 my_robot_bridge 用同一个):
        POST /api/interpret_command
        body: {utterance, context}
        return: {task_type, bottle_id, from_location, to_location, priority}
    """

    def __init__(self, base_url: Optional[str] = None, timeout_s: float = 8.0):
        self.base_url = (base_url or os.environ.get(
            'EB_AI_BRAIN_URL', 'http://192.0.2.103:8888')).rstrip('/')
        self.timeout = timeout_s

    def parse(self, utterance: str, context: Optional[Dict[str, Any]] = None) -> TaskResult:
        r = TaskResult(backend_used='remote', task_id=self._new_task_id(), raw_response='')
        try:
            import requests
        except ImportError:
            r.reason = 'requests not installed'
            return r

        try:
            resp = requests.post(
                self.base_url + '/api/interpret_command',
                json={'utterance': utterance, 'context': context or {}},
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                r.reason = f'remote http {resp.status_code}'
                return r
            j = resp.json()
            r.raw_response = json.dumps(j, ensure_ascii=False)
            return _fill_task_from_dict(j, r)
        except Exception as e:
            r.reason = f'remote error: {e}'
            return r


# ============== 工具: JSON → TaskResult ==============


_PRIORITY_MAP = {'low': 1, 'normal': 2, 'high': 3}


def _parse_json_to_task(text: str, r: TaskResult) -> TaskResult:
    """从 LLM 输出文本里抽 JSON. 容忍 markdown ``` 代码块包裹."""
    # 去 markdown 代码块
    s = text.strip()
    s = re.sub(r'^```(?:json)?\s*', '', s)
    s = re.sub(r'\s*```$', '', s)
    s = s.strip()

    try:
        d = json.loads(s)
    except Exception as e:
        r.reason = f'JSON parse failed: {e}'
        return r
    return _fill_task_from_dict(d, r)


def _fill_task_from_dict(d: Dict[str, Any], r: TaskResult) -> TaskResult:
    tt = (d.get('task_type') or '').strip()
    if tt not in ('fetch_sample', 'deliver_to_furnace', 'monitor_furnace', 'observe', 'patrol', 'home'):
        r.reason = f'invalid task_type: {tt}'
        return r

    r.task_type = tt
    r.bottle_id = (d.get('bottle_id') or '').strip()
    r.from_location = (d.get('from_location') or '').strip()
    r.to_location = (d.get('to_location') or '').strip()
    pri = (d.get('priority') or 'normal').strip().lower()
    r.priority = _PRIORITY_MAP.get(pri, 2)
    r.success = True
    return r


# ============== 工厂 ==============


def make_interpreter(backend: str, **kwargs) -> CommandInterpreter:
    """工厂方法: 一行切换 backend.

    Args:
        backend: 'rule' | 'local' | 'remote'
    """
    backend = backend.lower()
    if backend == 'rule':
        return RuleInterpreter()
    if backend == 'local':
        return LocalLLMInterpreter(**kwargs)
    if backend == 'remote':
        return RemoteLLMInterpreter(**kwargs)
    raise ValueError(f'unknown backend: {backend}')
