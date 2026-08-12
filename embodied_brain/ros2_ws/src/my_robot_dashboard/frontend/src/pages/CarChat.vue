<script setup lang="ts">
// CarChat · 问车 — 车载本地 LLM 工具调用对话 (第 3 期 #3)
// POST /api/chat → qid → EventSource /api/chat/stream
// 断网全本地: 0.5B 快档 (:9101) / 1.7B 深档 (:9100), 工具走 cockpit_bridge 真车.
import { computed, nextTick, onUnmounted, ref } from 'vue'
import { useTelemetryStore } from '@/stores/telemetry'
import KineticTitle from '@/components/premium/KineticTitle.vue'

const telemetry = useTelemetryStore()
const bridgeAlive = computed(() => telemetry.packet?.bridge?.alive ?? false)

interface ToolStep { name: string; args: Record<string, unknown>; result?: string; photo?: boolean; photoUrl?: string }
interface Msg {
  role: 'user' | 'assistant'
  text: string
  phase?: string
  tools: ToolStep[]
  model?: string
  latency?: number
  error?: string
  streaming?: boolean
}

const msgs = ref<Msg[]>([])
const input = ref('')
const deep = ref(false)
const allowMotion = ref(false)
const busy = ref(false)
const scrollRef = ref<HTMLDivElement | null>(null)
let es: EventSource | null = null

const QUICK = [
  '现在位姿和资源情况?',
  '最近有什么报警?',
  '读一下炉温',
  '拍张照看看周围有什么',
  '检索酒精灯使用安全规程',
]

function scrollDown() {
  nextTick(() => { scrollRef.value?.scrollTo({ top: scrollRef.value.scrollHeight, behavior: 'smooth' }) })
}

async function send(text?: string) {
  const q = (text ?? input.value).trim()
  if (!q || busy.value) return
  input.value = ''
  busy.value = true

  // 历史: 取最近 3 轮纯文本
  const history = msgs.value
    .filter((m) => m.text && !m.error)
    .slice(-6)
    .map((m) => ({ role: m.role, content: m.text }))

  msgs.value.push({ role: 'user', text: q, tools: [] })
  const a: Msg = { role: 'assistant', text: '', tools: [], streaming: true }
  msgs.value.push(a)
  scrollDown()

  try {
    const r = await fetch('/api/chat', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: q, history, deep: deep.value, allow_motion: allowMotion.value }),
    }).then((x) => x.json())
    if (!r.ok) throw new Error(r.error || 'chat start 失败')

    es = new EventSource(`/api/chat/stream?qid=${r.qid}`)
    es.onmessage = (evt) => {
      const d = JSON.parse(evt.data)
      if (d.type === 'phase') {
        a.phase = d.text
      } else if (d.type === 'tool_call') {
        a.phase = ''
        a.tools.push({ name: d.name, args: d.args ?? {} })
      } else if (d.type === 'tool_result') {
        const t = a.tools[a.tools.length - 1]
        if (t) {
          t.result = d.result
          if (d.photo) { t.photo = true; t.photoUrl = `/api/photo/latest.jpg?t=${Date.now()}` }
        }
      } else if (d.type === 'delta') {
        a.phase = ''
        a.text += d.text
      } else if (d.type === 'done') {
        a.model = d.model
        a.latency = d.latency_ms
        a.streaming = false
        finish()
      } else if (d.type === 'error') {
        a.error = d.error
        a.streaming = false
        finish()
      }
      msgs.value = [...msgs.value]   // shallow 触发
      scrollDown()
    }
    es.onerror = () => {
      if (a.streaming) { a.error = a.error || 'SSE 连接中断'; a.streaming = false; msgs.value = [...msgs.value] }
      finish()
    }
  } catch (e) {
    a.error = (e as Error).message
    a.streaming = false
    msgs.value = [...msgs.value]
    finish()
  }
}

function finish() {
  es?.close()
  es = null
  busy.value = false
}

function clearChat() { msgs.value = [] }

onUnmounted(() => es?.close())

const TOOL_ICON: Record<string, string> = {
  get_pose: '📡', get_alarms: '🚨', read_furnace: '🌡', capture_photo: '📷',
  vlm_ask: '👁', nav_goto: '🧭', search_sop: '📚',
}
</script>

<template>
  <section class="page chat-page">
    <header class="page-header">
      <div>
        <h1 class="page-title"><KineticTitle text="CarChat · 问车" gradient="aurora" /></h1>
        <p class="page-subtitle">
          车载本地 LLM 工具调用 · 断网全本地 · 7 工具 (位姿/报警/炉温/拍照/VLM/导航/SOP)
          <span class="chip" :class="bridgeAlive ? 'chip-ok' : 'chip-warn'" style="margin-left: 8px;">
            {{ bridgeAlive ? '● 车端桥在线' : '○ 桥离线 (工具会报离线)' }}
          </span>
        </p>
      </div>
      <div class="header-actions">
        <label class="toggle">
          <input v-model="deep" type="checkbox" />
          <span>{{ deep ? '🧠 1.7B 深档' : '⚡ 0.5B 快档' }}</span>
        </label>
        <label class="toggle" :class="{ 'toggle-danger': allowMotion }">
          <input v-model="allowMotion" type="checkbox" />
          <span>{{ allowMotion ? '🟠 允许运动' : '🔒 禁止运动' }}</span>
        </label>
        <button class="btn" @click="clearChat">🧹 清空</button>
      </div>
    </header>

    <div class="chat-shell card-elevated">
      <div ref="scrollRef" class="chat-scroll">
        <div v-if="!msgs.length" class="welcome">
          <div class="welcome-icon">🤖</div>
          <p>我是车载 RDK X5 的语言中枢 — 问我位姿、报警、炉温, 让我拍照、查 SOP, 授权后还能让我开过去。</p>
          <div class="quick-wrap">
            <button v-for="q in QUICK" :key="q" class="quick-chip" @click="send(q)">{{ q }}</button>
          </div>
        </div>

        <div v-for="(m, i) in msgs" :key="i" class="msg" :class="`msg-${m.role}`">
          <div class="bubble" :class="{ 'bubble-err': m.error }">
            <!-- 工具调用步骤 -->
            <div v-for="(t, j) in m.tools" :key="j" class="tool-step">
              <div class="tool-head mono">
                <span class="tool-icon">{{ TOOL_ICON[t.name] ?? '🔧' }}</span>
                <span class="tool-name">{{ t.name }}</span>
                <span v-if="Object.keys(t.args).length" class="tool-args">{{ JSON.stringify(t.args) }}</span>
                <span v-if="!t.result" class="tool-spin">⏳</span>
              </div>
              <div v-if="t.result" class="tool-result mono">{{ t.result }}</div>
              <img v-if="t.photoUrl" :src="t.photoUrl" class="tool-photo" alt="车载相机照片" />
            </div>

            <div v-if="m.phase" class="phase mono">{{ m.phase }}</div>
            <div v-if="m.text" class="msg-text">{{ m.text }}</div>
            <span v-if="m.streaming && m.text" class="caret">▋</span>
            <div v-if="m.error" class="msg-err mono">⚠ {{ m.error }}</div>
            <div v-if="m.model" class="msg-meta mono">{{ m.model }} · {{ ((m.latency ?? 0) / 1000).toFixed(1) }}s · 全本地</div>
          </div>
        </div>
      </div>

      <div class="input-row">
        <input
          v-model="input"
          class="chat-input"
          :placeholder="busy ? '回答中…' : '问点什么, 比如: 读一下炉温 / 拍张照'"
          :disabled="busy"
          @keyup.enter="send()"
        />
        <button class="send-btn" :disabled="busy || !input.trim()" @click="send()">
          {{ busy ? '…' : '发送 ➤' }}
        </button>
      </div>
    </div>
  </section>
</template>

<style scoped>
.chat-page { display: flex; flex-direction: column; height: 100%; }
.chat-shell {
  flex: 1; display: flex; flex-direction: column; min-height: 0;
  padding: 0; overflow: hidden;
  max-height: calc(100vh - 230px);
}
.chat-scroll { flex: 1; overflow-y: auto; padding: 22px; display: flex; flex-direction: column; gap: 14px; }

.welcome { text-align: center; margin: auto; max-width: 520px; color: var(--ink-tertiary); }
.welcome-icon { font-size: 2.6rem; margin-bottom: 10px; }
.welcome p { font-size: 0.85rem; line-height: 1.6; }
.quick-wrap { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin-top: 16px; }
.quick-chip {
  border: 1px solid var(--line-border); background: rgba(255,255,255,0.8); border-radius: 999px;
  padding: 7px 14px; cursor: pointer; font-size: 0.76rem; font-family: inherit; color: var(--ink-secondary);
  transition: all 0.15s var(--ease-out-quint);
}
.quick-chip:hover { border-color: rgba(124, 58, 237, 0.45); color: var(--accent-violet); transform: translateY(-1px); }

.msg { display: flex; }
.msg-user { justify-content: flex-end; }
.msg-assistant { justify-content: flex-start; }
.bubble {
  max-width: 76%; border-radius: 14px; padding: 11px 15px;
  font-size: 0.84rem; line-height: 1.6;
  box-shadow: 0 2px 10px -4px rgba(15, 23, 42, 0.12);
}
.msg-user .bubble {
  background: linear-gradient(135deg, var(--accent-blue), var(--accent-teal));
  color: white; border-bottom-right-radius: 4px;
}
.msg-assistant .bubble {
  background: rgba(255, 255, 255, 0.9); border: 1px solid var(--line-border);
  border-bottom-left-radius: 4px; color: var(--ink-primary);
}
.bubble-err { border-color: rgba(225, 29, 72, 0.35) !important; }

.msg-text { white-space: pre-wrap; }
.caret { color: var(--accent-violet); animation: blink 0.9s steps(1) infinite; }
@keyframes blink { 50% { opacity: 0; } }

.phase { font-size: 0.7rem; color: var(--ink-muted); animation: pulseSoft 1.4s infinite; }

.tool-step {
  border: 1px dashed rgba(124, 58, 237, 0.3); border-radius: 9px;
  padding: 7px 10px; margin-bottom: 8px; background: rgba(124, 58, 237, 0.03);
}
.tool-head { display: flex; align-items: center; gap: 7px; font-size: 0.7rem; }
.tool-name { font-weight: 700; color: var(--accent-violet); }
.tool-args { color: var(--ink-muted); max-width: 280px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.tool-spin { animation: pulseSoft 1s infinite; }
.tool-result {
  margin-top: 5px; font-size: 0.68rem; color: var(--ink-secondary);
  background: rgba(15, 23, 42, 0.035); border-radius: 6px; padding: 5px 8px;
  white-space: pre-wrap; max-height: 130px; overflow-y: auto;
}
.tool-photo { margin-top: 8px; max-width: 100%; border-radius: 9px; border: 1px solid var(--line-border); }

.msg-err { margin-top: 6px; font-size: 0.72rem; color: #e11d48; }
.msg-meta { margin-top: 7px; font-size: 0.64rem; color: var(--ink-muted); }

.input-row {
  display: flex; gap: 10px; padding: 14px 18px;
  border-top: 1px solid var(--line-divider);
  background: rgba(255, 255, 255, 0.75); backdrop-filter: blur(10px);
}
.chat-input {
  flex: 1; border: 1px solid var(--line-border); border-radius: 11px;
  padding: 11px 15px; font-size: 0.84rem; font-family: inherit;
  background: white;
}
.chat-input:focus { outline: 2px solid rgba(124, 58, 237, 0.3); }
.send-btn {
  border: none; border-radius: 11px; padding: 0 22px; cursor: pointer;
  background: linear-gradient(135deg, var(--accent-violet), var(--accent-rose, #e11d48));
  color: white; font-weight: 700; font-size: 0.82rem; font-family: inherit;
  box-shadow: 0 3px 12px -3px rgba(124, 58, 237, 0.45);
  transition: all 0.15s var(--ease-out-quint);
}
.send-btn:hover:not(:disabled) { transform: translateY(-1px); }
.send-btn:disabled { opacity: 0.45; cursor: not-allowed; }

.toggle {
  display: inline-flex; align-items: center; gap: 6px;
  border: 1px solid var(--line-border); border-radius: 999px;
  padding: 6px 13px; cursor: pointer; font-size: 0.74rem; font-weight: 600;
  color: var(--ink-secondary); background: var(--bg-elevated); user-select: none;
  transition: all 0.15s var(--ease-out-quint);
}
.toggle input { display: none; }
.toggle:has(input:checked) { border-color: rgba(124, 58, 237, 0.4); color: var(--accent-violet); }
.toggle-danger:has(input:checked) { border-color: rgba(217, 119, 6, 0.55); color: #d97706; background: rgba(217, 119, 6, 0.06); }

.btn {
  border: 1px solid var(--line-border); background: var(--bg-elevated); border-radius: 8px;
  padding: 7px 13px; cursor: pointer; font-size: 0.78rem; font-weight: 600;
  font-family: inherit; color: var(--ink-secondary);
}
.btn:hover { transform: translateY(-1px); }

.chip {
  display: inline-flex; align-items: center; gap: 4px;
  border-radius: 999px; padding: 2px 9px; font-size: 0.66rem; font-weight: 700;
}
.chip-ok   { background: rgba(5, 150, 105, 0.10); color: #059669; }
.chip-warn { background: rgba(217, 119, 6, 0.10); color: #d97706; }
</style>
