<script setup lang="ts">
// Missions · 任务编排 — 行为树设计器 + 暂停/恢复/中止 (第 3 期 #1)
// 后端: /api/missions* (mission.py MissionRunner), 叶节点经 cockpit_bridge 下发真车.
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { useTelemetryStore } from '@/stores/telemetry'
import { useToastStore } from '@/stores/toast'
import KineticTitle from '@/components/premium/KineticTitle.vue'
import BtNode, { type BtNodeData } from '@/components/mission/BtNode.vue'

const telemetry = useTelemetryStore()
const toasts = useToastStore()

const bridgeAlive = computed(() => telemetry.packet?.bridge?.alive ?? false)

// ---------------- 模板库 ----------------
const TEMPLATES: Array<{ id: string; name: string; tree: BtNodeData }> = [
  {
    id: 'patrol',
    name: '烧结炉巡检',
    tree: {
      type: 'sequence',
      children: [
        { type: 'speak', params: { text: '开始烧结炉巡检' } },
        { type: 'forward', params: { distance: 0.4, speed: 0.1 } },
        { type: 'read_furnace' },
        { type: 'photo' },
        { type: 'speak', params: { text: '巡检完成, 数据已记录' } },
      ],
    },
  },
  {
    id: 'find_bottle',
    name: '找瓶检测',
    tree: {
      type: 'sequence',
      children: [
        { type: 'speak', params: { text: '开始搜索试剂瓶' } },
        { type: 'spin', params: { angle: 1.57, speed: 0.4 } },
        { type: 'detect_wait', params: { label: 'bottle', timeout_s: 20 } },
        { type: 'photo' },
        { type: 'speak', params: { text: '已发现目标并拍照' } },
      ],
    },
  },
  {
    id: 'fault_demo',
    name: '故障冗余演示',
    tree: {
      type: 'sequence',
      children: [
        { type: 'speak', params: { text: '故障冗余流程开始' } },
        { type: 'retry', params: { times: 3 }, children: [{ type: 'read_furnace' }] },
        {
          type: 'fallback',
          children: [
            { type: 'detect_wait', params: { label: 'bottle', timeout_s: 5 } },
            { type: 'speak', params: { text: '主路径未发现目标, 切换备选路径' } },
          ],
        },
        { type: 'speak', params: { text: '冗余流程结束' } },
      ],
    },
  },
]

// ---------------- 编辑器状态 ----------------
const name = ref('烧结炉巡检')
const editingMid = ref<string | null>(null)
const jsonText = ref(JSON.stringify(TEMPLATES[0].tree, null, 2))
const parseErr = ref('')

const tree = computed<BtNodeData | null>(() => {
  try {
    const t = JSON.parse(jsonText.value) as BtNodeData
    return t && typeof t === 'object' && t.type ? t : null
  } catch {
    return null
  }
})
watch(jsonText, () => {
  try {
    JSON.parse(jsonText.value)
    parseErr.value = ''
  } catch (e) {
    parseErr.value = (e as Error).message
  }
})

function loadTemplate(id: string) {
  const t = TEMPLATES.find((x) => x.id === id)
  if (!t) return
  name.value = t.name
  editingMid.value = null
  jsonText.value = JSON.stringify(t.tree, null, 2)
}

// 快捷插入: 往根 children 末尾追加叶节点
const LEAF_PALETTE: Array<{ type: string; label: string; params?: Record<string, unknown> }> = [
  { type: 'speak', label: '🔊 播报', params: { text: '你好' } },
  { type: 'forward', label: '⬆ 前进', params: { distance: 0.3, speed: 0.1 } },
  { type: 'spin', label: '↺ 旋转', params: { angle: 1.57, speed: 0.4 } },
  { type: 'wait', label: '⏲ 等待', params: { seconds: 2 } },
  { type: 'photo', label: '📷 拍照' },
  { type: 'read_furnace', label: '🌡 读炉温' },
  { type: 'detect_wait', label: '🎯 等检测', params: { label: 'bottle', timeout_s: 15 } },
  { type: 'vlm', label: '👁 VLM', params: { prompt: 'Describe the scene' } },
  { type: 'goto', label: '🧭 导航', params: { backend: 'direct', x: 0.5, y: 0.0 } },
]
function appendLeaf(p: (typeof LEAF_PALETTE)[number]) {
  const t = tree.value
  if (!t) { toasts.push({ tone: 'warn', title: 'JSON 解析失败', detail: '先修好再插入' }); return }
  if (!t.children) t.children = []
  t.children.push({ type: p.type, ...(p.params ? { params: { ...p.params } } : {}) })
  jsonText.value = JSON.stringify(t, null, 2)
}

// ---------------- 任务库 ----------------
interface MissionEntry { mid: string; name: string; tree: BtNodeData; created_at: string }
const missions = ref<MissionEntry[]>([])
async function refreshMissions() {
  try {
    const r = await fetch('/api/missions').then((x) => x.json())
    missions.value = r.missions ?? []
  } catch { /* 离线静默 */ }
}
function loadMission(m: MissionEntry) {
  name.value = m.name
  editingMid.value = m.mid
  jsonText.value = JSON.stringify(m.tree, null, 2)
}
async function saveMission() {
  if (!tree.value) { toasts.push({ tone: 'err', title: 'JSON 无效' }); return }
  const r = await fetch('/api/missions', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ mid: editingMid.value || undefined, name: name.value, tree: tree.value }),
  }).then((x) => x.json())
  if (r.ok) {
    editingMid.value = r.mid
    toasts.push({ tone: 'ok', title: '已保存', detail: `${name.value} · ${r.mid}` })
    refreshMissions()
  } else {
    toasts.push({ tone: 'err', title: '保存失败', detail: r.error })
  }
}
async function deleteMission(mid: string) {
  await fetch(`/api/missions/${mid}`, { method: 'DELETE' })
  if (editingMid.value === mid) editingMid.value = null
  refreshMissions()
}

// ---------------- 运行控制 ----------------
interface RunStatus {
  state: string
  mid: string | null
  name: string
  node_states: Record<string, { status?: string; result?: Record<string, unknown> }>
  log: Array<{ t: number; msg: string }>
  started_at: number | null
  ended_at: number | null
}
const run = ref<RunStatus | null>(null)
let pollTimer: number | null = null

async function pollStatus() {
  try {
    const r = await fetch('/api/missions/status').then((x) => x.json())
    run.value = r
  } catch { /* noop */ }
}
const isActive = computed(() => run.value?.state === 'running' || run.value?.state === 'paused')

async function runAdhoc() {
  if (!tree.value) { toasts.push({ tone: 'err', title: 'JSON 无效' }); return }
  const r = await fetch('/api/missions/run_adhoc', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name: name.value, tree: tree.value }),
  }).then((x) => x.json())
  if (!r.ok) toasts.push({ tone: 'err', title: '启动失败', detail: r.error })
  pollStatus()
}
async function runSaved(mid: string) {
  const r = await fetch(`/api/missions/${mid}/run`, { method: 'POST' }).then((x) => x.json())
  if (!r.ok) toasts.push({ tone: 'err', title: '启动失败', detail: r.error })
  pollStatus()
}
async function ctrl(action: 'pause' | 'resume' | 'abort') {
  const r = await fetch(`/api/missions/${action}`, { method: 'POST' }).then((x) => x.json())
  if (!r.ok) toasts.push({ tone: 'warn', title: `${action} 失败`, detail: r.error })
  pollStatus()
}

const stateLabel = computed(() => ({
  idle: '空闲', running: '运行中', paused: '已暂停',
  done: '完成 ✓', failed: '失败 ✗', aborted: '已中止',
}[run.value?.state ?? 'idle'] ?? run.value?.state))

const elapsed = computed(() => {
  const r = run.value
  if (!r?.started_at) return ''
  const end = r.ended_at ?? Date.now() / 1000
  return `${(end - r.started_at).toFixed(1)}s`
})

onMounted(() => {
  refreshMissions()
  pollStatus()
  pollTimer = window.setInterval(pollStatus, 1000)
})
onUnmounted(() => { if (pollTimer !== null) window.clearInterval(pollTimer) })
</script>

<template>
  <section class="page">
    <header class="page-header">
      <div>
        <h1 class="page-title"><KineticTitle text="Missions · 任务编排" gradient="blue-violet" /></h1>
        <p class="page-subtitle">
          行为树设计器 · sequence / fallback / retry / repeat + 9 种动作叶 · 暂停 / 恢复 / 中止
          <span class="chip" :class="bridgeAlive ? 'chip-ok' : 'chip-warn'" style="margin-left: 8px;">
            {{ bridgeAlive ? '● 车端桥在线' : '○ 车端桥离线 (可设计, 不可执行)' }}
          </span>
        </p>
      </div>
      <div class="header-actions">
        <button class="btn btn-primary" :disabled="!tree || isActive" @click="runAdhoc">▶ 试跑当前树</button>
        <button class="btn" :disabled="!tree" @click="saveMission">💾 保存</button>
      </div>
    </header>

    <div class="grid">
      <!-- 左: 任务库 + 模板 -->
      <div class="col-left">
        <div class="card-elevated panel">
          <div class="panel-head"><span class="section-label">模板</span></div>
          <button v-for="t in TEMPLATES" :key="t.id" class="lib-item" @click="loadTemplate(t.id)">
            <span class="lib-name">{{ t.name }}</span>
            <span class="lib-meta mono">{{ t.tree.children?.length ?? 0 }} 步</span>
          </button>
        </div>

        <div class="card-elevated panel">
          <div class="panel-head">
            <span class="section-label">已存任务</span>
            <span class="chip chip-info">{{ missions.length }}</span>
          </div>
          <div v-if="!missions.length" class="empty mono">还没有保存的任务</div>
          <div v-for="m in missions" :key="m.mid" class="lib-item lib-saved" :class="{ active: editingMid === m.mid }">
            <span class="lib-name" @click="loadMission(m)">{{ m.name }}</span>
            <span class="lib-actions">
              <button class="mini-btn" title="执行" :disabled="isActive || !bridgeAlive" @click="runSaved(m.mid)">▶</button>
              <button class="mini-btn mini-del" title="删除" @click="deleteMission(m.mid)">🗑</button>
            </span>
          </div>
        </div>

        <div class="card-elevated panel">
          <div class="panel-head"><span class="section-label">快捷插入 (追加到根)</span></div>
          <div class="palette">
            <button v-for="p in LEAF_PALETTE" :key="p.type" class="pal-btn" @click="appendLeaf(p)">{{ p.label }}</button>
          </div>
        </div>
      </div>

      <!-- 中: 树预览 (运行态着色) -->
      <div class="col-mid">
        <div class="card-elevated panel">
          <div class="panel-head">
            <span class="section-label">行为树</span>
            <input v-model="name" class="name-input" placeholder="任务名" />
            <span
v-if="run && run.state !== 'idle'" class="chip" :class="{
              'chip-info': run.state === 'running',
              'chip-warn': run.state === 'paused',
              'chip-ok': run.state === 'done',
              'chip-err': run.state === 'failed' || run.state === 'aborted',
            }">{{ stateLabel }} <template v-if="elapsed">· {{ elapsed }}</template></span>
          </div>
          <div v-if="tree" class="tree-wrap">
            <BtNode :node="tree" path="r" :node-states="run?.node_states ?? {}" />
          </div>
          <div v-else class="empty mono">JSON 解析失败: {{ parseErr }}</div>

          <div class="run-ctrl">
            <button class="btn" :disabled="run?.state !== 'running'" @click="ctrl('pause')">⏸ 暂停</button>
            <button class="btn" :disabled="run?.state !== 'paused'" @click="ctrl('resume')">▶ 恢复</button>
            <button class="btn btn-danger" :disabled="!isActive" @click="ctrl('abort')">🟥 中止 (estop 脉冲)</button>
          </div>
        </div>

        <div class="card-elevated panel">
          <div class="panel-head"><span class="section-label">执行日志</span></div>
          <div class="log-box mono">
            <div v-if="!run?.log?.length" class="empty mono">暂无日志</div>
            <div v-for="(l, i) in run?.log ?? []" :key="i" class="log-line">
              <span class="log-t">{{ new Date(l.t * 1000).toLocaleTimeString() }}</span>
              <span>{{ l.msg }}</span>
            </div>
          </div>
        </div>
      </div>

      <!-- 右: JSON 源码 -->
      <div class="col-right">
        <div class="card-elevated panel panel-tall">
          <div class="panel-head">
            <span class="section-label">树 JSON (直接编辑)</span>
            <span v-if="parseErr" class="chip chip-err">✗ 语法错</span>
            <span v-else class="chip chip-ok">✓ 有效</span>
          </div>
          <textarea v-model="jsonText" class="json-editor mono" spellcheck="false"></textarea>
        </div>
      </div>
    </div>
  </section>
</template>

<style scoped>
.grid {
  display: grid;
  grid-template-columns: 250px 1fr 360px;
  gap: 16px;
  align-items: start;
}
@media (max-width: 1280px) { .grid { grid-template-columns: 220px 1fr; } .col-right { grid-column: 1 / -1; } }

.col-left, .col-mid, .col-right { display: flex; flex-direction: column; gap: 14px; }

.panel { padding: 14px; }
.panel-head { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
.panel-tall { min-height: 540px; display: flex; flex-direction: column; }

.lib-item {
  display: flex; align-items: center; justify-content: space-between; gap: 8px;
  width: 100%; padding: 8px 10px; margin-bottom: 4px;
  border: 1px solid var(--line-border); border-radius: 8px;
  background: rgba(255, 255, 255, 0.6); cursor: pointer;
  font-family: inherit; font-size: 0.78rem; color: var(--ink-secondary);
  transition: all 0.15s var(--ease-out-quint);
}
.lib-item:hover { border-color: rgba(37, 99, 235, 0.35); color: var(--ink-primary); transform: translateY(-1px); }
.lib-item.active { box-shadow: inset 0 0 0 1px rgba(37, 99, 235, 0.4); }
.lib-name { flex: 1; text-align: left; cursor: pointer; }
.lib-meta { font-size: 0.66rem; color: var(--ink-muted); }
.lib-actions { display: flex; gap: 4px; }
.mini-btn {
  border: 1px solid var(--line-border); background: white; border-radius: 6px;
  padding: 2px 7px; cursor: pointer; font-size: 0.72rem;
}
.mini-btn:disabled { opacity: 0.4; cursor: not-allowed; }
.mini-btn:hover:not(:disabled) { border-color: rgba(37, 99, 235, 0.4); }
.mini-del:hover { border-color: rgba(225, 29, 72, 0.5); }

.palette { display: flex; flex-wrap: wrap; gap: 6px; }
.pal-btn {
  border: 1px solid var(--line-border); background: rgba(255,255,255,0.7); border-radius: 7px;
  padding: 5px 9px; cursor: pointer; font-size: 0.72rem; font-family: inherit;
  transition: all 0.15s var(--ease-out-quint);
}
.pal-btn:hover { border-color: rgba(124, 58, 237, 0.4); transform: translateY(-1px); }

.name-input {
  flex: 1; min-width: 0; border: 1px solid var(--line-border); border-radius: 7px;
  padding: 5px 9px; font-size: 0.8rem; font-family: inherit; background: rgba(255,255,255,0.8);
}
.tree-wrap { max-height: 420px; overflow-y: auto; padding-right: 4px; }

.run-ctrl { display: flex; gap: 8px; margin-top: 12px; }
.btn {
  border: 1px solid var(--line-border); background: var(--bg-elevated); border-radius: 8px;
  padding: 7px 13px; cursor: pointer; font-size: 0.78rem; font-weight: 600;
  font-family: inherit; color: var(--ink-secondary);
  transition: all 0.15s var(--ease-out-quint);
}
.btn:hover:not(:disabled) { transform: translateY(-1px); border-color: rgba(37, 99, 235, 0.35); color: var(--accent-blue); }
.btn:disabled { opacity: 0.45; cursor: not-allowed; }
.btn-primary {
  background: linear-gradient(135deg, var(--accent-blue), var(--accent-teal));
  color: white; border-color: transparent;
  box-shadow: 0 3px 10px -3px rgba(37, 99, 235, 0.4);
}
.btn-primary:hover:not(:disabled) { color: white; }
.btn-danger:hover:not(:disabled) { border-color: rgba(225, 29, 72, 0.5); color: #e11d48; }

.log-box { max-height: 200px; overflow-y: auto; font-size: 0.7rem; }
.log-line { display: flex; gap: 10px; padding: 2px 0; color: var(--ink-secondary); }
.log-t { color: var(--ink-muted); flex-shrink: 0; }

.json-editor {
  flex: 1; width: 100%; min-height: 460px; resize: vertical;
  border: 1px solid var(--line-border); border-radius: 10px;
  padding: 12px; font-size: 0.72rem; line-height: 1.5;
  background: rgba(15, 23, 42, 0.025); color: var(--ink-primary);
}
.json-editor:focus { outline: 2px solid rgba(37, 99, 235, 0.25); }

.empty { color: var(--ink-muted); font-size: 0.72rem; padding: 8px 4px; }

.chip {
  display: inline-flex; align-items: center; gap: 4px;
  border-radius: 999px; padding: 2px 9px; font-size: 0.66rem; font-weight: 700;
}
.chip-ok   { background: rgba(5, 150, 105, 0.10); color: #059669; }
.chip-warn { background: rgba(217, 119, 6, 0.10); color: #d97706; }
.chip-err  { background: rgba(225, 29, 72, 0.10); color: #e11d48; }
.chip-info { background: rgba(37, 99, 235, 0.10); color: #2563eb; }
</style>
