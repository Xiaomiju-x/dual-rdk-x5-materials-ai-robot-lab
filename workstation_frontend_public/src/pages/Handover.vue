<script setup lang="ts">
// Handover — dual-arm cooperation theatre. Hero is the live workshop stage;
// below it the operator sees the two arms' joints reconciled (SyncBars +
// PhaseWheel), a cooperative task queue, and small "link health" peeks into
// the cross-network plumbing (kept tiny so we don't double up navcockpit's
// dedicated topology page).
import { computed, ref } from 'vue'
import { useTelemetryStore } from '@/stores/telemetry'
import type { HistorySample } from '@/stores/telemetry'
import DualArmScene from '@/components/three/DualArmScene.vue'
import NetworkTopology3D from '@/components/three/NetworkTopology3D.vue'
import TimeSeries from '@/components/charts/TimeSeries.vue'
import GlowCard from '@/components/fx/GlowCard.vue'
import BorderBeam from '@/components/fx/BorderBeam.vue'
import Odometer from '@/components/fx/Odometer.vue'
import RingChart from '@/components/charts/RingChart.vue'
import SyncBars from '@/components/charts/SyncBars.vue'
import PhaseWheel from '@/components/charts/PhaseWheel.vue'

const telemetry = useTelemetryStore()

// Sync metrics
const a01 = computed(() => telemetry.arm01.angles)
const a02 = computed(() => telemetry.arm02.angles)
const sumAbsDiff = computed(() => {
  let s = 0
  for (let i = 0; i < 6; i++) s += Math.abs((a01.value[i] ?? 0) - (a02.value[i] ?? 0))
  return s
})
const syncScore = computed(() => Math.max(0, 1 - sumAbsDiff.value / (90 * 6)))

// Task queue (mocked, looks like real plan)
interface Task {
  id: string; verb: 'pick' | 'place' | 'handover' | 'mirror'
  src: string; dst: string; eta: number; status: 'queued' | 'running' | 'done'
}
const TASKS_INIT: Task[] = [
  { id: 't1', verb: 'pick',     src: 'arm01 → rack[0]',     dst: '',                 eta: 4.2, status: 'done' },
  { id: 't2', verb: 'handover', src: 'arm01 → hot zone',    dst: '→ arm02',          eta: 5.4, status: 'running' },
  { id: 't3', verb: 'place',    src: 'arm02 → rack[2]',     dst: '',                 eta: 3.1, status: 'queued' },
  { id: 't4', verb: 'mirror',   src: 'arm02 mirror arm01',  dst: '',                 eta: 2.4, status: 'queued' },
]
const tasks = ref<Task[]>(TASKS_INIT)
const taskStats = computed(() => {
  const done = tasks.value.filter((t) => t.status === 'done').length
  const total = tasks.value.length
  return { done, total, pct: done / Math.max(1, total) }
})

const TASK_COLORS: Record<Task['verb'], string> = {
  pick: '#d97706', handover: '#10b981', place: '#2563eb', mirror: '#7c3aed',
}
const TASK_GLYPHS: Record<Task['verb'], string> = { pick: '✊', handover: '🤝', place: '☷', mirror: '⇄' }
function advance(t: Task) {
  if (t.status === 'queued') t.status = 'running'
  else if (t.status === 'running') t.status = 'done'
  else { t.status = 'queued' }
}
function clearTask(id: string) { tasks.value = tasks.value.filter((t) => t.id !== id) }

// Cross-net stats (real telemetry from /api/coop)
const events = computed(() => telemetry.coopEvents)
const totalEvents = computed(() => events.value.length)
const avgRtt = computed(() => {
  if (events.value.length === 0) return 0
  return events.value.reduce((a, e) => a + e.rtt_ms, 0) / events.value.length
})
const okRatio = computed(() => {
  if (events.value.length === 0) return 1
  return events.value.filter((e) => e.ok).length / events.value.length
})
const series = computed(() => {
  const tp = telemetry.coopThroughput
  if (!tp) return []
  const toSamples = (arr: number[]): HistorySample[] => {
    const now = tp.now * 1000
    return arr.map((v, i) => ({ t: now - (arr.length - 1 - i) * 1000, v }))
  }
  return [
    { name: '工位→AI脑', samples: toSamples(tp.arm_to_ai), accent: 'blue' as const },
    { name: '工位→车载脑', samples: toSamples(tp.arm_to_car), accent: 'amber' as const },
    { name: '车载脑→AI脑', samples: toSamples(tp.car_to_ai), accent: 'violet' as const },
  ]
})
const recentEvents = computed(() => telemetry.coopEvents.slice(0, 8))
const fmtBytes = (b: number) => (b >= 1024 ? `${(b/1024).toFixed(1)}KB` : `${b}B`)
const KIND_COLOR: Record<string, string> = {
  bpu_detect: '#2563eb', vlm_query: '#7c3aed', joint_sync: '#0891b2',
  ocr_read: '#d97706', recipe: '#059669', predict: '#e11d48', failure_lib: '#64748b',
}
const kindColor = (k: string) => KIND_COLOR[k] ?? '#94a3b8'
</script>

<template>
  <div class="handover-pro">
    <!-- header strip -->
    <section class="hv-hero card-elevated">
      <BorderBeam :duration="14" color-from="rgba(16,185,129,.9)" color-to="rgba(124,58,237,0)" />
      <div class="hv-hero-row">
        <div class="hv-stat">
          <div class="metric-label">SYNC SCORE</div>
          <div class="metric-hero"><Odometer :value="syncScore * 100" :precision="0" /><span class="frac">%</span></div>
          <div class="metric-unit">phase-aligned</div>
        </div>
        <div class="hv-stat">
          <div class="metric-label">总差 Σ|Δ|</div>
          <div class="metric-hero"><Odometer :value="sumAbsDiff" :precision="1" /><span class="frac">°</span></div>
          <div class="metric-unit">6 joints aggregate</div>
        </div>
        <div class="hv-stat">
          <div class="metric-label">任务进度</div>
          <div class="metric-hero"><Odometer :value="taskStats.done" />/<span class="frac">{{ taskStats.total }}</span></div>
          <div class="metric-unit">{{ tasks.find((t) => t.status === 'running')?.verb ?? 'idle' }}</div>
        </div>
        <div class="hv-stat">
          <div class="metric-label">跨网 RTT</div>
          <div class="metric-hero"><Odometer :value="avgRtt" :precision="0" /><span class="frac">ms</span></div>
          <div class="metric-unit">{{ totalEvents }} events</div>
        </div>
        <div class="hv-ring">
          <RingChart :value="syncScore" :inner="okRatio" accent="emerald" inner-accent="violet"
                     :label="(syncScore * 100).toFixed(0) + '%'" caption="sync · link" :size="128" />
        </div>
      </div>
    </section>

    <!-- hero stage -->
    <GlowCard accent="emerald" class="hv-stage-card">
      <div class="ck-card-head">
        <div>
          <div class="ck-card-title">Handover Theatre</div>
          <div class="ck-card-sub">中央 hot-zone · arm01 → arm02 取放协同</div>
        </div>
        <span class="chip chip-info kv-mono">double-click 单臂聚焦</span>
      </div>
      <div class="hv-stage-mount">
        <DualArmScene />
      </div>
    </GlowCard>

    <!-- mid grid: SyncBars + PhaseWheel + Task Queue -->
    <div class="hv-mid">
      <GlowCard accent="blue" class="hv-sync-card">
        <SyncBars :a01="a01" :a02="a02" />
      </GlowCard>

      <GlowCard accent="violet" class="hv-wheel-card">
        <div class="ck-card-head">
          <div>
            <div class="ck-card-title">Phase Wheel</div>
            <div class="ck-card-sub">6 关节相位差 · 满针 = ±90°</div>
          </div>
        </div>
        <div class="hv-wheel-mount">
          <PhaseWheel :a01="a01" :a02="a02" :size="240" :soft-cap="90" />
        </div>
      </GlowCard>

      <GlowCard accent="amber" class="hv-tasks-card">
        <div class="ck-card-head">
          <div>
            <div class="ck-card-title">协同任务队列</div>
            <div class="ck-card-sub">点击切换状态 · queued ▸ running ▸ done</div>
          </div>
        </div>
        <div class="hv-tasks-list">
          <div v-for="t in tasks" :key="t.id" class="hv-task" :class="`task-${t.status}`"
               @click="advance(t)">
            <span class="t-glyph" :style="{ background: TASK_COLORS[t.verb] + '1f', color: TASK_COLORS[t.verb] }">{{ TASK_GLYPHS[t.verb] }}</span>
            <div class="t-text">
              <div class="t-verb" :style="{ color: TASK_COLORS[t.verb] }">{{ t.verb }}</div>
              <div class="t-detail mono">{{ t.src }} {{ t.dst }}</div>
            </div>
            <div class="t-eta mono">{{ t.eta.toFixed(1) }}s</div>
            <span class="t-status" :class="`status-${t.status}`">{{ t.status }}</span>
            <button class="t-x" @click.stop="clearTask(t.id)" title="移除">×</button>
          </div>
        </div>
      </GlowCard>
    </div>

    <!-- bottom: throughput + small topology + recent events -->
    <div class="hv-bottom">
      <GlowCard accent="blue" class="hv-tp-card">
        <div class="tp-head">
          <span class="section-label">跨网吞吐 · KB/s · 30s 窗口</span>
        </div>
        <TimeSeries v-if="series.length" :series="series" y-label="KB/s" height="160px" />
        <div v-else class="tp-empty">等待吞吐数据…</div>
      </GlowCard>

      <GlowCard accent="violet" class="hv-link-card">
        <div class="ck-card-head">
          <div>
            <div class="ck-card-title">Link Health</div>
            <div class="ck-card-sub">3 节点 · curl flow · 副视图</div>
          </div>
        </div>
        <div class="hv-link-mount">
          <NetworkTopology3D />
        </div>
      </GlowCard>

      <GlowCard accent="teal" class="hv-log-card">
        <div class="log-head section-label">最近消息</div>
        <div class="log-list">
          <div v-for="(e, i) in recentEvents" :key="i" class="log-row">
            <span class="log-kind" :style="{ background: kindColor(e.kind) + '1a', color: kindColor(e.kind), borderColor: kindColor(e.kind) + '40' }">{{ e.kind }}</span>
            <span class="log-path mono">{{ e.src }}→{{ e.dst }}</span>
            <span class="log-rtt mono">{{ e.rtt_ms.toFixed(0) }}ms</span>
            <span class="log-bytes mono">{{ fmtBytes(e.bytes) }}</span>
            <span class="log-dot dot" :class="e.ok ? 'dot-ok' : 'dot-err'"></span>
          </div>
          <div v-if="!recentEvents.length" class="log-empty">等待消息…</div>
        </div>
      </GlowCard>
    </div>
  </div>
</template>

<style scoped>
.handover-pro { display: flex; flex-direction: column; gap: 14px; }

.hv-hero { position: relative; padding: 16px 22px; border-radius: 18px; overflow: hidden; }
.hv-hero-row { display: flex; align-items: center; gap: 28px; flex-wrap: wrap; }
.hv-stat { display: flex; flex-direction: column; gap: 4px; min-width: 130px; }
.hv-stat .metric-hero { display: flex; align-items: baseline; }
.frac { font-family: 'JetBrains Mono Variable', monospace; font-size: 0.86rem; color: var(--ink-tertiary); margin-left: 4px; font-weight: 500; }
.hv-ring { margin-left: auto; }

.hv-stage-card { padding: 14px; display: flex; flex-direction: column; }
.hv-stage-mount { flex: 1; min-height: 420px; border-radius: 12px; overflow: hidden; }

.ck-card-head { display: flex; align-items: flex-start; justify-content: space-between; margin-bottom: 8px; }
.ck-card-title { font-weight: 700; font-size: 0.95rem; letter-spacing: -0.01em; color: var(--ink-primary); }
.ck-card-sub { font-size: 0.7rem; color: var(--ink-tertiary); margin-top: 2px; }

.hv-mid { display: grid; grid-template-columns: minmax(0, 1.4fr) 280px minmax(0, 1.1fr); gap: 14px; }
@media (max-width: 1300px) { .hv-mid { grid-template-columns: 1fr 1fr; } .hv-tasks-card { grid-column: 1 / -1; } }
@media (max-width: 860px) { .hv-mid { grid-template-columns: 1fr; } .hv-tasks-card { grid-column: auto; } }
.hv-sync-card, .hv-wheel-card, .hv-tasks-card { padding: 14px 16px; }
.hv-wheel-mount { display: flex; align-items: center; justify-content: center; padding: 6px 0; }

.hv-tasks-list { display: flex; flex-direction: column; gap: 6px; max-height: 320px; overflow-y: auto; }
.hv-task { display: grid; grid-template-columns: 36px 1fr auto auto 22px; gap: 10px; align-items: center;
  padding: 8px 10px; border-radius: 10px;
  background: color-mix(in srgb, var(--bg-elevated) 80%, transparent);
  border: 1px solid var(--line-divider); cursor: pointer; transition: all .15s; }
.hv-task:hover { transform: translateY(-1px); box-shadow: var(--shadow-soft); }
.hv-task.task-running { border-color: rgba(16,185,129,.4); background: rgba(16,185,129,.06); }
.hv-task.task-done { opacity: 0.55; }
.t-glyph { display: inline-flex; align-items: center; justify-content: center;
  width: 32px; height: 32px; border-radius: 8px; font-size: 0.95rem; font-weight: 700; }
.t-text { min-width: 0; }
.t-verb { font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; }
.t-detail { font-size: 0.7rem; color: var(--ink-tertiary); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.t-eta { font-size: 0.72rem; color: var(--ink-secondary); font-weight: 600; }
.t-status { font-size: 0.62rem; padding: 2px 7px; border-radius: 999px;
  font-family: 'JetBrains Mono Variable', monospace; font-weight: 700; }
.status-queued { background: var(--bg-card); color: var(--ink-muted); border: 1px solid var(--line-border); }
.status-running { background: rgba(16,185,129,.15); color: #059669; animation: pulseSoft 1.4s ease-in-out infinite; }
.status-done { background: rgba(8,145,178,.15); color: #0891b2; }
.t-x { width: 22px; height: 22px; border-radius: 50%; border: none; background: rgba(225,29,72,.1); color: var(--accent-rose);
  font-weight: 800; cursor: pointer; font-size: 0.95rem; line-height: 1; }
.t-x:hover { background: rgba(225,29,72,.2); }

.hv-bottom { display: grid; grid-template-columns: minmax(0, 1.4fr) 380px minmax(0, 1fr); gap: 14px; }
@media (max-width: 1300px) { .hv-bottom { grid-template-columns: 1fr 1fr; } .hv-tp-card { grid-column: 1 / -1; } }
@media (max-width: 860px) { .hv-bottom { grid-template-columns: 1fr; } .hv-tp-card { grid-column: auto; } }

.hv-tp-card { padding: 12px 16px; }
.tp-head { margin-bottom: 6px; }
.tp-empty { padding: 24px; text-align: center; color: var(--ink-muted); font-size: 0.85rem; }

.hv-link-card { padding: 12px 14px; display: flex; flex-direction: column; }
.hv-link-mount { flex: 1; min-height: 200px; border-radius: 10px; overflow: hidden; }

.hv-log-card { padding: 12px 14px; }
.log-head { margin-bottom: 6px; }
.log-list { display: flex; flex-direction: column; gap: 2px; overflow-y: auto; max-height: 220px; }
.log-row { display: grid; grid-template-columns: auto 1fr auto auto 10px; gap: 8px; align-items: center; padding: 5px 4px; border-bottom: 1px dashed var(--line-hairline); font-size: 0.7rem; }
.log-kind { font-size: 0.62rem; font-weight: 700; padding: 2px 6px; border-radius: 5px; border: 1px solid; font-family: 'JetBrains Mono Variable', monospace; }
.log-path { color: var(--ink-tertiary); font-size: 0.64rem; }
.log-rtt { color: var(--ink-secondary); }
.log-bytes { color: var(--ink-muted); font-size: 0.64rem; }
.log-empty { padding: 16px; text-align: center; color: var(--ink-muted); font-size: 0.8rem; }
</style>
