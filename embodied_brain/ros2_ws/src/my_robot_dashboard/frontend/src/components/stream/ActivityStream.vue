<script setup lang="ts">
/**
 * ActivityStream — right-edge slide-out terminal log of WS events.
 *
 *  - filter chips: ALL / TELEMETRY / ALARM / TASK / HEARTBEAT
 *  - mini sparkline at top showing observed Hz
 *  - ring buffer 500 entries, auto-scroll to newest
 *  - touch: bottom sheet slide
 *  - keyboard `]` to toggle
 */
import { computed, nextTick, onBeforeUnmount, ref, watch } from 'vue'
import { useTelemetryStore } from '@/stores/telemetry'
import { useInputMode } from '@/composables/useInputMode'
import Sparkline from '@/components/charts/Sparkline.vue'

interface Props { open: boolean }
const props = defineProps<Props>()
const emit = defineEmits<{ (e: 'close'): void }>()

const telemetry = useTelemetryStore()
const { isTouch } = useInputMode()

type Channel = 'telemetry' | 'alarm' | 'task' | 'heartbeat'
type FilterId = 'all' | Channel

interface LogEntry {
  id: string
  ts: number
  ch: Channel
  tone: 'ok' | 'warn' | 'err' | 'info' | 'idle'
  text: string
}

const MAX_LOG = 500
const log = ref<LogEntry[]>([])
let logSeq = 0

function push(entry: Omit<LogEntry, 'id' | 'ts'>) {
  logSeq += 1
  log.value.push({ id: `e${logSeq}`, ts: Date.now(), ...entry })
  if (log.value.length > MAX_LOG) log.value.splice(0, log.value.length - MAX_LOG)
}

// hook telemetry: heartbeat-per-second + alarms + task transitions
let lastHbSeq = -1
const seenAlarms = new Set<string>()
const taskState = new Map<string, string>()

watch(() => telemetry.packet, (pkt) => {
  if (!pkt) return
  // heartbeat
  if (pkt.heartbeat.sequence !== lastHbSeq) {
    if (lastHbSeq >= 0 && pkt.heartbeat.sequence - lastHbSeq <= 3) {
      // suppress flood: 1 log per 10 seq
      if (pkt.heartbeat.sequence % 10 === 0) {
        push({ ch: 'heartbeat', tone: 'info', text: `hb seq=${pkt.heartbeat.sequence} uptime=${pkt.heartbeat.uptime_s.toFixed(0)}s ${telemetry.observedHz.toFixed(1)}Hz` })
      }
    } else if (lastHbSeq < 0) {
      push({ ch: 'heartbeat', tone: 'ok', text: `WS connected · seq=${pkt.heartbeat.sequence}` })
    }
    lastHbSeq = pkt.heartbeat.sequence
  }
  // alarms
  for (const a of pkt.alarms ?? []) {
    if (seenAlarms.has(a.id)) continue
    seenAlarms.add(a.id)
    push({ ch: 'alarm', tone: a.severity, text: `[${a.source}] ${a.title} — ${a.detail}` })
  }
  // tasks
  for (const t of pkt.tasks ?? []) {
    const prev = taskState.get(t.id)
    if (prev !== t.status) {
      taskState.set(t.id, t.status)
      const tone: LogEntry['tone'] = t.status === 'failed' ? 'err' : t.status === 'completed' ? 'ok' : t.status === 'running' ? 'info' : 'idle'
      push({ ch: 'task', tone, text: `task[${t.id}] ${t.name} → ${t.status} ${t.progress_pct.toFixed(0)}%` })
    }
  }
  // telemetry summary every 5s
  if (pkt.heartbeat.sequence % 50 === 0) {
    push({ ch: 'telemetry', tone: 'idle', text: `pose=(${pkt.pose.x.toFixed(2)},${pkt.pose.y.toFixed(2)}) vel=${pkt.velocity.linear.toFixed(2)}m/s bpu=${pkt.host.bpu_pct.toFixed(0)}% cpu=${pkt.host.cpu_pct.toFixed(0)}%` })
  }
})

// WS state changes
watch(() => telemetry.state, (st) => {
  if (st === 'connecting') push({ ch: 'heartbeat', tone: 'warn', text: 'WS connecting…' })
  else if (st === 'open') push({ ch: 'heartbeat', tone: 'ok', text: 'WS open' })
  else if (st === 'closed') push({ ch: 'heartbeat', tone: 'warn', text: 'WS closed — auto reconnect' })
  else if (st === 'error') push({ ch: 'heartbeat', tone: 'err', text: 'WS error' })
})

// filter
const filter = ref<FilterId>('all')
const filtered = computed(() => filter.value === 'all' ? log.value : log.value.filter((e) => e.ch === filter.value))

// auto-scroll
const listEl = ref<HTMLDivElement | null>(null)
const followTail = ref(true)
watch(filtered, async () => {
  if (!followTail.value) return
  await nextTick()
  if (listEl.value) listEl.value.scrollTop = listEl.value.scrollHeight
})
function onScroll() {
  if (!listEl.value) return
  const el = listEl.value
  const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 80
  followTail.value = nearBottom
}

// hz mini chart
const hzSamples = computed(() => telemetry.hostHistory.cpu.slice(-60).map((_, i, arr) => ({ t: arr[i].t, v: telemetry.observedHz })))

// counters per filter
const counters = computed(() => ({
  all: log.value.length,
  telemetry: log.value.filter((e) => e.ch === 'telemetry').length,
  alarm: log.value.filter((e) => e.ch === 'alarm').length,
  task: log.value.filter((e) => e.ch === 'task').length,
  heartbeat: log.value.filter((e) => e.ch === 'heartbeat').length,
}))

function fmt(ts: number) {
  const d = new Date(ts)
  const pad = (n: number) => n.toString().padStart(2, '0')
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`
}

function onKey(e: KeyboardEvent) {
  if (e.key === ']') {
    if (props.open) emit('close')
    else { /* parent toggles */ }
  }
}
window.addEventListener('keydown', onKey)
onBeforeUnmount(() => window.removeEventListener('keydown', onKey))
</script>

<template>
  <Teleport to="body">
    <Transition :name="isTouch ? 'sheet' : 'slide'">
      <aside
        v-if="props.open"
        class="as"
        :class="isTouch ? 'as-sheet' : 'as-side'"
        role="complementary"
        aria-label="Live activity stream"
      >
        <header class="as-head">
          <div class="as-head-l">
            <span class="dot dot-ok"></span>
            <span class="as-title">Live Stream</span>
            <span class="mono as-hz">{{ telemetry.observedHz.toFixed(1) }} Hz</span>
          </div>
          <button class="as-x" aria-label="close" @click="emit('close')">×</button>
        </header>

        <div class="as-spark"><Sparkline :samples="hzSamples" :y-range="[0, 12]" accent="emerald" /></div>

        <nav class="as-filters">
          <button
            v-for="opt in (['all','telemetry','alarm','task','heartbeat'] as const)"
            :key="opt"
            class="as-chip"
            :class="{ active: filter === opt }"
            @click="filter = opt"
          >
            <span>{{ opt }}</span>
            <span class="as-chip-n mono">{{ counters[opt] }}</span>
          </button>
        </nav>

        <div ref="listEl" class="as-list" @scroll="onScroll">
          <div
            v-for="e in filtered"
            :key="e.id"
            class="as-row"
            :class="`tone-${e.tone}`"
          >
            <span class="as-ts mono">{{ fmt(e.ts) }}</span>
            <span class="as-ch mono">[{{ e.ch.slice(0,3).toUpperCase() }}]</span>
            <span class="as-text">{{ e.text }}</span>
          </div>
          <div v-if="!filtered.length" class="as-empty mono">awaiting events…</div>
        </div>

        <footer class="as-foot mono">
          <span>{{ filtered.length }} / {{ log.length }} events</span>
          <span v-if="!followTail" class="as-resume" @click="followTail = true; if (listEl) listEl.scrollTop = listEl.scrollHeight">⤓ resume tail</span>
        </footer>
      </aside>
    </Transition>
  </Teleport>
</template>

<style scoped>
.as-side {
  position: fixed;
  top: 64px; right: 0; bottom: 0;
  width: 380px;
  z-index: 70;
  background: rgba(8, 11, 18, 0.96);
  backdrop-filter: blur(18px) saturate(160%);
  -webkit-backdrop-filter: blur(18px) saturate(160%);
  color: #cbd5e1;
  border-left: 1px solid rgba(255, 255, 255, 0.08);
  box-shadow: -16px 0 40px -10px rgba(0, 0, 0, 0.5);
  display: flex; flex-direction: column;
}
.as-sheet {
  position: fixed;
  left: 0; right: 0; bottom: 0;
  max-height: 60vh;
  z-index: 70;
  background: rgba(8, 11, 18, 0.96);
  backdrop-filter: blur(18px) saturate(160%);
  -webkit-backdrop-filter: blur(18px) saturate(160%);
  color: #cbd5e1;
  border-top: 1px solid rgba(255, 255, 255, 0.08);
  box-shadow: 0 -16px 40px -10px rgba(0, 0, 0, 0.5);
  border-radius: 18px 18px 0 0;
  display: flex; flex-direction: column;
}

.as-head {
  display: flex; justify-content: space-between; align-items: center;
  padding: 12px 14px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}
.as-head-l { display: flex; align-items: center; gap: 8px; }
.as-title { font-size: 0.78rem; font-weight: 700; letter-spacing: 0.06em; color: #f1f5f9; }
.as-hz { font-size: 0.66rem; color: #94a3b8; padding-left: 6px; border-left: 1px solid rgba(255, 255, 255, 0.10); margin-left: 4px; }
.as-x {
  width: 28px; height: 28px;
  display: flex; align-items: center; justify-content: center;
  background: transparent; border: 1px solid rgba(255,255,255,0.10);
  color: #94a3b8;
  font-size: 1.2rem; cursor: pointer; border-radius: 6px;
  transition: background 0.18s var(--ease-out-quint);
}
.as-x:hover { background: rgba(255, 255, 255, 0.06); color: #f1f5f9; }

.as-spark {
  height: 36px;
  padding: 4px 12px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.04);
}

.as-filters {
  display: flex; gap: 4px;
  padding: 8px 12px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.06);
  flex-wrap: wrap;
}
.as-chip {
  display: inline-flex; align-items: center; gap: 4px;
  padding: 3px 8px;
  font-size: 0.62rem;
  font-family: 'JetBrains Mono Variable', monospace;
  color: #94a3b8;
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid rgba(255, 255, 255, 0.06);
  border-radius: 999px;
  cursor: pointer;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  transition: all 0.18s var(--ease-out-quint);
}
.as-chip:hover { color: #e2e8f0; }
.as-chip.active { background: rgba(96, 165, 250, 0.18); color: #93c5fd; border-color: rgba(96, 165, 250, 0.40); }
.as-chip-n { font-size: 0.58rem; opacity: 0.7; }

.as-list {
  flex: 1; min-height: 0;
  overflow-y: auto;
  padding: 6px 10px;
  font-family: 'JetBrains Mono Variable', monospace;
  scroll-behavior: smooth;
}
.as-list::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.08); }

.as-row {
  display: grid;
  grid-template-columns: 64px 50px 1fr;
  gap: 6px; align-items: baseline;
  padding: 3px 4px;
  font-size: 0.66rem;
  line-height: 1.42;
  color: #cbd5e1;
  border-radius: 3px;
}
.as-row:hover { background: rgba(255, 255, 255, 0.03); }
.as-ts { color: #475569; }
.as-ch { color: #64748b; font-size: 0.58rem; }
.as-text { white-space: pre-wrap; word-break: break-word; }

.tone-ok   .as-text { color: #6ee7b7; }
.tone-warn .as-text { color: #fcd34d; }
.tone-err  .as-text { color: #fca5a5; }
.tone-info .as-text { color: #93c5fd; }
.tone-idle .as-text { color: #94a3b8; }

.as-empty { padding: 24px; text-align: center; color: #475569; font-size: 0.72rem; }

.as-foot {
  display: flex; justify-content: space-between; align-items: center;
  padding: 6px 12px;
  border-top: 1px solid rgba(255, 255, 255, 0.06);
  font-size: 0.6rem;
  color: #64748b;
}
.as-resume { color: #93c5fd; cursor: pointer; }
.as-resume:hover { text-decoration: underline; }

.slide-enter-from { transform: translateX(100%); }
.slide-leave-to   { transform: translateX(100%); }
.slide-enter-active, .slide-leave-active { transition: transform 0.32s var(--ease-out-quint); }

.sheet-enter-from { transform: translateY(100%); }
.sheet-leave-to   { transform: translateY(100%); }
.sheet-enter-active, .sheet-leave-active { transition: transform 0.32s var(--ease-out-quint); }
</style>
