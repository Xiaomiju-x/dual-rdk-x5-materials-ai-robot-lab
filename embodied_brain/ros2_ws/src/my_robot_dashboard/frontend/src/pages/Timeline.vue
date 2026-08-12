<script setup lang="ts">
import { computed, ref } from 'vue'
import { useTelemetryStore } from '@/stores/telemetry'
import GanttTimeline from '@/components/charts/GanttTimeline.vue'
import AlarmFeed from '@/components/AlarmFeed.vue'
import AiLinkPanel from '@/components/AiLinkPanel.vue'
import KineticTitle from '@/components/premium/KineticTitle.vue'
import WaveformTimeline from '@/components/premium/WaveformTimeline.vue'

const telemetry = useTelemetryStore()
const events = computed(() => telemetry.packet?.timeline ?? [])
const link = computed(() => telemetry.packet?.ai_link ?? null)
const alarmHistory = computed(() => telemetry.alarmHistory)
const cpuSamples = computed(() => telemetry.hostHistory.cpu.map((h) => h.v))
const bpuSamples = computed(() => telemetry.hostHistory.bpu.map((h) => h.v))
const ramSamples = computed(() => telemetry.hostHistory.ram.map((h) => h.v))

type TrackFilter = 'all' | 'dispatch' | 'nav' | 'perception' | 'ai_brain' | 'system'
const filter = ref<TrackFilter>('all')
const filtered = computed(() => filter.value === 'all' ? events.value : events.value.filter((e) => e.track === filter.value))

const trackStats = computed(() => {
  const out: Record<string, { count: number; dur: number }> = {}
  for (const e of events.value) {
    if (!out[e.track]) out[e.track] = { count: 0, dur: 0 }
    out[e.track].count += 1
    out[e.track].dur += Math.max(0, e.end_ms - e.start_ms)
  }
  return out
})

const totalDuration = computed(() => {
  if (!events.value.length) return 0
  const tMin = Math.min(...events.value.map((e) => e.start_ms))
  const tMax = Math.max(...events.value.map((e) => e.end_ms))
  return (tMax - tMin) / 1000
})

const TRACK_LABEL: Record<string, string> = {
  ai_brain: 'AI Brain',
  dispatch: 'Dispatch',
  nav: 'Navigation',
  perception: 'Perception',
  system: 'System',
}

const TRACK_COLOR: Record<string, string> = {
  ai_brain: '#7c3aed',
  dispatch: '#2563eb',
  nav: '#0891b2',
  perception: '#059669',
  system: '#d97706',
}
</script>

<template>
  <section class="page">
    <header class="page-header">
      <div>
        <h1 class="page-title">
          <KineticTitle text="Timeline · 时间轴" gradient="blue-violet" />
        </h1>
        <p class="page-subtitle">
          5 轨任务 Gantt · 报警流 · AI 脑跨网链路 ·
          <span class="mono">{{ events.length }} events / {{ totalDuration.toFixed(0) }}s span</span>
        </p>
      </div>
      <div class="seg">
        <button
          v-for="opt in (['all','ai_brain','dispatch','nav','perception','system'] as const)"
          :key="opt"
          class="seg-btn"
          :class="{ active: filter === opt }"
          @click="filter = opt"
        >{{ opt === 'all' ? 'all' : TRACK_LABEL[opt] }}</button>
      </div>
    </header>

    <!-- Live telemetry waveforms — SpaceX-style scrubbable time strips -->
    <div class="waveform-row card-elevated">
      <span class="section-label">Live Telemetry · 60s window</span>
      <div class="waveform-grid">
        <WaveformTimeline :samples="cpuSamples" label="CPU %" accent="blue" :height="56" />
        <WaveformTimeline :samples="bpuSamples" label="BPU %" accent="violet" :height="56" />
        <WaveformTimeline :samples="ramSamples" label="RAM GB" accent="teal" :height="56" />
      </div>
    </div>

    <!-- Track summary chips -->
    <div class="track-strip">
      <div v-for="(s, key) in trackStats" :key="key" class="track-chip card-elevated">
        <span class="track-color" :style="{ background: TRACK_COLOR[key] }"></span>
        <div class="track-info">
          <div class="track-name">{{ TRACK_LABEL[key] ?? key }}</div>
          <div class="track-stat mono">{{ s.count }} ev · {{ (s.dur / 1000).toFixed(0) }}s</div>
        </div>
      </div>
    </div>

    <!-- Gantt -->
    <div class="card-elevated gantt-panel">
      <div class="panel-head">
        <span class="section-label">Recent 8 Minutes · Gantt</span>
        <span class="mono panel-meta">{{ filtered.length }} visible / {{ events.length }} total</span>
      </div>
      <GanttTimeline :events="filtered" height="340px" />
    </div>

    <!-- Lower row -->
    <div class="lower-grid">
      <AiLinkPanel :link="link" :recent-events="events" />
      <AlarmFeed :history="alarmHistory" :max="30" class="alarm-side" />
    </div>
  </section>
</template>

<style scoped>
.page {
  max-width: 1680px;
  margin: 0 auto;
  display: flex; flex-direction: column;
  gap: 18px;
  animation: fadeUp 0.42s var(--ease-out-quint) both;
}
.page-header { display: flex; justify-content: space-between; align-items: flex-end; gap: 14px; }

.seg {
  display: inline-flex;
  background: var(--bg-elevated);
  border: 1px solid var(--line-border);
  border-radius: 8px;
  overflow: hidden;
  flex-wrap: wrap;
}
.seg-btn {
  background: transparent; border: none;
  padding: 6px 12px;
  font-size: 0.74rem; color: var(--ink-tertiary);
  cursor: pointer; font-weight: 500;
  transition: background 0.15s var(--ease-out-quint), color 0.15s var(--ease-out-quint);
}
.seg-btn:hover { color: var(--ink-primary); background: rgba(241, 245, 249, 0.6); }
.seg-btn.active {
  background: linear-gradient(135deg, var(--accent-blue), var(--accent-teal));
  color: white;
}

.track-strip {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 10px;
}
@media (max-width: 1024px) { .track-strip { grid-template-columns: repeat(3, 1fr); } }
@media (max-width: 720px)  { .track-strip { grid-template-columns: repeat(2, 1fr); } }

.track-chip {
  padding: 10px 14px;
  display: flex; align-items: center; gap: 10px;
}
.track-color { width: 6px; height: 28px; border-radius: 3px; flex-shrink: 0; }
.track-name { font-size: 0.78rem; font-weight: 600; color: var(--ink-primary); }
.track-stat { font-size: 0.66rem; color: var(--ink-muted); margin-top: 2px; }

.gantt-panel { padding: 14px 16px; display: flex; flex-direction: column; gap: 10px; }
.panel-head { display: flex; align-items: center; justify-content: space-between; }
.panel-meta { font-size: 0.7rem; color: var(--ink-muted); }

.lower-grid {
  display: grid;
  grid-template-columns: 1.2fr 1fr;
  gap: 14px;
  min-height: 420px;
}

.waveform-row { padding: 14px 16px; display: flex; flex-direction: column; gap: 10px; }
.waveform-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; }
@media (max-width: 1024px) { .waveform-grid { grid-template-columns: 1fr; } }
@media (max-width: 1024px) { .lower-grid { grid-template-columns: 1fr; } }
.alarm-side { min-height: 0; }
</style>
