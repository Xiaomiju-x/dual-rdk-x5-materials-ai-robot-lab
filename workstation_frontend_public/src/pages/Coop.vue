<script setup lang="ts">
import { computed } from 'vue'
import { useTelemetryStore } from '@/stores/telemetry'
import type { HistorySample } from '@/stores/telemetry'
import TimeSeries from '@/components/charts/TimeSeries.vue'
import NetworkTopology3D from '@/components/three/NetworkTopology3D.vue'
import GlowCard from '@/components/fx/GlowCard.vue'
import BorderBeam from '@/components/fx/BorderBeam.vue'
import Odometer from '@/components/fx/Odometer.vue'
import RingChart from '@/components/charts/RingChart.vue'

const telemetry = useTelemetryStore()

const KIND_COLOR: Record<string, string> = {
  bpu_detect: '#2563eb', vlm_query: '#7c3aed', joint_sync: '#0891b2',
  ocr_read: '#d97706', recipe: '#059669', predict: '#e11d48', failure_lib: '#64748b',
}
const kindColor = (k: string) => KIND_COLOR[k] ?? '#94a3b8'

// kpis
const events = computed(() => telemetry.coopEvents)
const totalEvents = computed(() => events.value.length)
const avgRtt = computed(() => {
  if (events.value.length === 0) return 0
  return events.value.reduce((a, e) => a + e.rtt_ms, 0) / events.value.length
})
const totalBytes = computed(() => events.value.reduce((a, e) => a + e.bytes, 0))
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
const recentEvents = computed(() => telemetry.coopEvents.slice(0, 16))
const fmtTime = (ts: number) => new Date(ts * 1000).toLocaleTimeString().slice(0, 8)
const fmtBytes = (b: number) => (b >= 1024 ? `${(b/1024).toFixed(1)}KB` : `${b}B`)
</script>

<template>
  <div class="coop-pro">
    <div class="cp-head">
      <div>
        <h1 class="page-title">三机协同 · 3D 拓扑</h1>
        <p class="page-subtitle">Three.js 实时拓扑 · 流光数据 · BPU bloom · 真实跨网消息</p>
      </div>
      <div class="cp-legend">
        <span v-for="(c, k) in KIND_COLOR" :key="k" class="lg-item">
          <span class="lg-dot" :style="{ background: c }"></span>{{ k }}
        </span>
      </div>
    </div>

    <!-- KPI hero strip -->
    <section class="cp-kpis card-elevated">
      <BorderBeam :duration="12" color-from="rgba(124,58,237,.9)" color-to="rgba(8,145,178,0)" />
      <div class="kpi"><div class="metric-label">总事件</div><div class="metric-hero"><Odometer :value="totalEvents" /></div><div class="metric-unit">events</div></div>
      <div class="kpi"><div class="metric-label">平均 RTT</div><div class="metric-hero"><Odometer :value="avgRtt" :precision="0" /></div><div class="metric-unit">ms · cross-net</div></div>
      <div class="kpi"><div class="metric-label">总流量</div><div class="metric-hero"><Odometer :value="totalBytes/1024" :precision="1" /></div><div class="metric-unit">KB · session</div></div>
      <div class="kpi-ring">
        <RingChart :value="okRatio" accent="emerald" :label="(okRatio*100).toFixed(0)+'%'" caption="success" :size="116" />
      </div>
    </section>

    <!-- main: 3D topology + log -->
    <div class="cp-main">
      <GlowCard accent="violet" class="cp-topo-card">
        <div class="ck-card-head">
          <div>
            <div class="ck-card-title">异构网络拓扑 · 实时数据流</div>
            <div class="ck-card-sub">3 节点 · 3 边 · curl flow shader · 220 particles</div>
          </div>
          <span class="chip chip-info kv-mono">three.js + bloom</span>
        </div>
        <div class="cp-topo-mount">
          <NetworkTopology3D />
        </div>
      </GlowCard>

      <GlowCard accent="teal" class="cp-log-card">
        <div class="log-head section-label">消息日志 · 最近 16 条</div>
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

    <!-- throughput chart -->
    <GlowCard accent="blue" class="cp-tp-card">
      <div class="tp-head">
        <span class="section-label">跨网吞吐 (KB/s · 30s 窗口)</span>
        <span class="mono tp-now" v-if="telemetry.coopThroughput">{{ fmtTime(telemetry.coopThroughput.now) }}</span>
      </div>
      <TimeSeries v-if="series.length" :series="series" y-label="KB/s" height="220px" />
      <div v-else class="tp-empty">等待吞吐数据…</div>
    </GlowCard>
  </div>
</template>

<style scoped>
.coop-pro { display: flex; flex-direction: column; gap: 14px; }
.cp-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; flex-wrap: wrap; }
.cp-legend { display: flex; gap: 10px; flex-wrap: wrap; }
.lg-item { display: inline-flex; align-items: center; gap: 5px; font-size: 0.68rem; color: var(--ink-tertiary); font-family: 'JetBrains Mono Variable', monospace; }
.lg-dot { width: 8px; height: 8px; border-radius: 2px; }

/* KPI band */
.cp-kpis { position: relative; padding: 14px 22px; display: flex; align-items: center; gap: 32px; flex-wrap: wrap; border-radius: 18px; overflow: hidden; }
.kpi { display: flex; flex-direction: column; gap: 4px; min-width: 140px; }
.kpi .metric-hero { display: flex; align-items: baseline; }
.kpi-ring { margin-left: auto; }

/* main grid */
.cp-main { display: grid; gap: 14px; grid-template-columns: 1fr 360px; }
@media (max-width: 1100px) { .cp-main { grid-template-columns: 1fr; } }
.cp-topo-card { padding: 14px; display: flex; flex-direction: column; }
.cp-topo-mount { flex: 1; min-height: 420px; border-radius: 12px; overflow: hidden; }

.ck-card-head { display: flex; align-items: flex-start; justify-content: space-between; margin-bottom: 8px; }
.ck-card-title { font-weight: 700; font-size: 0.95rem; letter-spacing: -0.01em; color: var(--ink-primary); }
.ck-card-sub { font-size: 0.7rem; color: var(--ink-tertiary); margin-top: 2px; }

.cp-log-card { padding: 14px 16px; display: flex; flex-direction: column; min-height: 0; }
.log-head { margin-bottom: 8px; }
.log-list { display: flex; flex-direction: column; gap: 2px; overflow-y: auto; max-height: 420px; }
.log-row { display: grid; grid-template-columns: auto 1fr auto auto 10px; gap: 8px; align-items: center; padding: 6px 4px; border-bottom: 1px dashed var(--line-hairline); font-size: 0.72rem; }
.log-kind { font-size: 0.62rem; font-weight: 700; padding: 2px 6px; border-radius: 5px; border: 1px solid; font-family: 'JetBrains Mono Variable', monospace; }
.log-path { color: var(--ink-tertiary); font-size: 0.66rem; }
.log-rtt { color: var(--ink-secondary); }
.log-bytes { color: var(--ink-muted); font-size: 0.66rem; }
.log-empty { padding: 16px; text-align: center; color: var(--ink-muted); font-size: 0.8rem; }

.cp-tp-card { padding: 14px 16px; }
.tp-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px; }
.tp-now { font-size: 0.72rem; color: var(--ink-muted); }
.tp-empty { padding: 40px; text-align: center; color: var(--ink-muted); font-size: 0.85rem; }
</style>
