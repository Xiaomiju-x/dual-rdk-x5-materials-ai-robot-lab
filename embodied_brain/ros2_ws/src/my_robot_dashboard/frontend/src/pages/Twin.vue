<script setup lang="ts">
import { computed, ref } from 'vue'
import { useTelemetryStore } from '@/stores/telemetry'
import SlamScene from '@/components/three/SlamScene.vue'
import WaypointMap from '@/components/WaypointMap.vue'
import Sparkline from '@/components/charts/Sparkline.vue'
import type { Waypoint } from '@/types/telemetry'
import KineticTitle from '@/components/premium/KineticTitle.vue'

const telemetry = useTelemetryStore()
const waypoints = computed<Waypoint[]>(() => telemetry.packet?.waypoints ?? [])
const sensors = computed(() => telemetry.packet?.sensors ?? [])

const focus = ref<'all' | 'real' | 'plan' | 'host'>('all')

const traveledKm = computed(() => {
  // mock — derive from uptime + cruise vel
  const t = telemetry.packet?.heartbeat.uptime_s ?? 0
  return (t * 0.42) / 1000
})
</script>

<template>
  <section class="page">
    <header class="page-header">
      <div>
        <h1 class="page-title">
          <KineticTitle text="Digital Twin · 数字孪生" gradient="aurora" />
        </h1>
        <p class="page-subtitle">
          实景 3D + 规划俯视 + 传感器健康 + 主机负载 4 视并列 · 一键对齐 ·
          <span class="mono">{{ traveledKm.toFixed(3) }} km traveled (mock)</span>
        </p>
      </div>
      <div class="seg">
        <button class="seg-btn" :class="{ active: focus === 'all' }" @click="focus = 'all'">All</button>
        <button class="seg-btn" :class="{ active: focus === 'real' }" @click="focus = 'real'">Real 3D</button>
        <button class="seg-btn" :class="{ active: focus === 'plan' }" @click="focus = 'plan'">Plan 2D</button>
        <button class="seg-btn" :class="{ active: focus === 'host' }" @click="focus = 'host'">Host</button>
      </div>
    </header>

    <div class="twin-grid" :class="`focus-${focus}`">
      <!-- A: Real 3D SLAM -->
      <div class="quad quad-real card-floating">
        <div class="quad-head">
          <span class="quad-tag" style="color: #2563eb">A · Real</span>
          <span class="section-label">Live SLAM · 3D Twin</span>
          <span v-if="telemetry.packet" class="chip chip-info mono">
            x {{ telemetry.packet.pose.x.toFixed(2) }} · y {{ telemetry.packet.pose.y.toFixed(2) }} · θ {{ ((telemetry.packet.pose.yaw * 180) / Math.PI).toFixed(0) }}°
          </span>
        </div>
        <div class="quad-body">
          <SlamScene :embed="true" mode="orbit-auto" />
        </div>
      </div>

      <!-- B: Plan 2D map view -->
      <div class="quad quad-plan card-floating">
        <div class="quad-head">
          <span class="quad-tag" style="color: #7c3aed">B · Plan</span>
          <span class="section-label">Top-Down · Plan vs Robot</span>
          <span class="chip chip-idle mono">read-only</span>
        </div>
        <div class="quad-body">
          <WaypointMap :waypoints="waypoints" :pose="telemetry.packet?.pose" :span="3.0" readonly @change="() => {}" />
        </div>
      </div>

      <!-- C: Sensor health matrix -->
      <div class="quad quad-sensors card-floating">
        <div class="quad-head">
          <span class="quad-tag" style="color: #059669">C · Sensors</span>
          <span class="section-label">Coverage · 8 lanes</span>
          <span class="chip chip-ok">{{ sensors.filter((s) => s.health === 'ok').length }} / {{ sensors.length }} ok</span>
        </div>
        <div class="quad-body sensor-cover">
          <div v-for="s in sensors" :key="s.id" class="sensor-row">
            <span class="dot" :class="`dot-${s.health}`"></span>
            <span class="sr-label">{{ s.label }}</span>
            <span class="sr-spark"><Sparkline :samples="telemetry.sensorBuffer(s.id)" :accent="s.kind === 'imu' ? 'violet' : s.kind === 'odom' ? 'amber' : 'blue'" /></span>
            <span class="sr-val mono">{{ s.hz > 0 ? `${s.hz.toFixed(1)} Hz` : 'idle' }}</span>
          </div>
        </div>
      </div>

      <!-- D: Host vitals -->
      <div class="quad quad-host card-floating">
        <div class="quad-head">
          <span class="quad-tag" style="color: #d97706">D · Host</span>
          <span class="section-label">CPU · BPU · RAM · CMA</span>
          <span v-if="telemetry.packet" class="chip chip-info mono">
            {{ telemetry.packet.host.ram_gb.toFixed(1) }}/{{ telemetry.packet.host.ram_total_gb }} GB
          </span>
        </div>
        <div class="quad-body host-vital">
          <div class="vital-row">
            <div class="vital-label">CPU</div>
            <div class="vital-bar"><div class="vital-fill" :class="'bar-blue'" :style="{ width: `${telemetry.packet?.host.cpu_pct ?? 0}%` }"></div></div>
            <div class="vital-val mono">{{ telemetry.packet?.host.cpu_pct.toFixed(0) ?? '—' }}%</div>
            <div class="vital-spark"><Sparkline :samples="telemetry.hostHistory.cpu" :yRange="[0, 100]" accent="blue" /></div>
          </div>
          <div class="vital-row">
            <div class="vital-label">BPU</div>
            <div class="vital-bar"><div class="vital-fill" :class="'bar-violet'" :style="{ width: `${telemetry.packet?.host.bpu_pct ?? 0}%` }"></div></div>
            <div class="vital-val mono">{{ telemetry.packet?.host.bpu_pct.toFixed(0) ?? '—' }}%</div>
            <div class="vital-spark"><Sparkline :samples="telemetry.hostHistory.bpu" :yRange="[0, 100]" accent="violet" /></div>
          </div>
          <div class="vital-row">
            <div class="vital-label">RAM</div>
            <div class="vital-bar"><div class="vital-fill" :class="'bar-teal'" :style="{ width: `${((telemetry.packet?.host.ram_gb ?? 0) / (telemetry.packet?.host.ram_total_gb || 8)) * 100}%` }"></div></div>
            <div class="vital-val mono">{{ telemetry.packet?.host.ram_gb.toFixed(1) ?? '—' }} GB</div>
            <div class="vital-spark"><Sparkline :samples="telemetry.hostHistory.ram" accent="teal" /></div>
          </div>
          <div class="vital-row">
            <div class="vital-label">CMA</div>
            <div class="vital-bar"><div class="vital-fill" :class="'bar-amber'" :style="{ width: `${((telemetry.packet?.host.cma_mb ?? 0) / (telemetry.packet?.host.cma_total_mb || 391)) * 100}%` }"></div></div>
            <div class="vital-val mono">{{ telemetry.packet?.host.cma_mb.toFixed(0) ?? '—' }} MB</div>
            <div class="vital-spark"><Sparkline :samples="telemetry.hostHistory.cma" accent="amber" /></div>
          </div>
        </div>
      </div>
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
  height: 100%;
}
.page-header { display: flex; justify-content: space-between; align-items: flex-end; }
.seg {
  display: inline-flex;
  background: var(--bg-elevated);
  border: 1px solid var(--line-border);
  border-radius: 8px;
  overflow: hidden;
}
.seg-btn {
  background: transparent; border: none;
  padding: 6px 14px;
  font-size: 0.78rem; color: var(--ink-tertiary);
  cursor: pointer; font-weight: 500;
  transition: background 0.15s var(--ease-out-quint), color 0.15s var(--ease-out-quint);
}
.seg-btn:hover { color: var(--ink-primary); background: rgba(241, 245, 249, 0.6); }
.seg-btn.active {
  background: linear-gradient(135deg, var(--accent-blue), var(--accent-teal));
  color: white;
}

.twin-grid {
  flex: 1;
  display: grid;
  grid-template-columns: 1fr 1fr;
  grid-template-rows: 1fr 1fr;
  gap: 14px;
  min-height: 0;
  transition: all 0.32s var(--ease-out-quint);
}
@media (max-width: 1024px) { .twin-grid { grid-template-columns: 1fr; grid-template-rows: auto; } }

.focus-real .quad-real,
.focus-plan .quad-plan,
.focus-host .quad-host {
  grid-row: 1 / span 2;
  grid-column: 1 / span 2;
}
.focus-real .quad-plan,
.focus-real .quad-sensors,
.focus-real .quad-host,
.focus-plan .quad-real,
.focus-plan .quad-sensors,
.focus-plan .quad-host,
.focus-host .quad-real,
.focus-host .quad-plan,
.focus-host .quad-sensors {
  display: none;
}

.quad {
  padding: 0;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  min-height: 0;
  min-width: 0;
}
.quad-head {
  display: flex; align-items: center; gap: 10px;
  padding: 10px 14px;
  border-bottom: 1px solid var(--line-divider);
  background: rgba(255, 255, 255, 0.7);
  backdrop-filter: blur(12px);
}
.quad-tag {
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.04em;
}
.quad-head .section-label { flex: 1; }
.quad-body {
  flex: 1;
  min-height: 0;
  position: relative;
}

/* sensor coverage */
.sensor-cover { padding: 12px 16px; display: flex; flex-direction: column; gap: 8px; overflow-y: auto; }
.sensor-row {
  display: grid;
  grid-template-columns: 12px 110px 1fr 80px;
  align-items: center;
  gap: 10px;
  padding: 6px 0;
}
.sr-label { font-size: 0.78rem; color: var(--ink-secondary); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.sr-spark { height: 28px; min-width: 0; }
.sr-val { font-size: 0.7rem; color: var(--ink-tertiary); text-align: right; }

/* host vitals */
.host-vital { padding: 14px 16px; display: flex; flex-direction: column; gap: 14px; justify-content: center; height: 100%; }
.vital-row {
  display: grid;
  grid-template-columns: 50px 1fr 65px 100px;
  align-items: center;
  gap: 10px;
}
.vital-label { font-size: 0.76rem; font-weight: 600; color: var(--ink-secondary); }
.vital-bar { height: 8px; border-radius: 4px; background: rgba(15, 23, 42, 0.05); overflow: hidden; }
.vital-fill { height: 100%; border-radius: 4px; transition: width 0.4s var(--ease-out-quint); }
.bar-blue    { background: linear-gradient(90deg, var(--accent-blue), #60a5fa); }
.bar-teal    { background: linear-gradient(90deg, var(--accent-teal), #22d3ee); }
.bar-violet  { background: linear-gradient(90deg, var(--accent-violet), #a78bfa); }
.bar-amber   { background: linear-gradient(90deg, var(--accent-amber), #fbbf24); }
.vital-val { font-size: 0.78rem; color: var(--ink-primary); font-weight: 600; text-align: right; }
.vital-spark { height: 28px; min-width: 0; }
</style>
