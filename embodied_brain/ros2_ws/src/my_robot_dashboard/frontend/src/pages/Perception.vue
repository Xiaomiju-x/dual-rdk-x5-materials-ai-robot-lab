<script setup lang="ts">
import { computed, ref } from 'vue'
import { useTelemetryStore } from '@/stores/telemetry'
import CameraTile, { type Detection } from '@/components/CameraTile.vue'
import BpuSlotCard from '@/components/BpuSlotCard.vue'
import DetectionLog from '@/components/DetectionLog.vue'
import KineticTitle from '@/components/premium/KineticTitle.vue'
import HolographicHUD from '@/components/premium/HolographicHUD.vue'
import BorderBeam from '@/components/premium/BorderBeam.vue'
import Odometer from '@/components/premium/Odometer.vue'

const telemetry = useTelemetryStore()
const cameras = computed(() => telemetry.packet?.cameras ?? [])
const bpuSlots = computed(() => telemetry.packet?.bpu_slots ?? [])

const CAM_MODE: Record<string, 'yolo' | 'apriltag' | 'depth'> = {
  lift_cam:  'yolo',
  astra_rgb: 'apriltag',
  furnace:   'depth',
}
const CAM_ACCENT: Record<string, 'blue' | 'teal' | 'emerald' | 'violet' | 'amber' | 'rose'> = {
  lift_cam:  'emerald',
  astra_rgb: 'violet',
  furnace:   'teal',
}
const SLOT_ACCENT: Record<string, 'blue' | 'teal' | 'emerald' | 'violet' | 'amber' | 'rose'> = {
  yolo_world: 'emerald',
  ppocr:      'amber',
  mppi_cost:  'blue',
  xfeat:      'violet',
}

// detection event bus (camera tiles emit, log subscribes)
const detections = ref<Detection[]>([])
const MAX_DET = 200

function onDetect(d: Detection) {
  const arr = detections.value.slice()
  arr.push(d)
  if (arr.length > MAX_DET) arr.splice(0, arr.length - MAX_DET)
  detections.value = arr
}

// per-class stats (last 60s window)
const stats = computed(() => {
  const cutoff = Date.now() - 60_000
  const recent = detections.value.filter((d) => d.at_ms >= cutoff)
  const byCls: Record<string, number> = {}
  let confSum = 0
  for (const d of recent) {
    byCls[d.cls] = (byCls[d.cls] ?? 0) + 1
    confSum += d.conf
  }
  return {
    total: recent.length,
    classes: Object.entries(byCls).sort((a, b) => b[1] - a[1]),
    avgConf: recent.length ? confSum / recent.length : 0,
  }
})

const totalBpuMb = computed(() => bpuSlots.value.reduce((s, b) => s + b.size_mb, 0))
</script>

<template>
  <section class="page">
    <header class="page-header">
      <div>
        <h1 class="page-title">
          <KineticTitle text="AR Perception · 视觉链路" gradient="teal-emerald" />
        </h1>
        <p class="page-subtitle">
          3 路摄像头实时帧 · YOLO/AprilTag 检测叠加 · 4 BPU slot 热力 ·
          <span class="mono">{{ stats.total }} det/60s</span>
        </p>
      </div>
      <div class="header-actions">
        <div class="stat-pill">
          <span class="stat-pill-label">avg conf</span>
          <span class="stat-pill-val mono"><Odometer :value="stats.avgConf * 100" :decimals="1" suffix="%" /></span>
        </div>
        <div class="stat-pill">
          <span class="stat-pill-label">CMA used</span>
          <span class="stat-pill-val mono"><Odometer :value="totalBpuMb" :decimals="1" suffix=" / 391 MB" /></span>
        </div>
      </div>
    </header>

    <!-- Camera grid -->
    <div class="cam-grid">
      <HolographicHUD
        v-for="cam in cameras"
        :key="cam.id"
        :tone="cam.online ? 'ok' : 'idle'"
        :crosshair="cam.id === 'astra_rgb'"
        :scanline="true"
        :inset="4"
      >
        <CameraTile
          :camera="cam"
          :mode="CAM_MODE[cam.id] ?? 'yolo'"
          :accent="CAM_ACCENT[cam.id] ?? 'blue'"
          :on-detect="onDetect"
        />
      </HolographicHUD>
    </div>

    <!-- Lower: BPU strip + detection log -->
    <div class="lower-grid">
      <div class="bpu-panel card-elevated">
        <BorderBeam :duration="14" :size="180" :radius="18" :colorFrom="'rgba(124, 58, 237, 0.85)'" :colorTo="'rgba(34, 211, 238, 0.85)'" />
        <div class="panel-head">
          <span class="section-label">BPU Slot · Bayes-e 10 TOPS</span>
          <span class="chip chip-info mono">{{ bpuSlots.length }} active</span>
        </div>
        <div class="bpu-grid">
          <BpuSlotCard
            v-for="slot in bpuSlots"
            :key="slot.id"
            :slot="slot"
            :accent="SLOT_ACCENT[slot.id] ?? 'violet'"
          />
        </div>
        <div class="cma-bar">
          <div class="cma-bar-label">CMA 391 MB allocation</div>
          <div class="cma-bar-track">
            <div
              v-for="(b, i) in bpuSlots" :key="b.id"
              class="cma-seg"
              :style="{
                width: `${(b.size_mb / 391) * 100}%`,
                background: ['#059669', '#d97706', '#2563eb', '#7c3aed'][i % 4],
              }"
              :title="`${b.label} · ${b.size_mb.toFixed(2)} MB`"
            ></div>
            <div class="cma-free" :style="{ width: `${((391 - totalBpuMb) / 391) * 100}%` }"></div>
          </div>
          <div class="cma-bar-legend mono">
            {{ totalBpuMb.toFixed(1) }} MB used · {{ (391 - totalBpuMb).toFixed(1) }} MB free
          </div>
        </div>
      </div>

      <DetectionLog :items="detections" :max="20" class="log-panel" />
    </div>

    <!-- Footer: class breakdown -->
    <div class="card-elevated cls-panel" v-if="stats.classes.length">
      <span class="section-label">Class Histogram · 60s window</span>
      <div class="cls-rows">
        <div v-for="[cls, n] in stats.classes" :key="cls" class="cls-row">
          <span class="cls-name">{{ cls }}</span>
          <div class="cls-bar-track">
            <div class="cls-bar-fill" :style="{ width: `${(n / stats.total) * 100}%` }"></div>
          </div>
          <span class="cls-count mono">{{ n }}</span>
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
}
.page-header { display: flex; justify-content: space-between; align-items: flex-end; }
.header-actions { display: flex; gap: 10px; align-items: center; }
.stat-pill {
  display: flex; flex-direction: column;
  padding: 6px 14px;
  border-radius: 8px;
  background: var(--bg-elevated);
  border: 1px solid var(--line-border);
}
.stat-pill-label { font-size: 0.62rem; color: var(--ink-muted); text-transform: uppercase; letter-spacing: 0.05em; }
.stat-pill-val { font-size: 0.86rem; color: var(--ink-primary); font-weight: 600; margin-top: 2px; }

.cam-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 14px;
}
@media (max-width: 1280px) { .cam-grid { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 880px)  { .cam-grid { grid-template-columns: 1fr; } }

.lower-grid {
  display: grid;
  grid-template-columns: 1.4fr 1fr;
  gap: 14px;
  min-height: 380px;
}
@media (max-width: 1280px) { .lower-grid { grid-template-columns: 1fr; } }

.bpu-panel { position: relative; padding: 14px 16px; display: flex; flex-direction: column; gap: 12px; }
.panel-head { display: flex; align-items: center; justify-content: space-between; }
.bpu-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 10px;
}
@media (max-width: 720px) { .bpu-grid { grid-template-columns: 1fr; } }

.cma-bar { display: flex; flex-direction: column; gap: 6px; padding-top: 4px; border-top: 1px solid var(--line-divider); }
.cma-bar-label { font-size: 0.7rem; color: var(--ink-tertiary); }
.cma-bar-track {
  height: 10px; border-radius: 5px;
  background: rgba(15, 23, 42, 0.05);
  overflow: hidden;
  display: flex;
}
.cma-seg { transition: width 0.4s var(--ease-out-quint); }
.cma-free { background: transparent; flex: 1; }
.cma-bar-legend { font-size: 0.66rem; color: var(--ink-muted); text-align: right; }

.log-panel { min-height: 0; }

.cls-panel { padding: 12px 16px; display: flex; flex-direction: column; gap: 8px; }
.cls-rows { display: flex; flex-direction: column; gap: 4px; }
.cls-row { display: grid; grid-template-columns: 110px 1fr 60px; gap: 12px; align-items: center; font-size: 0.78rem; }
.cls-name { color: var(--ink-primary); font-weight: 500; }
.cls-bar-track { height: 6px; background: rgba(15, 23, 42, 0.05); border-radius: 3px; overflow: hidden; }
.cls-bar-fill { height: 100%; background: linear-gradient(90deg, #2563eb, #7c3aed); border-radius: 3px; transition: width 0.3s var(--ease-out-quint); }
.cls-count { font-size: 0.74rem; color: var(--ink-secondary); text-align: right; }
</style>
