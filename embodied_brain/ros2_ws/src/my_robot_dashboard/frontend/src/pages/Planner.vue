<script setup lang="ts">
import { computed, ref, watch, onMounted } from 'vue'
import { useTelemetryStore } from '@/stores/telemetry'
import { useToastStore } from '@/stores/toast'
import { useAudioFeedback } from '@/composables/useAudioFeedback'
import WaypointMap from '@/components/WaypointMap.vue'
import type { Waypoint } from '@/types/telemetry'
import KineticTitle from '@/components/premium/KineticTitle.vue'
import Odometer from '@/components/premium/Odometer.vue'
import BorderBeam from '@/components/premium/BorderBeam.vue'

const telemetry = useTelemetryStore()
const toasts = useToastStore()
const audio = useAudioFeedback()
const mapRef = ref<InstanceType<typeof WaypointMap> | null>(null)

const localWaypoints = ref<Waypoint[]>([])
const isDirty = ref(false)
const dispatched = ref(false)
const selectedWp = ref<string | null>(null)

// seed from backend once, then ignore further server updates
let seeded = false
watch(() => telemetry.packet?.waypoints, (wp) => {
  if (seeded || !wp) return
  localWaypoints.value = wp.map((w) => ({ ...w }))
  seeded = true
}, { immediate: true })

onMounted(() => {
  if (!seeded && telemetry.packet?.waypoints) {
    localWaypoints.value = telemetry.packet.waypoints.map((w) => ({ ...w }))
    seeded = true
  }
})

const CRUISE_VEL = 0.42 // m/s, matches mock velocity
const TURN_PENALTY_S = 1.2 // per turn of >30°

const totalDistance = computed(() => {
  let total = 0
  for (let i = 1; i < localWaypoints.value.length; i++) {
    const a = localWaypoints.value[i - 1]
    const b = localWaypoints.value[i]
    total += Math.hypot(b.x - a.x, b.y - a.y)
  }
  return total
})

const turnCount = computed(() => {
  let n = 0
  for (let i = 2; i < localWaypoints.value.length; i++) {
    const a = localWaypoints.value[i - 2]
    const b = localWaypoints.value[i - 1]
    const c = localWaypoints.value[i]
    const v1x = b.x - a.x, v1y = b.y - a.y
    const v2x = c.x - b.x, v2y = c.y - b.y
    const d1 = Math.hypot(v1x, v1y) || 1
    const d2 = Math.hypot(v2x, v2y) || 1
    const cos = (v1x * v2x + v1y * v2y) / (d1 * d2)
    if (cos < Math.cos((30 * Math.PI) / 180)) n++
  }
  return n
})

const estimatedEta = computed(() => totalDistance.value / CRUISE_VEL + turnCount.value * TURN_PENALTY_S)

const cumulativeEtas = computed(() => {
  const out: number[] = [0]
  let acc = 0
  for (let i = 1; i < localWaypoints.value.length; i++) {
    const a = localWaypoints.value[i - 1]
    const b = localWaypoints.value[i]
    acc += Math.hypot(b.x - a.x, b.y - a.y) / CRUISE_VEL
    out.push(acc)
  }
  return out
})

function onChange(list: Waypoint[]) {
  localWaypoints.value = list
  isDirty.value = true
  dispatched.value = false
}

function onSelect(id: string | null) {
  selectedWp.value = id
}

function resetPlan() {
  const wp = telemetry.packet?.waypoints ?? []
  localWaypoints.value = wp.map((w) => ({ ...w }))
  isDirty.value = false
  dispatched.value = false
}

function dispatchPlan() {
  // mock: in Phase 10 this would POST to /api/dispatch_plan and stream MPPI rollouts back
  dispatched.value = true
  isDirty.value = false
  toasts.push({
    tone: 'info',
    title: `Plan dispatched · ${localWaypoints.value.length} wp · ${totalDistance.value.toFixed(2)} m`,
    detail: `ETA ${fmtEta(estimatedEta.value)} · MPPI cost MLP 接管路径优化`,
  })
  audio.play('success')
}

function onKeydown(evt: KeyboardEvent) {
  if (evt.key === 'Backspace' || evt.key === 'Delete') {
    mapRef.value?.removeSelected()
  }
}

function removeSelected() {
  mapRef.value?.removeSelected()
  selectedWp.value = null
}

const hasSelection = computed(() => selectedWp.value != null)

onMounted(() => {
  window.addEventListener('keydown', onKeydown)
})

const KIND_LABEL: Record<string, string> = {
  start: 'Start', pickup: 'Pickup', dropoff: 'Drop-off', patrol: 'Patrol', home: 'Home',
}

const KIND_COLOR: Record<string, string> = {
  start: '#2563eb',
  pickup: '#059669',
  dropoff: '#d97706',
  patrol: '#0891b2',
  home: '#7c3aed',
}

function fmtEta(s: number): string {
  if (s < 60) return `${s.toFixed(1)}s`
  const min = Math.floor(s / 60)
  const sec = s - min * 60
  return `${min}m ${sec.toFixed(0)}s`
}
</script>

<template>
  <section class="page">
    <header class="page-header">
      <div>
        <h1 class="page-title">
          <KineticTitle text="What-If Planner · 规划沙盒" gradient="teal-emerald" />
        </h1>
        <p class="page-subtitle">
          点空白加点 · 拖圆点移动 · 选中后用上方 × 删除 · cruise {{ CRUISE_VEL }} m/s ·
          <span class="mono">{{ localWaypoints.length }} wp · {{ totalDistance.toFixed(2) }} m</span>
        </p>
      </div>
      <div class="header-actions">
        <button
          v-if="hasSelection"
          class="btn btn-danger"
          @click="removeSelected"
        >× Remove WP</button>
        <button class="btn" :disabled="!isDirty" @click="resetPlan">↺ Reset</button>
        <button class="btn btn-primary" @click="dispatchPlan">
          {{ dispatched ? '✓ Dispatched' : '▶ Dispatch to Robot' }}
        </button>
      </div>
    </header>

    <div class="planner-grid">
      <!-- Map -->
      <div class="map-panel card-floating">
        <BorderBeam :duration="13" :size="220" :radius="22" />
        <WaypointMap
          ref="mapRef"
          :waypoints="localWaypoints"
          :pose="telemetry.packet?.pose"
          :span="3.0"
          @change="onChange"
          @select="onSelect"
        />
      </div>

      <!-- Side: stats + list -->
      <div class="side">
        <div class="card-elevated stat-panel">
          <span class="section-label">Plan Summary</span>
          <div class="stat-row">
            <span class="stat-label">Total distance</span>
            <span class="stat-val mono"><Odometer :value="totalDistance" :decimals="2" suffix=" m" /></span>
          </div>
          <div class="stat-row">
            <span class="stat-label">Turns &gt; 30°</span>
            <span class="stat-val mono">{{ turnCount }} (+{{ (turnCount * TURN_PENALTY_S).toFixed(1) }}s)</span>
          </div>
          <div class="stat-row stat-row-prime">
            <span class="stat-label">Estimated ETA</span>
            <span class="stat-val mono"><Odometer :value="estimatedEta" :decimals="1" suffix=" s" /></span>
          </div>
          <div class="eta-bar">
            <div class="eta-fill" :style="{ width: `${Math.min(100, (estimatedEta / 180) * 100)}%` }"></div>
          </div>
          <div class="eta-legend mono">3 min reference</div>
        </div>

        <div class="card-elevated wp-list-panel">
          <span class="section-label">Waypoint Sequence</span>
          <div v-if="!localWaypoints.length" class="wp-empty mono">点地图加第一个航点</div>
          <ol v-else class="wp-list">
            <li v-for="(w, i) in localWaypoints" :key="w.id" class="wp-row">
              <span class="wp-num" :style="{ color: KIND_COLOR[w.kind] }">{{ i + 1 }}</span>
              <div class="wp-info">
                <div class="wp-row-label">
                  {{ w.label || `WP${i + 1}` }}
                  <span class="wp-kind">· {{ KIND_LABEL[w.kind] }}</span>
                </div>
                <div class="wp-row-meta mono">
                  ({{ w.x.toFixed(2) }}, {{ w.y.toFixed(2) }}) ·
                  ETA <span class="wp-eta">{{ fmtEta(cumulativeEtas[i] ?? 0) }}</span>
                </div>
              </div>
            </li>
          </ol>
        </div>

        <div class="card-elevated tip-panel">
          <span class="section-label">MPPI Cost MLP</span>
          <p class="tip-line">
            264 KB INT8 模型在 X5 BPU 推理 <span class="mono">1.14 ms / batch</span>
            (880K traj·s⁻¹), 4.1× 于 ARM CPU. 实际部署时 Dispatch 触发 BPU
            roll-out, 把上面 ETA 替换为 MPPI 选优后的真实路径估算.
          </p>
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
.header-actions { display: flex; gap: 10px; }

.planner-grid {
  display: grid;
  grid-template-columns: 2fr 1fr;
  gap: 14px;
  flex: 1;
  min-height: 0;
}
@media (max-width: 1024px) { .planner-grid { grid-template-columns: 1fr; } }

.map-panel {
  padding: 0;
  overflow: hidden;
  min-height: 480px;
  position: relative;
}
.map-panel :deep(.beam) { z-index: 5; }

.side { display: flex; flex-direction: column; gap: 12px; min-width: 0; min-height: 0; }
.stat-panel { padding: 14px 16px; display: flex; flex-direction: column; gap: 8px; }
.stat-row { display: flex; align-items: baseline; justify-content: space-between; gap: 8px; font-size: 0.78rem; }
.stat-label { color: var(--ink-tertiary); }
.stat-val { color: var(--ink-primary); font-weight: 600; }
.stat-row-prime .stat-val { font-size: 1.15rem; color: var(--accent-blue); }
.eta-bar { height: 6px; background: rgba(15, 23, 42, 0.05); border-radius: 3px; overflow: hidden; margin-top: 2px; }
.eta-fill { height: 100%; background: linear-gradient(90deg, var(--accent-blue), var(--accent-violet)); border-radius: 3px; transition: width 0.32s var(--ease-out-quint); }
.eta-legend { font-size: 0.62rem; color: var(--ink-muted); text-align: right; }

.wp-list-panel { padding: 14px 16px; display: flex; flex-direction: column; gap: 8px; max-height: 420px; min-height: 0; overflow: hidden; }
.wp-empty { color: var(--ink-muted); font-size: 0.78rem; padding: 18px; text-align: center; }
.wp-list {
  list-style: none; margin: 0; padding: 0;
  display: flex; flex-direction: column; gap: 4px;
  overflow-y: auto;
}
.wp-row {
  display: grid;
  grid-template-columns: 30px 1fr;
  gap: 10px; align-items: center;
  padding: 6px 8px;
  border-radius: 8px;
  background: rgba(248, 250, 252, 0.7);
  border: 1px solid var(--line-hairline);
}
.wp-num {
  width: 26px; height: 26px;
  display: flex; align-items: center; justify-content: center;
  border-radius: 6px;
  background: rgba(15, 23, 42, 0.04);
  font-weight: 700; font-size: 0.82rem;
}
.wp-row-label { font-size: 0.78rem; color: var(--ink-primary); font-weight: 500; }
.wp-kind { color: var(--ink-muted); font-weight: 400; font-size: 0.72rem; }
.wp-row-meta { font-size: 0.66rem; color: var(--ink-tertiary); margin-top: 2px; }
.wp-eta { color: var(--accent-blue); font-weight: 600; }

.tip-panel { padding: 12px 16px; }
.tip-line { font-size: 0.72rem; color: var(--ink-tertiary); line-height: 1.55; margin: 6px 0 0; }
</style>
