<script setup lang="ts">
/**
 * MiniMap — always-floating bottom-right SVG mini-map (90×90 px).
 * Shows robot position + heading + waypoints + grid.
 *
 * Click to maximize → opens full WaypointMap modal.
 * Hide on /twin and /planner (they have their own big map).
 */
import { computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useTelemetryStore } from '@/stores/telemetry'

const telemetry = useTelemetryStore()
const route = useRoute()
const router = useRouter()

const HIDE_ON = new Set(['/twin', '/planner', '/topology'])
const visible = computed(() => !HIDE_ON.has(route.path))

const SPAN = 3.0  // ± m
const SIZE = 90

const pose = computed(() => telemetry.packet?.pose ?? { x: 0, y: 0, yaw: 0 })
const waypoints = computed(() => telemetry.packet?.waypoints ?? [])

function project(x: number, y: number): { px: number; py: number } {
  const halfSize = SIZE / 2
  const px = halfSize + (x / SPAN) * (halfSize - 8)
  const py = halfSize - (y / SPAN) * (halfSize - 8)
  return { px, py }
}

const robotXY = computed(() => project(pose.value.x, pose.value.y))
const headingDeg = computed(() => -((pose.value.yaw * 180) / Math.PI))

const KIND_COLOR: Record<string, string> = {
  start: '#2563eb', pickup: '#059669', dropoff: '#d97706', patrol: '#0891b2', home: '#7c3aed',
}
</script>

<template>
  <Teleport to="body">
    <button
      v-if="visible"
      class="mm"
      :style="{ width: `${SIZE}px`, height: `${SIZE}px` } as any"
      @click="router.push('/twin')"
      :title="`Robot · click for digital twin (${pose.x.toFixed(2)}, ${pose.y.toFixed(2)})`"
    >
      <svg :viewBox="`0 0 ${SIZE} ${SIZE}`" class="mm-svg">
        <!-- range rings -->
        <circle :cx="SIZE/2" :cy="SIZE/2" r="34" fill="none" stroke="var(--line-divider)" stroke-width="1" opacity="0.5"/>
        <circle :cx="SIZE/2" :cy="SIZE/2" r="22" fill="none" stroke="var(--line-divider)" stroke-width="1" opacity="0.3"/>
        <circle :cx="SIZE/2" :cy="SIZE/2" r="10" fill="none" stroke="var(--line-divider)" stroke-width="1" opacity="0.2"/>
        <!-- crosshair -->
        <line :x1="6" :y1="SIZE/2" :x2="SIZE-6" :y2="SIZE/2" stroke="var(--line-divider)" stroke-width="0.5" opacity="0.4"/>
        <line :x1="SIZE/2" :y1="6" :x2="SIZE/2" :y2="SIZE-6" stroke="var(--line-divider)" stroke-width="0.5" opacity="0.4"/>

        <!-- waypoints -->
        <circle
          v-for="w in waypoints" :key="w.id"
          :cx="project(w.x, w.y).px"
          :cy="project(w.x, w.y).py"
          :r="2.8"
          :fill="KIND_COLOR[w.kind] || '#94a3b8'"
          opacity="0.85"
        />

        <!-- robot — arrow triangle rotated to yaw -->
        <g :transform="`translate(${robotXY.px} ${robotXY.py}) rotate(${headingDeg})`">
          <polygon points="0,-6 5,5 0,2 -5,5" fill="var(--accent-blue)" />
          <circle cx="0" cy="0" r="9" fill="none" stroke="var(--accent-blue)" stroke-width="1" opacity="0.4" />
        </g>
      </svg>
      <span class="mm-tag mono">{{ pose.x.toFixed(1) }},{{ pose.y.toFixed(1) }}</span>
    </button>
  </Teleport>
</template>

<style scoped>
.mm {
  position: fixed;
  bottom: 50px;       /* above EventMarquee 30 + gap */
  right: 18px;
  background: var(--bg-glass-strong);
  backdrop-filter: blur(18px) saturate(160%);
  -webkit-backdrop-filter: blur(18px) saturate(160%);
  border: 1px solid var(--line-divider);
  border-radius: 12px;
  padding: 4px;
  cursor: pointer;
  z-index: 30;
  box-shadow: var(--shadow-card);
  transition: transform 0.18s var(--ease-out-quint), box-shadow 0.18s var(--ease-out-quint);
}
.mm:hover { transform: translateY(-2px); box-shadow: var(--shadow-elevated); }
.mm-svg { display: block; width: 100%; height: 100%; }
.mm-tag {
  position: absolute;
  bottom: 2px; left: 50%;
  transform: translateX(-50%);
  font-size: 0.54rem;
  color: var(--ink-muted);
  background: rgba(255, 255, 255, 0.7);
  padding: 1px 4px;
  border-radius: 3px;
  white-space: nowrap;
}
[data-theme='dark'] .mm-tag { background: rgba(20, 25, 38, 0.78); }
</style>
