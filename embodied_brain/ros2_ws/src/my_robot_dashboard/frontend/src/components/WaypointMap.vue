<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import type { Pose2D, Waypoint } from '@/types/telemetry'

interface Props {
  waypoints: Waypoint[]
  pose?: Pose2D | null
  /** metres per axis (world bounds is roughly ±this) */
  span?: number
  /** read-only: disable click-to-add and drag */
  readonly?: boolean
}
const props = withDefaults(defineProps<Props>(), { pose: null, span: 4.0, readonly: false })

const emit = defineEmits<{
  (e: 'change', list: Waypoint[]): void
  (e: 'select', id: string | null): void
}>()

const selected = ref<string | null>(null)
const dragging = ref<string | null>(null)
const svgRef = ref<SVGSVGElement | null>(null)

// view: 600 wide × 480 tall, origin centre, y inverts (north up = -y svg)
const VIEW_W = 600
const VIEW_H = 480
const ORIGIN_X = VIEW_W / 2
const ORIGIN_Y = VIEW_H / 2

function worldToSvg(x: number, y: number): [number, number] {
  const pxPerM = Math.min(VIEW_W, VIEW_H) / (props.span * 2)
  return [ORIGIN_X + x * pxPerM, ORIGIN_Y - y * pxPerM]
}

function svgToWorld(sx: number, sy: number): [number, number] {
  const pxPerM = Math.min(VIEW_W, VIEW_H) / (props.span * 2)
  return [(sx - ORIGIN_X) / pxPerM, -(sy - ORIGIN_Y) / pxPerM]
}

const robotSvg = computed(() => {
  if (!props.pose) return null
  const [x, y] = worldToSvg(props.pose.x, props.pose.y)
  return { x, y, yaw: -props.pose.yaw }  // svg y flipped → yaw flipped
})

const pathD = computed(() => {
  if (props.waypoints.length < 2) return ''
  const parts: string[] = []
  for (let i = 0; i < props.waypoints.length; i++) {
    const [x, y] = worldToSvg(props.waypoints[i].x, props.waypoints[i].y)
    parts.push(`${i === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)}`)
  }
  return parts.join(' ')
})

const totalDistance = computed(() => {
  if (props.waypoints.length < 2) return 0
  let total = 0
  for (let i = 1; i < props.waypoints.length; i++) {
    const a = props.waypoints[i - 1]
    const b = props.waypoints[i]
    total += Math.hypot(b.x - a.x, b.y - a.y)
  }
  return total
})

const KIND_COLOR: Record<string, string> = {
  start: '#2563eb',
  pickup: '#059669',
  dropoff: '#d97706',
  patrol: '#0891b2',
  home: '#7c3aed',
}

const KIND_GLYPH: Record<string, string> = {
  start: '▶',
  pickup: '↑',
  dropoff: '↓',
  patrol: '◊',
  home: '⌂',
}

// drag handling
function onPointerDown(evt: PointerEvent, id: string) {
  if (props.readonly) {
    selected.value = id
    return
  }
  evt.preventDefault()
  dragging.value = id
  selected.value = id
  ;(evt.target as Element).setPointerCapture?.(evt.pointerId)
}

function onPointerMove(evt: PointerEvent) {
  if (!dragging.value || !svgRef.value) return
  const rect = svgRef.value.getBoundingClientRect()
  const sx = ((evt.clientX - rect.left) / rect.width) * VIEW_W
  const sy = ((evt.clientY - rect.top) / rect.height) * VIEW_H
  const [wx, wy] = svgToWorld(sx, sy)
  const list = props.waypoints.map((w) =>
    w.id === dragging.value ? { ...w, x: Math.round(wx * 100) / 100, y: Math.round(wy * 100) / 100 } : w,
  )
  emit('change', list)
}

function onPointerUp(evt: PointerEvent) {
  if (dragging.value) {
    ;(evt.target as Element).releasePointerCapture?.(evt.pointerId)
  }
  dragging.value = null
}

function onCanvasClick(evt: MouseEvent) {
  if (props.readonly) return
  if (!svgRef.value) return
  // ignore clicks on existing handles
  if ((evt.target as Element).classList.contains('wp-handle')) return
  const rect = svgRef.value.getBoundingClientRect()
  const sx = ((evt.clientX - rect.left) / rect.width) * VIEW_W
  const sy = ((evt.clientY - rect.top) / rect.height) * VIEW_H
  const [wx, wy] = svgToWorld(sx, sy)
  const list = [...props.waypoints, {
    id: `wp-add-${Date.now()}`,
    x: Math.round(wx * 100) / 100,
    y: Math.round(wy * 100) / 100,
    label: 'New',
    kind: 'patrol' as const,
    eta_s: null,
  }]
  emit('change', list)
}

function removeSelected() {
  if (!selected.value) return
  emit('change', props.waypoints.filter((w) => w.id !== selected.value))
  selected.value = null
}

// emit selection changes so parents can show context-aware controls
watch(selected, (id) => emit('select', id))

defineExpose({ removeSelected, totalDistance })
</script>

<template>
  <div class="map-wrap">
    <svg
      ref="svgRef"
      :viewBox="`0 0 ${VIEW_W} ${VIEW_H}`"
      preserveAspectRatio="xMidYMid meet"
      class="map-svg"
      @click="onCanvasClick"
      @pointermove="onPointerMove"
      @pointerup="onPointerUp"
    >
      <!-- background gradient -->
      <defs>
        <radialGradient id="mapBg" cx="50%" cy="50%" r="70%">
          <stop offset="0%"  stop-color="#ffffff" stop-opacity="0.95" />
          <stop offset="100%" stop-color="#eef2f7" stop-opacity="0.85" />
        </radialGradient>
        <pattern id="grid-fine" width="20" height="20" patternUnits="userSpaceOnUse">
          <path d="M 20 0 L 0 0 0 20" fill="none" stroke="rgba(15,23,42,0.04)" stroke-width="0.5" />
        </pattern>
        <pattern id="grid-coarse" width="100" height="100" patternUnits="userSpaceOnUse">
          <path d="M 100 0 L 0 0 0 100" fill="none" stroke="rgba(15,23,42,0.07)" stroke-width="0.8" />
        </pattern>
      </defs>
      <rect width="100%" height="100%" fill="url(#mapBg)" />
      <rect width="100%" height="100%" fill="url(#grid-fine)" />
      <rect width="100%" height="100%" fill="url(#grid-coarse)" />

      <!-- axes -->
      <line :x1="ORIGIN_X" y1="0" :x2="ORIGIN_X" :y2="VIEW_H" stroke="rgba(15,23,42,0.10)" stroke-width="0.8" />
      <line x1="0" :y1="ORIGIN_Y" :x2="VIEW_W" :y2="ORIGIN_Y" stroke="rgba(15,23,42,0.10)" stroke-width="0.8" />
      <text :x="ORIGIN_X + 6" y="12" font-size="10" fill="#94a3b8" font-family="JetBrains Mono Variable, monospace">+y (north)</text>
      <text :x="VIEW_W - 64" :y="ORIGIN_Y - 6" font-size="10" fill="#94a3b8" font-family="JetBrains Mono Variable, monospace">+x (east)</text>

      <!-- distance circles -->
      <circle :cx="ORIGIN_X" :cy="ORIGIN_Y" :r="(Math.min(VIEW_W, VIEW_H) / (span * 2)) * 1" fill="none" stroke="rgba(37,99,235,0.10)" stroke-dasharray="3 4" stroke-width="0.8" />
      <circle :cx="ORIGIN_X" :cy="ORIGIN_Y" :r="(Math.min(VIEW_W, VIEW_H) / (span * 2)) * 2" fill="none" stroke="rgba(37,99,235,0.07)" stroke-dasharray="3 4" stroke-width="0.8" />
      <circle :cx="ORIGIN_X" :cy="ORIGIN_Y" :r="(Math.min(VIEW_W, VIEW_H) / (span * 2)) * 3" fill="none" stroke="rgba(37,99,235,0.04)" stroke-dasharray="3 4" stroke-width="0.8" />

      <!-- path -->
      <path
        v-if="pathD"
        :d="pathD"
        fill="none"
        stroke="url(#path-grad)"
        stroke-width="2.6"
        stroke-linecap="round"
        stroke-linejoin="round"
        stroke-dasharray="0"
      />
      <defs>
        <linearGradient id="path-grad" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stop-color="#2563eb" stop-opacity="0.85" />
          <stop offset="100%" stop-color="#7c3aed" stop-opacity="0.85" />
        </linearGradient>
      </defs>

      <!-- waypoints -->
      <g v-for="(w, i) in waypoints" :key="w.id">
        <g :transform="`translate(${worldToSvg(w.x, w.y)[0]} ${worldToSvg(w.x, w.y)[1]})`">
          <circle
            r="14" fill="white"
            :stroke="KIND_COLOR[w.kind] ?? '#94a3b8'"
            stroke-width="2"
            :opacity="selected === w.id ? 1 : 0.92"
            class="wp-handle"
            style="cursor: grab;"
            @pointerdown.stop="onPointerDown($event, w.id)"
            @click.stop="selected = w.id"
          />
          <text text-anchor="middle" dy="3.5" font-size="11" font-weight="700" :fill="KIND_COLOR[w.kind] ?? '#475569'" pointer-events="none">
            {{ KIND_GLYPH[w.kind] ?? (i + 1) }}
          </text>
          <text text-anchor="middle" dy="28" font-size="9" fill="#475569" font-family="Inter Variable, sans-serif" font-weight="600" pointer-events="none">
            {{ w.label || `WP${i + 1}` }}
          </text>
          <text text-anchor="middle" dy="40" font-size="8" fill="#94a3b8" font-family="JetBrains Mono Variable, monospace" pointer-events="none">
            ({{ w.x.toFixed(2) }}, {{ w.y.toFixed(2) }})
          </text>
          <circle v-if="selected === w.id" r="20" fill="none" :stroke="KIND_COLOR[w.kind] ?? '#94a3b8'" stroke-width="1" stroke-dasharray="3 3" opacity="0.6" pointer-events="none"/>
        </g>
      </g>

      <!-- robot pose -->
      <g v-if="robotSvg" :transform="`translate(${robotSvg.x} ${robotSvg.y}) rotate(${(robotSvg.yaw * 180) / Math.PI})`" pointer-events="none">
        <circle r="10" fill="rgba(37, 99, 235, 0.15)" />
        <circle r="5" fill="#2563eb" />
        <path d="M 0 0 L 14 0 L 9 -3 M 14 0 L 9 3" stroke="#2563eb" stroke-width="1.8" fill="none" stroke-linecap="round" />
      </g>
    </svg>

    <div v-if="!readonly" class="map-overlay-hint mono">click 空白处加点 · 拖拽圆点移动 · 选中后 ⌫ 删除</div>
    <div v-else class="map-overlay-hint mono">read-only · 来自 AI 脑下发的航点</div>
  </div>
</template>

<style scoped>
.map-wrap {
  position: relative;
  width: 100%;
  height: 100%;
  border-radius: 14px;
  overflow: hidden;
  background: var(--bg-card);
}
.map-svg {
  width: 100%;
  height: 100%;
  display: block;
}
.map-overlay-hint {
  position: absolute;
  bottom: 8px; left: 50%;
  transform: translateX(-50%);
  font-size: 0.66rem;
  color: var(--ink-muted);
  background: rgba(255, 255, 255, 0.78);
  padding: 4px 10px;
  border-radius: 999px;
  border: 1px solid var(--line-divider);
  pointer-events: none;
}
.wp-handle { transition: r 0.18s var(--ease-out-quint); }
.wp-handle:hover { r: 16; }
</style>
