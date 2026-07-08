<script setup lang="ts">
// Apple Watch-style circular slider. Maps angle 0..360 around a partial arc
// (270° sweep) to [min..max]. Drag-anywhere-on-track or drag-handle, with
// touch + keyboard (←/→). Shows target + actual diff as a secondary arc.
import { computed, ref, watch } from 'vue'

const props = withDefaults(defineProps<{
  modelValue: number
  min: number
  max: number
  size?: number
  stroke?: number
  accent?: string         // CSS var color
  actual?: number
  label?: string
  unit?: string
  step?: number
}>(), { size: 92, stroke: 8, accent: 'var(--accent-blue)', unit: '°', step: 1 })
const emit = defineEmits<{ (e: 'update:modelValue', v: number): void }>()

const sweep = 270  // degrees
const startDeg = 135  // bottom-left
const c = computed(() => props.size / 2)
const r = computed(() => c.value - props.stroke / 2 - 2)

function clamp(v: number) { return Math.max(props.min, Math.min(props.max, v)) }
function valueToDeg(v: number): number {
  const t = (clamp(v) - props.min) / (props.max - props.min)
  return startDeg + t * sweep
}
function degToValue(deg: number): number {
  // normalize to startDeg..startDeg+sweep
  let d = deg
  while (d < startDeg - 1e-6) d += 360
  while (d > startDeg + sweep + 1e-6) d -= 360
  const t = (d - startDeg) / sweep
  const raw = props.min + t * (props.max - props.min)
  return Math.round(raw / props.step) * props.step
}
const targetDeg = computed(() => valueToDeg(props.modelValue))
const actualDeg = computed(() => valueToDeg(props.actual ?? props.modelValue))

// SVG arc path from startDeg → endDeg
function arc(startD: number, endD: number): string {
  if (Math.abs(endD - startD) < 0.001) return ''
  const cx = c.value, cy = c.value, rr = r.value
  const toRad = (d: number) => ((d - 90) * Math.PI) / 180
  const s = { x: cx + rr * Math.cos(toRad(startD)), y: cy + rr * Math.sin(toRad(startD)) }
  const e = { x: cx + rr * Math.cos(toRad(endD)), y: cy + rr * Math.sin(toRad(endD)) }
  const large = Math.abs(endD - startD) > 180 ? 1 : 0
  const sweepFlag = endD > startD ? 1 : 0
  return `M ${s.x.toFixed(2)} ${s.y.toFixed(2)} A ${rr} ${rr} 0 ${large} ${sweepFlag} ${e.x.toFixed(2)} ${e.y.toFixed(2)}`
}
const trackPath = computed(() => arc(startDeg, startDeg + sweep))
const valuePath = computed(() => arc(startDeg, targetDeg.value))
const actualPath = computed(() => arc(startDeg, actualDeg.value))

const handlePos = computed(() => {
  const rad = ((targetDeg.value - 90) * Math.PI) / 180
  return { x: c.value + r.value * Math.cos(rad), y: c.value + r.value * Math.sin(rad) }
})

const svgRef = ref<SVGSVGElement>()
const dragging = ref(false)

function pointerToValue(ev: PointerEvent) {
  if (!svgRef.value) return
  const rect = svgRef.value.getBoundingClientRect()
  const px = ev.clientX - rect.left
  const py = ev.clientY - rect.top
  const dx = px - c.value
  const dy = py - c.value
  let deg = (Math.atan2(dy, dx) * 180) / Math.PI + 90
  if (deg < 0) deg += 360
  // clamp to sweep range
  const ws = startDeg
  const we = startDeg + sweep
  let dd = deg
  if (dd < ws) dd += 360
  if (dd > we) {
    // snap to nearest endpoint
    dd = Math.abs(dd - we) < Math.abs(dd - (ws + 360)) ? we : ws
  }
  const v = degToValue(dd)
  emit('update:modelValue', clamp(v))
}
function onPointerDown(ev: PointerEvent) {
  dragging.value = true
  svgRef.value?.setPointerCapture(ev.pointerId)
  pointerToValue(ev)
}
function onPointerMove(ev: PointerEvent) {
  if (!dragging.value) return
  pointerToValue(ev)
}
function onPointerUp(ev: PointerEvent) {
  dragging.value = false
  svgRef.value?.releasePointerCapture(ev.pointerId)
}
function onKey(e: KeyboardEvent) {
  if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') { emit('update:modelValue', clamp(props.modelValue - props.step)); e.preventDefault() }
  if (e.key === 'ArrowRight' || e.key === 'ArrowUp') { emit('update:modelValue', clamp(props.modelValue + props.step)); e.preventDefault() }
}
const isDirty = computed(() => props.actual !== undefined && Math.abs(props.actual - props.modelValue) > 1)

watch(() => props.modelValue, () => { /* no-op, hook for haptics if desired */ })
</script>

<template>
  <div class="ring-slider" :class="{ dirty: isDirty }" :style="{ width: size + 'px', height: size + 'px' }" tabindex="0" @keydown="onKey">
    <svg
      ref="svgRef"
      :width="size" :height="size" :viewBox="`0 0 ${size} ${size}`"
      class="rs-svg"
      @pointerdown="onPointerDown"
      @pointermove="onPointerMove"
      @pointerup="onPointerUp"
    >
      <path :d="trackPath" class="rs-track" :stroke-width="stroke" fill="none" stroke-linecap="round" />
      <path v-if="actual !== undefined" :d="actualPath" class="rs-actual" :stroke-width="stroke - 4" fill="none" stroke-linecap="round" />
      <path :d="valuePath" class="rs-value" :stroke-width="stroke" fill="none" stroke-linecap="round" :style="{ stroke: accent }" />
      <circle :cx="handlePos.x" :cy="handlePos.y" :r="stroke / 1.4" class="rs-handle" :style="{ fill: accent }" />
      <circle :cx="handlePos.x" :cy="handlePos.y" r="2.5" fill="white" />
    </svg>
    <div class="rs-center">
      <div class="rs-val mono" :style="{ color: accent }">{{ Math.round(modelValue) }}<span class="rs-unit">{{ unit }}</span></div>
      <div v-if="label" class="rs-label">{{ label }}</div>
    </div>
  </div>
</template>

<style scoped>
.ring-slider { position: relative; display: inline-block; outline: none; touch-action: none; cursor: pointer; transition: transform .15s var(--ease-out-quint); }
.ring-slider:focus-visible { transform: scale(1.04); }
.rs-svg { display: block; overflow: visible; }
.rs-track { stroke: color-mix(in srgb, var(--ink-muted) 26%, transparent); }
.rs-actual { stroke: color-mix(in srgb, var(--ink-tertiary) 50%, transparent); opacity: .85; }
.rs-value { transition: stroke .25s ease; }
.rs-handle {
  transition: r .12s ease;
  filter: drop-shadow(0 1px 4px rgba(15,23,42,.18));
}
.ring-slider:active .rs-handle { r: calc(var(--stroke, 8) / 1.2); }
.rs-center {
  position: absolute; inset: 0; pointer-events: none;
  display: flex; flex-direction: column; align-items: center; justify-content: center;
}
.rs-val { font-weight: 700; font-size: 1.15rem; line-height: 1; letter-spacing: -0.02em; }
.rs-unit { font-size: 0.6rem; font-weight: 500; opacity: .7; margin-left: 1px; }
.rs-label { font-size: 0.62rem; color: var(--ink-tertiary); margin-top: 2px; letter-spacing: 0.04em; font-weight: 600; }
.ring-slider.dirty .rs-val { color: var(--accent-amber) !important; }
.ring-slider.dirty::after {
  content: ''; position: absolute; inset: -3px; border-radius: 50%;
  background: radial-gradient(circle, rgba(217,119,6,0.15), transparent 65%);
  pointer-events: none; animation: pulseSoft 1.6s ease-in-out infinite;
}
</style>
