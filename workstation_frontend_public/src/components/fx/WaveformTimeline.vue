<script setup lang="ts">
// WaveformTimeline — draws a stacked sparkband of the recorded angle deltas as
// a "waveform", with a draggable scrubber + playhead. Used to scrub clips in
// the Teleop replay UI. Pure SVG, no canvas.
import { computed } from 'vue'

interface Frame { angles: number[]; gripper: number }
const props = withDefaults(defineProps<{
  frames: Frame[]
  position?: number              // 0..1
  accent?: string                // CSS color
  width?: number
  height?: number
}>(), { position: 0, accent: 'var(--accent-blue)', width: 280, height: 60 })
const emit = defineEmits<{ (e: 'update:position', v: number): void; (e: 'seek', i: number): void }>()

const N = computed(() => props.frames.length)
// summarise into a delta-magnitude band per frame
const peaks = computed<number[]>(() => {
  if (N.value === 0) return []
  let max = 0
  const arr: number[] = []
  for (let i = 0; i < N.value; i++) {
    if (i === 0) { arr.push(0); continue }
    const a = props.frames[i].angles
    const b = props.frames[i - 1].angles
    let s = 0
    for (let j = 0; j < 6; j++) s += Math.abs((a[j] ?? 0) - (b[j] ?? 0))
    s += Math.abs(props.frames[i].gripper - props.frames[i - 1].gripper) * 0.4
    if (s > max) max = s
    arr.push(s)
  }
  if (max < 0.001) return arr.map(() => 0)
  return arr.map((v) => v / max)
})
const path = computed(() => {
  if (peaks.value.length === 0) return ''
  const W = props.width
  const H = props.height
  const cy = H / 2
  let s = `M 0 ${cy}`
  peaks.value.forEach((p, i) => {
    const x = (i / Math.max(1, peaks.value.length - 1)) * W
    const y = cy - p * (cy - 4)
    s += ` L ${x.toFixed(2)} ${y.toFixed(2)}`
  })
  peaks.value.slice().reverse().forEach((p, ri) => {
    const i = peaks.value.length - 1 - ri
    const x = (i / Math.max(1, peaks.value.length - 1)) * W
    const y = cy + p * (cy - 4)
    s += ` L ${x.toFixed(2)} ${y.toFixed(2)}`
  })
  s += ' Z'
  return s
})
const playX = computed(() => Math.max(0, Math.min(1, props.position)) * props.width)

function onPointer(e: PointerEvent) {
  const tgt = e.currentTarget as SVGSVGElement
  const r = tgt.getBoundingClientRect()
  const t = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width))
  emit('update:position', t)
  if (N.value > 0) emit('seek', Math.floor(t * (N.value - 1)))
}
function onDown(e: PointerEvent) {
  ;(e.currentTarget as SVGSVGElement).setPointerCapture(e.pointerId)
  onPointer(e)
}
function onMove(e: PointerEvent) {
  if (e.buttons === 0) return
  onPointer(e)
}
</script>

<template>
  <svg
    :width="width" :height="height" :viewBox="`0 0 ${width} ${height}`"
    class="wf"
    @pointerdown="onDown"
    @pointermove="onMove"
  >
    <defs>
      <linearGradient :id="`wfg-${accent.replace(/[^a-z0-9]/gi,'')}`" x1="0" x2="0" y1="0" y2="1">
        <stop offset="0" :stop-color="accent" stop-opacity="0.45" />
        <stop offset="1" :stop-color="accent" stop-opacity="0.05" />
      </linearGradient>
    </defs>
    <line class="wf-mid" :x1="0" :x2="width" :y1="height/2" :y2="height/2" />
    <path :d="path" :fill="`url(#wfg-${accent.replace(/[^a-z0-9]/gi,'')})`" :stroke="accent" stroke-width="1" />
    <line class="wf-play" :x1="playX" :x2="playX" y1="0" :y2="height" :style="{ stroke: accent }" />
    <circle class="wf-head" :cx="playX" :cy="height/2" r="5" :style="{ fill: accent }" />
  </svg>
</template>

<style scoped>
.wf { display: block; touch-action: none; cursor: ew-resize; user-select: none; border-radius: 8px; background: color-mix(in srgb, var(--bg-elevated) 88%, transparent); }
.wf-mid { stroke: var(--line-divider); stroke-width: 1; }
.wf-play { stroke-width: 1.5; filter: drop-shadow(0 0 4px currentColor); }
.wf-head { filter: drop-shadow(0 1px 4px rgba(15,23,42,.25)); }
</style>
