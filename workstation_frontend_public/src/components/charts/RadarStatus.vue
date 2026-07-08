<script setup lang="ts">
// RadarStatus — 6-axis health radar (telemetry / health summary).
// Pure SVG, animates polygon points via CSS transitions.
import { computed } from 'vue'

interface Axis { label: string; value: number }   // value 0..1
const props = withDefaults(defineProps<{
  axes: Axis[]
  size?: number
  accent?: 'blue' | 'teal' | 'emerald' | 'violet' | 'amber' | 'rose'
}>(), { size: 260, accent: 'blue' })

const ACCENT_RGB: Record<string, string> = {
  blue: '37,99,235', teal: '8,145,178', emerald: '5,150,105',
  violet: '124,58,237', amber: '217,119,6', rose: '225,29,72',
}
const accentVar = computed(() => `rgb(${ACCENT_RGB[props.accent]})`)
const accentFill = computed(() => `rgba(${ACCENT_RGB[props.accent]}, .15)`)
const cx = computed(() => props.size / 2)
const cy = computed(() => props.size / 2)
const R = computed(() => props.size / 2 - 26)
const angle = (i: number, n: number) => (-Math.PI / 2) + (i * 2 * Math.PI) / n
function pt(i: number, n: number, v: number) {
  const a = angle(i, n)
  return [cx.value + Math.cos(a) * R.value * v, cy.value + Math.sin(a) * R.value * v]
}
const valuesPoints = computed(() =>
  props.axes.map((ax, i) => pt(i, props.axes.length, Math.max(0.04, Math.min(1, ax.value))).join(',')).join(' '),
)
const labelPositions = computed(() =>
  props.axes.map((ax, i) => {
    const [x, y] = pt(i, props.axes.length, 1.18)
    return { label: ax.label, x, y, anchor: x < cx.value - 4 ? 'end' : x > cx.value + 4 ? 'start' : 'middle' }
  }),
)
const gridLevels = [0.25, 0.5, 0.75, 1]
function gridPoly(level: number): string {
  return props.axes.map((_, i) => pt(i, props.axes.length, level).join(',')).join(' ')
}
function axisLine(i: number): { x1: number; y1: number; x2: number; y2: number } {
  const [x, y] = pt(i, props.axes.length, 1)
  return { x1: cx.value, y1: cy.value, x2: x, y2: y }
}
</script>

<template>
  <svg :width="size" :height="size" :viewBox="`0 0 ${size} ${size}`" class="radar">
    <g class="grid">
      <polygon v-for="l in gridLevels" :key="l" :points="gridPoly(l)" />
      <line v-for="(_, i) in axes" :key="'a'+i" v-bind="axisLine(i)" />
    </g>
    <polygon :points="valuesPoints" class="value-poly" :style="{ fill: accentFill, stroke: accentVar }" />
    <g class="value-dots">
      <circle
        v-for="(ax, i) in axes" :key="'d'+i"
        :cx="pt(i, axes.length, Math.max(0.04, Math.min(1, ax.value)))[0]"
        :cy="pt(i, axes.length, Math.max(0.04, Math.min(1, ax.value)))[1]"
        r="3.2" :fill="accentVar" stroke="white" stroke-width="1.4" />
    </g>
    <g class="labels">
      <text v-for="(l, i) in labelPositions" :key="'l'+i"
            :x="l.x" :y="l.y" :text-anchor="l.anchor" dominant-baseline="middle">{{ l.label }}</text>
    </g>
  </svg>
</template>

<style scoped>
.radar { display: block; }
.grid polygon { fill: none; stroke: color-mix(in srgb, var(--ink-muted) 25%, transparent); stroke-width: 0.7; }
.grid line { stroke: color-mix(in srgb, var(--ink-muted) 18%, transparent); stroke-width: 0.7; stroke-dasharray: 2 4; }
.value-poly { stroke-width: 1.8; transition: all .8s cubic-bezier(0.22,1,0.36,1); }
.value-dots circle { transition: cx .8s cubic-bezier(0.22,1,0.36,1), cy .8s cubic-bezier(0.22,1,0.36,1); }
.labels text { fill: var(--ink-tertiary); font-size: 11px; font-weight: 600; letter-spacing: 0.04em; }
</style>
