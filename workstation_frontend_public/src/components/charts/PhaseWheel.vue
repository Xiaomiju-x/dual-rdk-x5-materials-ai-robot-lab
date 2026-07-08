<script setup lang="ts">
// PhaseWheel — 6 joints arranged around a clock face. Each needle length =
// |arm01[j] - arm02[j]| / softCap, normalised to 0..1. Centre score = average
// in-phase (100% if all joints are equal, 0% if all are at ±softCap).
import { computed } from 'vue'

interface Props {
  a01: number[]
  a02: number[]
  size?: number
  softCap?: number       // a "fully out of phase" magnitude (deg)
}
const props = withDefaults(defineProps<Props>(), { size: 220, softCap: 90 })

const r = computed(() => props.size / 2)
const inner = computed(() => r.value * 0.16)
const outer = computed(() => r.value * 0.88)

const needles = computed(() => {
  return [0,1,2,3,4,5].map((j) => {
    const angDeg = -90 + j * 60      // top centre = J1
    const ang = (angDeg * Math.PI) / 180
    const diff = (props.a01[j] ?? 0) - (props.a02[j] ?? 0)
    const norm = Math.min(1, Math.abs(diff) / props.softCap)
    const ix = r.value + Math.cos(ang) * inner.value
    const iy = r.value + Math.sin(ang) * inner.value
    const ox = r.value + Math.cos(ang) * (inner.value + (outer.value - inner.value) * norm)
    const oy = r.value + Math.sin(ang) * (inner.value + (outer.value - inner.value) * norm)
    const lblX = r.value + Math.cos(ang) * (outer.value + 14)
    const lblY = r.value + Math.sin(ang) * (outer.value + 14)
    return { j, label: `J${j+1}`, diff, norm, ix, iy, ox, oy, lblX, lblY }
  })
})

const score = computed(() => {
  const sum = needles.value.reduce((acc, n) => acc + (1 - n.norm), 0)
  return Math.round((sum / needles.value.length) * 100)
})
const scoreColor = computed(() => {
  if (score.value >= 85) return '#10b981'
  if (score.value >= 60) return '#d97706'
  return '#ef4444'
})
</script>

<template>
  <div class="phase-wheel" :style="{ width: size + 'px', height: size + 'px' }">
    <svg :width="size" :height="size">
      <!-- rings -->
      <circle :cx="r" :cy="r" :r="outer" fill="none" stroke="var(--line-divider)" stroke-width="1" />
      <circle :cx="r" :cy="r" :r="outer * 0.66" fill="none" stroke="var(--line-hairline)" stroke-width="1" stroke-dasharray="2 4" />
      <circle :cx="r" :cy="r" :r="outer * 0.33" fill="none" stroke="var(--line-hairline)" stroke-width="1" stroke-dasharray="2 4" />
      <!-- spokes -->
      <g v-for="n in needles" :key="n.j">
        <line :x1="r" :y1="r" :x2="r + Math.cos((-90 + n.j*60) * Math.PI / 180) * outer"
              :y2="r + Math.sin((-90 + n.j*60) * Math.PI / 180) * outer"
              stroke="var(--line-hairline)" stroke-width="1" />
      </g>
      <!-- needles -->
      <g v-for="n in needles" :key="'nd-'+n.j">
        <line :x1="n.ix" :y1="n.iy" :x2="n.ox" :y2="n.oy"
              :stroke="n.norm > 0.7 ? '#ef4444' : n.norm > 0.35 ? '#d97706' : '#10b981'"
              stroke-width="3" stroke-linecap="round" />
        <circle :cx="n.ox" :cy="n.oy" r="3.5"
                :fill="n.norm > 0.7 ? '#ef4444' : n.norm > 0.35 ? '#d97706' : '#10b981'" />
      </g>
      <!-- labels -->
      <g v-for="n in needles" :key="'lb-'+n.j">
        <text :x="n.lblX" :y="n.lblY" text-anchor="middle" dominant-baseline="middle"
              font-size="11" font-family="JetBrains Mono Variable, monospace" fill="var(--ink-tertiary)" font-weight="700">{{ n.label }}</text>
      </g>
      <!-- centre score -->
      <circle :cx="r" :cy="r" :r="inner" fill="var(--bg-card)" stroke="var(--line-divider)" />
      <text :x="r" :y="r - 4" text-anchor="middle" dominant-baseline="middle"
            font-size="22" font-family="JetBrains Mono Variable, monospace" font-weight="800" :fill="scoreColor">{{ score }}</text>
      <text :x="r" :y="r + 14" text-anchor="middle" dominant-baseline="middle"
            font-size="9" font-family="Inter, sans-serif" fill="var(--ink-muted)" font-weight="700" letter-spacing="0.15em">SYNC</text>
    </svg>
  </div>
</template>

<style scoped>
.phase-wheel { position: relative; }
svg { display: block; }
</style>
