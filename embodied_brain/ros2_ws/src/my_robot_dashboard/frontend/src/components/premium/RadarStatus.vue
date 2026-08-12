<script setup lang="ts">
/**
 * RadarStatus — circular radar sweep with N items pinned around the
 * perimeter, each colored by health.
 *
 *   <RadarStatus :items="sensors" />
 *
 * items: { id, label, angle?, health: 'ok'|'warn'|'err'|'idle' }
 * (angle is optional; auto-distributed if omitted)
 */
import { computed } from 'vue'

interface Item {
  id: string
  label: string
  health: 'ok' | 'warn' | 'err' | 'idle'
  angle?: number          // degrees
  hint?: string
}
interface Props {
  items: Item[]
  size?: number           // viewbox size
  ringCount?: number
}
const props = withDefaults(defineProps<Props>(), { size: 220, ringCount: 3 })

const cx = computed(() => props.size / 2)
const cy = computed(() => props.size / 2)
const rOuter = computed(() => (props.size / 2) - 16)
const rings = computed(() => {
  const out: number[] = []
  for (let i = 1; i <= props.ringCount; i++) out.push((rOuter.value / props.ringCount) * i)
  return out
})

const positioned = computed(() => {
  const n = props.items.length || 1
  return props.items.map((it, i) => {
    const ang = it.angle ?? (i * (360 / n)) - 90
    const rad = (ang * Math.PI) / 180
    const r = rOuter.value - 6
    return {
      ...it,
      angle: ang,
      x: cx.value + Math.cos(rad) * r,
      y: cy.value + Math.sin(rad) * r,
    }
  })
})

const HEALTH_COLOR: Record<Item['health'], string> = {
  ok: '#10b981',
  warn: '#f59e0b',
  err: '#ef4444',
  idle: '#94a3b8',
}
</script>

<template>
  <div class="radar-wrap">
    <svg :viewBox="`0 0 ${props.size} ${props.size}`" class="radar">
      <defs>
        <radialGradient :id="`r-bg-${props.size}`" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stop-color="rgba(37,99,235,0.18)"/>
          <stop offset="60%" stop-color="rgba(37,99,235,0.04)"/>
          <stop offset="100%" stop-color="rgba(37,99,235,0)"/>
        </radialGradient>
        <linearGradient :id="`r-sweep-${props.size}`">
          <stop offset="0%" stop-color="rgba(37,99,235,0)"/>
          <stop offset="100%" stop-color="rgba(37,99,235,0.55)"/>
        </linearGradient>
      </defs>

      <circle :cx="cx" :cy="cy" :r="rOuter" :fill="`url(#r-bg-${props.size})`" />

      <!-- rings -->
      <circle
        v-for="(r, i) in rings"
        :key="i"
        :cx="cx" :cy="cy" :r="r"
        fill="none" stroke="var(--line-divider)" stroke-width="1"
      />
      <!-- crosshair -->
      <line :x1="cx - rOuter" :y1="cy" :x2="cx + rOuter" :y2="cy" stroke="var(--line-divider)" stroke-width="1"/>
      <line :x1="cx" :y1="cy - rOuter" :x2="cx" :y2="cy + rOuter" stroke="var(--line-divider)" stroke-width="1"/>

      <!-- sweep arm -->
      <g class="sweep" :transform-origin="`${cx} ${cy}`">
        <line :x1="cx" :y1="cy" :x2="cx + rOuter" :y2="cy" stroke="rgba(37,99,235,0.55)" stroke-width="1.5"/>
        <path
          :d="`M ${cx} ${cy} L ${cx + rOuter} ${cy} A ${rOuter} ${rOuter} 0 0 0 ${cx + rOuter * Math.cos(-Math.PI/3)} ${cy + rOuter * Math.sin(-Math.PI/3)} Z`"
          :fill="`url(#r-sweep-${props.size})`"
          opacity="0.55"
        />
      </g>

      <!-- items -->
      <g v-for="it in positioned" :key="it.id">
        <circle
          :cx="it.x" :cy="it.y" :r="6.5"
          :fill="HEALTH_COLOR[it.health]"
          :class="['dotpt', `pt-${it.health}`]"
        />
        <circle :cx="it.x" :cy="it.y" :r="3.5" fill="white" opacity="0.85" />
        <text
          :x="it.x" :y="it.y + (it.y > cy ? 18 : -10)"
          text-anchor="middle"
          class="radar-label"
        >{{ it.label }}</text>
      </g>
    </svg>
  </div>
</template>

<style scoped>
.radar-wrap { display: inline-flex; align-items: center; justify-content: center; }
.radar { width: 100%; height: 100%; max-width: 320px; }
.sweep { animation: spin 6s linear infinite; transform-box: fill-box; transform-origin: center; }
@keyframes spin {
  from { transform: rotate(0deg); }
  to   { transform: rotate(360deg); }
}
.radar-label {
  font-size: 9px;
  font-family: 'JetBrains Mono Variable', ui-monospace, monospace;
  fill: var(--ink-tertiary);
}
.dotpt { transition: r 200ms var(--ease-out-quint), fill 200ms var(--ease-out-quint); filter: drop-shadow(0 0 6px currentColor); }
.pt-warn, .pt-err { animation: pulse 1.8s ease-in-out infinite; transform-origin: center; transform-box: fill-box; }
.pt-err { animation-duration: 1.2s; }
@keyframes pulse {
  0%, 100% { transform: scale(1); }
  50%      { transform: scale(1.25); }
}
@media (prefers-reduced-motion: reduce) { .sweep, .dotpt { animation: none !important; } }
[data-reduce-motion='true'] .sweep, [data-reduce-motion='true'] .dotpt { animation: none !important; }
[data-theme='dark'] .radar-label { fill: var(--ink-muted); }
</style>
