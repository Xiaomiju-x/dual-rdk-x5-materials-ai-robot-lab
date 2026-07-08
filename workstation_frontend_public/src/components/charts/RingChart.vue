<script setup lang="ts">
// RingChart — premium concentric ring with center label + sub-label.
// Two ring layers (track + progress) + optional inner small ring (secondary metric).
import { computed } from 'vue'

const props = withDefaults(defineProps<{
  value: number               // 0..1 outer ring
  inner?: number              // 0..1 inner ring (optional)
  size?: number               // px
  label?: string
  caption?: string
  accent?: 'blue' | 'teal' | 'emerald' | 'violet' | 'amber' | 'rose'
  innerAccent?: 'blue' | 'teal' | 'emerald' | 'violet' | 'amber' | 'rose'
}>(), { size: 132, accent: 'blue', innerAccent: 'teal' })

const ACCENT: Record<string, string> = {
  blue: 'var(--accent-blue)', teal: 'var(--accent-teal)',
  emerald: 'var(--accent-emerald)', violet: 'var(--accent-violet)',
  amber: 'var(--accent-amber)', rose: 'var(--accent-rose)',
}
const r1 = computed(() => props.size / 2 - 8)
const r2 = computed(() => r1.value - 16)
const c1 = computed(() => 2 * Math.PI * r1.value)
const c2 = computed(() => 2 * Math.PI * r2.value)
const off1 = computed(() => c1.value * (1 - Math.max(0, Math.min(1, props.value))))
const off2 = computed(() => c2.value * (1 - Math.max(0, Math.min(1, props.inner ?? 0))))
</script>

<template>
  <div class="ring-chart" :style="{ width: size + 'px', height: size + 'px' }">
    <svg :width="size" :height="size" :viewBox="`0 0 ${size} ${size}`" class="ring-svg">
      <g :transform="`rotate(-90 ${size/2} ${size/2})`">
        <circle :cx="size/2" :cy="size/2" :r="r1" fill="none" class="ring-track" stroke-width="6" />
        <circle :cx="size/2" :cy="size/2" :r="r1" fill="none"
                :stroke="ACCENT[accent]" stroke-width="6" stroke-linecap="round"
                :stroke-dasharray="c1" :stroke-dashoffset="off1" class="ring-progress" />
        <circle v-if="inner !== undefined" :cx="size/2" :cy="size/2" :r="r2" fill="none" class="ring-track" stroke-width="4" />
        <circle v-if="inner !== undefined" :cx="size/2" :cy="size/2" :r="r2" fill="none"
                :stroke="ACCENT[innerAccent]" stroke-width="4" stroke-linecap="round"
                :stroke-dasharray="c2" :stroke-dashoffset="off2" class="ring-progress" />
      </g>
    </svg>
    <div class="ring-center">
      <div class="ring-label">{{ label }}</div>
      <div class="ring-cap kv-mono">{{ caption }}</div>
    </div>
  </div>
</template>

<style scoped>
.ring-chart { position: relative; display: inline-block; flex-shrink: 0; }
.ring-svg { display: block; }
.ring-center {
  position: absolute; inset: 0;
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  pointer-events: none;
}
.ring-label { font-weight: 700; font-size: 1.05rem; color: var(--ink-primary); letter-spacing: -0.01em; }
.ring-cap { opacity: .75; margin-top: 2px; }
</style>
