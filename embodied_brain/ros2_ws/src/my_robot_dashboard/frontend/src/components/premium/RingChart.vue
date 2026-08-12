<script setup lang="ts">
/**
 * RingChart — animated arc gauge with center label.
 *
 *   <RingChart :pct="batteryPct" label="Battery" suffix="%" />
 *   <RingChart :pct="bpuUtil" :stroke="14" accent="violet" />
 */
import { computed, watch, ref } from 'vue'

interface Props {
  pct: number          // 0..100
  label?: string
  suffix?: string
  size?: number
  stroke?: number
  /** semantic accent driving the gradient */
  accent?: 'blue' | 'teal' | 'emerald' | 'violet' | 'amber' | 'rose'
  /** ms */
  duration?: number
  /** decimals on center number */
  decimals?: number
}
const props = withDefaults(defineProps<Props>(), {
  label: '',
  suffix: '%',
  size: 120,
  stroke: 10,
  accent: 'blue',
  duration: 820,
  decimals: 0,
})

const radius = computed(() => (props.size - props.stroke) / 2)
const circ   = computed(() => 2 * Math.PI * radius.value)

// animated value
const display = ref(0)
let raf = 0
function animateTo(target: number) {
  cancelAnimationFrame(raf)
  const from = display.value
  const t0 = performance.now()
  const tick = (now: number) => {
    const t = Math.min(1, (now - t0) / props.duration)
    const eased = 1 - Math.pow(1 - t, 5)
    display.value = from + (target - from) * eased
    if (t < 1) raf = requestAnimationFrame(tick)
  }
  raf = requestAnimationFrame(tick)
}
watch(() => props.pct, (v) => animateTo(Math.max(0, Math.min(100, v))), { immediate: true })

const offset = computed(() => circ.value - (display.value / 100) * circ.value)

const ACCENT_FROM: Record<Required<Props>['accent'], string> = {
  blue: '#2563eb', teal: '#0891b2', emerald: '#059669', violet: '#7c3aed', amber: '#d97706', rose: '#e11d48',
}
const ACCENT_TO: Record<Required<Props>['accent'], string> = {
  blue: '#7c3aed', teal: '#10b981', emerald: '#0891b2', violet: '#ec4899', amber: '#e11d48', rose: '#7c3aed',
}
const gradId = computed(() => `rc-${props.accent}-${props.size}-${Math.random().toString(36).slice(2, 6)}`)
</script>

<template>
  <div class="rc-wrap" :style="{ width: `${props.size}px`, height: `${props.size}px` } as any">
    <svg :width="props.size" :height="props.size" :viewBox="`0 0 ${props.size} ${props.size}`">
      <defs>
        <linearGradient :id="gradId" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" :stop-color="ACCENT_FROM[props.accent]" />
          <stop offset="100%" :stop-color="ACCENT_TO[props.accent]" />
        </linearGradient>
      </defs>
      <circle
        :cx="props.size / 2" :cy="props.size / 2" :r="radius"
        fill="none"
        stroke="var(--line-divider)"
        :stroke-width="props.stroke"
      />
      <circle
        :cx="props.size / 2" :cy="props.size / 2" :r="radius"
        fill="none"
        :stroke="`url(#${gradId})`"
        :stroke-width="props.stroke"
        :stroke-dasharray="circ"
        :stroke-dashoffset="offset"
        stroke-linecap="round"
        :transform="`rotate(-90 ${props.size / 2} ${props.size / 2})`"
        class="rc-arc"
      />
    </svg>
    <div class="rc-text">
      <div class="rc-val mono">{{ display.toFixed(props.decimals) }}<span class="rc-suf">{{ props.suffix }}</span></div>
      <div v-if="props.label" class="rc-label">{{ props.label }}</div>
    </div>
  </div>
</template>

<style scoped>
.rc-wrap { position: relative; display: inline-block; }
.rc-arc { transition: stroke-dashoffset 0.4s var(--ease-out-quint); filter: drop-shadow(0 0 6px currentColor); }
.rc-text {
  position: absolute; inset: 0;
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  pointer-events: none;
}
.rc-val { font-size: 1.45rem; font-weight: 700; letter-spacing: -0.02em; color: var(--ink-primary); }
.rc-suf { font-size: 0.7em; color: var(--ink-tertiary); margin-left: 1px; }
.rc-label { font-size: 0.6rem; color: var(--ink-muted); letter-spacing: 0.12em; text-transform: uppercase; margin-top: 2px; }
</style>
