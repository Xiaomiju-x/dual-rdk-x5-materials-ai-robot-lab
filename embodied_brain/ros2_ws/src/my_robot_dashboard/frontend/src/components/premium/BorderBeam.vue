<script setup lang="ts">
/**
 * BorderBeam — animated conic-gradient border beam (Vercel style).
 *
 *   <div class="card relative">
 *     <BorderBeam :duration="8" :size="160" />
 *     ...content
 *   </div>
 *
 * Drop into any positioned card. Two beams travel opposite directions.
 */
interface Props {
  /** seconds for a full revolution */
  duration?: number
  /** beam length in px along the perimeter */
  size?: number
  /** primary gradient color */
  colorFrom?: string
  /** trailing gradient color */
  colorTo?: string
  /** border-radius in px (match parent) */
  radius?: number
  /** whether to fire the counter-beam */
  dual?: boolean
}
const props = withDefaults(defineProps<Props>(), {
  duration: 9,
  size: 160,
  colorFrom: 'rgba(37, 99, 235, 0.92)',
  colorTo:   'rgba(8, 145, 178, 0.92)',
  radius: 18,
  dual: true,
})
</script>

<template>
  <div
    class="beam"
    :style="{
      '--bb-duration': `${props.duration}s`,
      '--bb-size': `${props.size}px`,
      '--bb-from': props.colorFrom,
      '--bb-to':   props.colorTo,
      '--bb-radius': `${props.radius}px`,
    } as any"
    aria-hidden="true"
  >
    <span class="beam-a"></span>
    <span v-if="dual" class="beam-b"></span>
  </div>
</template>

<style scoped>
.beam {
  position: absolute;
  inset: 0;
  border-radius: var(--bb-radius);
  pointer-events: none;
  overflow: hidden;
  /* mask to "ring only" — show the rotating beam at the perimeter, not the fill */
  -webkit-mask:
    linear-gradient(#000 0 0) content-box,
    linear-gradient(#000 0 0);
  -webkit-mask-composite: xor;
          mask-composite: exclude;
  padding: 1px;
}
.beam-a, .beam-b {
  position: absolute;
  inset: 0;
  border-radius: inherit;
  background: conic-gradient(from 0deg,
    transparent 0deg,
    var(--bb-from) 35deg,
    var(--bb-to) 70deg,
    transparent 110deg);
  animation: spin var(--bb-duration) linear infinite;
  will-change: transform;
}
.beam-b {
  background: conic-gradient(from 180deg,
    transparent 0deg,
    var(--bb-to) 35deg,
    var(--bb-from) 70deg,
    transparent 110deg);
  animation-direction: reverse;
  opacity: 0.7;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

@media (prefers-reduced-motion: reduce) {
  .beam-a, .beam-b { animation: none; opacity: 0.20; }
}
[data-reduce-motion='true'] .beam-a,
[data-reduce-motion='true'] .beam-b { animation: none; opacity: 0.20; }
</style>
