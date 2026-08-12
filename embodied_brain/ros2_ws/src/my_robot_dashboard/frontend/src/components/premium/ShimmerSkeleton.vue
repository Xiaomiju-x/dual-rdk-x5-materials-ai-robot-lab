<script setup lang="ts">
/**
 * ShimmerSkeleton — premium loading placeholder.
 *
 *   <ShimmerSkeleton width="220px" height="14px" />
 *   <ShimmerSkeleton variant="card" />
 */
interface Props {
  width?: string
  height?: string
  variant?: 'line' | 'card' | 'circle' | 'pill'
  rounded?: string
}
const props = withDefaults(defineProps<Props>(), {
  width: '100%',
  height: '12px',
  variant: 'line',
  rounded: '',
})

const radius = () => {
  if (props.rounded) return props.rounded
  if (props.variant === 'circle') return '50%'
  if (props.variant === 'pill') return '999px'
  if (props.variant === 'card') return '14px'
  return '6px'
}
</script>

<template>
  <div
    class="shimmer"
    :style="{ width: props.width, height: props.height, borderRadius: radius() }"
    aria-hidden="true"
  ></div>
</template>

<style scoped>
.shimmer {
  display: inline-block;
  background: linear-gradient(110deg,
    var(--bg-elevated) 0%,
    rgba(15, 23, 42, 0.06) 40%,
    var(--bg-elevated) 80%);
  background-size: 220% 100%;
  animation: skel-shimmer 1.6s linear infinite;
}
[data-theme='dark'] .shimmer {
  background: linear-gradient(110deg,
    rgba(255, 255, 255, 0.04) 0%,
    rgba(255, 255, 255, 0.10) 40%,
    rgba(255, 255, 255, 0.04) 80%);
  background-size: 220% 100%;
}
@keyframes skel-shimmer {
  0%   { background-position: 220% 0; }
  100% { background-position: -120% 0; }
}
@media (prefers-reduced-motion: reduce) { .shimmer { animation: none; } }
[data-reduce-motion='true'] .shimmer { animation: none; }
</style>
