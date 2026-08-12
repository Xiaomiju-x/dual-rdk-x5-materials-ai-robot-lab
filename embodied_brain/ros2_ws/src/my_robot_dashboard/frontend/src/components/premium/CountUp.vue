<script setup lang="ts">
/**
 * CountUp — animate a number from 0 (or any start) to a target on mount
 *           or whenever the prop changes.
 *
 *   <CountUp :end="29" suffix="s" />
 *   <CountUp :end="totalDistance" :decimals="2" suffix=" m" :duration="1100" />
 */
import { onMounted, ref, watch } from 'vue'

interface Props {
  end: number
  start?: number
  duration?: number   // ms
  decimals?: number
  prefix?: string
  suffix?: string
  /** ease curve from 0..1; default = ease-out-quint */
  ease?: (t: number) => number
}
const props = withDefaults(defineProps<Props>(), {
  start: 0,
  duration: 1200,
  decimals: 0,
  prefix: '',
  suffix: '',
  ease: (t: number) => 1 - Math.pow(1 - t, 5),
})

const value = ref(props.start)
let raf = 0

function run(from: number, to: number) {
  cancelAnimationFrame(raf)
  const t0 = performance.now()
  const tick = (now: number) => {
    const t = Math.min(1, (now - t0) / props.duration)
    value.value = from + (to - from) * props.ease(t)
    if (t < 1) raf = requestAnimationFrame(tick)
    else value.value = to
  }
  raf = requestAnimationFrame(tick)
}

onMounted(() => run(props.start, props.end))
watch(() => props.end, (v, prev) => run(typeof prev === 'number' ? value.value : props.start, v))
</script>

<template>
  <span class="count-up">
    {{ props.prefix }}{{ value.toFixed(props.decimals) }}{{ props.suffix }}
  </span>
</template>

<style scoped>
.count-up {
  font-variant-numeric: tabular-nums;
  font-feature-settings: 'tnum';
}
</style>
