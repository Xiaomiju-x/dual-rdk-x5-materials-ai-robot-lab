<script setup lang="ts">
// Odometer — gas-pump digit flip with smooth interpolation.
// Each digit slot animates independently; integer + fractional supported.
// Falls back to a plain number when reduce-motion is on.
import { computed, ref, watch, onBeforeUnmount } from 'vue'
import { useSettingsStore } from '@/stores/settings'

const props = withDefaults(defineProps<{
  value: number
  precision?: number       // fractional digits, default 0
  pad?: number             // pad integer side to at least N digits (e.g. 3 → 007)
  duration?: number        // ms to reach target
  prefix?: string
  suffix?: string
  loose?: boolean          // narrower digit spacing
}>(), { precision: 0, pad: 0, duration: 700, prefix: '', suffix: '', loose: false })

const settings = useSettingsStore()
const display = ref(props.value)
let raf = 0
let from = props.value
let start = 0

function easeOutQuint(t: number) { return 1 - Math.pow(1 - t, 5) }

function animate(to: number) {
  if (settings.reduceMotion) { display.value = to; return }
  cancelAnimationFrame(raf)
  from = display.value
  start = performance.now()
  const dur = props.duration
  const step = () => {
    const t = Math.min(1, (performance.now() - start) / dur)
    display.value = from + (to - from) * easeOutQuint(t)
    if (t < 1) raf = requestAnimationFrame(step)
  }
  raf = requestAnimationFrame(step)
}

watch(() => props.value, (n) => animate(n))
onBeforeUnmount(() => cancelAnimationFrame(raf))

const intDigits = computed<string[]>(() => {
  const sign = display.value < 0 ? '-' : ''
  const abs = Math.abs(display.value)
  let intStr = Math.trunc(abs).toString()
  if (props.pad > 0) intStr = intStr.padStart(props.pad, '0')
  return (sign + intStr).split('')
})
const fracDigits = computed<string[]>(() => {
  if (props.precision <= 0) return []
  const abs = Math.abs(display.value)
  const frac = abs - Math.trunc(abs)
  return frac.toFixed(props.precision).slice(2).split('')
})

function rollFor(ch: string): number {
  const n = parseInt(ch, 10)
  return isNaN(n) ? 0 : n
}
</script>

<template>
  <span class="odo" :class="{ loose }">
    <span v-if="prefix" class="odo-fix">{{ prefix }}</span>
    <span v-for="(c, i) in intDigits" :key="'i'+i" class="odo-slot" :class="{ 'odo-sym': isNaN(parseInt(c,10)) }">
      <template v-if="isNaN(parseInt(c,10))">{{ c }}</template>
      <span v-else class="odo-roll" :style="{ transform: `translateY(${-rollFor(c) * 10}%)` }">
        <span v-for="d in 10" :key="d">{{ d - 1 }}</span>
      </span>
    </span>
    <template v-if="fracDigits.length">
      <span class="odo-sym">.</span>
      <span v-for="(c, i) in fracDigits" :key="'f'+i" class="odo-slot">
        <span class="odo-roll" :style="{ transform: `translateY(${-rollFor(c) * 10}%)` }">
          <span v-for="d in 10" :key="d">{{ d - 1 }}</span>
        </span>
      </span>
    </template>
    <span v-if="suffix" class="odo-fix">{{ suffix }}</span>
  </span>
</template>

<style scoped>
.odo {
  font-family: 'JetBrains Mono Variable', ui-monospace, SFMono-Regular, Menlo, monospace;
  font-feature-settings: 'tnum';
  display: inline-flex;
  align-items: baseline;
  letter-spacing: -0.02em;
  line-height: 1;
}
.odo.loose .odo-slot { width: .58em; }
.odo-slot {
  display: inline-block;
  height: 1em;
  width: .62em;
  overflow: hidden;
  position: relative;
  text-align: center;
  vertical-align: baseline;
}
.odo-slot.odo-sym { width: auto; padding: 0 .03em; }
.odo-roll {
  display: inline-flex;
  flex-direction: column;
  transition: transform .5s cubic-bezier(0.34, 1.2, 0.4, 1);
  will-change: transform;
}
.odo-roll > span {
  display: block;
  height: 1em;
  line-height: 1em;
}
.odo-fix { font-family: inherit; padding: 0 .15em; opacity: .75; }
</style>
