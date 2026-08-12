<script setup lang="ts">
/**
 * Odometer — sliding-digit numeric display, Tesla cluster style.
 *
 *   <Odometer :value="batteryPct" :decimals="1" />
 *   <Odometer :value="totalDistance" :decimals="2" prefix="↗" suffix="m" />
 *
 * Each character (digit / dot / minus) lives in a 1-line tall strip
 * that scrolls. Sub-second debounce + tight cubic-bezier easing.
 */
import { computed, watch, ref } from 'vue'

interface Props {
  value: number | string
  decimals?: number
  prefix?: string
  suffix?: string
  /** char height in em — matches font line-height */
  charHeight?: string
}

const props = withDefaults(defineProps<Props>(), {
  decimals: 0,
  prefix: '',
  suffix: '',
  charHeight: '1em',
})

function fmt(v: number | string): string {
  if (typeof v === 'string') return v
  if (Number.isNaN(v) || !Number.isFinite(v)) return '—'
  return v.toFixed(props.decimals)
}

const displayed = ref(fmt(props.value))
watch(() => props.value, (v) => { displayed.value = fmt(v) }, { immediate: true })

const chars = computed(() => displayed.value.split(''))

/** for digits use 0..9 column scroll, for non-digits just show the char */
function isDigit(c: string) { return c >= '0' && c <= '9' }
</script>

<template>
  <span class="odo" :style="{ '--ch': props.charHeight } as any">
    <span v-if="props.prefix" class="odo-fix">{{ props.prefix }}</span>
    <span
      v-for="(c, i) in chars"
      :key="`${i}-${c === '.' ? 'dot' : 'd'}`"
      class="odo-char"
      :class="{ 'odo-digit': isDigit(c) }"
    >
      <template v-if="isDigit(c)">
        <span class="odo-col" :style="{ transform: `translateY(-${parseInt(c, 10)}em)` }">
          <span class="odo-cell">0</span>
          <span class="odo-cell">1</span>
          <span class="odo-cell">2</span>
          <span class="odo-cell">3</span>
          <span class="odo-cell">4</span>
          <span class="odo-cell">5</span>
          <span class="odo-cell">6</span>
          <span class="odo-cell">7</span>
          <span class="odo-cell">8</span>
          <span class="odo-cell">9</span>
        </span>
      </template>
      <template v-else>{{ c }}</template>
    </span>
    <span v-if="props.suffix" class="odo-fix">{{ props.suffix }}</span>
  </span>
</template>

<style scoped>
.odo {
  font-variant-numeric: tabular-nums;
  font-feature-settings: 'tnum';
  display: inline-flex;
  align-items: baseline;
  letter-spacing: -0.01em;
}
.odo-fix { color: inherit; padding-right: 0.18em; }
.odo-char { display: inline-block; }
.odo-char.odo-digit {
  display: inline-block;
  height: var(--ch);
  line-height: var(--ch);
  overflow: hidden;
  vertical-align: top;
}
.odo-col {
  display: flex;
  flex-direction: column;
  transition: transform 0.62s cubic-bezier(0.22, 1, 0.36, 1);
  will-change: transform;
}
.odo-cell { display: block; height: 1em; line-height: 1em; }

[data-reduce-motion='true'] .odo-col { transition: none; }
@media (prefers-reduced-motion: reduce) { .odo-col { transition: none; } }
</style>
