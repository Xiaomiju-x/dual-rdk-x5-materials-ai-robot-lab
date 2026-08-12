<script setup lang="ts">
import { computed } from 'vue'
import VChart from 'vue-echarts'
import type { EChartsOption } from 'echarts'
import { ensureEchartsRegistered } from './echartsRegister'
import type { HistorySample } from '@/stores/telemetry'

ensureEchartsRegistered()

interface Props {
  samples: HistorySample[]
  /** target/expected value — drawn as a faint dashed reference line */
  target?: number
  accent?: 'blue' | 'teal' | 'emerald' | 'violet' | 'amber' | 'rose'
  /** y-axis span hint; pass [min, max] for fixed range. omit for auto */
  yRange?: [number, number] | null
}
const props = withDefaults(defineProps<Props>(), { target: undefined, accent: 'blue', yRange: null })

const accentColor = computed(() => {
  return {
    blue: '#2563eb',
    teal: '#0891b2',
    emerald: '#059669',
    violet: '#7c3aed',
    amber: '#d97706',
    rose: '#e11d48',
  }[props.accent]
})

const accentRGBA = computed(() => {
  return {
    blue: '37, 99, 235',
    teal: '8, 145, 178',
    emerald: '5, 150, 105',
    violet: '124, 58, 237',
    amber: '217, 119, 6',
    rose: '225, 29, 72',
  }[props.accent]
})

const option = computed<EChartsOption>(() => {
  const data: [number, number][] = props.samples.map((s) => [s.t, s.v])
  const opt: EChartsOption = {
    animation: false,
    grid: { left: 4, right: 4, top: 6, bottom: 4, containLabel: false },
    xAxis: {
      type: 'time',
      show: false,
      splitLine: { show: false },
    },
    yAxis: {
      type: 'value',
      show: false,
      scale: !props.yRange,
      min: props.yRange ? props.yRange[0] : 'dataMin',
      max: props.yRange ? props.yRange[1] : 'dataMax',
      splitLine: { show: false },
    },
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(255,255,255,0.96)',
      borderWidth: 0,
      textStyle: { fontSize: 11, color: '#0b1220' },
      padding: [4, 8],
      formatter: (params: any) => {
        const p = Array.isArray(params) ? params[0] : params
        return `<span style="font-family: 'JetBrains Mono Variable', monospace">${p.value[1].toFixed(2)}</span>`
      },
      axisPointer: { type: 'line', lineStyle: { color: 'rgba(15,23,42,0.18)', width: 1 } },
    },
    series: [
      {
        type: 'line',
        data,
        showSymbol: false,
        smooth: 0.25,
        lineStyle: { color: accentColor.value, width: 1.6 },
        areaStyle: {
          color: {
            type: 'linear', x: 0, y: 0, x2: 0, y2: 1,
            colorStops: [
              { offset: 0, color: `rgba(${accentRGBA.value}, 0.34)` },
              { offset: 1, color: `rgba(${accentRGBA.value}, 0.02)` },
            ],
          },
        },
        markLine: props.target != null ? {
          symbol: 'none',
          silent: true,
          data: [{ yAxis: props.target }],
          lineStyle: { color: `rgba(${accentRGBA.value}, 0.38)`, type: 'dashed', width: 1 },
          label: { show: false },
        } : undefined,
      },
    ],
  }
  return opt
})
</script>

<template>
  <div class="sparkline">
    <VChart v-if="samples.length > 0" :option="option" autoresize />
    <div v-else class="empty mono">collecting…</div>
  </div>
</template>

<style scoped>
.sparkline { width: 100%; height: 100%; }
.empty {
  display: flex; align-items: center; justify-content: center;
  width: 100%; height: 100%;
  font-size: 0.68rem; color: var(--ink-muted);
}
</style>
