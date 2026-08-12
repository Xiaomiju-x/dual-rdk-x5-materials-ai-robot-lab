<script setup lang="ts">
import { computed } from 'vue'
import VChart from 'vue-echarts'
import type { EChartsOption } from 'echarts'
import { ensureEchartsRegistered } from './echartsRegister'
import type { HistorySample } from '@/stores/telemetry'

ensureEchartsRegistered()

interface SeriesSpec {
  name: string
  samples: HistorySample[]
  accent?: 'blue' | 'teal' | 'emerald' | 'violet' | 'amber' | 'rose'
  target?: number
}

interface Props {
  series: SeriesSpec[]
  yLabel?: string
  height?: string
}
const props = withDefaults(defineProps<Props>(), { yLabel: '', height: '260px' })

const ACCENT = {
  blue: '#2563eb',
  teal: '#0891b2',
  emerald: '#059669',
  violet: '#7c3aed',
  amber: '#d97706',
  rose: '#e11d48',
} as const

const ACCENT_RGBA = {
  blue: '37, 99, 235',
  teal: '8, 145, 178',
  emerald: '5, 150, 105',
  violet: '124, 58, 237',
  amber: '217, 119, 6',
  rose: '225, 29, 72',
} as const

const option = computed<EChartsOption>(() => ({
  animation: false,
  grid: { left: 56, right: 18, top: 22, bottom: 26, containLabel: false },
  legend: {
    show: props.series.length > 1,
    top: 0,
    right: 8,
    textStyle: { fontSize: 11, color: '#475569' },
    icon: 'roundRect',
    itemWidth: 10,
    itemHeight: 6,
  },
  xAxis: {
    type: 'time',
    axisLine: { lineStyle: { color: 'rgba(15,23,42,0.10)' } },
    axisLabel: { fontSize: 10, color: '#94a3b8', formatter: (v: number) => new Date(v).toLocaleTimeString().slice(0, 8) },
    splitLine: { show: false },
  },
  yAxis: {
    type: 'value',
    name: props.yLabel,
    nameTextStyle: { fontSize: 10, color: '#64748b' },
    axisLine: { show: false },
    axisTick: { show: false },
    axisLabel: { fontSize: 10, color: '#94a3b8' },
    splitLine: { lineStyle: { color: 'rgba(15,23,42,0.05)' } },
  },
  tooltip: {
    trigger: 'axis',
    backgroundColor: 'rgba(255,255,255,0.96)',
    borderWidth: 0,
    textStyle: { fontSize: 11, color: '#0b1220' },
    padding: [6, 10],
  },
  dataZoom: [
    { type: 'inside' },
    { type: 'slider', height: 16, bottom: 4, borderColor: 'transparent', backgroundColor: 'rgba(15,23,42,0.03)' },
  ],
  series: props.series.map((s) => ({
    name: s.name,
    type: 'line',
    showSymbol: false,
    smooth: 0.2,
    data: s.samples.map((x) => [x.t, x.v]),
    lineStyle: { color: ACCENT[s.accent ?? 'blue'], width: 1.6 },
    areaStyle: {
      color: {
        type: 'linear', x: 0, y: 0, x2: 0, y2: 1,
        colorStops: [
          { offset: 0, color: `rgba(${ACCENT_RGBA[s.accent ?? 'blue']}, 0.22)` },
          { offset: 1, color: `rgba(${ACCENT_RGBA[s.accent ?? 'blue']}, 0.01)` },
        ],
      },
    },
    markLine: s.target != null ? {
      symbol: 'none',
      silent: true,
      data: [{ yAxis: s.target }],
      lineStyle: { color: `rgba(${ACCENT_RGBA[s.accent ?? 'blue']}, 0.5)`, type: 'dashed', width: 1 },
      label: { show: true, formatter: `target ${s.target}`, position: 'insideEndTop', fontSize: 10, color: ACCENT[s.accent ?? 'blue'] },
    } : undefined,
  })),
}))
</script>

<template>
  <div class="ts-wrap" :style="{ height }">
    <VChart :option="option" autoresize />
  </div>
</template>

<style scoped>
.ts-wrap { width: 100%; }
</style>
