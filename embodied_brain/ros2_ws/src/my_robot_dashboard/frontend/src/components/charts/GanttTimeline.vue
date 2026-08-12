<script setup lang="ts">
import { computed } from 'vue'
import VChart from 'vue-echarts'
import type { EChartsOption } from 'echarts'
import { ensureEchartsRegistered } from './echartsRegister'
import type { TimelineEvent } from '@/types/telemetry'

ensureEchartsRegistered()

interface Props {
  events: TimelineEvent[]
  height?: string
}
const props = withDefaults(defineProps<Props>(), { height: '320px' })

const TRACKS = [
  { id: 'ai_brain',   label: 'AI Brain',   color: '#7c3aed' },
  { id: 'dispatch',   label: 'Dispatch',   color: '#2563eb' },
  { id: 'nav',        label: 'Navigation', color: '#0891b2' },
  { id: 'perception', label: 'Perception', color: '#059669' },
  { id: 'system',     label: 'System',     color: '#d97706' },
] as const

type TrackId = (typeof TRACKS)[number]['id']

const STATUS_TINT: Record<string, number> = {
  ok: 1.0, info: 0.95, warn: 0.85, err: 0.75, idle: 0.65,
}

const option = computed<EChartsOption>(() => {
  const trackIdx = (id: TrackId) => TRACKS.findIndex((t) => t.id === id)
  const data = props.events.map((e) => ({
    name: e.label,
    value: [trackIdx(e.track as TrackId), e.start_ms, e.end_ms, e.status, e.detail, e.label],
    itemStyle: {
      color: TRACKS.find((t) => t.id === e.track)?.color ?? '#94a3b8',
      opacity: STATUS_TINT[e.status] ?? 0.85,
    },
  }))

  const tMin = props.events.length ? Math.min(...props.events.map((e) => e.start_ms)) : Date.now() - 8 * 60 * 1000
  const tMax = props.events.length ? Math.max(...props.events.map((e) => e.end_ms)) : Date.now()
  const span = tMax - tMin
  const pad = Math.max(15_000, span * 0.04)

  return {
    animation: false,
    grid: { left: 110, right: 24, top: 16, bottom: 38 },
    tooltip: {
      backgroundColor: 'rgba(255,255,255,0.96)',
      borderWidth: 0,
      textStyle: { fontSize: 11, color: '#0b1220' },
      padding: [8, 12],
      formatter: (p: any) => {
        const v = p.value
        const start = new Date(v[1]).toLocaleTimeString()
        const end = new Date(v[2]).toLocaleTimeString()
        const dur = Math.max(0, (v[2] - v[1]) / 1000)
        const status = v[3]
        const detail = v[4]
        const label = v[5]
        return `
          <div style="font-weight:600;color:#0b1220;margin-bottom:4px">${label}</div>
          <div style="color:#475569;font-size:10px;font-family:JetBrains Mono Variable, monospace">
            ${start} → ${end} · ${dur.toFixed(1)} s<br/>
            status <span style="color:${status === 'err' ? '#b91c1c' : status === 'warn' ? '#b45309' : '#059669'}">${status}</span>
            ${detail ? `<br/>${detail}` : ''}
          </div>
        `
      },
    },
    xAxis: {
      type: 'time',
      min: tMin - pad,
      max: tMax + pad,
      axisLine: { lineStyle: { color: 'rgba(15,23,42,0.10)' } },
      axisLabel: {
        fontSize: 10,
        color: '#94a3b8',
        formatter: (v: number) => new Date(v).toLocaleTimeString().slice(0, 5),
      },
      splitLine: { lineStyle: { color: 'rgba(15,23,42,0.04)' } },
    },
    yAxis: {
      type: 'category',
      data: TRACKS.map((t) => t.label),
      inverse: true,
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: { fontSize: 11, color: '#475569', fontWeight: 500, margin: 14 },
      splitArea: {
        show: true,
        areaStyle: { color: ['rgba(248,250,252,0.0)', 'rgba(248,250,252,0.6)'] },
      },
    },
    dataZoom: [
      { type: 'inside', xAxisIndex: 0 },
      { type: 'slider', xAxisIndex: 0, height: 16, bottom: 6, borderColor: 'transparent', backgroundColor: 'rgba(15,23,42,0.03)' },
    ],
    series: [
      {
        type: 'custom',
        renderItem: (_params: any, api: any): any => {
          const trackIdx = api.value(0)
          const startCoord = api.coord([api.value(1), trackIdx])
          const endCoord = api.coord([api.value(2), trackIdx])
          const height = api.size([0, 1])[1] * 0.5
          const x = startCoord[0]
          const w = Math.max(2, endCoord[0] - startCoord[0])
          const y = startCoord[1] - height / 2
          const status = api.value(3)
          const isWarn = status === 'warn' || status === 'err'
          return {
            type: 'group',
            children: [
              {
                type: 'rect',
                shape: { x, y, width: w, height, r: 4 },
                style: {
                  fill: api.visual('color'),
                  opacity: 0.92,
                },
              },
              ...(isWarn ? [{
                type: 'rect',
                shape: { x, y, width: 3, height, r: [2, 0, 0, 2] },
                style: { fill: status === 'err' ? '#b91c1c' : '#b45309' },
              }] : []),
              {
                type: 'text',
                position: [x + 8, y + height / 2],
                style: {
                  text: w > 60 ? String(api.value(5)).slice(0, Math.floor(w / 7)) : '',
                  textVerticalAlign: 'middle',
                  textAlign: 'left',
                  fontSize: 10,
                  fontFamily: 'Inter Variable, sans-serif',
                  fontWeight: 600,
                  fill: '#ffffff',
                },
              },
            ],
          }
        },
        encode: { x: [1, 2], y: 0, tooltip: [0, 1, 2, 3, 4, 5] },
        data: data as any,
      },
    ],
  }
})
</script>

<template>
  <div class="gantt-wrap" :style="{ height }">
    <VChart :option="option" autoresize />
  </div>
</template>

<style scoped>
.gantt-wrap { width: 100%; }
</style>
