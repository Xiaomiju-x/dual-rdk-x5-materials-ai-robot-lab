<script setup lang="ts">
import { computed } from 'vue'
import VChart from 'vue-echarts'
import type {
  CustomSeriesRenderItemAPI,
  CustomSeriesRenderItemParams,
  CustomSeriesRenderItemReturn,
  EChartsOption,
  TooltipComponentFormatterCallbackParams,
} from 'echarts'
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
type CustomGroupReturn = Extract<NonNullable<CustomSeriesRenderItemReturn>, { type: 'group' }>
type CustomElement = CustomGroupReturn['children'][number]

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
      formatter: (params: TooltipComponentFormatterCallbackParams) => {
        const p = Array.isArray(params) ? params[0] : params
        const v = Array.isArray(p?.value) ? p.value : []
        const startMs = Number(v[1] ?? 0)
        const endMs = Number(v[2] ?? 0)
        const start = new Date(startMs).toLocaleTimeString()
        const end = new Date(endMs).toLocaleTimeString()
        const dur = Math.max(0, (endMs - startMs) / 1000)
        const status = String(v[3] ?? '')
        const detail = String(v[4] ?? '')
        const label = String(v[5] ?? '')
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
        renderItem: (
          _params: CustomSeriesRenderItemParams,
          api: CustomSeriesRenderItemAPI,
        ): CustomSeriesRenderItemReturn => {
          const trackIdx = Number(api.value(0))
          const startCoord = api.coord([api.value(1), trackIdx])
          const endCoord = api.coord([api.value(2), trackIdx])
          const size = api.size?.([0, 1])
          const height = (Array.isArray(size) ? size[1] : Number(size ?? 0)) * 0.5
          const x = startCoord[0]
          const w = Math.max(2, endCoord[0] - startCoord[0])
          const y = startCoord[1] - height / 2
          const status = String(api.value(3) ?? '')
          const isWarn = status === 'warn' || status === 'err'
          const children: CustomElement[] = [
            {
              type: 'rect',
              shape: { x, y, width: w, height, r: 4 },
              style: {
                fill: String(api.visual('color') ?? '#94a3b8'),
                opacity: 0.92,
              },
            },
          ]
          if (isWarn) {
            children.push({
              type: 'rect',
              shape: { x, y, width: 3, height, r: [2, 0, 0, 2] },
              style: { fill: status === 'err' ? '#b91c1c' : '#b45309' },
            })
          }
          children.push({
            type: 'text',
            x: x + 8,
            y: y + height / 2,
            style: {
              text: w > 60 ? String(api.value(5)).slice(0, Math.floor(w / 7)) : '',
              verticalAlign: 'middle',
              align: 'left',
              fontSize: 10,
              fontFamily: 'Inter Variable, sans-serif',
              fontWeight: 600,
              fill: '#ffffff',
            },
          })
          return {
            type: 'group',
            children,
          }
        },
        encode: { x: [1, 2], y: 0, tooltip: [0, 1, 2, 3, 4, 5] },
        data,
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
