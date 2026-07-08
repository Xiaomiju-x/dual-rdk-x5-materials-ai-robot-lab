<script setup lang="ts">
// SyncBars — 6 horizontal twin-track bars, one per joint. Each row shows
// arm01 (amber) and arm02 (blue) on a [min, max] axis with their delta as a
// pulse halo. Useful to spot which joint is lagging during cooperative motion.
import { computed } from 'vue'

interface Props {
  a01: number[]
  a02: number[]
  limits?: [number, number][]   // per joint, [min, max]
  names?: string[]
}
const props = withDefaults(defineProps<Props>(), {
  limits: () => [[-168,168],[-135,135],[-150,150],[-145,145],[-165,165],[-180,180]],
  names: () => ['J1','J2','J3','J4','J5','J6'],
})

const rows = computed(() => {
  return [0,1,2,3,4,5].map((i) => {
    const [lo, hi] = props.limits[i]
    const span = hi - lo
    const a = props.a01[i] ?? 0
    const b = props.a02[i] ?? 0
    const pA = ((a - lo) / span) * 100
    const pB = ((b - lo) / span) * 100
    const diff = a - b
    const absDiff = Math.abs(diff)
    return { i, name: props.names[i], lo, hi, a, b, pA, pB, diff, absDiff }
  })
})
const maxDiff = computed(() => {
  let m = 0
  for (const r of rows.value) if (r.absDiff > m) m = r.absDiff
  return m
})
</script>

<template>
  <div class="sync-bars">
    <div class="sb-head">
      <span class="section-label">关节同步度 · 双臂逐关节对照</span>
      <span class="mono sb-max">max Δ {{ maxDiff.toFixed(1) }}°</span>
    </div>
    <div class="sb-row" v-for="r in rows" :key="r.i">
      <div class="sb-name mono">{{ r.name }}</div>
      <div class="sb-track">
        <div class="sb-axis"></div>
        <div class="sb-marker sb-a01" :style="{ left: r.pA + '%' }" :title="`arm01: ${r.a.toFixed(1)}°`">
          <span class="sb-tick"></span><span class="sb-label">{{ r.a.toFixed(0) }}</span>
        </div>
        <div class="sb-marker sb-a02" :style="{ left: r.pB + '%' }" :title="`arm02: ${r.b.toFixed(1)}°`">
          <span class="sb-tick"></span><span class="sb-label">{{ r.b.toFixed(0) }}</span>
        </div>
        <div class="sb-bridge"
             :style="{
               left: Math.min(r.pA, r.pB) + '%',
               width: Math.abs(r.pA - r.pB) + '%',
               background: `linear-gradient(90deg, rgba(217,119,6,${0.18 + r.absDiff*0.005}) 0%, rgba(37,99,235,${0.18 + r.absDiff*0.005}) 100%)`
             }"></div>
      </div>
      <div class="sb-delta mono" :class="{ warn: r.absDiff > 15, err: r.absDiff > 45 }">
        Δ{{ r.diff >= 0 ? '+' : '' }}{{ r.diff.toFixed(1) }}°
      </div>
    </div>
  </div>
</template>

<style scoped>
.sync-bars { display: flex; flex-direction: column; gap: 6px; }
.sb-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }
.sb-max { font-size: 0.7rem; color: var(--ink-muted); }
.sb-row { display: grid; grid-template-columns: 40px 1fr 70px; gap: 10px; align-items: center; padding: 6px 0; }
.sb-name { font-size: 0.74rem; font-weight: 700; color: var(--ink-secondary); }
.sb-track { position: relative; height: 28px; border-radius: 8px; background: var(--bg-elevated); }
.sb-axis { position: absolute; top: 50%; left: 8px; right: 8px; height: 1px; background: var(--line-divider); transform: translateY(-50%); }
.sb-bridge { position: absolute; top: 50%; height: 4px; border-radius: 2px; transform: translateY(-50%); transition: all .3s; }
.sb-marker { position: absolute; top: 50%; transform: translate(-50%, -50%); display: flex; flex-direction: column; align-items: center; gap: 1px; pointer-events: auto; cursor: help; transition: left .25s; }
.sb-tick { width: 10px; height: 18px; border-radius: 3px; box-shadow: 0 1px 4px rgba(15,23,42,.25); }
.sb-label { font-size: 0.6rem; color: var(--ink-tertiary); font-family: 'JetBrains Mono Variable', monospace; line-height: 1; }
.sb-a01 .sb-tick { background: var(--accent-amber); }
.sb-a02 .sb-tick { background: var(--accent-blue); }
.sb-a01 .sb-label { color: var(--accent-amber); }
.sb-a02 .sb-label { color: var(--accent-blue); }
.sb-delta { font-size: 0.74rem; font-weight: 700; color: var(--ink-secondary); text-align: right; }
.sb-delta.warn { color: var(--accent-amber); }
.sb-delta.err  { color: var(--accent-rose); }
</style>
