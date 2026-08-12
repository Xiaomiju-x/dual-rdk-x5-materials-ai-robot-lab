<script setup lang="ts">
/**
 * WaveformTimeline — SpaceX-style time-strip showing a 60-bar
 * amplitude history with a draggable playhead and time labels.
 *
 *   <WaveformTimeline :samples="cpuHistory" label="CPU" :height="64" />
 *
 * samples = array of numbers (last N points, oldest first); auto-normalised.
 */
import { computed, ref } from 'vue'

interface Props {
  samples: number[]
  label?: string
  height?: number
  bars?: number
  accent?: 'blue' | 'teal' | 'emerald' | 'violet' | 'amber' | 'rose'
}
const props = withDefaults(defineProps<Props>(), {
  label: '',
  height: 64,
  bars: 60,
  accent: 'blue',
})

const ACCENT: Record<Required<Props>['accent'], string> = {
  blue: '#2563eb', teal: '#0891b2', emerald: '#059669', violet: '#7c3aed', amber: '#d97706', rose: '#e11d48',
}

const playhead = ref(1)  // 0..1, default at right edge (now)

const downsampled = computed(() => {
  const src = props.samples ?? []
  const n = props.bars
  if (src.length === 0) return new Array(n).fill(0)
  if (src.length <= n) return [...new Array(n - src.length).fill(0), ...src]
  const out: number[] = []
  const step = src.length / n
  for (let i = 0; i < n; i++) {
    const idx = Math.floor(i * step)
    out.push(src[idx])
  }
  return out
})

const maxVal = computed(() => Math.max(0.001, ...downsampled.value, 1))

function barHeight(v: number) {
  const norm = Math.max(0, Math.min(1, v / maxVal.value))
  return Math.max(2, norm * props.height * 0.85)
}

function onPointerDown(e: PointerEvent) {
  const t = e.currentTarget as HTMLElement
  t.setPointerCapture(e.pointerId)
  updateHead(e, t)
}
function onPointerMove(e: PointerEvent) {
  if (e.buttons === 0 && e.pointerType !== 'touch') return
  const t = e.currentTarget as HTMLElement
  updateHead(e, t)
}
function updateHead(e: PointerEvent, target: HTMLElement) {
  const r = target.getBoundingClientRect()
  playhead.value = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width))
}
</script>

<template>
  <div class="wave-wrap" :style="{ height: `${props.height + 24}px` } as any">
    <div v-if="props.label" class="wave-head">
      <span class="section-label">{{ props.label }}</span>
      <span class="wave-time mono">t-{{ Math.round((1 - playhead) * downsampled.length) }}f</span>
    </div>
    <div
      class="wave-strip"
      :style="{ height: `${props.height}px`, '--bar-color': ACCENT[props.accent] } as any"
      @pointerdown="onPointerDown"
      @pointermove="onPointerMove"
    >
      <span
        v-for="(v, i) in downsampled"
        :key="i"
        class="bar"
        :style="{
          height: `${barHeight(v)}px`,
          opacity: i / downsampled.length < playhead ? 1 : 0.32,
        } as any"
      ></span>
      <span class="wave-head-cursor" :style="{ left: `${playhead * 100}%` } as any"></span>
    </div>
  </div>
</template>

<style scoped>
.wave-wrap { display: flex; flex-direction: column; gap: 6px; user-select: none; }
.wave-head { display: flex; justify-content: space-between; align-items: baseline; }
.wave-time { font-size: 0.66rem; color: var(--ink-tertiary); }
.wave-strip {
  position: relative;
  display: flex;
  align-items: flex-end;
  gap: 2px;
  padding: 0 4px;
  background: var(--bg-elevated);
  border-radius: 8px;
  cursor: ew-resize;
  overflow: hidden;
  border: 1px solid var(--line-divider);
}
.bar {
  flex: 1 1 0;
  background: linear-gradient(to top, var(--bar-color), color-mix(in srgb, var(--bar-color) 65%, transparent));
  border-radius: 2px 2px 0 0;
  transition: opacity 0.18s var(--ease-out-quint);
  min-width: 2px;
}
.wave-head-cursor {
  position: absolute; top: 0; bottom: 0;
  width: 2px;
  background: linear-gradient(to bottom, transparent, var(--bar-color) 30%, var(--bar-color) 70%, transparent);
  box-shadow: 0 0 12px var(--bar-color);
  pointer-events: none;
  transition: left 0.1s linear;
}
[data-theme='dark'] .wave-strip { background: rgba(255, 255, 255, 0.03); }
</style>
