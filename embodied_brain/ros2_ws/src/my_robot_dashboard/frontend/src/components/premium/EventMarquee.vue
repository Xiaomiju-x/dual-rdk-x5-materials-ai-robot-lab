<script setup lang="ts">
/**
 * EventMarquee — horizontal scrolling event ticker (SpaceX / Vercel /
 * NASA stripe). Pauses on hover. Always duplicates the items so the
 * loop is seamless regardless of count.
 *
 *   <EventMarquee :items="latestEvents" />
 *
 * item = { id, label, tone?, glyph?, ts? }
 */
import { computed } from 'vue'

interface MarqueeItem {
  id: string
  label: string
  tone?: 'ok' | 'warn' | 'err' | 'info' | 'idle'
  glyph?: string
  ts?: number  // ms epoch
}
interface Props {
  items: MarqueeItem[]
  /** seconds for one cycle */
  speed?: number
  /** marquee height in px */
  height?: number
}
const props = withDefaults(defineProps<Props>(), { speed: 60, height: 32 })

const tone = (t?: MarqueeItem['tone']) => t ?? 'idle'
const doubled = computed(() => [...props.items, ...props.items])

function fmtTime(ts?: number) {
  if (!ts) return ''
  const d = new Date(ts)
  const pad = (n: number) => n.toString().padStart(2, '0')
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`
}
</script>

<template>
  <div
    class="marq-outer"
    :style="{ height: `${props.height}px`, '--m-speed': `${props.speed}s` } as any"
  >
    <div class="marq-rail">
      <span v-for="(it, i) in doubled" :key="`${it.id}-${i}`" class="marq-item">
        <span v-if="it.glyph" class="marq-glyph" :class="`tone-${tone(it.tone)}`">{{ it.glyph }}</span>
        <span v-else class="dot" :class="`dot-${tone(it.tone) === 'idle' ? 'idle' : tone(it.tone)}`"></span>
        <span v-if="it.ts" class="marq-ts mono">{{ fmtTime(it.ts) }}</span>
        <span class="marq-label">{{ it.label }}</span>
        <span class="marq-divider">·</span>
      </span>
    </div>
    <span class="marq-fade marq-fade-l"></span>
    <span class="marq-fade marq-fade-r"></span>
  </div>
</template>

<style scoped>
.marq-outer {
  position: relative;
  overflow: hidden;
  width: 100%;
  background: var(--bg-glass);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border-top: 1px solid var(--line-divider);
  border-bottom: 1px solid var(--line-divider);
}
.marq-rail {
  display: inline-flex;
  align-items: center;
  gap: 16px;
  height: 100%;
  white-space: nowrap;
  padding: 0 16px;
  animation: marq var(--m-speed) linear infinite;
  will-change: transform;
}
.marq-outer:hover .marq-rail { animation-play-state: paused; }
.marq-item { display: inline-flex; align-items: center; gap: 8px; font-size: 0.74rem; color: var(--ink-tertiary); }
.marq-glyph { font-size: 0.95rem; }
.marq-ts { font-size: 0.66rem; color: var(--ink-muted); }
.marq-label { color: var(--ink-secondary); }
.marq-divider { color: var(--ink-disabled); padding: 0 4px; }

.tone-ok { color: var(--status-ok); }
.tone-warn { color: var(--status-warn); }
.tone-err { color: var(--status-err); }
.tone-info { color: var(--status-info); }
.tone-idle { color: var(--status-idle); }

.marq-fade {
  position: absolute; top: 0; bottom: 0; width: 60px;
  pointer-events: none;
}
.marq-fade-l { left: 0; background: linear-gradient(to right, var(--bg-base), transparent); }
.marq-fade-r { right: 0; background: linear-gradient(to left, var(--bg-base), transparent); }
[data-theme='dark'] .marq-fade-l { background: linear-gradient(to right, rgba(10, 13, 20, 0.95), transparent); }
[data-theme='dark'] .marq-fade-r { background: linear-gradient(to left,  rgba(10, 13, 20, 0.95), transparent); }

@keyframes marq {
  from { transform: translate3d(0, 0, 0); }
  to   { transform: translate3d(-50%, 0, 0); }
}
@media (prefers-reduced-motion: reduce) { .marq-rail { animation: none; } }
[data-reduce-motion='true'] .marq-rail { animation: none; }
</style>
