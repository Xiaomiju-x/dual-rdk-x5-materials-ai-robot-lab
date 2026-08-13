<script setup lang="ts">
import { computed } from 'vue'
import type { AiLinkState, TimelineEvent } from '@/types/telemetry'

interface Props {
  link: AiLinkState | null
  recentEvents: TimelineEvent[]
}
const props = defineProps<Props>()

const dispatches = computed(() =>
  props.recentEvents
    .filter((e) => e.track === 'ai_brain')
    .slice()
    .sort((a, b) => b.start_ms - a.start_ms)
    .slice(0, 6),
)

function fmtTime(ms: number): string {
  const d = new Date(ms)
  return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}:${String(d.getSeconds()).padStart(2, '0')}`
}

const rttBand = computed(() => {
  if (!props.link) return 'idle'
  if (props.link.rtt_ms < 50) return 'ok'
  if (props.link.rtt_ms < 120) return 'warn'
  return 'err'
})
</script>

<template>
  <div class="ai-link card-elevated">
    <div class="link-head">
      <div class="link-head-left">
        <div class="brain-mark">
          <svg viewBox="0 0 48 48" width="32" height="32" aria-hidden="true">
            <defs>
              <linearGradient id="brainGrad" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stop-color="#2563eb" />
                <stop offset="60%" stop-color="#0891b2" />
                <stop offset="100%" stop-color="#7c3aed" />
              </linearGradient>
            </defs>
            <path
d="M24 6 C 14 6 8 14 10 22 C 6 26 8 34 16 36 C 18 42 28 42 30 36 C 38 34 40 26 36 22 C 38 14 32 6 24 6 Z"
                  fill="none" stroke="url(#brainGrad)" stroke-width="2" />
            <circle cx="18" cy="20" r="2" fill="#2563eb" />
            <circle cx="30" cy="20" r="2" fill="#7c3aed" />
            <circle cx="24" cy="30" r="2" fill="#0891b2" />
            <path d="M18 20 L 24 30 L 30 20" fill="none" stroke="#94a3b8" stroke-width="0.6" />
          </svg>
        </div>
        <div>
          <div class="link-title">AI Brain Bridge</div>
          <div class="link-endpoint mono">{{ link?.endpoint ?? '—' }}</div>
        </div>
      </div>
      <span class="chip" :class="link?.online ? 'chip-ok' : 'chip-err'">
        <span class="dot" :class="link?.online ? 'dot-ok' : 'dot-err'"></span>
        {{ link?.online ? 'online' : 'offline' }}
      </span>
    </div>

    <div class="link-stats">
      <div class="stat">
        <div class="stat-label">RTT</div>
        <div class="stat-val mono" :class="`tone-${rttBand}`">
          {{ link ? `${link.rtt_ms.toFixed(0)} ms` : '—' }}
        </div>
      </div>
      <div class="stat">
        <div class="stat-label">Dispatches · 24h</div>
        <div class="stat-val mono">{{ link?.dispatches_24h ?? 0 }}</div>
      </div>
      <div class="stat">
        <div class="stat-label">Last seen</div>
        <div class="stat-val mono">{{ link ? fmtTime(link.last_seen_ms) : '—' }}</div>
      </div>
    </div>

    <div class="recent-head">
      <span class="section-label">Recent AI Dispatches</span>
      <span class="mono link-last">{{ link?.last_dispatch_label ?? '—' }}</span>
    </div>
    <ul class="dispatch-list">
      <li v-for="d in dispatches" :key="d.id" class="dispatch">
        <span class="d-time mono">{{ fmtTime(d.start_ms) }}</span>
        <span class="d-glyph">⇢</span>
        <span class="d-label">{{ d.label }}</span>
        <span class="d-detail mono">{{ d.detail }}</span>
        <span class="chip" :class="`chip-${d.status}`">{{ d.status }}</span>
      </li>
      <li v-if="!dispatches.length" class="dispatch-empty mono">no recent dispatches</li>
    </ul>
  </div>
</template>

<style scoped>
.ai-link { padding: 14px 16px; display: flex; flex-direction: column; gap: 14px; }
.link-head { display: flex; align-items: center; justify-content: space-between; }
.link-head-left { display: flex; gap: 12px; align-items: center; min-width: 0; }
.brain-mark {
  width: 40px; height: 40px;
  border-radius: 12px;
  display: flex; align-items: center; justify-content: center;
  background: radial-gradient(circle at 30% 20%, rgba(37, 99, 235, 0.10), rgba(124, 58, 237, 0.04) 70%);
  border: 1px solid rgba(37, 99, 235, 0.15);
  animation: breath 3.5s ease-in-out infinite;
}
.link-title { font-size: 0.92rem; font-weight: 600; color: var(--ink-primary); }
.link-endpoint { font-size: 0.68rem; color: var(--ink-muted); margin-top: 2px; }

.link-stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
.stat {
  padding: 8px 10px;
  background: var(--bg-elevated);
  border: 1px solid var(--line-divider);
  border-radius: 10px;
  display: flex; flex-direction: column; gap: 4px;
}
.stat-label { font-size: 0.62rem; color: var(--ink-muted); text-transform: uppercase; letter-spacing: 0.05em; }
.stat-val { font-size: 1.05rem; font-weight: 600; color: var(--ink-primary); }
.tone-ok   { color: #047857; }
.tone-warn { color: #b45309; }
.tone-err  { color: #b91c1c; }
.tone-idle { color: #475569; }

.recent-head { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.link-last { font-size: 0.68rem; color: var(--ink-tertiary); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 60%; }

.dispatch-list { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 4px; }
.dispatch {
  display: grid;
  grid-template-columns: 70px 16px 1fr 1fr 50px;
  align-items: center;
  gap: 8px;
  padding: 5px 8px;
  border-radius: 6px;
  font-size: 0.74rem;
  background: rgba(248, 250, 252, 0.6);
  border: 1px solid var(--line-hairline);
}
.d-time { color: var(--ink-muted); font-size: 0.68rem; }
.d-glyph { color: var(--accent-violet); font-size: 0.85rem; text-align: center; }
.d-label { color: var(--ink-primary); font-weight: 500; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.d-detail { color: var(--ink-tertiary); font-size: 0.68rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.dispatch-empty { color: var(--ink-muted); font-size: 0.74rem; padding: 8px; text-align: center; }
</style>
