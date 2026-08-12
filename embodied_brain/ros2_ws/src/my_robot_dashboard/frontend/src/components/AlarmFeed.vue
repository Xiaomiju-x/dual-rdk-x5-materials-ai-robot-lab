<script setup lang="ts">
import { computed } from 'vue'
import type { Alarm } from '@/types/telemetry'

interface Props {
  history: Alarm[]
  max?: number
}
const props = withDefaults(defineProps<Props>(), { max: 30 })

const visible = computed(() => props.history.slice(-props.max).reverse())

function fmtTime(ms: number): string {
  const d = new Date(ms)
  const hh = String(d.getHours()).padStart(2, '0')
  const mm = String(d.getMinutes()).padStart(2, '0')
  const ss = String(d.getSeconds()).padStart(2, '0')
  return `${hh}:${mm}:${ss}`
}

function fmtAgo(ms: number): string {
  const diff = Math.max(0, (Date.now() - ms) / 1000)
  if (diff < 60) return `${diff.toFixed(0)}s ago`
  if (diff < 3600) return `${(diff / 60).toFixed(1)}m ago`
  return `${(diff / 3600).toFixed(1)}h ago`
}

function sevGlyph(s: string): string {
  if (s === 'err') return '⛌'
  if (s === 'warn') return '⚠'
  if (s === 'info') return 'ℹ'
  if (s === 'ok') return '✓'
  return '·'
}
</script>

<template>
  <div class="alarm-feed card-elevated">
    <div class="feed-head">
      <span class="section-label">Alarm Stream</span>
      <span class="chip" :class="visible.length ? 'chip-warn' : 'chip-ok'">
        {{ visible.length || 'all clear' }}
      </span>
    </div>
    <div class="feed-body">
      <div v-if="!visible.length" class="feed-empty mono">no alarms in last 24h</div>
      <transition-group v-else name="al" tag="div" class="feed-list">
        <article
          v-for="a in visible"
          :key="a.id"
          class="alarm"
          :class="`sev-${a.severity}`"
        >
          <div class="rail" />
          <div class="al-glyph">{{ sevGlyph(a.severity) }}</div>
          <div class="al-body">
            <div class="al-head">
              <span class="al-title">{{ a.title }}</span>
              <span class="al-time mono">{{ fmtTime(a.at_ms) }}</span>
            </div>
            <div class="al-detail">{{ a.detail }}</div>
            <div class="al-meta">
              <span class="al-src mono">{{ a.source }}</span>
              <span class="al-ago mono">{{ fmtAgo(a.at_ms) }}</span>
            </div>
          </div>
        </article>
      </transition-group>
    </div>
  </div>
</template>

<style scoped>
.alarm-feed { padding: 14px 16px; display: flex; flex-direction: column; gap: 10px; min-height: 0; }
.feed-head { display: flex; align-items: center; justify-content: space-between; }
.feed-body { flex: 1; min-height: 0; overflow-y: auto; padding-right: 4px; }
.feed-empty {
  height: 100%;
  display: flex; align-items: center; justify-content: center;
  color: var(--ink-muted); font-size: 0.78rem;
}
.feed-list { display: flex; flex-direction: column; gap: 6px; }

.alarm {
  display: grid;
  grid-template-columns: 4px 28px 1fr;
  gap: 10px;
  padding: 8px 10px 8px 0;
  background: rgba(248, 250, 252, 0.7);
  border: 1px solid var(--line-divider);
  border-radius: 10px;
  align-items: flex-start;
}
.rail { width: 4px; align-self: stretch; border-radius: 4px 0 0 4px; }
.sev-err   .rail { background: linear-gradient(180deg, #ef4444, #b91c1c); }
.sev-warn  .rail { background: linear-gradient(180deg, #f59e0b, #b45309); }
.sev-info  .rail { background: linear-gradient(180deg, #3b82f6, #1d4ed8); }
.sev-ok    .rail { background: linear-gradient(180deg, #10b981, #047857); }
.sev-idle  .rail { background: linear-gradient(180deg, #94a3b8, #475569); }

.al-glyph {
  width: 28px; height: 28px;
  display: flex; align-items: center; justify-content: center;
  border-radius: 8px;
  font-size: 1rem; font-weight: 600;
}
.sev-err  .al-glyph { color: #b91c1c; background: rgba(239, 68, 68, 0.10); }
.sev-warn .al-glyph { color: #b45309; background: rgba(245, 158, 11, 0.10); }
.sev-info .al-glyph { color: #1d4ed8; background: rgba(59, 130, 246, 0.10); }
.sev-ok   .al-glyph { color: #047857; background: rgba(16, 185, 129, 0.10); }
.sev-idle .al-glyph { color: #475569; background: rgba(148, 163, 184, 0.10); }

.al-body { display: flex; flex-direction: column; gap: 2px; min-width: 0; }
.al-head { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.al-title { font-size: 0.82rem; font-weight: 600; color: var(--ink-primary); }
.al-time { font-size: 0.7rem; color: var(--ink-muted); }
.al-detail { font-size: 0.74rem; color: var(--ink-secondary); }
.al-meta { display: flex; gap: 10px; font-size: 0.66rem; color: var(--ink-muted); margin-top: 2px; }
.al-ago { margin-left: auto; }

.al-enter-from { opacity: 0; transform: translateX(-6px); }
.al-enter-active { transition: opacity 0.28s var(--ease-out-quint), transform 0.28s var(--ease-out-quint); }
.al-leave-to { opacity: 0; transform: scale(0.96); }
.al-leave-active { transition: all 0.18s ease; position: absolute; }
.al-move { transition: transform 0.28s var(--ease-out-quint); }

.feed-body::-webkit-scrollbar { width: 6px; }
.feed-body::-webkit-scrollbar-thumb { background: rgba(15, 23, 42, 0.12); border-radius: 3px; }
</style>
