<script setup lang="ts">
import { computed } from 'vue'
import { useTelemetryStore } from '@/stores/telemetry'

const telemetry = useTelemetryStore()

const state = computed(() => {
  if (telemetry.state === 'open' && !telemetry.isStale) return null
  if (telemetry.state === 'open' && telemetry.isStale) {
    return { tone: 'warn' as const, glyph: '⚠', title: 'Stream stalled', detail: '最近 1.5 秒没有帧, 后端 mock_loop 可能在 GC 或网络抖动' }
  }
  if (telemetry.state === 'connecting') {
    return { tone: 'info' as const, glyph: '↻', title: 'Connecting…', detail: '正在握手 /ws/telemetry' }
  }
  if (telemetry.state === 'closed') {
    return { tone: 'warn' as const, glyph: '↻', title: 'Reconnecting', detail: '指数退避中, 自动重连' }
  }
  if (telemetry.state === 'error') {
    return { tone: 'err' as const, glyph: '⛌', title: 'Connection error', detail: telemetry.lastError || 'websocket failed; check backend on :8890' }
  }
  if (telemetry.state === 'idle') {
    return { tone: 'idle' as const, glyph: '·', title: 'Idle', detail: 'WebSocket 未启动' }
  }
  return null
})

function retry() {
  telemetry.disconnect()
  setTimeout(() => telemetry.connect(), 60)
}
</script>

<template>
  <transition name="banner">
    <div v-if="state" class="banner" :class="`tone-${state.tone}`">
      <span class="b-glyph">{{ state.glyph }}</span>
      <span class="b-title">{{ state.title }}</span>
      <span class="b-detail">{{ state.detail }}</span>
      <button v-if="state.tone !== 'info'" class="b-retry mono" @click="retry">retry</button>
    </div>
  </transition>
</template>

<style scoped>
.banner {
  position: fixed;
  top: 64px;   /* below topbar */
  left: 50%;
  transform: translateX(-50%);
  z-index: 70;
  display: inline-flex;
  align-items: center;
  gap: 10px;
  padding: 8px 14px;
  border-radius: 999px;
  font-size: 0.78rem;
  font-weight: 500;
  box-shadow: var(--shadow-card);
  border: 1px solid;
  backdrop-filter: blur(20px) saturate(160%);
  -webkit-backdrop-filter: blur(20px) saturate(160%);
  max-width: calc(100vw - 32px);
}
.tone-warn { background: rgba(254, 243, 199, 0.92); color: #92400e; border-color: rgba(245, 158, 11, 0.4); }
.tone-err  { background: rgba(254, 226, 226, 0.92); color: #991b1b; border-color: rgba(239, 68, 68, 0.4); }
.tone-info { background: rgba(219, 234, 254, 0.92); color: #1e40af; border-color: rgba(59, 130, 246, 0.4); }
.tone-idle { background: rgba(241, 245, 249, 0.92); color: #334155; border-color: rgba(148, 163, 184, 0.4); }

[data-theme='dark'] .tone-warn { background: rgba(120, 53, 15, 0.6); color: #fde68a; border-color: rgba(245, 158, 11, 0.5); }
[data-theme='dark'] .tone-err  { background: rgba(127, 29, 29, 0.6); color: #fecaca; border-color: rgba(239, 68, 68, 0.6); }
[data-theme='dark'] .tone-info { background: rgba(30, 58, 138, 0.6); color: #dbeafe; border-color: rgba(59, 130, 246, 0.6); }
[data-theme='dark'] .tone-idle { background: rgba(30, 41, 59, 0.6); color: #cbd5e1; border-color: rgba(148, 163, 184, 0.4); }

.b-glyph { font-size: 0.92rem; font-weight: 600; }
.b-title { font-weight: 600; }
.b-detail { font-size: 0.7rem; opacity: 0.85; }
.b-retry {
  border: 1px solid currentColor;
  background: transparent;
  color: inherit;
  padding: 3px 9px;
  border-radius: 999px;
  font-size: 0.66rem;
  cursor: pointer;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  transition: background 0.15s var(--ease-out-quint);
}
.b-retry:hover { background: rgba(255, 255, 255, 0.5); }
[data-theme='dark'] .b-retry:hover { background: rgba(255, 255, 255, 0.1); }

.banner-enter-from { opacity: 0; transform: translateX(-50%) translateY(-10px); }
.banner-leave-to { opacity: 0; transform: translateX(-50%) translateY(-6px); }
.banner-enter-active, .banner-leave-active { transition: opacity 0.24s var(--ease-out-quint), transform 0.24s var(--ease-out-quint); }
</style>
