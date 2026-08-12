<script setup lang="ts">
import { useToastStore } from '@/stores/toast'

const toasts = useToastStore()

function glyph(tone: string): string {
  if (tone === 'ok')   return '✓'
  if (tone === 'warn') return '⚠'
  if (tone === 'err')  return '✕'
  if (tone === 'info') return 'ℹ'
  return '•'
}
</script>

<template>
  <transition-group name="toast" tag="div" class="toast-stack">
    <div
      v-for="t in toasts.toasts"
      :key="t.id"
      class="toast card-floating"
      :class="`tone-${t.tone}`"
      @click="toasts.dismiss(t.id)"
      role="status"
      aria-live="polite"
    >
      <div class="t-rail"></div>
      <div class="t-glyph">{{ glyph(t.tone) }}</div>
      <div class="t-body">
        <div class="t-title">{{ t.title }}</div>
        <div v-if="t.detail" class="t-detail">{{ t.detail }}</div>
      </div>
      <button class="t-close" @click.stop="toasts.dismiss(t.id)" aria-label="dismiss">×</button>
    </div>
  </transition-group>
</template>

<style scoped>
.toast-stack {
  position: fixed;
  top: 78px;
  right: 18px;
  z-index: 80;
  display: flex;
  flex-direction: column;
  gap: 10px;
  pointer-events: none;
  width: 360px;
  max-width: calc(100vw - 36px);
}
.toast {
  display: grid;
  grid-template-columns: 4px 28px 1fr 24px;
  align-items: flex-start;
  gap: 10px;
  padding: 10px 12px 10px 0;
  pointer-events: auto;
  cursor: pointer;
  transition: transform 0.18s var(--ease-out-quint), box-shadow 0.18s var(--ease-out-quint);
}
.toast:hover { transform: translateX(-2px); box-shadow: var(--shadow-floating); }
.t-rail { width: 4px; height: 100%; min-height: 30px; border-radius: 4px 0 0 4px; }
.tone-ok    .t-rail { background: linear-gradient(180deg, #10b981, #047857); }
.tone-warn  .t-rail { background: linear-gradient(180deg, #f59e0b, #b45309); }
.tone-err   .t-rail { background: linear-gradient(180deg, #ef4444, #b91c1c); }
.tone-info  .t-rail { background: linear-gradient(180deg, #3b82f6, #1d4ed8); }
.tone-idle  .t-rail { background: linear-gradient(180deg, #94a3b8, #475569); }

.t-glyph {
  width: 28px; height: 28px;
  display: flex; align-items: center; justify-content: center;
  border-radius: 8px;
  font-size: 0.95rem; font-weight: 600;
  margin-top: 2px;
}
.tone-ok   .t-glyph { color: #047857; background: rgba(16, 185, 129, 0.12); }
.tone-warn .t-glyph { color: #b45309; background: rgba(245, 158, 11, 0.12); }
.tone-err  .t-glyph { color: #b91c1c; background: rgba(239, 68, 68, 0.12); }
.tone-info .t-glyph { color: #1d4ed8; background: rgba(59, 130, 246, 0.12); }
.tone-idle .t-glyph { color: #475569; background: rgba(148, 163, 184, 0.10); }

.t-body { min-width: 0; }
.t-title { font-size: 0.82rem; font-weight: 600; color: var(--ink-primary); }
.t-detail { font-size: 0.72rem; color: var(--ink-tertiary); margin-top: 3px; line-height: 1.4; }

.t-close {
  background: transparent; border: none;
  width: 24px; height: 24px;
  display: flex; align-items: center; justify-content: center;
  font-size: 1.1rem; color: var(--ink-muted);
  cursor: pointer; border-radius: 4px;
  transition: background 0.15s var(--ease-out-quint), color 0.15s var(--ease-out-quint);
}
.t-close:hover { background: rgba(15, 23, 42, 0.05); color: var(--ink-secondary); }

.toast-enter-from { opacity: 0; transform: translateX(20px) scale(0.95); }
.toast-enter-active { transition: opacity 0.28s var(--ease-out-quint), transform 0.28s var(--ease-out-quint); }
.toast-leave-to { opacity: 0; transform: translateX(20px) scale(0.95); }
.toast-leave-active { transition: opacity 0.18s ease, transform 0.18s ease; }
.toast-move { transition: transform 0.24s var(--ease-out-quint); }
</style>
