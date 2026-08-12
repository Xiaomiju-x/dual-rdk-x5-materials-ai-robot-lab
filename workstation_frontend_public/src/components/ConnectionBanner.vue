<script setup lang="ts">
import { computed } from 'vue'
import { useTelemetryStore } from '@/stores/telemetry'

const telemetry = useTelemetryStore()
const show = computed(() => telemetry.state === 'error' || telemetry.state === 'connecting' || telemetry.isStale)
const text = computed(() => {
  if (telemetry.state === 'connecting') return '正在连接工位服务…'
  if (telemetry.state === 'error') return `与工位服务断开,自动重连中 · ${telemetry.lastError}`
  if (telemetry.isStale) return '数据流停滞,可能已暂停或网络抖动'
  return ''
})
const tone = computed(() => (telemetry.state === 'error' ? 'err' : 'warn'))
</script>

<template>
  <transition name="banner">
    <div v-if="show" class="conn-banner" :class="`tone-${tone}`">
      <span class="dot" :class="tone === 'err' ? 'dot-err' : 'dot-warn'"></span>
      <span class="banner-text mono">{{ text }}</span>
    </div>
  </transition>
</template>

<style scoped>
.conn-banner {
  position: fixed; left: 50%; bottom: 18px; transform: translateX(-50%);
  display: flex; align-items: center; gap: 10px;
  padding: 8px 16px; border-radius: 999px; z-index: 55;
  font-size: 0.76rem; backdrop-filter: blur(16px) saturate(160%);
  box-shadow: var(--shadow-elevated);
}
.tone-warn { background: rgba(245, 158, 11, 0.12); color: #b45309; border: 1px solid rgba(245, 158, 11, 0.3); }
.tone-err { background: rgba(239, 68, 68, 0.12); color: #b91c1c; border: 1px solid rgba(239, 68, 68, 0.3); }
.banner-text { letter-spacing: 0.01em; }
.banner-enter-active, .banner-leave-active { transition: opacity 0.3s var(--ease-out-quint), transform 0.3s var(--ease-out-quint); }
.banner-enter-from, .banner-leave-to { opacity: 0; transform: translateX(-50%) translateY(8px); }
</style>
