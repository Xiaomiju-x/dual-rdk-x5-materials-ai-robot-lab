<script setup lang="ts">
/**
 * NotificationCenter — bell button + slide-down panel listing all past
 * toasts + alarms. Read/unread tracking via localStorage.
 *
 * Bell button lives in topbar. Panel is teleported.
 */
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useTelemetryStore } from '@/stores/telemetry'
import { useUiStore } from '@/stores/ui'

const telemetry = useTelemetryStore()
const ui = useUiStore()
const anchor = ref<HTMLElement | null>(null)
const panel = ref<HTMLElement | null>(null)

const READ_KEY = 'navcockpit.notifs.read.v1'
const readIds = ref<Set<string>>(new Set())

function loadRead() {
  try {
    const raw = localStorage.getItem(READ_KEY)
    if (raw) readIds.value = new Set(JSON.parse(raw))
  } catch { /* noop */ }
}
function persistRead() {
  try { localStorage.setItem(READ_KEY, JSON.stringify([...readIds.value])) } catch { /* noop */ }
}
loadRead()

const allAlarms = computed(() => telemetry.alarmHistory.slice().reverse())
const unreadCount = computed(() => allAlarms.value.filter((a) => !readIds.value.has(a.id)).length)

function markAllRead() {
  for (const a of telemetry.alarmHistory) readIds.value.add(a.id)
  persistRead()
}
function clearOne(id: string) {
  readIds.value.add(id)
  persistRead()
}

function onDoc(evt: MouseEvent) {
  if (!ui.notifCenterOpen) return
  const t = evt.target as Node
  if (anchor.value?.contains(t) || panel.value?.contains(t)) return
  ui.notifCenterOpen = false
}
onMounted(() => document.addEventListener('mousedown', onDoc))
onBeforeUnmount(() => document.removeEventListener('mousedown', onDoc))

// auto-mark visible items as read on open
watch(() => ui.notifCenterOpen, (v) => {
  if (v) {
    setTimeout(() => {
      for (const a of allAlarms.value.slice(0, 8)) readIds.value.add(a.id)
      persistRead()
    }, 1200)
  }
})

function fmt(ts: number) {
  const d = new Date(ts)
  const now = Date.now()
  const diff = Math.floor((now - ts) / 1000)
  if (diff < 60) return `${diff}s ago`
  if (diff < 3600) return `${Math.floor(diff/60)}m ago`
  if (diff < 86400) return `${Math.floor(diff/3600)}h ago`
  const pad = (n: number) => n.toString().padStart(2, '0')
  return `${d.getMonth()+1}/${d.getDate()} ${pad(d.getHours())}:${pad(d.getMinutes())}`
}
</script>

<template>
  <div class="nc-wrap">
    <button
      ref="anchor"
      class="nc-btn"
      :class="{ active: ui.notifCenterOpen, hasUnread: unreadCount > 0 }"
      :title="`Notifications (${unreadCount} unread)`"
      @click="ui.toggleNotifCenter"
    >
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
        <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"></path>
        <path d="M13.73 21a2 2 0 0 1-3.46 0"></path>
      </svg>
      <span v-if="unreadCount > 0" class="nc-badge">{{ unreadCount > 99 ? '99+' : unreadCount }}</span>
    </button>
    <Teleport to="body">
      <Transition name="nc">
        <div v-if="ui.notifCenterOpen" ref="panel" class="nc-panel glass-strong">
          <header class="nc-head">
            <div class="nc-head-l">
              <span class="section-label">Notifications</span>
              <span class="chip" :class="unreadCount ? 'chip-warn' : 'chip-idle'">{{ unreadCount }} unread</span>
            </div>
            <button v-if="unreadCount" class="nc-mark" @click="markAllRead">mark all read</button>
          </header>
          <div class="nc-list">
            <div v-if="!allAlarms.length" class="nc-empty mono">all clear</div>
            <div
              v-for="a in allAlarms.slice(0, 50)"
              :key="a.id"
              class="nc-item"
              :class="[`tone-${a.severity}`, { unread: !readIds.has(a.id) }]"
            >
              <span class="nc-rail" :class="`rail-${a.severity}`"></span>
              <div class="nc-body">
                <div class="nc-row1">
                  <span class="nc-title">{{ a.title }}</span>
                  <span class="mono nc-ago">{{ fmt(a.at_ms) }}</span>
                </div>
                <div class="nc-detail">{{ a.detail }}</div>
                <div class="nc-src mono">{{ a.source }}</div>
              </div>
              <button v-if="!readIds.has(a.id)" class="nc-x" aria-label="dismiss" @click="clearOne(a.id)">✓</button>
            </div>
          </div>
        </div>
      </Transition>
    </Teleport>
  </div>
</template>

<style scoped>
.nc-wrap { position: relative; }
.nc-btn {
  position: relative;
  width: 32px; height: 32px;
  display: flex; align-items: center; justify-content: center;
  border: 1px solid var(--line-border);
  background: var(--bg-card);
  color: var(--ink-tertiary);
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.18s var(--ease-out-quint);
}
.nc-btn:hover { color: var(--ink-primary); transform: translateY(-1px); }
.nc-btn.active { color: var(--accent-blue); border-color: rgba(37, 99, 235, 0.30); background: rgba(37, 99, 235, 0.06); }
.nc-btn.hasUnread { color: var(--accent-amber); }
.nc-btn.hasUnread svg { animation: bell-ring 4s ease-in-out infinite; transform-origin: top center; }
@keyframes bell-ring {
  0%, 80%, 100% { transform: rotate(0); }
  84%, 92%      { transform: rotate(-9deg); }
  88%, 96%      { transform: rotate(9deg); }
}

.nc-badge {
  position: absolute;
  top: -4px; right: -4px;
  min-width: 14px; height: 14px;
  padding: 0 3px;
  background: var(--status-err);
  color: white;
  border-radius: 999px;
  font-size: 0.54rem;
  font-weight: 700;
  font-family: 'JetBrains Mono Variable', monospace;
  display: flex; align-items: center; justify-content: center;
  line-height: 1;
  box-shadow: 0 0 0 2px var(--bg-card);
}

.nc-panel {
  position: fixed;
  top: 64px;
  right: 18px;
  width: 380px;
  max-height: 70vh;
  border-radius: 14px;
  z-index: 80;
  display: flex; flex-direction: column;
  overflow: hidden;
}
.nc-head { display: flex; justify-content: space-between; align-items: center; padding: 12px 14px; border-bottom: 1px solid var(--line-divider); }
.nc-head-l { display: flex; align-items: center; gap: 8px; }
.nc-mark { background: transparent; border: none; color: var(--accent-blue); cursor: pointer; font-size: 0.7rem; }
.nc-mark:hover { text-decoration: underline; }

.nc-list { flex: 1; overflow-y: auto; padding: 6px; }
.nc-empty { padding: 32px; text-align: center; color: var(--ink-muted); font-size: 0.78rem; }
.nc-item {
  display: grid;
  grid-template-columns: 4px 1fr auto;
  gap: 10px; align-items: flex-start;
  padding: 10px 12px;
  border-radius: 8px;
  font-size: 0.76rem;
  transition: background 0.18s var(--ease-out-quint);
}
.nc-item:hover { background: rgba(241, 245, 249, 0.6); }
[data-theme='dark'] .nc-item:hover { background: rgba(255, 255, 255, 0.03); }
.nc-item.unread { background: rgba(37, 99, 235, 0.04); }
[data-theme='dark'] .nc-item.unread { background: rgba(37, 99, 235, 0.10); }
.nc-rail { width: 3px; align-self: stretch; border-radius: 2px; }
.rail-ok   { background: var(--status-ok); }
.rail-warn { background: var(--status-warn); }
.rail-err  { background: var(--status-err); }
.rail-info { background: var(--status-info); }
.rail-idle { background: var(--status-idle); }

.nc-body { display: flex; flex-direction: column; gap: 2px; min-width: 0; }
.nc-row1 { display: flex; align-items: baseline; justify-content: space-between; gap: 6px; }
.nc-title { font-weight: 600; color: var(--ink-primary); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.nc-ago { font-size: 0.6rem; color: var(--ink-muted); flex-shrink: 0; }
.nc-detail { color: var(--ink-tertiary); font-size: 0.7rem; }
.nc-src { color: var(--ink-muted); font-size: 0.6rem; }

.nc-x {
  width: 22px; height: 22px;
  display: flex; align-items: center; justify-content: center;
  background: transparent; border: 1px solid var(--line-border);
  border-radius: 6px;
  color: var(--ink-tertiary);
  font-size: 0.72rem;
  cursor: pointer;
  flex-shrink: 0;
}
.nc-x:hover { color: var(--status-ok); border-color: rgba(16, 185, 129, 0.30); background: rgba(16, 185, 129, 0.10); }

.nc-enter-from { opacity: 0; transform: translateY(-10px) scale(0.97); }
.nc-leave-to { opacity: 0; transform: translateY(-6px); }
.nc-enter-active, .nc-leave-active { transition: opacity 0.22s var(--ease-out-quint), transform 0.22s var(--ease-out-quint); }
</style>
