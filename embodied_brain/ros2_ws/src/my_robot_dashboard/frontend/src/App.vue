<script setup lang="ts">
import { RouterLink, RouterView, useRoute } from 'vue-router'
import { computed, onMounted, ref, watch } from 'vue'
import router from '@/router'
import { useTelemetryStore } from '@/stores/telemetry'
import { useSettingsStore } from '@/stores/settings'
import { useToastStore } from '@/stores/toast'
import { useUiStore } from '@/stores/ui'
import { useKeyboard } from '@/composables/useKeyboard'
import { useAudioFeedback } from '@/composables/useAudioFeedback'
import { useInputMode } from '@/composables/useInputMode'
import CommandPalette from '@/components/CommandPalette.vue'
import SettingsPopover from '@/components/SettingsPopover.vue'
import ToastStack from '@/components/ToastStack.vue'
import ConnectionBanner from '@/components/ConnectionBanner.vue'
import HotkeyReference from '@/components/HotkeyReference.vue'
import AboutDialog from '@/components/AboutDialog.vue'
import DispatchTaskModal from '@/components/DispatchTaskModal.vue'
import PwaInstallButton from '@/components/PwaInstallButton.vue'
import AuroraBg from '@/components/premium/AuroraBg.vue'
import SpotlightCursor from '@/components/premium/SpotlightCursor.vue'
import EventMarquee from '@/components/premium/EventMarquee.vue'
import MissionClock from '@/components/mission/MissionClock.vue'
import MissionPhase from '@/components/mission/MissionPhase.vue'
import GoNoGoGrid from '@/components/mission/GoNoGoGrid.vue'
import MasterAlarm from '@/components/mission/MasterAlarm.vue'
import BootSplash from '@/components/boot/BootSplash.vue'
import ActivityStream from '@/components/stream/ActivityStream.vue'
import KpiDrilldown from '@/components/drill/KpiDrilldown.vue'
import MiniMap from '@/components/drill/MiniMap.vue'
import NotificationCenter from '@/components/drill/NotificationCenter.vue'

const route = useRoute()
const pages = computed(() => router.getRoutes().filter((r) => r.meta?.title))
const currentTitle = computed(() => (route.meta.title as string | undefined) ?? 'NavCockpit')

const telemetry = useTelemetryStore()
const settings = useSettingsStore()
const toasts = useToastStore()
const ui = useUiStore()
const audio = useAudioFeedback()
const { isTouch } = useInputMode()

const wsTone = computed<'ok' | 'warn' | 'err' | 'idle'>(() => {
  if (telemetry.state === 'open') return telemetry.isStale ? 'warn' : 'ok'
  if (telemetry.state === 'connecting' || telemetry.state === 'closed') return 'warn'
  if (telemetry.state === 'error') return 'err'
  return 'idle'
})

const sourceMode = computed(() => telemetry.packet?.provenance?.mode ?? 'fixture_only')
const sourceLabel = computed(() => sourceMode.value === 'live_partial' ? 'LIVE PARTIAL' : 'FIXTURE ONLY')
const batteryText = computed(() => {
  if (telemetry.packet?.provenance?.unavailable_fields.includes('battery')) return 'unavailable'
  const pct = telemetry.packet?.battery?.pct
  return pct === undefined ? '—' : `${pct.toFixed(0)}%`
})

const clock = ref('')
onMounted(() => {
  const tick = () => {
    const d = new Date()
    const pad = (n: number) => n.toString().padStart(2, '0')
    clock.value = `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`
  }
  tick()
  setInterval(tick, 1000)
})

// ---------- modals state (proxied through ui store) ----------
function openPalette() { ui.openPalette(); audio.play('ping') }
function closePalette() { ui.paletteOpen = false }

// ---------- alarm badge counter (Timeline only, recent alarms) ----------
const ALARM_RECENT_WINDOW_MS = 5 * 60 * 1000
const alarmBadge = computed(() => {
  const cutoff = Date.now() - ALARM_RECENT_WINDOW_MS
  return telemetry.alarmHistory.filter((a) => a.at_ms >= cutoff).length
})

function badgeFor(path: string): { n: number; tone: 'err' | 'warn' | 'info' } | null {
  if (path === '/timeline' && alarmBadge.value > 0) {
    const hasErr = telemetry.alarmHistory.some((a) => a.severity === 'err' && a.at_ms >= Date.now() - ALARM_RECENT_WINDOW_MS)
    return { n: alarmBadge.value, tone: hasErr ? 'err' : 'warn' }
  }
  return null
}

// ---------- global keybindings ----------
useKeyboard({
  'mod+k':   () => { ui.paletteOpen = !ui.paletteOpen },
  '/':       () => { ui.paletteOpen = true },
  '?':       () => { ui.hotkeysOpen = !ui.hotkeysOpen },
  't':       () => settings.toggleTheme(),
  's':       () => settings.toggleSound(),
  'd':       () => { ui.dispatchOpen = true },
  'g c':     () => router.push('/'),
  'g i':     () => router.push('/immersive'),
  'g r':     () => router.push('/perception'),
  'g n':     () => router.push('/inspector'),
  'g t':     () => router.push('/timeline'),
  'g d':     () => router.push('/twin'),
  'g p':     () => router.push('/planner'),
  'g o':     () => router.push('/topology'),
  'g m':     () => router.push('/missions'),
  'g l':     () => router.push('/livemap'),
  'g a':     () => router.push('/chat'),
  'g b':     () => router.push('/blackbox'),
  ']':       () => ui.toggleActivityStream(),
  'escape':  () => ui.closeAll(),
})

// ---------- alarm → toast bridge ----------
const seenAlarms = new Set<string>()
let primed = false
watch(() => telemetry.alarmHistory, (history) => {
  for (const a of history) {
    if (seenAlarms.has(a.id)) continue
    seenAlarms.add(a.id)
    if (!primed) continue  // first-paint priming: don't spam toasts for backlog
    toasts.push({
      tone: a.severity,
      title: a.title,
      detail: a.detail,
      durationMs: a.severity === 'err' ? 8000 : 5000,
    })
    audio.play(a.severity === 'err' ? 'err' : a.severity === 'warn' ? 'warn' : 'ping')
  }
  primed = true
}, { deep: false })

// route change audio
watch(() => route.fullPath, () => audio.play('navigate'))

// pause stream forwarder
watch(() => ui.streamPaused, (p) => telemetry.setPaused(p))

// ---------- brand mark heartbeat — pulse on every WS packet (debounced) ----------
const heartbeatTick = ref(false)
watch(() => telemetry.packet?.heartbeat.sequence, () => {
  heartbeatTick.value = true
  setTimeout(() => { heartbeatTick.value = false }, 280)
})

// ---------- embedded in 指挥中心 portal iframe? (chrome 精简 + splash 去重) ----------
const isEmbedded = (() => { try { return window.self !== window.top } catch { return true } })()

// 指挥中心剧场模式深度联动: 收 portal scene 消息 → 路由跳到对应幕 (如取料幕 → /livemap)
onMounted(() => {
  if (!isEmbedded) return
  window.addEventListener('message', (e: MessageEvent) => {
    if (e.origin !== 'https://xiaomiju.xyz') return
    const d = e.data
    if (!d || d.source !== 'xrd-cmdcenter') return
    if (d.action === 'scene' && d.route && router.currentRoute.value.path !== d.route) {
      router.push(d.route).catch(() => {})
    }
  })
})

// ---------- boot splash — session-once gate ----------
const BOOT_KEY = 'navcockpit.bootShown.v1'
const bootVisible = ref(false)
function checkBoot() {
  try {
    // explicit URL escape hatch
    const params = new URLSearchParams(window.location.search)
    if (params.get('boot') === 'force') { bootVisible.value = true; return }
    if (params.get('boot') === 'skip') return
    // 内嵌在指挥中心 iframe 中: 门户已统一播过渡动画, 本舱不重复 (避免双层 splash)
    if (window.self !== window.top) return
    if (sessionStorage.getItem(BOOT_KEY) === '1') return
    bootVisible.value = true
  } catch { /* SSR / quota */ }
}
function bootDone() {
  bootVisible.value = false
  try { sessionStorage.setItem(BOOT_KEY, '1') } catch { /* noop */ }
}
onMounted(checkBoot)

// ---------- marquee feed: latest alarms + dispatched tasks + heartbeat blips ----------
const marqueeItems = computed(() => {
  const items: Array<{ id: string; label: string; tone: 'ok' | 'warn' | 'err' | 'info' | 'idle'; glyph?: string; ts?: number }> = []
  // recent alarms
  for (const a of telemetry.alarmHistory.slice(-6)) {
    items.push({ id: `alm-${a.id}`, label: a.title, tone: a.severity, glyph: a.severity === 'err' ? '⚠' : a.severity === 'warn' ? '◆' : '•', ts: a.at_ms })
  }
  // tasks
  const tasks = telemetry.packet?.tasks ?? []
  for (const t of tasks.slice(0, 4)) {
    const tone = t.status === 'completed' ? 'ok' : t.status === 'failed' ? 'err' : t.status === 'running' ? 'info' : 'idle'
    items.push({ id: `tsk-${t.id}`, label: `${t.name} · ${t.status} ${t.progress_pct.toFixed(0)}%`, tone, glyph: '▶' })
  }
  // ai_link
  const ai = telemetry.packet?.ai_link
  if (ai) {
    items.push({ id: 'ai', label: `AI 脑 ${ai.online ? `online ${ai.rtt_ms.toFixed(0)}ms` : 'offline'} · ${ai.dispatches_24h} dispatches/24h`, tone: ai.online ? 'ok' : 'idle', glyph: '◉' })
  }
  // heartbeat
  items.push({ id: 'hb', label: `Heartbeat · seq ${telemetry.packet?.heartbeat.sequence ?? 0} · uptime ${telemetry.packet?.heartbeat.uptime_s.toFixed(0) ?? 0}s`, tone: 'info', glyph: '♥', ts: Date.now() })
  // pose
  const p = telemetry.packet?.pose
  if (p) {
    items.push({ id: 'pose', label: `Pose (${p.x.toFixed(2)}, ${p.y.toFixed(2)}) yaw ${((p.yaw * 180) / Math.PI).toFixed(0)}°`, tone: 'idle', glyph: '◎' })
  }
  return items.length ? items : [{ id: 'idle', label: 'awaiting telemetry…', tone: 'idle' as const }]
})
</script>

<template>
  <div class="layout mesh-bg" :class="{ embedded: isEmbedded }">
    <header class="topbar glass">
      <div class="brand" role="button" title="About NavCockpit" @click="ui.openAbout">
        <span class="brand-mark" :class="{ beat: telemetry.isConnected && heartbeatTick }">⬢</span>
        <div class="brand-text">
          <div class="brand-name">NavCockpit</div>
          <div class="brand-sub mono">embodied · X5 · 198.51.100.85</div>
        </div>
      </div>

      <div class="topbar-center">
        <span class="mission-clock-wrap"><MissionClock /></span>
        <MissionPhase />
        <span class="hairline-v"></span>
        <span class="gonogo-wrap"><GoNoGoGrid layout="compact" /></span>
        <span class="hairline-v"></span>
        <MasterAlarm />
      </div>

      <div class="topbar-right">
        <span class="source-badge" :class="`source-${sourceMode}`" :title="telemetry.packet?.provenance?.note">
          {{ sourceLabel }}
        </span>
        <span class="page-title-mini">{{ currentTitle }}</span>
        <span class="clock mono">{{ clock }}</span>
        <button class="dispatch-btn" :title="isTouch ? 'Dispatch task' : 'Dispatch task (d)'" @click="ui.openDispatch">
          <span>▶</span><span>Dispatch</span>
        </button>
        <span class="pwa-wrap"><PwaInstallButton /></span>
        <span class="notification-top-wrap"><NotificationCenter /></span>
        <button
          class="cmdk-btn stream-btn"
          :class="{ active: ui.activityStreamOpen }"
          :title="isTouch ? 'Live stream' : 'Live activity stream (])'"
          @click="ui.toggleActivityStream"
        >
          <span class="cmdk-icon">≡</span><span>Stream</span>
        </button>
        <button class="cmdk-btn palette-btn" :title="isTouch ? 'Search & actions' : 'Command palette (⌘K)'" @click="openPalette">
          <template v-if="isTouch">
            <span class="cmdk-icon">⌕</span><span>Search</span>
          </template>
          <template v-else>
            <span>⌘</span><span>K</span>
          </template>
        </button>
        <button class="help-btn" title="Help · 触控/快捷键手册" @click="ui.openHotkeys">
          <span>?</span>
        </button>
        <span class="settings-top-wrap"><SettingsPopover /></span>
      </div>
    </header>

    <div class="body">
      <nav class="sidebar">
        <div class="nav-section-label section-label">Pages</div>
        <RouterLink
          v-for="p in pages"
          :key="p.name as string"
          :to="p.path"
          class="nav-item"
          active-class="nav-item-active"
        >
          <span class="nav-icon">{{ p.meta?.icon }}</span>
          <span class="nav-label">{{ p.meta?.title }}</span>
          <span v-if="badgeFor(p.path)" class="nav-badge" :class="`badge-${badgeFor(p.path)!.tone}`">{{ badgeFor(p.path)!.n }}</span>
        </RouterLink>

        <div class="nav-section-label section-label" style="margin-top: 22px;">System</div>
        <div class="nav-meta">
          <div class="nav-meta-row">
            <span class="dot" :class="`dot-${wsTone}`"></span>
            <span>websocket</span>
            <span class="nav-meta-val mono">{{ telemetry.observedHz.toFixed(1) }} Hz</span>
          </div>
          <div class="nav-meta-row">
            <span class="dot" :class="telemetry.packet?.ai_link?.online ? 'dot-ok' : 'dot-idle'"></span>
            <span>ai brain</span>
            <span class="nav-meta-val mono">{{ telemetry.packet?.ai_link?.online ? `${telemetry.packet.ai_link.rtt_ms.toFixed(0)}ms` : 'offline' }}</span>
          </div>
          <div class="nav-meta-row">
            <span class="dot" :class="telemetry.packet?.bridge?.alive ? 'dot-ok' : 'dot-idle'"></span>
            <span>ros2 bridge</span>
            <span class="nav-meta-val mono">{{ telemetry.packet?.bridge?.alive ? (telemetry.packet?.bridge?.estop ? 'ESTOP' : 'live') : 'offline' }}</span>
          </div>
          <div class="nav-meta-row">
            <span class="dot dot-info" style="background: var(--accent-blue); box-shadow: 0 0 0 3px rgba(59,130,246,0.18);"></span>
            <span>battery</span>
            <span class="nav-meta-val mono">{{ batteryText }}</span>
          </div>
          <div class="nav-meta-row">
            <span class="dot dot-info" style="background: var(--accent-violet); box-shadow: 0 0 0 3px rgba(124,58,237,0.18);"></span>
            <span>bpu util</span>
            <span class="nav-meta-val mono">{{ telemetry.packet?.host.bpu_pct.toFixed(0) ?? '—' }}%</span>
          </div>
        </div>

        <div class="sidebar-foot">
          <button class="foot-link" @click="ui.openHotkeys">
            <span class="foot-icon">?</span>
            <span>{{ isTouch ? 'Help · 触控手册' : 'Keyboard' }}</span>
          </button>
          <button class="foot-link" @click="ui.openPalette">
            <span class="foot-icon">⌕</span>
            <span>{{ isTouch ? 'Search & actions' : 'Palette · ⌘K' }}</span>
          </button>
          <button class="foot-link" @click="ui.openAbout">
            <span class="foot-icon">ⓘ</span>
            <span>About</span>
          </button>
        </div>
      </nav>

      <main class="outlet">
        <RouterView v-slot="{ Component }">
          <Transition name="page" mode="out-in">
            <component :is="Component" />
          </Transition>
        </RouterView>
      </main>
    </div>

    <!-- bottom event ticker — SpaceX/Vercel style -->
    <EventMarquee :items="marqueeItems" :speed="80" :height="30" />

    <ConnectionBanner />
    <CommandPalette :open="ui.paletteOpen" @close="closePalette" />
    <HotkeyReference :open="ui.hotkeysOpen" @close="ui.hotkeysOpen = false" />
    <AboutDialog :open="ui.aboutOpen" @close="ui.aboutOpen = false" />
    <DispatchTaskModal :open="ui.dispatchOpen" @close="ui.dispatchOpen = false" />
    <ToastStack />

    <!-- Cinematic background effects (teleported to body) -->
    <AuroraBg />
    <SpotlightCursor />

    <!-- Boot splash (teleported to body, session-once) -->
    <BootSplash v-if="bootVisible" @done="bootDone" />

    <!-- Live activity stream (teleported, toggle from topbar or `]`) -->
    <ActivityStream :open="ui.activityStreamOpen" @close="ui.activityStreamOpen = false" />

    <!-- KPI deep-dive modal (teleported) -->
    <KpiDrilldown />

    <!-- Floating mini-map (auto-hidden on /twin /planner /topology) -->
    <MiniMap />
  </div>
</template>

<style scoped>
.layout {
  display: flex;
  flex-direction: column;
  height: 100vh;
  /* transparent so AuroraBg (z-index 0 on body) bleeds subtly through page gutters */
  background: transparent;
  color: var(--ink-primary);
  position: relative;
  z-index: 2;
}

/* 内嵌进指挥中心门户时: 门户已有品牌栏, 隐藏本舱 logo/名/IP 块, 看起来是一个产品 */
.layout.embedded .brand { display: none; }
.layout.embedded .topbar { padding-left: 16px; }

.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: var(--topbar-h, 64px);
  padding: 0 22px;
  border-radius: 0;
  border-left: 0;
  border-right: 0;
  border-top: 0;
  border-bottom: 1px solid var(--line-divider);
  position: sticky;
  top: 0;
  z-index: 40;
  width: 100%;
  max-width: 100%;
  min-width: 0;
  overflow: hidden;
  box-sizing: border-box;
}

.brand {
  display: flex;
  align-items: center;
  gap: 12px;
  min-width: 0;
  flex: 0 1 232px;
  cursor: pointer;
  transition: opacity 0.15s var(--ease-out-quint);
}
.brand:hover { opacity: 0.78; }
.brand-mark {
  font-size: 1.6rem;
  background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary) 60%, var(--accent-emerald));
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
  filter: drop-shadow(0 2px 8px color-mix(in srgb, var(--accent-primary) 35%, transparent));
  display: inline-block;
  transition: transform 0.18s var(--ease-spring), filter 0.18s var(--ease-spring);
}
.brand-mark.beat {
  transform: scale(1.16);
  filter: drop-shadow(0 4px 14px color-mix(in srgb, var(--accent-primary) 60%, transparent));
}
.brand-text { display: flex; flex-direction: column; line-height: 1.15; }
.brand-name {
  font-weight: 700;
  font-size: 1.05rem;
  letter-spacing: 0;
  background: linear-gradient(135deg, #0b1220 0%, #334155 100%);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}
.brand-sub {
  font-size: 0.68rem;
  color: var(--ink-muted);
  letter-spacing: 0.02em;
}

.topbar-center {
  display: flex; align-items: center; gap: 10px;
  min-width: 0;
  flex: 0 1 auto;
}
.hairline-v {
  display: inline-block; width: 1px; height: 16px;
  background: var(--line-divider);
}

.topbar-right {
  display: flex; align-items: center; gap: 8px;
  min-width: 0;
  flex: 0 1 auto;
  overflow: hidden;
}
.source-badge {
  display: inline-flex;
  align-items: center;
  height: 24px;
  padding: 0 8px;
  border-radius: 5px;
  font-family: 'JetBrains Mono Variable', monospace;
  font-size: 0.62rem;
  font-weight: 700;
  white-space: nowrap;
}
.source-live_partial {
  color: #047857;
  background: rgba(5, 150, 105, 0.09);
  border: 1px solid rgba(5, 150, 105, 0.22);
}
.source-fixture_only {
  color: #92400e;
  background: rgba(217, 119, 6, 0.09);
  border: 1px solid rgba(217, 119, 6, 0.24);
}
.page-title-mini {
  font-size: 0.78rem; color: var(--ink-tertiary); font-weight: 500;
  margin-right: 4px;
}
.clock {
  font-size: 0.85rem; color: var(--ink-secondary);
  font-variant-numeric: tabular-nums;
  letter-spacing: 0.02em;
  margin-right: 4px;
}

.dispatch-btn, .cmdk-btn, .help-btn {
  display: inline-flex; align-items: center; gap: 5px;
  padding: 6px 10px;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.18s var(--ease-out-quint);
  font-family: inherit;
}
.dispatch-btn {
  border: 1px solid transparent;
  background: linear-gradient(135deg, var(--accent-blue) 0%, var(--accent-teal) 100%);
  color: white;
  font-size: 0.78rem;
  font-weight: 600;
  box-shadow: 0 3px 10px -3px rgba(37, 99, 235, 0.40);
}
.dispatch-btn:hover { transform: translateY(-1px); box-shadow: 0 6px 16px -4px rgba(37, 99, 235, 0.5); }
.cmdk-btn {
  background: var(--bg-elevated);
  border: 1px solid var(--line-border);
  color: var(--ink-tertiary);
  font-family: 'JetBrains Mono Variable', monospace;
  font-size: 0.72rem;
}
.cmdk-btn:hover {
  color: var(--accent-blue);
  border-color: rgba(37, 99, 235, 0.3);
  background: rgba(37, 99, 235, 0.04);
  transform: translateY(-1px);
}
.cmdk-icon { font-size: 0.88rem; }
.help-btn {
  background: var(--bg-elevated);
  border: 1px solid var(--line-border);
  color: var(--ink-tertiary);
  font-size: 0.84rem;
  font-weight: 700;
  min-width: 32px;
  justify-content: center;
}
.help-btn:hover {
  color: var(--accent-violet);
  border-color: rgba(124, 58, 237, 0.3);
  background: rgba(124, 58, 237, 0.04);
  transform: translateY(-1px);
}

.body {
  display: flex;
  flex: 1;
  min-height: 0;
  min-width: 0;
}
.sidebar {
  width: var(--sidebar-w, 232px);
  padding: 18px 14px;
  border-right: 1px solid var(--line-divider);
  background: rgba(255, 255, 255, 0.62);
  backdrop-filter: blur(16px) saturate(160%);
  -webkit-backdrop-filter: blur(16px) saturate(160%);
  overflow-y: auto;
  display: flex; flex-direction: column;
}

.nav-section-label {
  padding: 0 8px 8px;
  color: var(--ink-muted);
}

.nav-item {
  position: relative;
  display: flex; align-items: center; gap: 10px;
  padding: 9px 12px;
  border-radius: 10px;
  text-decoration: none;
  color: var(--ink-secondary);
  font-size: 0.82rem;
  font-weight: 500;
  margin-bottom: 2px;
  transition: background 0.18s var(--ease-out-quint), color 0.18s var(--ease-out-quint), transform 0.18s var(--ease-out-quint);
}
.nav-item:hover {
  background: rgba(241, 245, 249, 0.85);
  color: var(--ink-primary);
}
.nav-item-active {
  background: linear-gradient(135deg, rgba(37, 99, 235, 0.10), rgba(8, 145, 178, 0.08));
  color: var(--accent-blue);
  font-weight: 600;
  box-shadow: inset 0 0 0 1px rgba(37, 99, 235, 0.18);
}
.nav-item-active::before {
  content: '';
  position: absolute; left: -14px; top: 8px; bottom: 8px;
  width: 3px; border-radius: 999px;
  background: linear-gradient(180deg, var(--accent-blue), var(--accent-teal));
}
.nav-icon {
  font-size: 1.0rem; width: 22px; text-align: center; flex-shrink: 0;
}
.nav-label { flex: 1; }

.nav-badge {
  display: inline-flex; align-items: center; justify-content: center;
  min-width: 18px; height: 16px; padding: 0 5px;
  border-radius: 999px;
  font-size: 0.62rem; font-weight: 700;
  font-variant-numeric: tabular-nums;
  font-family: 'JetBrains Mono Variable', monospace;
  line-height: 1;
}
.badge-err  { background: var(--status-err);  color: white; box-shadow: 0 0 0 2px rgba(239, 68, 68, 0.18); animation: pulseSoft 1.6s ease-in-out infinite; }
.badge-warn { background: var(--status-warn); color: white; box-shadow: 0 0 0 2px rgba(245, 158, 11, 0.18); }
.badge-info { background: var(--status-info); color: white; box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.18); }

.nav-meta {
  padding: 4px 8px;
  font-size: 0.72rem;
  color: var(--ink-tertiary);
  display: flex; flex-direction: column; gap: 8px;
}
.nav-meta-row { display: flex; align-items: center; gap: 8px; }
.nav-meta-val { margin-left: auto; color: var(--ink-muted); }

.sidebar-foot {
  margin-top: auto;
  padding: 12px 4px 4px;
  border-top: 1px solid var(--line-divider);
  display: flex; flex-direction: column; gap: 2px;
}
.foot-link {
  display: flex; align-items: center; gap: 10px;
  padding: 8px 10px;
  background: transparent;
  border: none;
  color: var(--ink-secondary);
  font-size: 0.76rem;
  text-align: left;
  cursor: pointer;
  border-radius: 8px;
  transition: background 0.15s var(--ease-out-quint), color 0.15s var(--ease-out-quint);
  font-family: inherit;
}
.foot-link:hover { background: rgba(15, 23, 42, 0.04); color: var(--ink-primary); }
.foot-icon {
  display: inline-flex; align-items: center; justify-content: center;
  width: 22px; height: 22px;
  font-weight: 700;
  color: var(--ink-tertiary);
  background: var(--bg-elevated);
  border-radius: 5px;
  font-size: 0.8rem;
  flex-shrink: 0;
}

.outlet {
  flex: 1;
  overflow-y: auto;
  overflow-x: hidden;
  min-width: 0;
  padding: 28px 32px;
}

@media (max-width: 1400px) {
  .topbar { padding: 0 14px; gap: 10px; }
  .brand { flex-basis: 190px; }
  .page-title-mini,
  .clock,
  .pwa-wrap,
  .stream-btn,
  .palette-btn,
  .source-badge,
  .help-btn { display: none; }
}

@media (max-width: 1180px) {
  .brand { flex-basis: 132px; }
  .brand-sub,
  .mission-clock-wrap,
  .gonogo-wrap,
  .topbar-center .hairline-v { display: none; }
  .outlet { padding: 22px 20px; }
}

@media (max-width: 900px) {
  .brand-text,
  .palette-btn { display: none; }
  .brand { flex: 0 0 34px; }
  .sidebar { width: 72px; padding-inline: 10px; }
  .nav-label,
  .nav-section-label,
  .nav-meta,
  .sidebar-foot span:not(.foot-icon) { display: none; }
  .nav-item { justify-content: center; padding-inline: 8px; }
}

@media (max-width: 620px) {
  .topbar { padding-inline: 10px; }
  .settings-top-wrap { display: none; }
  .sidebar { width: 56px; padding-inline: 7px; }
  .outlet { padding: 18px 10px; }
}

.page-enter-active,
.page-leave-active {
  transition: opacity 0.32s var(--ease-out-quint), transform 0.32s var(--ease-out-quint), filter 0.32s var(--ease-out-quint);
}
.page-enter-from {
  opacity: 0;
  transform: translateY(10px);
  filter: blur(4px);
}
.page-leave-to {
  opacity: 0;
  transform: translateY(-4px);
  filter: blur(2px);
}
</style>
