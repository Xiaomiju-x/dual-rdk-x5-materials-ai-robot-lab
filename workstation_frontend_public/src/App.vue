<script setup lang="ts">
import { RouterLink, RouterView, useRoute } from 'vue-router'
import { computed, onMounted, ref, watch } from 'vue'
import router from '@/router'
import { useTelemetryStore } from '@/stores/telemetry'
import { useSettingsStore } from '@/stores/settings'
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
import PwaInstallButton from '@/components/PwaInstallButton.vue'
import AuroraBackground from '@/components/fx/AuroraBackground.vue'
import SpotlightCursor from '@/components/fx/SpotlightCursor.vue'

const route = useRoute()
const pages = computed(() => router.getRoutes().filter((r) => r.meta?.title))
const currentTitle = computed(() => (route.meta.title as string | undefined) ?? 'WorkCockpit')

// 内嵌进指挥中心门户 iframe? → 隐藏本舱品牌块, 看起来是一个产品
const isEmbedded = (() => { try { return window.self !== window.top } catch (_e) { return true } })()

// 指挥中心剧场模式深度联动: 收 portal scene 消息 → 路由跳到对应幕 (研磨灌装幕 → /pipeline)
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

const telemetry = useTelemetryStore()
const settings = useSettingsStore()
const ui = useUiStore()
const audio = useAudioFeedback()
const { isTouch } = useInputMode()

const wsTone = computed<'ok' | 'warn' | 'err' | 'idle'>(() => {
  if (telemetry.state === 'open') return telemetry.isStale ? 'warn' : 'ok'
  if (telemetry.state === 'connecting' || telemetry.state === 'closed') return 'warn'
  if (telemetry.state === 'error') return 'err'
  return 'idle'
})
const wsLabel = computed(() => {
  if (telemetry.state === 'open') return telemetry.isStale ? 'stale' : `live · ${telemetry.observedHz.toFixed(1)} Hz`
  if (telemetry.state === 'connecting') return 'connecting…'
  if (telemetry.state === 'error') return 'reconnecting…'
  return 'idle'
})

const clock = ref('')
onMounted(() => {
  const tick = () => {
    const d = new Date()
    const pad = (n: number) => n.toString().padStart(2, '0')
    clock.value = `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`
  }
  tick(); setInterval(tick, 1000)
})

function openPalette() { ui.openPalette(); audio.play('ping') }

useKeyboard({
  'mod+k': () => { ui.paletteOpen = !ui.paletteOpen },
  '/': () => { ui.paletteOpen = true },
  '?': () => { ui.hotkeysOpen = !ui.hotkeysOpen },
  't': () => settings.toggleTheme(),
  's': () => settings.toggleSound(),
  'p': () => telemetry.togglePaused(),
  'g c': () => router.push('/'),
  'g o': () => router.push('/teleop'),
  'g h': () => router.push('/handover'),
  'g i': () => router.push('/inspect'),
  'g k': () => router.push('/calibration'),
  'g d': () => router.push('/defense'),
  'escape': () => ui.closeAll(),
})

watch(() => route.fullPath, () => audio.play('navigate'))
watch(() => ui.streamPaused, (p) => telemetry.setPaused(p))

const tempStr = (t: number | null | undefined) => (t == null ? '—' : `${t.toFixed(0)}°C`)
</script>

<template>
  <div class="layout" :class="{ embedded: isEmbedded }">
    <AuroraBackground />
    <SpotlightCursor />
    <header class="topbar glass">
      <div class="brand" @click="ui.openAbout" role="button" title="About WorkCockpit">
        <span class="brand-mark">🦾</span>
        <div class="brand-text">
          <div class="brand-name">WorkCockpit</div>
          <div class="brand-sub mono">dual myCobot · arm01 · 192.0.2.64</div>
        </div>
      </div>

      <div class="topbar-center">
        <span class="dot" :class="`dot-${wsTone}`"></span>
        <span class="mono text-[0.72rem] text-[var(--ink-tertiary)]">{{ wsLabel }}</span>
        <span class="hairline-v"></span>
        <span class="chip" :class="telemetry.mode === 'real' ? 'chip-ok' : 'chip-info'">
          {{ telemetry.mode }} · 10 Hz
        </span>
        <span class="mono text-[0.7rem] text-[var(--ink-muted)]">f{{ telemetry.frameCount }}</span>
      </div>

      <div class="topbar-right">
        <span class="page-title-mini">{{ currentTitle }}</span>
        <span class="clock mono">{{ clock }}</span>
        <button class="pause-btn" :class="{ active: telemetry.paused }" @click="telemetry.togglePaused()"
                :title="isTouch ? '冻结画面' : 'Pause stream (p)'">
          <span>{{ telemetry.paused ? '▶' : '⏸' }}</span>
          <span>{{ telemetry.paused ? 'Resume' : 'Pause' }}</span>
        </button>
        <PwaInstallButton />
        <button class="cmdk-btn" @click="openPalette" :title="isTouch ? 'Search & actions' : 'Command palette (⌘K)'">
          <template v-if="isTouch"><span class="cmdk-icon">⌕</span><span>Search</span></template>
          <template v-else><span>⌘</span><span>K</span></template>
        </button>
        <button class="help-btn" @click="ui.openHotkeys" title="Help · 触控/快捷键手册"><span>?</span></button>
        <SettingsPopover />
      </div>
    </header>

    <div class="body">
      <nav class="sidebar">
        <div class="nav-section-label section-label">Pages</div>
        <RouterLink v-for="p in pages" :key="p.name as string" :to="p.path"
                    class="nav-item" active-class="nav-item-active">
          <span class="nav-icon">{{ p.meta?.icon }}</span>
          <span class="nav-label">{{ p.meta?.title }}</span>
        </RouterLink>

        <div class="nav-section-label section-label" style="margin-top: 22px;">Workstation</div>
        <div class="nav-meta">
          <div class="nav-meta-row">
            <span class="dot" :class="telemetry.arm01.online ? 'dot-ok' : 'dot-idle'"></span>
            <span>arm01</span>
            <span class="nav-meta-val mono">{{ telemetry.arm01.online ? tempStr(telemetry.status?.arm01.temp_c) : 'offline' }}</span>
          </div>
          <div class="nav-meta-row">
            <span class="dot" :class="telemetry.arm02.online ? 'dot-ok' : 'dot-idle'"
                  style="background: var(--accent-blue); box-shadow: 0 0 0 3px rgba(37,99,235,0.18);"></span>
            <span>arm02</span>
            <span class="nav-meta-val mono">{{ telemetry.arm02.online ? tempStr(telemetry.status?.arm02.temp_c) : 'offline' }}</span>
          </div>
          <div class="nav-meta-row">
            <span class="dot dot-info" style="background: var(--accent-teal); box-shadow: 0 0 0 3px rgba(8,145,178,0.18);"></span>
            <span>cameras</span>
            <span class="nav-meta-val mono">{{ (telemetry.status?.cam01_fps ?? 0).toFixed(0) }}/{{ (telemetry.status?.cam02_fps ?? 0).toFixed(0) }} fps</span>
          </div>
          <div class="nav-meta-row">
            <span class="dot" :class="telemetry.aiOnline ? 'dot-ok' : 'dot-idle'"
                  style="background: var(--accent-violet); box-shadow: 0 0 0 3px rgba(124,58,237,0.18);"></span>
            <span>ai brain</span>
            <span class="nav-meta-val mono">{{ telemetry.aiOnline ? `${telemetry.status?.ai_brain_ms?.toFixed(0)}ms` : '—' }}</span>
          </div>
          <div class="nav-meta-row">
            <span class="dot" :class="telemetry.carOnline ? 'dot-ok' : 'dot-idle'"
                  style="background: var(--accent-amber); box-shadow: 0 0 0 3px rgba(217,119,6,0.18);"></span>
            <span>car brain</span>
            <span class="nav-meta-val mono">{{ telemetry.carOnline ? `${telemetry.status?.car_brain_ms?.toFixed(0)}ms` : '—' }}</span>
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
            <span class="foot-icon">ⓘ</span><span>About</span>
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

    <ConnectionBanner />
    <CommandPalette :open="ui.paletteOpen" @close="ui.paletteOpen = false" />
    <HotkeyReference :open="ui.hotkeysOpen" @close="ui.hotkeysOpen = false" />
    <AboutDialog :open="ui.aboutOpen" @close="ui.aboutOpen = false" />
    <ToastStack />
  </div>
</template>

<style scoped>
.layout { display: flex; flex-direction: column; height: 100vh; background: transparent; color: var(--ink-primary); position: relative; z-index: 1; }
/* 内嵌进指挥中心门户时: 门户已有品牌栏, 隐藏本舱 logo/名/IP 块 */
.layout.embedded .brand { display: none; }
.layout.embedded .topbar { padding-left: 16px; }

.topbar {
  display: flex; align-items: center; justify-content: space-between;
  height: var(--topbar-h, 64px); padding: 0 22px;
  border-radius: 0; border-left: 0; border-right: 0; border-top: 0;
  border-bottom: 1px solid var(--line-divider);
  position: sticky; top: 0; z-index: 40;
  background: var(--bg-glass-strong);
  backdrop-filter: blur(20px) saturate(180%);
  -webkit-backdrop-filter: blur(20px) saturate(180%);
}

.brand { display: flex; align-items: center; gap: 12px; min-width: 252px; cursor: pointer; transition: opacity 0.15s var(--ease-out-quint); }
.brand:hover { opacity: 0.78; }
.brand-mark { font-size: 1.5rem; filter: drop-shadow(0 2px 8px rgba(37, 99, 235, 0.25)); }
.brand-text { display: flex; flex-direction: column; line-height: 1.15; }
.brand-name {
  font-weight: 700; font-size: 1.05rem; letter-spacing: -0.01em;
  background: linear-gradient(135deg, #0b1220 0%, #334155 100%);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.brand-sub { font-size: 0.68rem; color: var(--ink-muted); letter-spacing: 0.02em; }

.topbar-center { display: flex; align-items: center; gap: 10px; }
.hairline-v { display: inline-block; width: 1px; height: 16px; background: var(--line-divider); }

.topbar-right { display: flex; align-items: center; gap: 8px; }
.page-title-mini { font-size: 0.78rem; color: var(--ink-tertiary); font-weight: 500; margin-right: 4px; }
.clock { font-size: 0.85rem; color: var(--ink-secondary); font-variant-numeric: tabular-nums; letter-spacing: 0.02em; margin-right: 4px; }

.pause-btn, .cmdk-btn, .help-btn {
  display: inline-flex; align-items: center; gap: 5px; padding: 6px 10px;
  border-radius: 6px; cursor: pointer; transition: all 0.18s var(--ease-out-quint); font-family: inherit;
}
.pause-btn {
  background: var(--bg-elevated); border: 1px solid var(--line-border);
  color: var(--ink-tertiary); font-size: 0.74rem; font-weight: 500;
}
.pause-btn:hover { color: var(--accent-amber); border-color: rgba(217,119,6,0.3); transform: translateY(-1px); }
.pause-btn.active { background: rgba(217,119,6,0.10); color: var(--accent-amber); border-color: rgba(217,119,6,0.35); }
.cmdk-btn {
  background: var(--bg-elevated); border: 1px solid var(--line-border); color: var(--ink-tertiary);
  font-family: 'JetBrains Mono Variable', monospace; font-size: 0.72rem;
}
.cmdk-btn:hover { color: var(--accent-blue); border-color: rgba(37, 99, 235, 0.3); background: rgba(37, 99, 235, 0.04); transform: translateY(-1px); }
.cmdk-icon { font-size: 0.88rem; }
.help-btn { background: var(--bg-elevated); border: 1px solid var(--line-border); color: var(--ink-tertiary); font-size: 0.84rem; font-weight: 700; min-width: 32px; justify-content: center; }
.help-btn:hover { color: var(--accent-violet); border-color: rgba(124, 58, 237, 0.3); background: rgba(124, 58, 237, 0.04); transform: translateY(-1px); }

.body { display: flex; flex: 1; min-height: 0; }
.sidebar {
  width: var(--sidebar-w, 232px); padding: 18px 14px;
  border-right: 1px solid var(--line-divider);
  background: var(--bg-glass);
  backdrop-filter: blur(22px) saturate(170%); -webkit-backdrop-filter: blur(22px) saturate(170%);
  overflow-y: auto; display: flex; flex-direction: column;
}
[data-theme='dark'] .sidebar { background: rgba(17, 21, 31, 0.62); }
.nav-section-label { padding: 0 8px 8px; color: var(--ink-muted); }
.nav-item {
  position: relative; display: flex; align-items: center; gap: 10px;
  padding: 9px 12px; border-radius: 10px; text-decoration: none;
  color: var(--ink-secondary); font-size: 0.82rem; font-weight: 500; margin-bottom: 2px;
  transition: background 0.18s var(--ease-out-quint), color 0.18s var(--ease-out-quint);
}
.nav-item:hover { background: rgba(241, 245, 249, 0.85); color: var(--ink-primary); }
.nav-item-active {
  background: linear-gradient(135deg, rgba(37, 99, 235, 0.10), rgba(8, 145, 178, 0.08));
  color: var(--accent-blue); font-weight: 600; box-shadow: inset 0 0 0 1px rgba(37, 99, 235, 0.18);
}
.nav-item-active::before {
  content: ''; position: absolute; left: -14px; top: 8px; bottom: 8px; width: 3px; border-radius: 999px;
  background: linear-gradient(180deg, var(--accent-blue), var(--accent-teal));
}
.nav-icon { font-size: 1.0rem; width: 22px; text-align: center; flex-shrink: 0; }
.nav-label { flex: 1; }

.nav-meta { padding: 4px 8px; font-size: 0.72rem; color: var(--ink-tertiary); display: flex; flex-direction: column; gap: 8px; }
.nav-meta-row { display: flex; align-items: center; gap: 8px; }
.nav-meta-val { margin-left: auto; color: var(--ink-muted); }

.sidebar-foot { margin-top: auto; padding: 12px 4px 4px; border-top: 1px solid var(--line-divider); display: flex; flex-direction: column; gap: 2px; }
.foot-link {
  display: flex; align-items: center; gap: 10px; padding: 8px 10px; background: transparent; border: none;
  color: var(--ink-secondary); font-size: 0.76rem; text-align: left; cursor: pointer; border-radius: 8px;
  transition: background 0.15s var(--ease-out-quint); font-family: inherit;
}
.foot-link:hover { background: rgba(15, 23, 42, 0.04); color: var(--ink-primary); }
.foot-icon {
  display: inline-flex; align-items: center; justify-content: center; width: 22px; height: 22px;
  font-weight: 700; color: var(--ink-tertiary); background: var(--bg-elevated); border-radius: 5px; font-size: 0.8rem; flex-shrink: 0;
}

.outlet { flex: 1; overflow-y: auto; padding: 24px 28px; }

.page-enter-active, .page-leave-active { transition: opacity 0.32s var(--ease-out-quint), transform 0.32s var(--ease-out-quint), filter 0.32s var(--ease-out-quint); }
.page-enter-from { opacity: 0; transform: translateY(10px); filter: blur(4px); }
.page-leave-to { opacity: 0; transform: translateY(-4px); filter: blur(2px); }
</style>
