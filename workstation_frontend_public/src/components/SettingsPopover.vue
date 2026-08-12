<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from 'vue'
import { useSettingsStore } from '@/stores/settings'

const settings = useSettingsStore()
const open = ref(false)
const anchorRef = ref<HTMLElement | null>(null)
const popRef = ref<HTMLElement | null>(null)

function toggle() { open.value = !open.value }

function onDoc(evt: MouseEvent) {
  if (!open.value) return
  const t = evt.target as Node
  if (anchorRef.value?.contains(t)) return
  if (popRef.value?.contains(t)) return
  open.value = false
}

onMounted(() => document.addEventListener('mousedown', onDoc))
onBeforeUnmount(() => document.removeEventListener('mousedown', onDoc))
</script>

<template>
  <div class="settings-wrap">
    <button
      ref="anchorRef"
      class="settings-btn"
      :class="{ active: open }"
      @click="toggle"
      aria-label="settings"
    >
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="3"></circle>
        <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
      </svg>
    </button>

    <transition name="pop">
      <div v-if="open" ref="popRef" class="settings-pop glass-strong">
        <div class="pop-head">
          <span class="section-label">Display</span>
        </div>

        <div class="setting">
          <span class="s-label">Theme</span>
          <div class="seg-tiny">
            <button class="seg-tiny-btn" :class="{ active: settings.theme === 'light' }" @click="settings.theme = 'light'">☀ Light</button>
            <button class="seg-tiny-btn" :class="{ active: settings.theme === 'dark' }"  @click="settings.theme = 'dark'">🌙 Dark</button>
          </div>
        </div>

        <div class="setting">
          <span class="s-label">Density</span>
          <div class="seg-tiny">
            <button class="seg-tiny-btn" :class="{ active: settings.density === 'comfortable' }" @click="settings.density = 'comfortable'">Comfortable</button>
            <button class="seg-tiny-btn" :class="{ active: settings.density === 'compact' }"     @click="settings.density = 'compact'">Compact</button>
          </div>
        </div>

        <div class="pop-divider"></div>
        <div class="pop-head">
          <span class="section-label">Feedback</span>
        </div>

        <label class="setting-toggle">
          <span>Audio feedback</span>
          <input type="checkbox" :checked="settings.sound" @change="settings.toggleSound" />
          <span class="switch"><span class="switch-knob"></span></span>
        </label>

        <label class="setting-toggle">
          <span>Reduce motion</span>
          <input type="checkbox" :checked="settings.reduceMotion" @change="settings.reduceMotion = !settings.reduceMotion" />
          <span class="switch"><span class="switch-knob"></span></span>
        </label>

        <label class="setting-toggle">
          <span>Show FPS overlay</span>
          <input type="checkbox" :checked="settings.showFps" @change="settings.showFps = !settings.showFps" />
          <span class="switch"><span class="switch-knob"></span></span>
        </label>

        <div class="pop-foot mono">
          <span>设置自动保存到 localStorage</span>
        </div>
      </div>
    </transition>
  </div>
</template>

<style scoped>
.settings-wrap { position: relative; }
.settings-btn {
  width: 32px; height: 32px;
  display: flex; align-items: center; justify-content: center;
  border: 1px solid var(--line-border);
  background: var(--bg-card);
  color: var(--ink-tertiary);
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.18s var(--ease-out-quint);
}
.settings-btn:hover { color: var(--ink-primary); transform: translateY(-1px); box-shadow: var(--shadow-soft); }
.settings-btn.active {
  color: var(--accent-blue);
  border-color: rgba(37, 99, 235, 0.3);
  background: rgba(37, 99, 235, 0.06);
}

.settings-pop {
  position: absolute;
  top: calc(100% + 8px);
  right: 0;
  width: 286px;
  padding: 12px;
  border-radius: 14px;
  z-index: 50;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.pop-head { padding: 4px 4px 0; }
.pop-divider { height: 1px; background: var(--line-divider); margin: 6px 0; }

.setting { display: flex; align-items: center; justify-content: space-between; padding: 4px 4px; gap: 8px; }
.s-label { font-size: 0.8rem; color: var(--ink-secondary); }

.seg-tiny {
  display: inline-flex;
  background: var(--bg-elevated);
  border: 1px solid var(--line-border);
  border-radius: 6px;
  overflow: hidden;
}
.seg-tiny-btn {
  background: transparent; border: none;
  padding: 4px 8px;
  font-size: 0.7rem; color: var(--ink-tertiary);
  cursor: pointer;
}
.seg-tiny-btn:hover { color: var(--ink-primary); }
.seg-tiny-btn.active { background: linear-gradient(135deg, var(--accent-blue), var(--accent-teal)); color: white; }

.setting-toggle {
  display: flex; align-items: center; justify-content: space-between;
  padding: 4px 4px;
  cursor: pointer;
  font-size: 0.8rem; color: var(--ink-secondary);
}
.setting-toggle input { position: absolute; opacity: 0; pointer-events: none; }
.switch {
  display: inline-flex; align-items: center;
  width: 32px; height: 18px;
  border-radius: 999px;
  background: rgba(15, 23, 42, 0.10);
  position: relative;
  transition: background 0.2s var(--ease-out-quint);
}
.switch-knob {
  position: absolute;
  width: 14px; height: 14px;
  border-radius: 50%;
  background: white;
  left: 2px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.2);
  transition: left 0.2s var(--ease-out-quint);
}
.setting-toggle input:checked ~ .switch {
  background: linear-gradient(135deg, var(--accent-blue), var(--accent-teal));
}
.setting-toggle input:checked ~ .switch .switch-knob { left: 16px; }

.pop-foot {
  margin-top: 4px;
  padding: 6px 4px 0;
  border-top: 1px solid var(--line-divider);
  font-size: 0.62rem;
  color: var(--ink-muted);
}

.pop-enter-from { opacity: 0; transform: translateY(-6px) scale(0.98); }
.pop-leave-to { opacity: 0; transform: translateY(-4px) scale(0.99); }
.pop-enter-active, .pop-leave-active { transition: opacity 0.18s var(--ease-out-quint), transform 0.18s var(--ease-out-quint); }
</style>
