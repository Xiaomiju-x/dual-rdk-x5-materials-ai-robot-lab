<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { useSettingsStore } from '@/stores/settings'
import { useTelemetryStore } from '@/stores/telemetry'
import { useUiStore } from '@/stores/ui'

interface Props { open: boolean }
const props = defineProps<Props>()
const emit = defineEmits<{ (e: 'close'): void }>()

const router = useRouter()
const settings = useSettingsStore()
const telemetry = useTelemetryStore()
const ui = useUiStore()

interface Cmd { id: string; icon: string; label: string; hint?: string; run: () => void }

const commands = computed<Cmd[]>(() => [
  { id: 'go-cockpit', icon: '🦾', label: '前往 · 驾驶舱', hint: 'g c', run: () => router.push('/') },
  { id: 'go-teleop', icon: '🎛', label: '前往 · 触控操控', hint: 'g o', run: () => router.push('/teleop') },
  { id: 'go-coop', icon: '🛰', label: '前往 · 三机协同', hint: 'g x', run: () => router.push('/coop') },
  { id: 'go-defense', icon: '🎬', label: '前往 · 答辩自演', hint: 'g d', run: () => router.push('/defense') },
  { id: 'theme', icon: settings.theme === 'dark' ? '☀' : '🌙', label: `切换主题 · ${settings.theme === 'dark' ? '浅色' : '深色'}`, hint: 't', run: () => settings.toggleTheme() },
  { id: 'sound', icon: settings.sound ? '🔇' : '🔊', label: `音效 · ${settings.sound ? '关闭' : '开启'}`, hint: 's', run: () => settings.toggleSound() },
  { id: 'pause', icon: telemetry.paused ? '▶' : '⏸', label: `数据流 · ${telemetry.paused ? '恢复' : '冻结'}`, hint: 'p', run: () => telemetry.togglePaused() },
  { id: 'help', icon: '⌨', label: '快捷键 / 触控手册', hint: '?', run: () => ui.openHotkeys() },
  { id: 'about', icon: 'ⓘ', label: '关于 WorkCockpit', run: () => ui.openAbout() },
])

const query = ref('')
const cursor = ref(0)
const inputEl = ref<HTMLInputElement | null>(null)

const filtered = computed(() => {
  const q = query.value.trim().toLowerCase()
  if (!q) return commands.value
  return commands.value.filter((c) => c.label.toLowerCase().includes(q) || c.id.includes(q))
})

watch(() => props.open, async (o) => {
  if (o) { query.value = ''; cursor.value = 0; await nextTick(); inputEl.value?.focus() }
})
watch(filtered, () => { cursor.value = 0 })

function exec(c?: Cmd) {
  const cmd = c ?? filtered.value[cursor.value]
  if (!cmd) return
  emit('close')
  cmd.run()
}
function onKey(e: KeyboardEvent) {
  if (e.key === 'ArrowDown') { e.preventDefault(); cursor.value = Math.min(cursor.value + 1, filtered.value.length - 1) }
  else if (e.key === 'ArrowUp') { e.preventDefault(); cursor.value = Math.max(cursor.value - 1, 0) }
  else if (e.key === 'Enter') { e.preventDefault(); exec() }
  else if (e.key === 'Escape') { emit('close') }
}
</script>

<template>
  <transition name="cp">
    <div v-if="open" class="cp-backdrop" @click.self="emit('close')">
      <div class="cp-panel glass-strong" role="dialog">
        <div class="cp-input-row">
          <span class="cp-glyph">⌕</span>
          <input ref="inputEl" v-model="query" class="cp-input" placeholder="搜索页面与操作…" @keydown="onKey" />
          <kbd class="cp-esc">esc</kbd>
        </div>
        <ul class="cp-list">
          <li v-for="(c, i) in filtered" :key="c.id" class="cp-item" :class="{ active: i === cursor }"
              @mouseenter="cursor = i" @click="exec(c)">
            <span class="cp-item-icon">{{ c.icon }}</span>
            <span class="cp-item-label">{{ c.label }}</span>
            <kbd v-if="c.hint" class="cp-item-hint">{{ c.hint }}</kbd>
          </li>
          <li v-if="!filtered.length" class="cp-empty">无匹配项</li>
        </ul>
      </div>
    </div>
  </transition>
</template>

<style scoped>
.cp-backdrop {
  position: fixed; inset: 0; z-index: 100; display: flex; align-items: flex-start; justify-content: center;
  padding-top: 14vh; background: rgba(11, 18, 32, 0.40);
  backdrop-filter: blur(10px) saturate(160%); -webkit-backdrop-filter: blur(10px) saturate(160%);
}
.cp-panel { width: 600px; max-width: 92vw; border-radius: 16px; overflow: hidden; }
.cp-input-row { display: flex; align-items: center; gap: 10px; padding: 14px 18px; border-bottom: 1px solid var(--line-divider); }
.cp-glyph { font-size: 1.1rem; color: var(--ink-tertiary); }
.cp-input { flex: 1; border: none; background: transparent; outline: none; font-size: 0.95rem; color: var(--ink-primary); font-family: inherit; }
.cp-esc { font-family: 'JetBrains Mono Variable', monospace; font-size: 0.66rem; padding: 2px 6px; border: 1px solid var(--line-border); border-radius: 5px; color: var(--ink-muted); }
.cp-list { list-style: none; margin: 0; padding: 6px; max-height: 52vh; overflow-y: auto; }
.cp-item {
  display: flex; align-items: center; gap: 12px; padding: 10px 12px; border-radius: 10px; cursor: pointer;
  transition: background 0.12s var(--ease-out-quint);
}
.cp-item.active { background: linear-gradient(135deg, rgba(37,99,235,0.10), rgba(8,145,178,0.07)); }
.cp-item-icon { width: 22px; text-align: center; font-size: 1.0rem; }
.cp-item-label { flex: 1; font-size: 0.85rem; color: var(--ink-secondary); }
.cp-item.active .cp-item-label { color: var(--accent-blue); font-weight: 600; }
.cp-item-hint { font-family: 'JetBrains Mono Variable', monospace; font-size: 0.66rem; padding: 2px 6px; border: 1px solid var(--line-border); border-radius: 5px; color: var(--ink-muted); }
.cp-empty { padding: 18px; text-align: center; font-size: 0.8rem; color: var(--ink-muted); }
.cp-enter-from, .cp-leave-to { opacity: 0; }
.cp-enter-from .cp-panel, .cp-leave-to .cp-panel { transform: translateY(-10px) scale(0.97); }
.cp-enter-active, .cp-leave-active { transition: opacity 0.2s var(--ease-out-quint); }
.cp-enter-active .cp-panel, .cp-leave-active .cp-panel { transition: transform 0.2s var(--ease-out-quint); }
</style>
