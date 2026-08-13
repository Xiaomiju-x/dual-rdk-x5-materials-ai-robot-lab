<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { useSettingsStore } from '@/stores/settings'
import { useToastStore } from '@/stores/toast'
import { useTelemetryStore } from '@/stores/telemetry'
import { useAudioFeedback } from '@/composables/useAudioFeedback'
import { useInputMode } from '@/composables/useInputMode'

interface Props {
  open: boolean
}
const props = defineProps<Props>()
const emit = defineEmits<{ (e: 'close'): void }>()

const router = useRouter()
const settings = useSettingsStore()
const toasts = useToastStore()
const telemetry = useTelemetryStore()
const audio = useAudioFeedback()
const { isTouch } = useInputMode()

const query = ref('')
const inputRef = ref<HTMLInputElement | null>(null)
const cursor = ref(0)

interface Action {
  id: string
  title: string
  hint?: string
  group: 'Navigate' | 'Toggle' | 'Action'
  glyph: string
  run: () => void
}

const actions = computed<Action[]>(() => {
  const navItems = router.getRoutes()
    .filter((r) => r.meta?.title)
    .map<Action>((r) => ({
      id: `nav-${r.path}`,
      title: r.meta.title as string,
      hint: r.path,
      group: 'Navigate',
      glyph: (r.meta.icon as string) ?? '→',
      run: () => router.push(r.path),
    }))

  const toggles: Action[] = [
    {
      id: 'theme',
      title: settings.theme === 'light' ? 'Switch to Dark Theme' : 'Switch to Light Theme',
      hint: settings.theme,
      group: 'Toggle',
      glyph: settings.theme === 'light' ? '🌙' : '☀',
      run: () => settings.toggleTheme(),
    },
    {
      id: 'sound',
      title: settings.sound ? 'Mute Audio Feedback' : 'Enable Audio Feedback',
      hint: settings.sound ? 'on' : 'off',
      group: 'Toggle',
      glyph: settings.sound ? '🔊' : '🔇',
      run: () => settings.toggleSound(),
    },
    {
      id: 'density',
      title: `Density: ${settings.density === 'comfortable' ? 'Comfortable → Compact' : 'Compact → Comfortable'}`,
      hint: settings.density,
      group: 'Toggle',
      glyph: '☰',
      run: () => { settings.density = settings.density === 'comfortable' ? 'compact' : 'comfortable' },
    },
    {
      id: 'reduce-motion',
      title: settings.reduceMotion ? 'Restore Animations' : 'Reduce Motion',
      hint: settings.reduceMotion ? 'on' : 'off',
      group: 'Toggle',
      glyph: '⥁',
      run: () => { settings.reduceMotion = !settings.reduceMotion },
    },
  ]

  const acts: Action[] = [
    {
      id: 'reconnect',
      title: 'Reconnect WebSocket',
      hint: telemetry.state,
      group: 'Action',
      glyph: '↻',
      run: () => {
        telemetry.disconnect()
        setTimeout(() => telemetry.connect(), 80)
        toasts.push({ tone: 'info', title: 'Reconnecting WebSocket', detail: 'forcing fresh /ws/telemetry handshake' })
      },
    },
    {
      id: 'toast-demo',
      title: 'Trigger Demo Toast',
      hint: 'show example notification',
      group: 'Action',
      glyph: '★',
      run: () => toasts.push({ tone: 'ok', title: 'Hello from Command Palette', detail: '⌘K → 任何动作一秒触发' }),
    },
    {
      id: 'clear-toasts',
      title: 'Dismiss All Toasts',
      hint: `${toasts.toasts.length} active`,
      group: 'Action',
      glyph: '✕',
      run: () => toasts.clear(),
    },
  ]

  return [...navItems, ...toggles, ...acts]
})

const filtered = computed<Action[]>(() => {
  const q = query.value.trim().toLowerCase()
  if (!q) return actions.value
  return actions.value.filter((a) =>
    a.title.toLowerCase().includes(q) || a.hint?.toLowerCase().includes(q) || a.id.includes(q),
  )
})

const grouped = computed(() => {
  const out: Array<{ group: string; items: Action[] }> = []
  for (const a of filtered.value) {
    let g = out.find((x) => x.group === a.group)
    if (!g) { g = { group: a.group, items: [] }; out.push(g) }
    g.items.push(a)
  }
  return out
})

const flatItems = computed(() => grouped.value.flatMap((g) => g.items))

watch(() => props.open, async (o) => {
  if (o) {
    query.value = ''
    cursor.value = 0
    await nextTick()
    inputRef.value?.focus()
  }
})

watch(filtered, () => { cursor.value = 0 })

function move(delta: number) {
  const n = flatItems.value.length
  if (n === 0) return
  cursor.value = (cursor.value + delta + n) % n
}

function run(a: Action) {
  a.run()
  audio.play('navigate')
  emit('close')
}

function onKey(evt: KeyboardEvent) {
  if (evt.key === 'Escape') { evt.preventDefault(); emit('close'); return }
  if (evt.key === 'ArrowDown') { evt.preventDefault(); move(1); return }
  if (evt.key === 'ArrowUp') { evt.preventDefault(); move(-1); return }
  if (evt.key === 'Enter') {
    evt.preventDefault()
    const a = flatItems.value[cursor.value]
    if (a) run(a)
    return
  }
}

function backdropClick(evt: MouseEvent) {
  if ((evt.target as HTMLElement).classList.contains('palette-backdrop')) emit('close')
}
</script>

<template>
  <transition name="palette">
    <div v-if="open" class="palette-backdrop" @click="backdropClick">
      <div class="palette glass-strong" role="dialog" aria-label="Command palette">
        <div class="palette-search">
          <span class="search-icon">{{ isTouch ? '⌕' : '⌘' }}</span>
          <input
            ref="inputRef"
            v-model="query"
            class="search-input"
            placeholder="搜索页面 · 切换主题 · 触发动作…"
            spellcheck="false"
            autocomplete="off"
            @keydown="onKey"
          />
          <button v-if="isTouch" class="search-close" aria-label="close" @click="emit('close')">×</button>
          <span v-else class="search-hint mono">esc</span>
        </div>
        <div class="palette-body">
          <div v-for="g in grouped" :key="g.group" class="group">
            <div class="group-label section-label">{{ g.group }}</div>
            <button
              v-for="a in g.items"
              :key="a.id"
              class="row"
              :class="{ active: flatItems[cursor]?.id === a.id }"
              @mouseenter="cursor = flatItems.findIndex((x) => x.id === a.id)"
              @click="run(a)"
            >
              <span class="row-glyph">{{ a.glyph }}</span>
              <span class="row-title">{{ a.title }}</span>
              <span v-if="a.hint" class="row-hint mono">{{ a.hint }}</span>
            </button>
          </div>
          <div v-if="!filtered.length" class="empty mono">未找到匹配项</div>
        </div>
        <div class="palette-foot mono">
          <template v-if="isTouch">
            <span>点条目执行</span>
            <span>点空白关闭</span>
          </template>
          <template v-else>
            <span>↑↓ 选择</span>
            <span>↵ 执行</span>
            <span>esc 关闭</span>
          </template>
          <span class="foot-spacer"></span>
          <span>{{ filtered.length }} / {{ actions.length }}</span>
        </div>
      </div>
    </div>
  </transition>
</template>

<style scoped>
.palette-backdrop {
  position: fixed; inset: 0;
  background: rgba(11, 18, 32, 0.35);
  backdrop-filter: blur(8px) saturate(150%);
  -webkit-backdrop-filter: blur(8px) saturate(150%);
  z-index: 100;
  display: flex; align-items: flex-start; justify-content: center;
  padding-top: 14vh;
}
.palette {
  width: 640px;
  max-width: calc(100vw - 32px);
  border-radius: 16px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  max-height: 70vh;
}
.palette-search {
  display: flex; align-items: center; gap: 10px;
  padding: 14px 16px;
  border-bottom: 1px solid var(--line-divider);
}
.search-icon { font-size: 1.1rem; color: var(--ink-tertiary); width: 22px; text-align: center; }
.search-input {
  flex: 1;
  border: none; outline: none;
  background: transparent;
  font-size: 0.96rem;
  color: var(--ink-primary);
  font-family: inherit;
}
.search-input::placeholder { color: var(--ink-muted); }
.search-hint {
  font-size: 0.66rem; color: var(--ink-muted);
  padding: 2px 6px;
  border: 1px solid var(--line-border);
  border-radius: 4px;
}
.search-close {
  width: 32px; height: 32px;
  display: flex; align-items: center; justify-content: center;
  background: transparent; border: none;
  font-size: 1.4rem;
  color: var(--ink-tertiary);
  border-radius: 8px;
  cursor: pointer;
}
.search-close:hover { background: rgba(15, 23, 42, 0.06); color: var(--ink-primary); }

.palette-body {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  padding: 8px 10px 12px;
}
.group { margin-bottom: 6px; }
.group-label { padding: 8px 8px 6px; color: var(--ink-muted); }

.row {
  display: grid;
  grid-template-columns: 28px 1fr auto;
  align-items: center;
  gap: 12px;
  width: 100%;
  padding: 8px 10px;
  border: none;
  background: transparent;
  border-radius: 8px;
  cursor: pointer;
  text-align: left;
  font-size: 0.84rem;
  color: var(--ink-primary);
  transition: background 0.12s var(--ease-out-quint);
}
.row:hover, .row.active {
  background: linear-gradient(135deg, rgba(37, 99, 235, 0.08), rgba(8, 145, 178, 0.06));
}
.row.active { box-shadow: inset 0 0 0 1px rgba(37, 99, 235, 0.15); }
.row-glyph { font-size: 1rem; color: var(--ink-tertiary); text-align: center; }
.row.active .row-glyph { color: var(--accent-blue); }
.row-title { font-weight: 500; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.row-hint { font-size: 0.7rem; color: var(--ink-muted); }

.empty { padding: 24px; text-align: center; color: var(--ink-muted); font-size: 0.78rem; }

.palette-foot {
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 8px 16px;
  border-top: 1px solid var(--line-divider);
  font-size: 0.66rem;
  color: var(--ink-muted);
}
.foot-spacer { flex: 1; }

.palette-enter-from { opacity: 0; }
.palette-enter-from .palette { transform: translateY(-10px) scale(0.96); }
.palette-leave-to { opacity: 0; }
.palette-leave-to .palette { transform: translateY(-6px) scale(0.98); }
.palette-enter-active, .palette-leave-active { transition: opacity 0.22s var(--ease-out-quint); }
.palette-enter-active .palette,
.palette-leave-active .palette { transition: transform 0.22s var(--ease-out-quint); }
</style>
