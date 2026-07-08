import { defineStore } from 'pinia'
import { ref, watch } from 'vue'

export type Theme = 'light' | 'dark'
export type Density = 'comfortable' | 'compact'

const KEY = 'workcockpit.settings.v1'

interface Persisted {
  theme: Theme
  density: Density
  reduceMotion: boolean
  sound: boolean
  showFps: boolean
}

function load(): Persisted {
  let base: Persisted = defaults
  try {
    const raw = localStorage.getItem(KEY)
    if (raw) base = { ...defaults, ...JSON.parse(raw) }
  } catch (_e) { /* ignore */ }
  // Optional URL escape hatch — handy for headless screenshots & sharing
  // permalinks. `?theme=dark` / `?density=compact` etc.
  if (typeof window !== 'undefined') {
    const params = new URLSearchParams(window.location.search)
    const t = params.get('theme')
    if (t === 'light' || t === 'dark') base = { ...base, theme: t }
    const d = params.get('density')
    if (d === 'compact' || d === 'comfortable') base = { ...base, density: d }
  }
  return base
}

const defaults: Persisted = {
  theme: 'light',
  density: 'comfortable',
  reduceMotion: false,
  sound: false,
  showFps: true,
}

export const useSettingsStore = defineStore('settings', () => {
  const initial = load()
  const theme = ref<Theme>(initial.theme)
  const density = ref<Density>(initial.density)
  const reduceMotion = ref<boolean>(initial.reduceMotion)
  const sound = ref<boolean>(initial.sound)
  const showFps = ref<boolean>(initial.showFps)

  function persist() {
    try {
      localStorage.setItem(KEY, JSON.stringify({
        theme: theme.value,
        density: density.value,
        reduceMotion: reduceMotion.value,
        sound: sound.value,
        showFps: showFps.value,
      }))
    } catch (_e) { /* quota exceeded etc. */ }
  }

  function applyToDom() {
    const root = document.documentElement
    root.dataset.theme = theme.value
    root.dataset.density = density.value
    root.dataset.reduceMotion = String(reduceMotion.value)
  }

  watch([theme, density, reduceMotion, sound, showFps], () => {
    persist()
    applyToDom()
  }, { flush: 'post' })

  function toggleTheme() { theme.value = theme.value === 'light' ? 'dark' : 'light' }
  function toggleSound() { sound.value = !sound.value }

  // initial apply
  if (typeof document !== 'undefined') applyToDom()

  return { theme, density, reduceMotion, sound, showFps, toggleTheme, toggleSound }
})
