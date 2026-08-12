import { defineStore } from 'pinia'
import { ref, watch } from 'vue'

export type Theme = 'light' | 'dark'
export type Density = 'comfortable' | 'compact'
export type Accent = 'blue' | 'teal' | 'emerald' | 'violet' | 'amber' | 'rose' | 'pink' | 'lime'
export type Preset = 'default' | 'compact-pad' | 'cinematic-show' | 'defense-stage'

const KEY = 'navcockpit.settings.v2'

interface Persisted {
  theme: Theme
  density: Density
  reduceMotion: boolean
  sound: boolean
  showFps: boolean
  /** master switch for premium animated effects (Aurora / SpotlightCursor / Bloom / particles) */
  cinematic: boolean
  /** primary accent color — drives buttons, KPI bars, hero rings */
  accent: Accent
}

/** Predefined accent palettes — applied as inline CSS vars on <html>. */
export const ACCENT_HEX: Record<Accent, { primary: string; secondary: string; label: string }> = {
  blue:    { primary: '#2563eb', secondary: '#0891b2', label: 'Cobalt' },
  teal:    { primary: '#0891b2', secondary: '#059669', label: 'Teal' },
  emerald: { primary: '#059669', secondary: '#0891b2', label: 'Emerald' },
  violet:  { primary: '#7c3aed', secondary: '#ec4899', label: 'Violet' },
  amber:   { primary: '#d97706', secondary: '#e11d48', label: 'Amber' },
  rose:    { primary: '#e11d48', secondary: '#7c3aed', label: 'Rose' },
  pink:    { primary: '#ec4899', secondary: '#a78bfa', label: 'Pink' },
  lime:    { primary: '#84cc16', secondary: '#10b981', label: 'Lime' },
}

const PRESETS: Record<Preset, Partial<Persisted>> = {
  'default':        { density: 'comfortable', cinematic: true,  reduceMotion: false, showFps: true,  theme: 'light' },
  'compact-pad':    { density: 'compact',     cinematic: false, reduceMotion: false, showFps: false, theme: 'light' },
  'cinematic-show': { density: 'comfortable', cinematic: true,  reduceMotion: false, showFps: false, theme: 'dark' },
  'defense-stage':  { density: 'comfortable', cinematic: true,  reduceMotion: false, showFps: false, theme: 'dark', accent: 'violet' },
}

function load(): Persisted {
  let base: Persisted = defaults
  try {
    const raw = localStorage.getItem(KEY)
    if (raw) base = { ...defaults, ...JSON.parse(raw) }
  } catch (_e) { /* ignore */ }
  // Optional URL escape hatch — handy for headless screenshots & sharing
  // permalinks. `?theme=dark` / `?density=compact` / `?cinematic=off` etc.
  if (typeof window !== 'undefined') {
    const params = new URLSearchParams(window.location.search)
    const t = params.get('theme')
    if (t === 'light' || t === 'dark') base = { ...base, theme: t }
    const d = params.get('density')
    if (d === 'compact' || d === 'comfortable') base = { ...base, density: d }
    const c = params.get('cinematic')
    if (c === 'on') base = { ...base, cinematic: true }
    else if (c === 'off') base = { ...base, cinematic: false }
  }
  return base
}

const defaults: Persisted = {
  theme: 'light',
  density: 'comfortable',
  reduceMotion: false,
  sound: false,
  showFps: true,
  cinematic: true,
  accent: 'blue',
}

export const useSettingsStore = defineStore('settings', () => {
  const initial = load()
  const theme = ref<Theme>(initial.theme)
  const density = ref<Density>(initial.density)
  const reduceMotion = ref<boolean>(initial.reduceMotion)
  const sound = ref<boolean>(initial.sound)
  const showFps = ref<boolean>(initial.showFps)
  const cinematic = ref<boolean>(initial.cinematic)
  const accent = ref<Accent>(initial.accent)

  function persist() {
    try {
      localStorage.setItem(KEY, JSON.stringify({
        theme: theme.value,
        density: density.value,
        reduceMotion: reduceMotion.value,
        sound: sound.value,
        showFps: showFps.value,
        cinematic: cinematic.value,
        accent: accent.value,
      }))
    } catch (_e) { /* quota exceeded etc. */ }
  }

  function applyToDom() {
    const root = document.documentElement
    root.dataset.theme = theme.value
    root.dataset.density = density.value
    root.dataset.reduceMotion = String(reduceMotion.value)
    root.dataset.cinematic = String(cinematic.value)
    root.dataset.accent = accent.value
    // override the two primary accent variables — propagates to every
    // .btn-primary / KPI bar / focus ring / topbar gradient.
    const { primary, secondary } = ACCENT_HEX[accent.value]
    root.style.setProperty('--accent-primary', primary)
    root.style.setProperty('--accent-secondary', secondary)
  }

  watch([theme, density, reduceMotion, sound, showFps, cinematic, accent], () => {
    persist()
    applyToDom()
  }, { flush: 'post' })

  function toggleTheme() { theme.value = theme.value === 'light' ? 'dark' : 'light' }
  function toggleSound() { sound.value = !sound.value }
  function toggleCinematic() { cinematic.value = !cinematic.value }
  function setAccent(a: Accent) { accent.value = a }
  function applyPreset(p: Preset) {
    const patch = PRESETS[p]
    if (patch.theme) theme.value = patch.theme
    if (patch.density) density.value = patch.density
    if (typeof patch.cinematic === 'boolean') cinematic.value = patch.cinematic
    if (typeof patch.reduceMotion === 'boolean') reduceMotion.value = patch.reduceMotion
    if (typeof patch.showFps === 'boolean') showFps.value = patch.showFps
    if (patch.accent) accent.value = patch.accent
  }

  // initial apply
  if (typeof document !== 'undefined') applyToDom()

  return { theme, density, reduceMotion, sound, showFps, cinematic, accent,
    toggleTheme, toggleSound, toggleCinematic, setAccent, applyPreset }
})
