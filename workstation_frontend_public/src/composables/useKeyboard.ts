/**
 * Global keyboard composable.
 *
 * Supports plain bindings (e.g. 'mod+k') and Vim-style chord bindings
 * (e.g. 'g c'). Plain bindings fire on keydown; chords wait up to 800 ms
 * for the second key after the leader.
 *
 *   useKeyboard({
 *     'mod+k': () => openPalette(),
 *     'g c': () => router.push('/'),
 *   })
 */
import { onBeforeUnmount, onMounted } from 'vue'

type Handler = (evt: KeyboardEvent) => void
type Bindings = Record<string, Handler>

const CHORD_WINDOW_MS = 900

function isTypingTarget(t: EventTarget | null): boolean {
  if (!t || !(t instanceof HTMLElement)) return false
  const tag = t.tagName
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || t.isContentEditable
}

function normalisePlain(spec: string): string {
  return spec
    .toLowerCase()
    .replace(/\s+/g, '')
    .replace('mod', /Mac/.test(navigator.platform) ? 'meta' : 'ctrl')
}

function eventToPlain(evt: KeyboardEvent): string {
  const parts: string[] = []
  if (evt.ctrlKey) parts.push('ctrl')
  if (evt.metaKey) parts.push('meta')
  if (evt.altKey) parts.push('alt')
  if (evt.shiftKey) parts.push('shift')
  const key = evt.key.toLowerCase()
  // skip modifier-only events
  if (['control', 'meta', 'alt', 'shift'].includes(key)) return ''
  parts.push(key)
  return parts.join('+')
}

export function useKeyboard(bindings: Bindings, opts: { allowInInputs?: boolean } = {}) {
  const plainMap = new Map<string, Handler>()
  const chordMap = new Map<string, Map<string, Handler>>()
  for (const [spec, fn] of Object.entries(bindings)) {
    const trimmed = spec.trim()
    if (trimmed.includes(' ')) {
      // chord: 'g c' → first='g', second='c'
      const [first, second] = trimmed.toLowerCase().split(/\s+/)
      if (!chordMap.has(first)) chordMap.set(first, new Map())
      chordMap.get(first)!.set(second, fn)
    } else {
      plainMap.set(normalisePlain(trimmed), fn)
    }
  }

  let chordLeader: string | null = null
  let chordTimer: number | null = null

  function clearChord() {
    chordLeader = null
    if (chordTimer !== null) {
      window.clearTimeout(chordTimer)
      chordTimer = null
    }
  }

  function onKey(evt: KeyboardEvent) {
    if (isTypingTarget(evt.target) && !opts.allowInInputs) return

    // first: try plain binding (with modifiers)
    const plainSpec = eventToPlain(evt)
    if (!plainSpec) return
    const plainFn = plainMap.get(plainSpec)
    if (plainFn) {
      evt.preventDefault()
      plainFn(evt)
      clearChord()
      return
    }

    // chord — only when no modifier (besides shift)
    if (evt.ctrlKey || evt.metaKey || evt.altKey) {
      clearChord()
      return
    }

    const key = evt.key.toLowerCase()
    if (chordLeader && chordMap.get(chordLeader)?.has(key)) {
      evt.preventDefault()
      chordMap.get(chordLeader)!.get(key)!(evt)
      clearChord()
      return
    }
    if (chordMap.has(key)) {
      chordLeader = key
      if (chordTimer !== null) window.clearTimeout(chordTimer)
      chordTimer = window.setTimeout(clearChord, CHORD_WINDOW_MS)
      return
    }
    clearChord()
  }

  onMounted(() => window.addEventListener('keydown', onKey))
  onBeforeUnmount(() => {
    window.removeEventListener('keydown', onKey)
    clearChord()
  })
}
