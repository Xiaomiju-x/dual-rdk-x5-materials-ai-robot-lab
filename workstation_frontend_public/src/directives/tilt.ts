// v-tilt — Linear / Vision Pro style parallax card tilt directive.
// Adds CSS vars for the inner glow follow + 3D rotate on pointer.
// Usage: <div v-tilt class="glass-card">…</div>   or  v-tilt="{ max: 6, scale: 1.01 }"
import type { Directive } from 'vue'

interface Opts { max?: number; scale?: number; speed?: number; glow?: boolean }
interface State { opts: Required<Opts>; onMove: (e: PointerEvent) => void; onLeave: () => void; onEnter: () => void; raf: number }

const DEFAULT: Required<Opts> = { max: 6, scale: 1.012, speed: 220, glow: true }
const STATE = new WeakMap<HTMLElement, State>()
const REDUCED = typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches

function isTouch(): boolean {
  return typeof window !== 'undefined' && window.matchMedia?.('(hover: none)').matches
}

function install(el: HTMLElement, raw: Opts | undefined) {
  if (REDUCED || isTouch()) return
  const opts: Required<Opts> = { ...DEFAULT, ...(raw ?? {}) }
  let targetRx = 0, targetRy = 0, targetGx = 50, targetGy = 50
  let rx = 0, ry = 0, gx = 50, gy = 50
  let active = false

  el.style.willChange = 'transform'
  el.style.transformStyle = 'preserve-3d'

  const onMove = (e: PointerEvent) => {
    if (e.pointerType === 'touch') return
    const r = el.getBoundingClientRect()
    const px = (e.clientX - r.left) / r.width
    const py = (e.clientY - r.top) / r.height
    targetRy = (px - 0.5) * opts.max * 2
    targetRx = (0.5 - py) * opts.max * 2
    targetGx = px * 100
    targetGy = py * 100
  }
  const onEnter = () => { active = true; raf() }
  const onLeave = () => { active = false; targetRx = 0; targetRy = 0; raf() }

  function raf() {
    state.raf = requestAnimationFrame(() => {
      const k = active ? 0.18 : 0.10
      rx += (targetRx - rx) * k
      ry += (targetRy - ry) * k
      gx += (targetGx - gx) * k
      gy += (targetGy - gy) * k
      const s = active ? opts.scale : 1
      el.style.transform = `perspective(900px) rotateX(${rx.toFixed(2)}deg) rotateY(${ry.toFixed(2)}deg) scale(${s})`
      if (opts.glow) {
        el.style.setProperty('--tilt-gx', `${gx}%`)
        el.style.setProperty('--tilt-gy', `${gy}%`)
      }
      if (Math.abs(rx - targetRx) > 0.05 || Math.abs(ry - targetRy) > 0.05 || active) raf()
    })
  }

  el.addEventListener('pointermove', onMove)
  el.addEventListener('pointerenter', onEnter)
  el.addEventListener('pointerleave', onLeave)
  const state: State = { opts, onMove, onLeave, onEnter, raf: 0 }
  STATE.set(el, state)
}

function uninstall(el: HTMLElement) {
  const s = STATE.get(el)
  if (!s) return
  if (s.raf) cancelAnimationFrame(s.raf)
  el.removeEventListener('pointermove', s.onMove)
  el.removeEventListener('pointerenter', s.onEnter)
  el.removeEventListener('pointerleave', s.onLeave)
  el.style.transform = ''
  el.style.willChange = ''
  STATE.delete(el)
}

export const vTilt: Directive<HTMLElement, Opts | undefined> = {
  mounted(el, binding) { install(el, binding.value) },
  updated(el, binding) {
    uninstall(el)
    install(el, binding.value)
  },
  unmounted(el) { uninstall(el) },
}
