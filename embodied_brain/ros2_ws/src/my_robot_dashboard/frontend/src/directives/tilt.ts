/**
 * v-tilt — parallax tilt directive (Apple/Linear style hover).
 *
 *   <div v-tilt class="card">…</div>
 *   <div v-tilt="{ max: 8, scale: 1.02 }">…</div>
 *
 * Desktop only; bypasses on coarse pointer.
 */
import type { Directive, DirectiveBinding } from 'vue'

interface TiltOpts {
  max?: number      // max rotation in degrees
  scale?: number    // hover scale
  speed?: number    // transition ms for snap-back
  reverse?: boolean // invert axis
}

interface TiltState {
  opts: Required<TiltOpts>
  onMove: (e: PointerEvent) => void
  onLeave: () => void
}

const isCoarse = () =>
  typeof window !== 'undefined' && window.matchMedia('(pointer: coarse)').matches

const defaults: Required<TiltOpts> = { max: 6, scale: 1.0, speed: 360, reverse: false }

function setup(el: HTMLElement, binding: DirectiveBinding<TiltOpts | undefined>): TiltState {
  const opts: Required<TiltOpts> = { ...defaults, ...(binding.value ?? {}) }
  el.style.transition = `transform ${opts.speed}ms cubic-bezier(0.22, 1, 0.36, 1)`
  el.style.transformStyle = 'preserve-3d'
  el.style.willChange = 'transform'

  const onMove = (e: PointerEvent) => {
    if (isCoarse()) return
    const r = el.getBoundingClientRect()
    const x = (e.clientX - r.left) / r.width
    const y = (e.clientY - r.top) / r.height
    const dx = (x - 0.5) * 2  // -1..1
    const dy = (y - 0.5) * 2
    const sign = opts.reverse ? -1 : 1
    const rotY = sign * dx * opts.max
    const rotX = sign * -dy * opts.max
    el.style.transform = `perspective(900px) rotateX(${rotX}deg) rotateY(${rotY}deg) scale(${opts.scale})`
  }
  const onLeave = () => {
    el.style.transform = `perspective(900px) rotateX(0) rotateY(0) scale(1)`
  }
  el.addEventListener('pointermove', onMove)
  el.addEventListener('pointerleave', onLeave)
  el.addEventListener('pointerdown', onLeave)
  return { opts, onMove, onLeave }
}

const stateMap = new WeakMap<HTMLElement, TiltState>()

export const vTilt: Directive<HTMLElement, TiltOpts | undefined> = {
  mounted(el, binding) {
    if (isCoarse()) return
    stateMap.set(el, setup(el, binding))
  },
  unmounted(el) {
    const s = stateMap.get(el)
    if (!s) return
    el.removeEventListener('pointermove', s.onMove)
    el.removeEventListener('pointerleave', s.onLeave)
    el.removeEventListener('pointerdown', s.onLeave)
    stateMap.delete(el)
  },
}
