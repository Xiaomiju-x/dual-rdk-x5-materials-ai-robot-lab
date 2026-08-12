/**
 * useInputMode — detect whether the user's primary input is touch or mouse/keyboard.
 *
 * Uses CSS media queries `(pointer: coarse)` + `(hover: none)` which together
 * are the standard way to identify touch-only devices (phones, tablets) vs
 * laptops/desktops with cursor pointers. Also reacts to runtime input changes
 * (someone plugging in a BT keyboard mid-session).
 *
 *   const { isTouch, isKeyboard, hasHover } = useInputMode()
 */
import { onBeforeUnmount, onMounted, ref } from 'vue'

export function useInputMode() {
  const coarsePointer = ref(false)
  const noHover = ref(false)

  let mqlPointer: MediaQueryList | null = null
  let mqlHover: MediaQueryList | null = null
  const onPointer = () => { coarsePointer.value = mqlPointer?.matches ?? false }
  const onHover   = () => { noHover.value      = mqlHover?.matches ?? false }

  onMounted(() => {
    if (typeof window === 'undefined') return
    mqlPointer = window.matchMedia('(pointer: coarse)')
    mqlHover   = window.matchMedia('(hover: none)')
    coarsePointer.value = mqlPointer.matches
    noHover.value       = mqlHover.matches
    mqlPointer.addEventListener('change', onPointer)
    mqlHover.addEventListener('change', onHover)
    // URL escape hatch: ?input=touch / ?input=keyboard (handy for QA + kiosk lock)
    const forced = new URLSearchParams(window.location.search).get('input')
    if (forced === 'touch') {
      coarsePointer.value = true
      noHover.value = true
    } else if (forced === 'keyboard') {
      coarsePointer.value = false
      noHover.value = false
    }
  })
  onBeforeUnmount(() => {
    mqlPointer?.removeEventListener('change', onPointer)
    mqlHover?.removeEventListener('change', onHover)
  })

  // tablet/phone/kiosk pattern: coarse pointer + no hover
  // (a laptop with touchscreen still has a precise pointer)
  return {
    coarsePointer,
    noHover,
    isTouch: computedBoth(coarsePointer, noHover),
    isKeyboard: invertedComputed(coarsePointer, noHover),
    hasHover: invertedHover(noHover),
  }
}

// tiny local computed helpers (avoid importing vue's computed for clarity)
import { computed } from 'vue'
import type { Ref } from 'vue'

function computedBoth(a: Ref<boolean>, b: Ref<boolean>) {
  return computed(() => a.value && b.value)
}
function invertedComputed(a: Ref<boolean>, b: Ref<boolean>) {
  return computed(() => !(a.value && b.value))
}
function invertedHover(noHover: Ref<boolean>) {
  return computed(() => !noHover.value)
}
