<script setup lang="ts">
/**
 * MagneticBtn — wrapper that gently attracts its child toward the cursor.
 *
 *   <MagneticBtn :strength="0.35">
 *     <button class="btn btn-primary">Dispatch</button>
 *   </MagneticBtn>
 *
 * Desktop only (skipped on coarse pointer).
 */
import { onBeforeUnmount, onMounted, ref } from 'vue'
import { useInputMode } from '@/composables/useInputMode'

interface Props {
  /** how much (0..1) of the cursor offset translates the element */
  strength?: number
  /** active radius in px around the element center */
  radius?: number
}
const props = withDefaults(defineProps<Props>(), { strength: 0.35, radius: 110 })
const { isKeyboard } = useInputMode()

const wrap = ref<HTMLDivElement | null>(null)
let tx = 0, ty = 0, raf = 0

function setTransform(x: number, y: number) {
  if (wrap.value) wrap.value.style.transform = `translate3d(${x}px, ${y}px, 0)`
}

function onMove(e: PointerEvent) {
  if (!wrap.value) return
  const r = wrap.value.getBoundingClientRect()
  const cx = r.left + r.width / 2
  const cy = r.top + r.height / 2
  const dx = e.clientX - cx
  const dy = e.clientY - cy
  const dist = Math.hypot(dx, dy)
  if (dist > props.radius) {
    tx = 0; ty = 0
  } else {
    tx = dx * props.strength
    ty = dy * props.strength
  }
  loop()
}
function onLeave() { tx = 0; ty = 0; loop() }

let cx2 = 0, cy2 = 0
function loop() {
  cancelAnimationFrame(raf)
  const step = () => {
    cx2 += (tx - cx2) * 0.22
    cy2 += (ty - cy2) * 0.22
    setTransform(cx2, cy2)
    if (Math.abs(tx - cx2) > 0.1 || Math.abs(ty - cy2) > 0.1) {
      raf = requestAnimationFrame(step)
    }
  }
  raf = requestAnimationFrame(step)
}

onMounted(() => {
  if (!isKeyboard.value) return
  window.addEventListener('pointermove', onMove, { passive: true })
  window.addEventListener('pointerout', onLeave, { passive: true })
})
onBeforeUnmount(() => {
  window.removeEventListener('pointermove', onMove)
  window.removeEventListener('pointerout', onLeave)
  cancelAnimationFrame(raf)
})
</script>

<template>
  <div ref="wrap" class="magnetic-wrap"><slot /></div>
</template>

<style scoped>
.magnetic-wrap {
  display: inline-flex;
  will-change: transform;
  transition: transform 0.18s var(--ease-out-quint);
}
</style>
