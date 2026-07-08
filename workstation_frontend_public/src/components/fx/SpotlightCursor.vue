<script setup lang="ts">
// Vercel-style spotlight that follows the cursor. Plain CSS radial-gradient on a
// fixed overlay, GPU-cheap, fades when idle, disabled on touch / reduce-motion.
import { ref, onMounted, onBeforeUnmount, computed } from 'vue'
import { useSettingsStore } from '@/stores/settings'

const settings = useSettingsStore()
const x = ref(-300)
const y = ref(-300)
const visible = ref(false)
let raf = 0

const target = { x: -300, y: -300 }
let idleTimer: number | null = null

function onPointer(e: PointerEvent) {
  if (e.pointerType !== 'mouse') return
  target.x = e.clientX
  target.y = e.clientY
  visible.value = true
  if (idleTimer) clearTimeout(idleTimer)
  idleTimer = window.setTimeout(() => (visible.value = false), 1800)
}

function loop() {
  x.value += (target.x - x.value) * 0.18
  y.value += (target.y - y.value) * 0.18
  raf = requestAnimationFrame(loop)
}

const style = computed(() => ({
  background: `radial-gradient(380px circle at ${x.value}px ${y.value}px, var(--spot, rgba(37,99,235,0.10)), transparent 70%)`,
  opacity: visible.value ? 1 : 0,
}))

onMounted(() => {
  if (settings.reduceMotion) return
  window.addEventListener('pointermove', onPointer, { passive: true })
  raf = requestAnimationFrame(loop)
})
onBeforeUnmount(() => {
  window.removeEventListener('pointermove', onPointer)
  if (raf) cancelAnimationFrame(raf)
  if (idleTimer) clearTimeout(idleTimer)
})
</script>

<template>
  <div class="spotlight" :style="style" aria-hidden="true"></div>
</template>

<style scoped>
.spotlight {
  position: fixed;
  inset: 0;
  pointer-events: none;
  z-index: 0;
  transition: opacity .6s ease;
  mix-blend-mode: plus-lighter;
}
[data-theme='dark'] .spotlight { --spot: rgba(96, 165, 250, 0.16); }
</style>
