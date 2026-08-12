<script setup lang="ts">
/**
 * SpotlightCursor — soft radial glow that follows the pointer.
 *
 * Desktop-only (auto-disables on coarse pointers via useInputMode).
 * Uses requestAnimationFrame lerp for buttery follow.
 * Hidden by reduce-motion or settings.cinematic=false.
 */
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useInputMode } from '@/composables/useInputMode'
import { useSettingsStore } from '@/stores/settings'

const { isKeyboard } = useInputMode()
const settings = useSettingsStore()

const el = ref<HTMLDivElement | null>(null)
const x = ref(-9999)
const y = ref(-9999)
let raf = 0
let cx = -9999, cy = -9999

function onMove(e: MouseEvent) {
  x.value = e.clientX
  y.value = e.clientY
}

function tick() {
  // lerp 0.18 — silky, not laggy
  cx += (x.value - cx) * 0.18
  cy += (y.value - cy) * 0.18
  if (el.value) {
    el.value.style.transform = `translate3d(${cx - 220}px, ${cy - 220}px, 0)`
  }
  raf = requestAnimationFrame(tick)
}

function attach() {
  window.addEventListener('mousemove', onMove, { passive: true })
  raf = requestAnimationFrame(tick)
}
function detach() {
  window.removeEventListener('mousemove', onMove)
  cancelAnimationFrame(raf)
}

onMounted(() => { if (isKeyboard.value && settings.cinematic !== false) attach() })
onBeforeUnmount(detach)
watch([isKeyboard, () => settings.cinematic], ([kb, cine]) => {
  detach()
  if (kb && cine !== false) attach()
})
</script>

<template>
  <Teleport to="body">
    <div
      v-if="isKeyboard && settings.cinematic !== false && !settings.reduceMotion"
      ref="el"
      class="spotlight"
      aria-hidden="true"
    ></div>
  </Teleport>
</template>

<style scoped>
.spotlight {
  position: fixed;
  top: 0; left: 0;
  width: 440px;
  height: 440px;
  border-radius: 50%;
  pointer-events: none;
  z-index: 1;
  background: radial-gradient(closest-side,
    rgba(37, 99, 235, 0.18) 0%,
    rgba(37, 99, 235, 0.10) 28%,
    rgba(37, 99, 235, 0.04) 55%,
    transparent 72%);
  mix-blend-mode: plus-lighter;
  will-change: transform;
  contain: strict;
  transition: opacity 0.4s var(--ease-out-quint);
}
[data-theme='dark'] .spotlight {
  background: radial-gradient(closest-side,
    rgba(59, 130, 246, 0.20) 0%,
    rgba(139, 92, 246, 0.12) 32%,
    rgba(59, 130, 246, 0.06) 58%,
    transparent 75%);
  mix-blend-mode: screen;
}
</style>
