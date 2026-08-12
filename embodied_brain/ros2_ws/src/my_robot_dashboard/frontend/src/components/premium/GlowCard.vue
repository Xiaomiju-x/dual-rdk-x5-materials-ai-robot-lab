<script setup lang="ts">
/**
 * GlowCard — a card-like wrapper that lights a soft radial under
 * the cursor as it hovers. Reads --gc-x / --gc-y CSS vars set on
 * pointermove. Desktop only (no-op on touch).
 *
 *   <GlowCard>
 *     <div class="card-elevated p-4">…</div>
 *   </GlowCard>
 */
import { ref } from 'vue'
import { useInputMode } from '@/composables/useInputMode'

interface Props {
  /** spotlight color (RGBA) */
  color?: string
  /** spotlight intensity 0..1 */
  intensity?: number
  /** radius in px */
  size?: number
}
const props = withDefaults(defineProps<Props>(), {
  color: 'rgba(37, 99, 235, 0.18)',
  intensity: 1,
  size: 360,
})
const { isKeyboard } = useInputMode()
const el = ref<HTMLDivElement | null>(null)

function onMove(e: PointerEvent) {
  if (!el.value || !isKeyboard.value) return
  const r = el.value.getBoundingClientRect()
  const x = ((e.clientX - r.left) / r.width) * 100
  const y = ((e.clientY - r.top) / r.height) * 100
  el.value.style.setProperty('--gc-x', `${x}%`)
  el.value.style.setProperty('--gc-y', `${y}%`)
  el.value.style.setProperty('--gc-opacity', '1')
}
function onLeave() {
  if (!el.value) return
  el.value.style.setProperty('--gc-opacity', '0')
}
</script>

<template>
  <div
    ref="el"
    class="glow-card"
    :style="{
      '--gc-color': props.color,
      '--gc-size': `${props.size}px`,
      '--gc-intensity': props.intensity,
    } as any"
    @pointermove="onMove"
    @pointerleave="onLeave"
  >
    <slot />
    <span class="glow-layer" aria-hidden="true"></span>
  </div>
</template>

<style scoped>
.glow-card {
  position: relative;
  isolation: isolate;
}
.glow-layer {
  position: absolute;
  inset: 0;
  border-radius: inherit;
  pointer-events: none;
  opacity: 0;
  transition: opacity 0.32s var(--ease-out-quint);
  background: radial-gradient(var(--gc-size, 360px) circle at var(--gc-x, 50%) var(--gc-y, 50%),
    var(--gc-color, rgba(37, 99, 235, 0.18)) 0%,
    transparent 65%);
  opacity: var(--gc-opacity, 0);
  mix-blend-mode: plus-lighter;
  z-index: 1;
}
[data-theme='dark'] .glow-layer { mix-blend-mode: screen; }

@media (pointer: coarse) { .glow-layer { display: none; } }
</style>
