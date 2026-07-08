<script setup lang="ts">
// Magnetic button — Linear / Awwwards classic. Cursor within `radius` exerts an
// attractive force on the button (capped by `pull`). Disabled on touch/reduce-motion.
import { ref, onMounted, onBeforeUnmount } from 'vue'
import { useSettingsStore } from '@/stores/settings'

const props = withDefaults(defineProps<{
  radius?: number; pull?: number; as?: 'button' | 'a'
}>(), { radius: 120, pull: 14, as: 'button' })
const emit = defineEmits<{ (e: 'click', ev: MouseEvent): void }>()

const settings = useSettingsStore()
const root = ref<HTMLElement>()
const tx = ref(0); const ty = ref(0)
let raf = 0
let targetX = 0, targetY = 0
const isTouch = typeof window !== 'undefined' && window.matchMedia?.('(hover: none)').matches

function onMove(e: PointerEvent) {
  if (!root.value || e.pointerType === 'touch') return
  const r = root.value.getBoundingClientRect()
  const cx = r.left + r.width / 2
  const cy = r.top + r.height / 2
  const dx = e.clientX - cx
  const dy = e.clientY - cy
  const d = Math.hypot(dx, dy)
  if (d > props.radius) { targetX = 0; targetY = 0; return }
  const k = (1 - d / props.radius) * props.pull / Math.max(d, 1)
  targetX = dx * k
  targetY = dy * k
}
function loop() {
  tx.value += (targetX - tx.value) * 0.18
  ty.value += (targetY - ty.value) * 0.18
  raf = requestAnimationFrame(loop)
}
function onLeave() { targetX = 0; targetY = 0 }

onMounted(() => {
  if (settings.reduceMotion || isTouch) return
  window.addEventListener('pointermove', onMove, { passive: true })
  root.value?.addEventListener('pointerleave', onLeave)
  raf = requestAnimationFrame(loop)
})
onBeforeUnmount(() => {
  if (raf) cancelAnimationFrame(raf)
  window.removeEventListener('pointermove', onMove)
  root.value?.removeEventListener('pointerleave', onLeave)
})
</script>

<template>
  <component
    :is="as"
    ref="root"
    class="magnetic-btn"
    :style="{ transform: `translate3d(${tx}px, ${ty}px, 0)` }"
    @click="(e: MouseEvent) => emit('click', e)"
  >
    <span class="mb-content"><slot /></span>
    <span class="mb-ripple" aria-hidden="true"></span>
  </component>
</template>

<style scoped>
.magnetic-btn {
  position: relative;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 10px 18px;
  border-radius: 12px;
  border: 1px solid var(--line-border, #e2e8f0);
  background: var(--bg-glass, rgba(255,255,255,.7));
  color: var(--ink-primary, #0b1220);
  font: inherit;
  font-weight: 600;
  font-size: 14px;
  cursor: pointer;
  overflow: hidden;
  isolation: isolate;
  transition: background .25s ease, border-color .25s ease, box-shadow .25s ease;
  backdrop-filter: blur(10px) saturate(180%);
  -webkit-backdrop-filter: blur(10px) saturate(180%);
}
.magnetic-btn:hover {
  border-color: var(--accent-blue, #2563eb);
  box-shadow: 0 8px 24px -8px rgba(37,99,235,.35), 0 0 0 1px rgba(37,99,235,.4) inset;
}
.magnetic-btn:active { transform: translate3d(0,0,0) scale(.985) !important; }
.mb-content { position: relative; z-index: 2; }
.mb-ripple {
  position: absolute; inset: 0; z-index: 1;
  background: radial-gradient(160px circle at var(--mb-x, 50%) var(--mb-y, 50%), rgba(37,99,235,.22), transparent 65%);
  opacity: 0; transition: opacity .35s ease;
}
.magnetic-btn:hover .mb-ripple { opacity: 1; }
[data-theme='dark'] .magnetic-btn {
  background: rgba(20,25,38,.6);
  border-color: rgba(148,163,184,.18);
}
[data-theme='dark'] .magnetic-btn:hover {
  border-color: #60a5fa;
  box-shadow: 0 8px 28px -8px rgba(96,165,250,.55), 0 0 0 1px rgba(96,165,250,.45) inset;
}
</style>
