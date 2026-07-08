<script setup lang="ts">
// GlowCard — premium 3-layer glass card with cursor-following glow + optional tilt.
// Combines the v-tilt directive (sets --tilt-gx/gy) with a soft radial glow that
// follows the cursor on hover (Vision Pro style).
import { vTilt } from '@/directives/tilt'
withDefaults(defineProps<{
  tilt?: boolean
  glow?: boolean
  accent?: 'blue' | 'teal' | 'emerald' | 'violet' | 'amber' | 'rose'
  beam?: boolean
}>(), { tilt: true, glow: true, accent: 'blue', beam: false })
const ACCENT_RGB: Record<string, string> = {
  blue: '37, 99, 235',
  teal: '8, 145, 178',
  emerald: '5, 150, 105',
  violet: '124, 58, 237',
  amber: '217, 119, 6',
  rose: '225, 29, 72',
}
</script>

<template>
  <div
    v-tilt="tilt ? { max: 5, scale: 1.008 } : undefined"
    class="glow-card"
    :class="{ 'gc-glow': glow }"
    :style="{ '--accent-rgb': ACCENT_RGB[accent] } as any"
  >
    <span class="gc-noise" aria-hidden="true"></span>
    <span v-if="glow" class="gc-hover-glow" aria-hidden="true"></span>
    <span v-if="beam" class="gc-beam-host">
      <slot name="beam">
        <!-- consumer can provide their own <BorderBeam/> -->
      </slot>
    </span>
    <span class="gc-edge" aria-hidden="true"></span>
    <div class="gc-content"><slot /></div>
  </div>
</template>

<style scoped>
.glow-card {
  position: relative;
  border-radius: 18px;
  background: var(--bg-card, #ffffff);
  border: 1px solid var(--line-border, #e2e8f0);
  box-shadow: var(--shadow-card);
  overflow: hidden;
  isolation: isolate;
  transition: box-shadow .35s var(--ease-out-quint), border-color .35s ease, background .35s ease;
}
.glow-card:hover {
  box-shadow: var(--shadow-elevated), 0 0 0 1px rgba(var(--accent-rgb), 0.15);
}
.gc-content {
  position: relative;
  z-index: 3;
  height: 100%;
}
/* subtle filmgrain for premium texture */
.gc-noise {
  position: absolute; inset: 0; z-index: 0;
  pointer-events: none;
  opacity: .03;
  background-image: repeating-conic-gradient(#000 0 .3deg, transparent .3deg .6deg);
  mix-blend-mode: multiply;
}
[data-theme='dark'] .gc-noise { mix-blend-mode: screen; opacity: .04; }
/* cursor-following inner glow (uses --tilt-gx/gy from v-tilt) */
.gc-hover-glow {
  position: absolute; inset: 0; z-index: 1;
  pointer-events: none;
  opacity: 0;
  transition: opacity .4s ease;
  background: radial-gradient(280px circle at var(--tilt-gx, 50%) var(--tilt-gy, 50%),
    rgba(var(--accent-rgb), 0.18), transparent 65%);
}
.glow-card:hover .gc-hover-glow { opacity: 1; }
/* edge highlight (top + left subtle inner shine) */
.gc-edge {
  position: absolute; inset: 0; z-index: 2;
  pointer-events: none;
  border-radius: inherit;
  background: linear-gradient(135deg, rgba(255,255,255,.5) 0%, transparent 25%);
  mix-blend-mode: overlay;
  opacity: .6;
}
[data-theme='dark'] .gc-edge {
  background: linear-gradient(135deg, rgba(255,255,255,.08) 0%, transparent 30%);
  opacity: 1;
}
.gc-beam-host {
  position: absolute; inset: 0; z-index: 2;
  border-radius: inherit;
  pointer-events: none;
}
</style>
