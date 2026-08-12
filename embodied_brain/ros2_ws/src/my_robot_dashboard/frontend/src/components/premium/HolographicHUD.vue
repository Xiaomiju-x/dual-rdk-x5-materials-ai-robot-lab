<script setup lang="ts">
/**
 * HolographicHUD — corner brackets + scanlines + optional crosshair
 * overlay. Drop on top of camera tiles, immersive 3D, or hero panels
 * to give a NASA-MCC / Iron Man HUD feel.
 *
 *   <HolographicHUD :crosshair="true" tone="ok">
 *     <video … />
 *   </HolographicHUD>
 */
interface Props {
  tone?: 'ok' | 'warn' | 'err' | 'info' | 'idle'
  crosshair?: boolean
  scanline?: boolean
  cornerSize?: number
  /** corner indent in % */
  inset?: number
}
const props = withDefaults(defineProps<Props>(), {
  tone: 'info',
  crosshair: false,
  scanline: true,
  cornerSize: 16,
  inset: 6,
})

const TONE_COLOR: Record<Required<Props>['tone'], string> = {
  ok: '#10b981',
  warn: '#f59e0b',
  err: '#ef4444',
  info: '#3b82f6',
  idle: '#94a3b8',
}
</script>

<template>
  <div class="hud" :style="{ '--hud-c': TONE_COLOR[props.tone], '--hud-cs': `${props.cornerSize}px`, '--hud-inset': `${props.inset}px` } as any">
    <slot />
    <svg class="hud-corners" aria-hidden="true">
      <g stroke="var(--hud-c)" stroke-width="1.5" fill="none">
        <!-- top-left -->
        <polyline :points="`${props.inset},${props.inset + props.cornerSize} ${props.inset},${props.inset} ${props.inset + props.cornerSize},${props.inset}`" />
        <!-- top-right -->
        <polyline :points="`calc(100% - ${props.inset + props.cornerSize}px),${props.inset} calc(100% - ${props.inset}px),${props.inset} calc(100% - ${props.inset}px),${props.inset + props.cornerSize}`" />
        <!-- bottom-left -->
        <polyline :points="`${props.inset},calc(100% - ${props.inset + props.cornerSize}px) ${props.inset},calc(100% - ${props.inset}px) ${props.inset + props.cornerSize},calc(100% - ${props.inset}px)`" />
        <!-- bottom-right -->
        <polyline :points="`calc(100% - ${props.inset + props.cornerSize}px),calc(100% - ${props.inset}px) calc(100% - ${props.inset}px),calc(100% - ${props.inset}px) calc(100% - ${props.inset}px),calc(100% - ${props.inset + props.cornerSize}px)`" />
      </g>
    </svg>
    <div v-if="props.scanline" class="hud-scan" aria-hidden="true"></div>
    <div v-if="props.crosshair" class="hud-cross" aria-hidden="true">
      <span class="cross-h"></span>
      <span class="cross-v"></span>
      <span class="cross-c"></span>
    </div>
  </div>
</template>

<style scoped>
.hud {
  position: relative;
  isolation: isolate;
  overflow: hidden;
}
.hud-corners {
  position: absolute; inset: 0;
  pointer-events: none;
  width: 100%; height: 100%;
  z-index: 3;
  filter: drop-shadow(0 0 4px var(--hud-c));
  opacity: 0.85;
}
.hud-scan {
  position: absolute; inset: 0;
  pointer-events: none;
  z-index: 2;
  background:
    repeating-linear-gradient(
      to bottom,
      transparent 0px,
      transparent 3px,
      rgba(255,255,255,0.025) 3px,
      rgba(255,255,255,0.025) 4px);
}
.hud-scan::after {
  content: '';
  position: absolute; left: 0; right: 0; height: 3px;
  background: linear-gradient(to bottom, transparent, var(--hud-c), transparent);
  opacity: 0.45;
  animation: scan 5.5s linear infinite;
}
@keyframes scan {
  0%   { top: 0%; }
  100% { top: 100%; }
}

.hud-cross {
  position: absolute; inset: 0; pointer-events: none; z-index: 3;
}
.cross-h, .cross-v, .cross-c {
  position: absolute;
  background: var(--hud-c);
  opacity: 0.5;
}
.cross-h { left: 32%; right: 32%; top: 50%; height: 1px; transform: translateY(-0.5px); }
.cross-v { top: 32%; bottom: 32%; left: 50%; width: 1px; transform: translateX(-0.5px); }
.cross-c { left: 50%; top: 50%; width: 8px; height: 8px; transform: translate(-50%, -50%); border: 1px solid var(--hud-c); background: transparent; border-radius: 50%; opacity: 0.8; }

@media (prefers-reduced-motion: reduce) { .hud-scan::after { display: none; } }
[data-reduce-motion='true'] .hud-scan::after { display: none; }
</style>
