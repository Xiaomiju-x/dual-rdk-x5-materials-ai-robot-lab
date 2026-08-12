<script setup lang="ts">
/**
 * AuroraBg — restrained animated gradient mesh as page background.
 *
 *   <AuroraBg />        // sits behind everything via teleport-to-body
 *
 * Inspired by Linear/Vercel hero gradients. Three radial-blob layers
 * drift on long ~28s loops, opacity ~0.35, blur 60px so primary
 * content reads cleanly. Theme-aware via CSS custom properties.
 *
 * Disables under prefers-reduced-motion + data-reduce-motion=true.
 */
import { useSettingsStore } from '@/stores/settings'
import { computed } from 'vue'

const settings = useSettingsStore()
const active = computed(() => settings.cinematic !== false && !settings.reduceMotion)
</script>

<template>
  <Teleport to="body">
    <div v-if="active" class="aurora" aria-hidden="true">
      <span class="blob blob-a"></span>
      <span class="blob blob-b"></span>
      <span class="blob blob-c"></span>
      <span class="blob blob-d"></span>
    </div>
  </Teleport>
</template>

<style scoped>
.aurora {
  position: fixed;
  inset: 0;
  z-index: 0;
  pointer-events: none;
  overflow: hidden;
  contain: strict;
}
.blob {
  position: absolute;
  width: 56vmax;
  height: 56vmax;
  border-radius: 50%;
  filter: blur(60px) saturate(140%);
  opacity: 0.35;
  mix-blend-mode: plus-lighter;
  will-change: transform, opacity;
}

/* light theme palette — premium SaaS gradient */
.blob-a { background: radial-gradient(closest-side, rgba(37, 99, 235, 0.55), rgba(37, 99, 235, 0) 70%); top: -18%; left: -10%; animation: drift-a 28s ease-in-out infinite; }
.blob-b { background: radial-gradient(closest-side, rgba(8, 145, 178, 0.50), rgba(8, 145, 178, 0) 70%); top: 30%; right: -18%; animation: drift-b 32s ease-in-out infinite; }
.blob-c { background: radial-gradient(closest-side, rgba(124, 58, 237, 0.45), rgba(124, 58, 237, 0) 70%); bottom: -22%; left: 22%; animation: drift-c 36s ease-in-out infinite; }
.blob-d { background: radial-gradient(closest-side, rgba(217, 119, 6, 0.30), rgba(217, 119, 6, 0) 70%); top: 14%; left: 38%; animation: drift-d 40s ease-in-out infinite; }

[data-theme='dark'] .blob { opacity: 0.55; mix-blend-mode: screen; filter: blur(70px) saturate(180%); }
[data-theme='dark'] .blob-a { background: radial-gradient(closest-side, rgba(59, 130, 246, 0.55), rgba(59, 130, 246, 0) 70%); }
[data-theme='dark'] .blob-b { background: radial-gradient(closest-side, rgba(14, 165, 233, 0.50), rgba(14, 165, 233, 0) 70%); }
[data-theme='dark'] .blob-c { background: radial-gradient(closest-side, rgba(139, 92, 246, 0.55), rgba(139, 92, 246, 0) 70%); }
[data-theme='dark'] .blob-d { background: radial-gradient(closest-side, rgba(244, 114, 182, 0.40), rgba(244, 114, 182, 0) 70%); }

@keyframes drift-a {
  0%, 100% { transform: translate3d(0, 0, 0) scale(1); }
  33% { transform: translate3d(8vw, 6vh, 0) scale(1.12); }
  66% { transform: translate3d(-4vw, 10vh, 0) scale(0.95); }
}
@keyframes drift-b {
  0%, 100% { transform: translate3d(0, 0, 0) scale(1); }
  40% { transform: translate3d(-10vw, -5vh, 0) scale(1.08); }
  75% { transform: translate3d(4vw, 8vh, 0) scale(0.92); }
}
@keyframes drift-c {
  0%, 100% { transform: translate3d(0, 0, 0) scale(1); }
  50% { transform: translate3d(12vw, -10vh, 0) scale(1.10); }
}
@keyframes drift-d {
  0%, 100% { transform: translate3d(0, 0, 0) scale(1) rotate(0deg); }
  50% { transform: translate3d(-8vw, 6vh, 0) scale(1.04) rotate(8deg); }
}

@media (prefers-reduced-motion: reduce) {
  .blob { animation: none !important; }
}
[data-reduce-motion='true'] .blob { animation: none !important; }
</style>
