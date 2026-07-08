<script setup lang="ts">
// BorderBeam — animated gradient ring traveling along the parent border.
// Mount inside a position: relative element; positions absolute.
// Inspired by magic-ui's border-beam — pure CSS conic gradient + spin animation.
withDefaults(defineProps<{
  size?: number; duration?: number; colorFrom?: string; colorTo?: string
}>(), {
  size: 220,
  duration: 9,
  colorFrom: 'rgba(37,99,235,0.9)',
  colorTo: 'rgba(8,145,178,0.0)',
})
</script>

<template>
  <span class="border-beam" :style="{
    '--beam-size': size + 'px',
    '--beam-dur': duration + 's',
    '--beam-from': colorFrom,
    '--beam-to': colorTo,
  } as any" aria-hidden="true"></span>
</template>

<style scoped>
.border-beam {
  position: absolute;
  inset: 0;
  border-radius: inherit;
  pointer-events: none;
  overflow: hidden;
  z-index: 1;
}
.border-beam::before {
  content: '';
  position: absolute;
  inset: -1px;
  border-radius: inherit;
  padding: 1.5px;
  background: conic-gradient(from var(--beam-angle, 0deg),
    transparent 0%,
    var(--beam-from) 8%,
    var(--beam-to) 18%,
    transparent 30%,
    transparent 100%);
  -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
  mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
  -webkit-mask-composite: xor;
  mask-composite: exclude;
  animation: beam-spin var(--beam-dur) linear infinite;
}
@property --beam-angle {
  syntax: '<angle>';
  inherits: false;
  initial-value: 0deg;
}
@keyframes beam-spin {
  to { --beam-angle: 360deg; }
}
@media (prefers-reduced-motion: reduce) {
  .border-beam::before { animation: none; }
}
</style>
