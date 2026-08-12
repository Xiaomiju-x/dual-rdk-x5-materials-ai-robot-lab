<script setup lang="ts">
/**
 * GradientText — display text with a slow shimmering gradient sweep.
 *
 *   <GradientText tag="h1" class="text-3xl">NavCockpit</GradientText>
 */
interface Props {
  /** rendered HTML tag, default span */
  tag?: string
  /** preset palette */
  palette?: 'blue-violet' | 'teal-emerald' | 'amber-rose' | 'aurora'
}
const props = withDefaults(defineProps<Props>(), { tag: 'span', palette: 'blue-violet' })
</script>

<template>
  <component :is="props.tag" class="gtext" :class="`p-${props.palette}`">
    <slot />
  </component>
</template>

<style scoped>
.gtext {
  background-size: 200% 100%;
  background-clip: text;
  -webkit-background-clip: text;
  color: transparent;
  animation: sweep 8s ease-in-out infinite;
  display: inline-block;
  letter-spacing: -0.015em;
}
.p-blue-violet  { background-image: linear-gradient(110deg, #2563eb, #7c3aed 40%, #0891b2 70%, #2563eb); }
.p-teal-emerald { background-image: linear-gradient(110deg, #0891b2, #059669 50%, #10b981 80%, #0891b2); }
.p-amber-rose   { background-image: linear-gradient(110deg, #d97706, #e11d48 50%, #c026d3 80%, #d97706); }
.p-aurora       { background-image: linear-gradient(110deg, #2563eb, #06b6d4 25%, #8b5cf6 50%, #ec4899 75%, #2563eb); }

@keyframes sweep {
  0%   { background-position: 0% 50%; }
  50%  { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}
@media (prefers-reduced-motion: reduce) {
  .gtext { animation: none; background-position: 25% 50%; }
}
[data-reduce-motion='true'] .gtext { animation: none; background-position: 25% 50%; }
</style>
