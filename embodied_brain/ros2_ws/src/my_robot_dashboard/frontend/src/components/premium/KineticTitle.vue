<script setup lang="ts">
/**
 * KineticTitle — letter-by-letter stagger reveal for page titles.
 *
 *   <KineticTitle text="Cockpit · 主驾驶舱" />
 *   <KineticTitle text="Cockpit · 主驾驶舱" gradient="aurora" />
 *
 * The optional `gradient` prop bakes a premium animated gradient
 * directly into every character — this is needed because nesting
 * inside <GradientText> doesn't work: each char span has its own
 * transform (creates a stacking context), so the parent's
 * background-clip: text can't reach it and the chars render as
 * `color: transparent` → invisible. Baking the gradient per-char
 * keeps them visible AND staggered.
 */
import { computed } from 'vue'

type Palette = 'blue-violet' | 'teal-emerald' | 'amber-rose' | 'aurora'

interface Props {
  text: string
  /** ms between adjacent letters */
  stagger?: number
  /** duration per letter (ms) */
  duration?: number
  /** initial Y offset (px) */
  fromY?: number
  /** optional gradient — applied per character so it survives transforms */
  gradient?: Palette
}
const props = withDefaults(defineProps<Props>(), {
  stagger: 28,
  duration: 720,
  fromY: 14,
  gradient: undefined,
})

const chars = computed(() => Array.from(props.text))
</script>

<template>
  <span class="kt" :class="props.gradient ? `kt-grad-${props.gradient}` : null" :key="props.text">
    <span
      v-for="(c, i) in chars"
      :key="i + c"
      class="kt-ch"
      :style="{
        animationDelay: `${i * props.stagger}ms`,
        animationDuration: `${props.duration}ms`,
        '--y': `${props.fromY}px`,
      } as any"
    >{{ c === ' ' ? ' ' : c }}</span>
  </span>
</template>

<style scoped>
.kt { display: inline-flex; flex-wrap: wrap; line-height: 1.15; }
.kt-ch {
  display: inline-block;
  opacity: 0;
  transform: translateY(var(--y, 12px));
  filter: blur(4px);
  animation-name: kt-rise;
  animation-fill-mode: forwards;
  animation-timing-function: cubic-bezier(0.34, 1.56, 0.64, 1);
  will-change: opacity, transform, filter;
}
@keyframes kt-rise {
  to { opacity: 1; transform: translateY(0); filter: blur(0); }
}

/* per-char gradient — survives transforms (cf. block comment above) */
.kt-grad-blue-violet  .kt-ch { background-image: linear-gradient(180deg, #2563eb 0%, #7c3aed 100%); }
.kt-grad-teal-emerald .kt-ch { background-image: linear-gradient(180deg, #0891b2 0%, #059669 100%); }
.kt-grad-amber-rose   .kt-ch { background-image: linear-gradient(180deg, #d97706 0%, #e11d48 100%); }
.kt-grad-aurora       .kt-ch { background-image: linear-gradient(180deg, #2563eb 0%, #7c3aed 50%, #06b6d4 100%); }
.kt-grad-blue-violet  .kt-ch,
.kt-grad-teal-emerald .kt-ch,
.kt-grad-amber-rose   .kt-ch,
.kt-grad-aurora       .kt-ch {
  -webkit-background-clip: text;
          background-clip: text;
  color: transparent;
}

@media (prefers-reduced-motion: reduce) {
  .kt-ch { opacity: 1; transform: none; filter: none; animation: none; }
}
[data-reduce-motion='true'] .kt-ch { opacity: 1; transform: none; filter: none; animation: none; }
</style>
