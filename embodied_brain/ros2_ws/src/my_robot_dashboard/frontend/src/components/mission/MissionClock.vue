<script setup lang="ts">
/**
 * MissionClock — big 7-segment-style MET (Mission Elapsed Time)
 * display in the topbar. SpaceX Dragon / NASA MCC vibe.
 *
 * Reads from useMissionStore which derives MET from telemetry uptime.
 */
import { computed } from 'vue'
import { useMissionStore } from '@/stores/mission'

const mission = useMissionStore()

const parts = computed(() => {
  const [hh, mm, ss] = mission.metFormatted.split(':')
  return { hh, mm, ss }
})
</script>

<template>
  <div class="mc" title="Mission Elapsed Time">
    <span class="mc-tag">MET</span>
    <span v-for="(p, k) in parts" :key="String(k)" class="mc-seg">{{ p }}</span>
  </div>
</template>

<style scoped>
.mc {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 4px 10px;
  background: rgba(0, 0, 0, 0.06);
  border: 1px solid var(--line-divider);
  border-radius: 8px;
  font-family: 'JetBrains Mono Variable', ui-monospace, monospace;
  font-variant-numeric: tabular-nums;
}
[data-theme='dark'] .mc { background: rgba(0, 0, 0, 0.35); }
.mc-tag {
  font-size: 0.58rem;
  letter-spacing: 0.18em;
  color: var(--ink-muted);
  font-weight: 700;
  padding-right: 6px;
  border-right: 1px solid var(--line-divider);
  margin-right: 2px;
}
.mc-seg {
  font-size: 0.92rem;
  font-weight: 700;
  color: var(--accent-blue);
  letter-spacing: 0.02em;
  text-shadow: 0 0 12px rgba(37, 99, 235, 0.35);
  min-width: 1.8em;
  text-align: center;
}
.mc-seg + .mc-seg::before {
  content: ':';
  color: var(--ink-muted);
  margin: 0 -2px 0 -4px;
  text-shadow: none;
  animation: mc-blink 1s steps(2, start) infinite;
}
[data-theme='dark'] .mc-seg { color: #93c5fd; text-shadow: 0 0 14px rgba(96, 165, 250, 0.50); }

@keyframes mc-blink {
  to { opacity: 0.35; }
}
@media (prefers-reduced-motion: reduce) { .mc-seg + .mc-seg::before { animation: none; } }
[data-reduce-motion='true'] .mc-seg + .mc-seg::before { animation: none; }
</style>
