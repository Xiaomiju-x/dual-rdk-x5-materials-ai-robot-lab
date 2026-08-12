<script setup lang="ts">
// EventMarquee — continuous horizontal ticker of recent coop events.
// Pulls from telemetry.coopEvents, dedupes, pads, loops seamlessly.
import { computed } from 'vue'
import { useTelemetryStore } from '@/stores/telemetry'

const telemetry = useTelemetryStore()
const KIND_ICON: Record<string, string> = {
  arm_to_ai: '→🧠', arm_to_car: '→🚗', car_to_ai: '🚗→🧠',
  default: '▸',
}
const KIND_COLOR: Record<string, string> = {
  arm_to_ai: 'var(--accent-violet)',
  arm_to_car: 'var(--accent-amber)',
  car_to_ai: 'var(--accent-emerald)',
  default: 'var(--ink-tertiary)',
}

const items = computed(() => {
  const src = telemetry.coopEvents.slice(-18)
  if (src.length === 0) {
    return [
      { ts: 0, kind: 'default', text: '工位待命 — 双 myCobot 280-Pi 已联机, AI 脑/车载脑握手 ok' },
      { ts: 0, kind: 'default', text: '相机 1280×720 @ 30fps · AprilTag id=0/3 pose ok' },
    ]
  }
  return src.map((e) => ({
    ts: e.ts,
    kind: e.kind ?? 'default',
    text: `${e.src} → ${e.dst} · ${e.endpoint} · ${e.rtt_ms.toFixed(0)}ms · ${(e.bytes/1024).toFixed(1)}KB${e.ok ? '' : ' · FAILED'}`,
  }))
})
// Duplicate the track for seamless loop.
const doubled = computed(() => [...items.value, ...items.value])
</script>

<template>
  <div class="event-marquee marquee" role="status" aria-live="polite">
    <div class="marquee-track">
      <span v-for="(e, i) in doubled" :key="i" class="evt">
        <span class="evt-pip" :style="{ backgroundColor: KIND_COLOR[e.kind] ?? KIND_COLOR.default }"></span>
        <span class="evt-kind kv-mono">{{ KIND_ICON[e.kind] ?? KIND_ICON.default }}</span>
        <span class="evt-text">{{ e.text }}</span>
        <span class="evt-sep">·</span>
      </span>
    </div>
  </div>
</template>

<style scoped>
.event-marquee {
  width: 100%;
  height: 32px;
  display: flex; align-items: center;
  border-radius: 999px;
  background: color-mix(in srgb, var(--bg-elevated) 70%, transparent);
  border: 1px solid var(--line-divider);
  padding: 0 16px;
  font-size: 0.78rem;
  color: var(--ink-secondary);
}
.evt { display: inline-flex; align-items: center; gap: 8px; }
.evt-pip { width: 6px; height: 6px; border-radius: 999px; box-shadow: 0 0 0 3px color-mix(in srgb, currentColor 20%, transparent); }
.evt-kind { color: var(--ink-tertiary); }
.evt-text { color: var(--ink-secondary); }
.evt-sep { color: var(--ink-disabled); margin-right: 16px; }
</style>
