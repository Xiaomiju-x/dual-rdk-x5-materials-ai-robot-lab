<script setup lang="ts">
import { computed } from 'vue'
import { useTelemetryStore } from '@/stores/telemetry'
import type { ArmId } from '@/types/telemetry'

interface Props {
  arm: ArmId
  label: string
  /** show animated AprilTag overlay in mock mode */
  tagLabel?: string
}
const props = withDefaults(defineProps<Props>(), { tagLabel: '' })

const telemetry = useTelemetryStore()
const isReal = computed(() => telemetry.mode === 'real')
const fps = computed(() =>
  props.arm === 'arm01' ? telemetry.status?.cam01_fps : telemetry.status?.cam02_fps)
const accent = computed(() => (props.arm === 'arm01' ? 'amber' : 'blue'))
</script>

<template>
  <div class="cam-tile" :class="`accent-${accent}`">
    <img v-if="isReal" :src="`/video/${arm}`" :alt="`${arm} live`" class="cam-img" />
    <div v-else class="cam-mock">
      <div class="scan-line"></div>
      <div class="tag-marker" :class="`accent-${accent}`">
        <span class="tag-lbl mono">{{ tagLabel || 'apriltag' }}</span>
      </div>
    </div>

    <div class="cam-tl mono">{{ label }} · /dev/video0 · 1280×720</div>
    <div class="cam-tr mono">
      <span class="live-dot"></span>{{ (fps ?? 0).toFixed(1) }} fps
    </div>
    <div class="cam-bl mono">{{ arm }}-eye · tag36h11</div>
  </div>
</template>

<style scoped>
.cam-tile {
  position: relative; border-radius: 16px; overflow: hidden; min-height: 0; height: 100%;
  background: #0a0e1a; box-shadow: var(--shadow-card);
}
.cam-img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; }
.cam-mock {
  position: absolute; inset: 0;
  background:
    radial-gradient(circle at 32% 40%, rgba(217,119,6,0.16), transparent 32%),
    radial-gradient(circle at 70% 62%, rgba(37,99,235,0.14), transparent 36%),
    repeating-linear-gradient(0deg, transparent 0 3px, rgba(255,255,255,0.015) 3px 4px),
    #0a0e1a;
}
.accent-blue.cam-mock, .accent-blue .cam-mock { }
.scan-line {
  position: absolute; left: 0; right: 0; height: 1px; top: 0;
  background: linear-gradient(90deg, transparent, rgba(125,211,252,0.7), transparent);
  box-shadow: 0 0 12px rgba(125,211,252,0.5);
  animation: scanY 4s linear infinite;
}
@keyframes scanY { 0% { top: 2%; } 100% { top: 98%; } }
.tag-marker {
  position: absolute; left: 32%; top: 34%; width: 22%; aspect-ratio: 1;
  border: 2px solid var(--accent-amber); border-radius: 5px;
  box-shadow: 0 0 16px rgba(217,119,6,0.5);
  animation: tagDrift 7s ease-in-out infinite;
}
.tag-marker.accent-blue { border-color: var(--accent-blue); box-shadow: 0 0 16px rgba(37,99,235,0.5); }
@keyframes tagDrift {
  0%,100% { transform: translate(0,0) rotate(-1deg); }
  50% { transform: translate(14%, 10%) rotate(2deg); }
}
.tag-lbl {
  position: absolute; top: -20px; left: 0; font-size: 10px; color: #fcd34d;
  background: rgba(0,0,0,0.6); padding: 2px 6px; border-radius: 3px; white-space: nowrap;
}
.tag-marker.accent-blue .tag-lbl { color: #93c5fd; }
.cam-tl, .cam-tr, .cam-bl {
  position: absolute; font-size: 10px; color: rgba(255,255,255,0.72);
  background: rgba(0,0,0,0.4); padding: 3px 7px; border-radius: 5px;
}
.cam-tl { top: 8px; left: 8px; }
.cam-tr { top: 8px; right: 8px; color: #6ee7b7; display: flex; align-items: center; gap: 5px; }
.cam-bl { bottom: 8px; left: 8px; color: rgba(255,255,255,0.5); }
.live-dot { width: 6px; height: 6px; border-radius: 50%; background: #34d399; box-shadow: 0 0 6px #34d399; animation: pulseSoft 1.6s ease-in-out infinite; }
</style>
