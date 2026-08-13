<script setup lang="ts">
/**
 * MasterAlarm — flight-director-style master caution lamp.
 * Latches on when an err alarm arrives, requires manual silence.
 */
import { computed } from 'vue'
import { useMissionStore } from '@/stores/mission'
import { useTelemetryStore } from '@/stores/telemetry'
import { useAudioFeedback } from '@/composables/useAudioFeedback'

const mission = useMissionStore()
const telemetry = useTelemetryStore()
const audio = useAudioFeedback()

const latestErr = computed(() => {
  const recent = telemetry.alarmHistory.slice(-10).filter((a) => a.severity === 'err')
  return recent[recent.length - 1] ?? null
})

function silence() {
  mission.silenceMasterAlarm()
  audio.play('click')
}
function rearm() {
  mission.rearmMasterAlarm()
}
</script>

<template>
  <button
    class="ma"
    :class="[
      mission.masterAlarmActive ? 'ma-on' : mission.masterAlarmSilenced ? 'ma-silenced' : 'ma-off',
    ]"
    :title="mission.masterAlarmActive
      ? `MASTER ALARM — ${latestErr?.title ?? ''} (click to silence)`
      : mission.masterAlarmSilenced
        ? 'Silenced — click to re-arm'
        : 'Master alarm — all clear'"
    @click="mission.masterAlarmActive ? silence() : rearm()"
  >
    <span class="ma-lamp">
      <span class="ma-dot"></span>
      <span class="ma-ring"></span>
    </span>
    <span class="ma-text">
      <span class="ma-tag">MASTER</span>
      <span class="ma-state">{{ mission.masterAlarmActive ? 'ALARM' : mission.masterAlarmSilenced ? 'SILENT' : 'CLEAR' }}</span>
    </span>
  </button>
</template>

<style scoped>
.ma {
  display: inline-flex; align-items: center; gap: 8px;
  padding: 4px 10px 4px 6px;
  border-radius: 8px;
  background: var(--bg-card);
  border: 1px solid var(--line-border);
  font-family: 'JetBrains Mono Variable', monospace;
  cursor: pointer;
  transition: all 0.18s var(--ease-out-quint);
}
.ma:hover { transform: translateY(-1px); }

.ma-lamp { position: relative; width: 14px; height: 14px; display: inline-block; }
.ma-dot { position: absolute; inset: 3px; border-radius: 50%; background: var(--ink-disabled); transition: all 0.3s var(--ease-out-quint); }
.ma-ring { position: absolute; inset: 0; border-radius: 50%; border: 1px solid rgba(15,23,42,0.10); }

.ma-text { display: inline-flex; flex-direction: column; line-height: 1; gap: 2px; }
.ma-tag { font-size: 0.52rem; letter-spacing: 0.18em; color: var(--ink-muted); font-weight: 700; }
.ma-state { font-size: 0.7rem; font-weight: 700; letter-spacing: 0.06em; color: var(--ink-secondary); }

/* OFF / CLEAR */
.ma-off .ma-dot { background: var(--status-ok); box-shadow: 0 0 6px rgba(16, 185, 129, 0.6); }
.ma-off .ma-state { color: var(--status-ok); }

/* SILENCED */
.ma-silenced .ma-dot { background: var(--accent-amber); }
.ma-silenced .ma-state { color: var(--accent-amber); }

/* ON / ALARM */
.ma-on { background: rgba(239, 68, 68, 0.10); border-color: rgba(239, 68, 68, 0.35); animation: ma-flash 0.7s ease-in-out infinite; }
.ma-on .ma-dot { background: var(--status-err); box-shadow: 0 0 12px rgba(239, 68, 68, 0.7); }
.ma-on .ma-state { color: var(--status-err); }
.ma-on .ma-ring { border-color: rgba(239, 68, 68, 0.55); animation: ma-ring 1.4s ease-out infinite; }

@keyframes ma-flash {
  0%, 100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
  50%      { box-shadow: 0 0 0 4px rgba(239, 68, 68, 0.18); }
}
@keyframes ma-ring {
  0%   { transform: scale(1); opacity: 1; }
  100% { transform: scale(1.8); opacity: 0; }
}
@media (prefers-reduced-motion: reduce) { .ma-on, .ma-ring { animation: none; } }
[data-reduce-motion='true'] .ma-on, [data-reduce-motion='true'] .ma-ring { animation: none; }
</style>
