<script setup lang="ts">
import { computed } from 'vue'
import { useTelemetryStore } from '@/stores/telemetry'

interface Props {
  open: boolean
}
defineProps<Props>()
const emit = defineEmits<{ (e: 'close'): void }>()

const telemetry = useTelemetryStore()

const stats = computed(() => ({
  uptime: telemetry.packet?.heartbeat.uptime_s.toFixed(0) ?? '—',
  seq: telemetry.packet?.heartbeat.sequence ?? '—',
  hz: telemetry.observedHz.toFixed(1),
  sensors: telemetry.packet?.sensors.length ?? 0,
  cams: telemetry.packet?.cameras.length ?? 0,
  bpu: telemetry.packet?.bpu_slots.length ?? 0,
  alarms: telemetry.alarmHistory.length,
  state: telemetry.state,
  endpoint: telemetry.packet?.ai_link?.endpoint ?? '—',
}))

const STACK = [
  { area: 'Frontend', items: ['Vue 3.5 + Vite 5', 'TypeScript 5.5', 'Pinia 2.2', 'Three.js R168', 'ECharts 5.5', 'Tailwind 3.4'] },
  { area: 'Backend',  items: ['FastAPI 0.115', 'uvicorn 0.32', 'WebSocket 13.1', 'Pydantic 2.10'] },
  { area: 'Target',   items: ['RDK X5 8G @ 192.0.2.85', 'ROS2 Humble', 'Bayes-e BPU 10 TOPS', 'CMA 391MB'] },
]

function backdropClick(evt: MouseEvent) {
  if ((evt.target as HTMLElement).classList.contains('ab-backdrop')) emit('close')
}
</script>

<template>
  <transition name="ab">
    <div v-if="open" class="ab-backdrop" @click="backdropClick">
      <div class="ab-card glass-strong" role="dialog" aria-label="About NavCockpit">
        <div class="ab-head">
          <div class="ab-brand">
            <span class="ab-mark">⬢</span>
            <div>
              <div class="ab-name">NavCockpit</div>
              <div class="ab-sub mono">Premium real-time dashboard for the 具身脑 embodied X5</div>
            </div>
          </div>
          <button class="ab-close" @click="emit('close')" aria-label="close">×</button>
        </div>

        <div class="ab-body">
          <section class="ab-section">
            <div class="section-label">Live Stats</div>
            <div class="stat-grid">
              <div class="stat">
                <div class="stat-label">WS State</div>
                <div class="stat-val mono">{{ stats.state }}</div>
              </div>
              <div class="stat">
                <div class="stat-label">Observed Hz</div>
                <div class="stat-val mono">{{ stats.hz }}</div>
              </div>
              <div class="stat">
                <div class="stat-label">Backend Uptime</div>
                <div class="stat-val mono">{{ stats.uptime }} s</div>
              </div>
              <div class="stat">
                <div class="stat-label">Sequence</div>
                <div class="stat-val mono">{{ stats.seq }}</div>
              </div>
              <div class="stat">
                <div class="stat-label">Sensors</div>
                <div class="stat-val mono">{{ stats.sensors }}</div>
              </div>
              <div class="stat">
                <div class="stat-label">Cameras</div>
                <div class="stat-val mono">{{ stats.cams }}</div>
              </div>
              <div class="stat">
                <div class="stat-label">BPU Slots</div>
                <div class="stat-val mono">{{ stats.bpu }}</div>
              </div>
              <div class="stat">
                <div class="stat-label">Alarm History</div>
                <div class="stat-val mono">{{ stats.alarms }}</div>
              </div>
            </div>
          </section>

          <section class="ab-section">
            <div class="section-label">Tech Stack</div>
            <div class="stack-grid">
              <div v-for="s in STACK" :key="s.area" class="stack-col">
                <div class="stack-area">{{ s.area }}</div>
                <ul class="stack-list">
                  <li v-for="it in s.items" :key="it" class="mono">{{ it }}</li>
                </ul>
              </div>
            </div>
          </section>

          <section class="ab-section">
            <div class="section-label">AI Brain Link</div>
            <div class="link-box mono">→ {{ stats.endpoint }}</div>
          </section>
        </div>

        <div class="ab-foot mono">
          <span>NavCockpit v1.0 · Phase 0-9 complete · 荧光具身智研</span>
          <span class="foot-spacer"></span>
          <span>2026 全国大学生嵌入式芯片与系统设计竞赛</span>
        </div>
      </div>
    </div>
  </transition>
</template>

<style scoped>
.ab-backdrop {
  position: fixed; inset: 0;
  background: rgba(11, 18, 32, 0.4);
  backdrop-filter: blur(10px) saturate(160%);
  -webkit-backdrop-filter: blur(10px) saturate(160%);
  z-index: 100;
  display: flex; align-items: center; justify-content: center;
  padding: 24px;
}
.ab-card {
  width: 680px; max-width: 100%; max-height: 86vh;
  border-radius: 20px;
  display: flex; flex-direction: column;
  overflow: hidden;
}

.ab-head {
  display: flex; justify-content: space-between; align-items: flex-start;
  padding: 20px 24px 16px;
  border-bottom: 1px solid var(--line-divider);
}
.ab-brand { display: flex; align-items: center; gap: 14px; }
.ab-mark {
  font-size: 2.4rem;
  background: linear-gradient(135deg, var(--accent-blue), var(--accent-teal) 50%, var(--accent-emerald));
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
  filter: drop-shadow(0 4px 12px rgba(37, 99, 235, 0.3));
}
.ab-name { font-size: 1.2rem; font-weight: 700; letter-spacing: -0.01em; color: var(--ink-primary); }
.ab-sub { font-size: 0.72rem; color: var(--ink-tertiary); margin-top: 2px; }
.ab-close {
  background: transparent; border: none;
  width: 32px; height: 32px;
  font-size: 1.4rem; color: var(--ink-tertiary);
  border-radius: 8px; cursor: pointer;
}
.ab-close:hover { background: rgba(15, 23, 42, 0.06); color: var(--ink-primary); }

.ab-body { flex: 1; overflow-y: auto; padding: 16px 24px; display: flex; flex-direction: column; gap: 20px; }
.ab-section { display: flex; flex-direction: column; gap: 8px; }
.section-label { color: var(--ink-muted); }

.stat-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 8px;
}
@media (max-width: 640px) { .stat-grid { grid-template-columns: repeat(2, 1fr); } }
.stat {
  padding: 8px 10px;
  background: var(--bg-elevated);
  border: 1px solid var(--line-divider);
  border-radius: 8px;
}
.stat-label { font-size: 0.62rem; color: var(--ink-muted); text-transform: uppercase; letter-spacing: 0.05em; }
.stat-val { font-size: 0.88rem; color: var(--ink-primary); font-weight: 600; margin-top: 3px; }

.stack-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
@media (max-width: 640px) { .stack-grid { grid-template-columns: 1fr; } }
.stack-col {
  padding: 10px 12px;
  background: var(--bg-elevated);
  border: 1px solid var(--line-divider);
  border-radius: 10px;
}
.stack-area { font-size: 0.72rem; font-weight: 700; color: var(--accent-blue); margin-bottom: 6px; }
.stack-list { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 3px; }
.stack-list li { font-size: 0.7rem; color: var(--ink-tertiary); }

.link-box {
  padding: 10px 12px;
  background: var(--bg-elevated);
  border: 1px solid var(--line-divider);
  border-radius: 8px;
  font-size: 0.74rem;
  color: var(--accent-violet);
}

.ab-foot {
  display: flex; gap: 10px; align-items: center;
  padding: 10px 24px;
  border-top: 1px solid var(--line-divider);
  font-size: 0.66rem; color: var(--ink-muted);
}
.foot-spacer { flex: 1; }

.ab-enter-from { opacity: 0; }
.ab-enter-from .ab-card { transform: translateY(-8px) scale(0.96); }
.ab-leave-to { opacity: 0; }
.ab-leave-to .ab-card { transform: translateY(-4px) scale(0.98); }
.ab-enter-active, .ab-leave-active { transition: opacity 0.22s var(--ease-out-quint); }
.ab-enter-active .ab-card, .ab-leave-active .ab-card { transition: transform 0.22s var(--ease-out-quint); }
</style>
