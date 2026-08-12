<script setup lang="ts">
/**
 * BootSplash — cinematic boot sequence shown on first navigation
 * to the cockpit each session.
 *
 *  - 5 staged checks (~500ms each, ~2.5s total)
 *  - tap / any key to skip
 *  - sessionStorage gate; settings.bootSplash toggle persists
 *
 * Mounted at App.vue root; visibility derived from store + session.
 */
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { useTelemetryStore } from '@/stores/telemetry'

const emit = defineEmits<{ (e: 'done'): void }>()

const stages = [
  { id: 'gpu',  label: 'Initializing WebGL R168 + PMREM environment',   ms: 480 },
  { id: 'ws',   label: 'Connecting telemetry stream · ws://embodied/ws', ms: 520 },
  { id: 'ring', label: 'Loading 60s ring buffers · sensors × 8 · bpu × 4', ms: 460 },
  { id: 'bpu',  label: 'Warming up 5 BPU slots · 8 perception nodes',    ms: 540 },
  { id: 'go',   label: 'All systems GO · launching cockpit',             ms: 380 },
]

const current = ref(0)
const skipped = ref(false)
const finished = ref(false)
const telemetry = useTelemetryStore()
let timers: number[] = []

function advance() {
  if (finished.value || skipped.value) return
  current.value += 1
  if (current.value >= stages.length) {
    finished.value = true
    setTimeout(() => emit('done'), 280)
  }
}

function skip(reason: 'key' | 'tap' | 'auto') {
  if (finished.value) return
  skipped.value = true
  finished.value = true
  for (const t of timers) clearTimeout(t)
  setTimeout(() => emit('done'), 120)
  void reason
}

function onKey(e: KeyboardEvent) {
  if (e.key === 'Escape' || e.key === 'Enter' || e.key === ' ') skip('key')
}

onMounted(() => {
  // run each stage
  let acc = 0
  for (let i = 0; i < stages.length; i++) {
    acc += stages[i].ms
    timers.push(window.setTimeout(advance, acc))
  }
  window.addEventListener('keydown', onKey)
})
onBeforeUnmount(() => {
  for (const t of timers) clearTimeout(t)
  window.removeEventListener('keydown', onKey)
})

const wsReady = computed(() => telemetry.isConnected)
</script>

<template>
  <Teleport to="body">
    <Transition name="boot">
      <div v-if="!finished" class="boot" @click="skip('tap')" role="status" aria-label="Boot sequence">
        <!-- bg layers -->
        <div class="boot-grid" aria-hidden="true"></div>
        <div class="boot-glow" aria-hidden="true"></div>

        <!-- centerpiece -->
        <div class="boot-stack">
          <div class="boot-logo">
            <span class="logo-hex">⬢</span>
            <div class="logo-rings">
              <span class="logo-ring r1"></span>
              <span class="logo-ring r2"></span>
              <span class="logo-ring r3"></span>
            </div>
          </div>

          <div class="boot-title">NavCockpit</div>
          <div class="boot-sub mono">embodied · X5 · 192.0.2.85</div>

          <div class="boot-stages">
            <div
              v-for="(s, i) in stages"
              :key="s.id"
              class="boot-row"
              :class="{
                'done': i < current,
                'active': i === current,
                'queued': i > current,
              }"
            >
              <span class="boot-marker">
                <span v-if="i < current">●</span>
                <span v-else-if="i === current" class="boot-spin">◐</span>
                <span v-else>○</span>
              </span>
              <span class="boot-row-text">{{ s.label }}</span>
              <span class="boot-row-tag mono">{{ i < current ? 'OK' : i === current ? '…' : '' }}</span>
            </div>
          </div>

          <div class="boot-bar">
            <div class="boot-bar-fill" :style="{ width: `${(current / stages.length) * 100}%` }"></div>
          </div>

          <div class="boot-foot mono">
            <span>WS {{ wsReady ? 'live' : 'connecting' }}</span>
            <span class="boot-foot-sep">·</span>
            <span>tap / esc to skip</span>
          </div>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<style scoped>
.boot {
  position: fixed; inset: 0;
  z-index: 9999;
  display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  cursor: pointer;
  background:
    radial-gradient(900px 620px at 50% 38%, rgba(37, 99, 235, 0.15), transparent 60%),
    radial-gradient(760px 560px at 78% 80%, rgba(6, 182, 212, 0.12), transparent 60%),
    linear-gradient(160deg, #ffffff, #eef4ff);
  color: #0f172a;
  user-select: none;
  -webkit-user-select: none;
}
.boot-enter-from { opacity: 0; }
.boot-leave-to   { opacity: 0; transform: scale(1.03); filter: blur(8px); }
.boot-enter-active { transition: opacity 0.32s var(--ease-out-quint); }
.boot-leave-active { transition: opacity 0.42s var(--ease-out-quint), transform 0.42s var(--ease-out-quint), filter 0.42s var(--ease-out-quint); }

.boot-grid {
  position: absolute; inset: 0;
  background-image:
    linear-gradient(to right, rgba(37, 99, 235, 0.10) 1px, transparent 1px),
    linear-gradient(to bottom, rgba(37, 99, 235, 0.10) 1px, transparent 1px);
  background-size: 48px 48px;
  mask-image: radial-gradient(ellipse at center, black 30%, transparent 80%);
  -webkit-mask-image: radial-gradient(ellipse at center, black 30%, transparent 80%);
}
.boot-glow {
  position: absolute; inset: 0;
  background: radial-gradient(circle at 50% 40%, rgba(37, 99, 235, 0.18), transparent 55%);
  filter: blur(40px);
  animation: boot-pulse 3.5s ease-in-out infinite;
}
@keyframes boot-pulse {
  0%, 100% { opacity: 0.45; transform: scale(1); }
  50%      { opacity: 0.7; transform: scale(1.08); }
}

.boot-stack {
  position: relative;
  z-index: 2;
  display: flex; flex-direction: column; align-items: center; gap: 14px;
  max-width: 520px;
  width: calc(100vw - 48px);
}

.boot-logo {
  position: relative;
  width: 96px; height: 96px;
  display: flex; align-items: center; justify-content: center;
  margin-bottom: 4px;
}
.logo-hex {
  font-size: 4.2rem;
  background: linear-gradient(135deg, #2563eb, #06b6d4 50%, #7c3aed);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
  filter: drop-shadow(0 6px 16px rgba(37, 99, 235, 0.35));
  animation: logo-breath 2.6s ease-in-out infinite;
}
@keyframes logo-breath {
  0%, 100% { transform: scale(1); filter: drop-shadow(0 6px 16px rgba(37, 99, 235, 0.3)); }
  50%      { transform: scale(1.06); filter: drop-shadow(0 8px 22px rgba(37, 99, 235, 0.5)); }
}
.logo-rings { position: absolute; inset: 0; pointer-events: none; }
.logo-ring {
  position: absolute; inset: 0;
  border-radius: 50%;
  border: 2px solid rgba(37, 99, 235, 0.4);
  animation: ring-out 2.8s ease-out infinite;
}
.logo-ring.r2 { animation-delay: 0.9s; }
.logo-ring.r3 { animation-delay: 1.8s; }
@keyframes ring-out {
  0%   { transform: scale(0.7); opacity: 0; }
  20%  { opacity: 0.85; }
  100% { transform: scale(1.85); opacity: 0; }
}

.boot-title {
  font-size: 1.5rem; font-weight: 800; letter-spacing: 0.04em;
  background: linear-gradient(110deg, #2563eb 25%, #06b6d4 50%, #2563eb 75%);
  background-size: 200% 100%;
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
  animation: title-sweep 5s linear infinite;
}
@keyframes title-sweep {
  0% { background-position: -120% 0; }
  100% { background-position: 120% 0; }
}
.boot-sub { font-size: 0.74rem; color: #64748b; letter-spacing: 0.06em; margin-top: -2px; }

.boot-stages {
  width: 100%;
  display: flex; flex-direction: column; gap: 4px;
  margin-top: 14px;
  font-family: 'JetBrains Mono Variable', monospace;
}
.boot-row {
  display: grid;
  grid-template-columns: 22px 1fr 38px;
  gap: 12px; align-items: center;
  padding: 6px 10px;
  border-radius: 6px;
  font-size: 0.76rem;
  border: 1px solid transparent;
  transition: background 0.2s var(--ease-out-quint), border-color 0.2s var(--ease-out-quint);
}
.boot-row { background: rgba(255, 255, 255, 0.5); }
.boot-row.queued { color: #94a3b8; opacity: 0.75; }
.boot-row.active {
  color: #2563eb;
  background: #ffffff;
  border-color: rgba(37, 99, 235, 0.3);
  box-shadow: 0 4px 16px rgba(37, 99, 235, 0.15);
}
.boot-row.done { color: #059669; background: rgba(255, 255, 255, 0.7); }
.boot-marker { text-align: center; font-size: 0.85rem; }
.boot-spin { display: inline-block; animation: boot-spin 0.9s linear infinite; }
@keyframes boot-spin { to { transform: rotate(360deg); } }
.boot-row-text { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.boot-row-tag { font-size: 0.62rem; color: inherit; opacity: 0.8; text-align: right; }

.boot-bar {
  width: 100%; height: 4px; border-radius: 999px;
  background: rgba(15, 23, 42, 0.08);
  overflow: hidden;
  margin-top: 6px;
}
.boot-bar-fill {
  height: 100%;
  background: linear-gradient(90deg, #2563eb, #06b6d4 50%, #7c3aed);
  border-radius: 999px;
  transition: width 0.4s var(--ease-out-quint);
}

.boot-foot {
  margin-top: 8px;
  font-size: 0.62rem;
  color: #94a3b8;
  letter-spacing: 0.05em;
  display: flex; gap: 8px;
}
.boot-foot-sep { color: #cbd5e1; }

@media (prefers-reduced-motion: reduce) {
  .boot-glow, .logo-hex, .logo-ring, .boot-spin, .boot-title { animation: none !important; }
}
[data-reduce-motion='true'] .boot-glow,
[data-reduce-motion='true'] .logo-hex,
[data-reduce-motion='true'] .logo-ring,
[data-reduce-motion='true'] .boot-spin,
[data-reduce-motion='true'] .boot-title { animation: none !important; }
</style>
