<script setup lang="ts">
/**
 * MissionPhase — current phase chip in topbar.
 * Click → dropdown to override (demo/handoff use).
 */
import { onBeforeUnmount, onMounted, ref } from 'vue'
import { useMissionStore, type MissionPhase as Phase } from '@/stores/mission'

const mission = useMissionStore()
const open = ref(false)
const anchor = ref<HTMLElement | null>(null)
const menu = ref<HTMLElement | null>(null)

const PHASES: { id: Phase; label: string; glyph: string }[] = [
  { id: 'IDLE',    label: 'IDLE',    glyph: '○' },
  { id: 'TRANSIT', label: 'TRANSIT', glyph: '➔' },
  { id: 'PICKUP',  label: 'PICKUP',  glyph: '✥' },
  { id: 'RETURN',  label: 'RETURN',  glyph: '↩' },
  { id: 'DOCK',    label: 'DOCK',    glyph: '⌂' },
  { id: 'FAULT',   label: 'FAULT',   glyph: '⚠' },
]

function pick(p: Phase | null) {
  mission.setPhaseOverride(p)
  open.value = false
}

function onDoc(evt: MouseEvent) {
  if (!open.value) return
  const t = evt.target as Node
  if (anchor.value?.contains(t) || menu.value?.contains(t)) return
  open.value = false
}
onMounted(() => document.addEventListener('mousedown', onDoc))
onBeforeUnmount(() => document.removeEventListener('mousedown', onDoc))
</script>

<template>
  <div class="mp-wrap">
    <button
      ref="anchor"
      class="mp"
      :class="`mp-${mission.phaseTone}`"
      :title="`Phase: ${mission.phaseInferred}${mission.phaseOverride ? ' (manual)' : ' (inferred)'}`"
      @click="open = !open"
    >
      <span class="mp-tag">PHASE</span>
      <span class="mp-val">{{ mission.phaseInferred }}</span>
      <span v-if="mission.phaseOverride" class="mp-pin" title="manually overridden">⚑</span>
    </button>
    <Transition name="pp">
      <div v-if="open" ref="menu" class="mp-menu glass-strong">
        <div class="mp-menu-head section-label">Override Phase</div>
        <button
          v-for="p in PHASES"
          :key="p.id"
          class="mp-opt"
          :class="{ active: mission.phaseOverride === p.id }"
          @click="pick(p.id)"
        ><span>{{ p.glyph }}</span><span>{{ p.label }}</span></button>
        <div class="mp-divider"></div>
        <button class="mp-opt mp-clear" @click="pick(null)">
          <span>⟲</span><span>Auto (inferred)</span>
        </button>
      </div>
    </Transition>
  </div>
</template>

<style scoped>
.mp-wrap { position: relative; }
.mp {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 4px 10px;
  border-radius: 8px;
  background: var(--bg-card);
  border: 1px solid var(--line-border);
  font-family: 'JetBrains Mono Variable', monospace;
  cursor: pointer;
  transition: all 0.18s var(--ease-out-quint);
}
.mp:hover { transform: translateY(-1px); box-shadow: var(--shadow-soft); }
.mp-tag { font-size: 0.58rem; letter-spacing: 0.18em; color: var(--ink-muted); font-weight: 700; padding-right: 6px; border-right: 1px solid var(--line-divider); }
.mp-val { font-size: 0.82rem; font-weight: 700; letter-spacing: 0.06em; }
.mp-pin { font-size: 0.78rem; color: var(--accent-amber); }

.mp-info  .mp-val { color: var(--accent-blue); }
.mp-warn  .mp-val { color: var(--accent-amber); }
.mp-err   .mp-val { color: var(--status-err); animation: pulseSoft 1.2s ease-in-out infinite; }
.mp-ok    .mp-val { color: var(--accent-emerald); }
.mp-idle  .mp-val { color: var(--ink-muted); }

.mp-menu {
  position: absolute;
  top: calc(100% + 8px);
  left: 0;
  min-width: 200px;
  padding: 8px;
  border-radius: 12px;
  z-index: 60;
  display: flex; flex-direction: column;
}
.mp-menu-head { padding: 4px 8px 8px; color: var(--ink-muted); }
.mp-opt {
  display: flex; align-items: center; gap: 10px;
  padding: 6px 10px;
  background: transparent;
  border: none;
  border-radius: 6px;
  font-family: inherit;
  font-size: 0.8rem;
  color: var(--ink-secondary);
  cursor: pointer;
  text-align: left;
  transition: background 0.12s var(--ease-out-quint);
}
.mp-opt:hover { background: rgba(37, 99, 235, 0.08); color: var(--ink-primary); }
.mp-opt.active { background: linear-gradient(135deg, rgba(37, 99, 235, 0.15), rgba(8, 145, 178, 0.10)); color: var(--accent-blue); }
.mp-opt span:first-child { width: 20px; text-align: center; }
.mp-clear { color: var(--ink-tertiary); }
.mp-divider { height: 1px; background: var(--line-divider); margin: 6px 0; }

.pp-enter-from { opacity: 0; transform: translateY(-6px) scale(0.98); }
.pp-leave-to { opacity: 0; transform: translateY(-4px) scale(0.99); }
.pp-enter-active, .pp-leave-active { transition: opacity 0.18s var(--ease-out-quint), transform 0.18s var(--ease-out-quint); }
</style>
