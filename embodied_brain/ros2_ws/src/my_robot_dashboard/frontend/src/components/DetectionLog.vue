<script setup lang="ts">
import { computed } from 'vue'
import type { Detection } from './CameraTile.vue'

interface Props {
  items: Detection[]
  max?: number
}
const props = withDefaults(defineProps<Props>(), { max: 24 })

const visible = computed(() => props.items.slice(-props.max).reverse())

function fmtTime(ms: number): string {
  const d = new Date(ms)
  const hh = String(d.getHours()).padStart(2, '0')
  const mm = String(d.getMinutes()).padStart(2, '0')
  const ss = String(d.getSeconds()).padStart(2, '0')
  const tt = String(d.getMilliseconds()).padStart(3, '0').slice(0, 2)
  return `${hh}:${mm}:${ss}.${tt}`
}

function clsTone(cls: string): string {
  if (cls === 'apriltag') return 'tone-violet'
  if (cls === 'person')   return 'tone-rose'
  if (cls === 'bottle')   return 'tone-teal'
  return 'tone-blue'
}

function clsIcon(cls: string): string {
  if (cls === 'apriltag') return '⊞'
  if (cls === 'person')   return '☺'
  if (cls === 'bottle')   return '🜨'
  return '◇'
}
</script>

<template>
  <div class="det-log card-elevated">
    <div class="log-head">
      <span class="section-label">Detection Stream</span>
      <span class="chip chip-info mono">{{ items.length }} events</span>
    </div>
    <div class="log-body">
      <div v-if="visible.length === 0" class="log-empty mono">awaiting first detection…</div>
      <transition-group v-else name="log" tag="div" class="log-list">
        <div v-for="d in visible" :key="d.id" class="log-row">
          <span class="ts mono">{{ fmtTime(d.at_ms) }}</span>
          <span class="cls-glyph" :class="clsTone(d.cls)">{{ clsIcon(d.cls) }}</span>
          <span class="cls">{{ d.cls }}<span v-if="d.tag_id != null" class="tag-id mono"> id={{ d.tag_id }}</span></span>
          <span class="cam mono">{{ d.camera_id }}</span>
          <span class="conf mono">
            <span class="bar"><span class="bar-fill" :style="{ width: `${(d.conf * 100).toFixed(0)}%` }"></span></span>
            {{ (d.conf * 100).toFixed(1) }}%
          </span>
        </div>
      </transition-group>
    </div>
  </div>
</template>

<style scoped>
.det-log { padding: 14px 16px; display: flex; flex-direction: column; gap: 10px; min-height: 0; }
.log-head { display: flex; align-items: center; justify-content: space-between; }
.log-body { flex: 1; min-height: 0; overflow-y: auto; padding-right: 4px; }
.log-empty {
  height: 100%;
  display: flex; align-items: center; justify-content: center;
  color: var(--ink-muted); font-size: 0.74rem;
}
.log-list { display: flex; flex-direction: column; gap: 4px; }
.log-row {
  display: grid;
  grid-template-columns: 86px 24px 1fr 90px 130px;
  align-items: center;
  gap: 8px;
  padding: 5px 8px;
  border-radius: 6px;
  background: rgba(248, 250, 252, 0.6);
  border: 1px solid rgba(15, 23, 42, 0.04);
  font-size: 0.74rem;
}
.ts { color: var(--ink-muted); font-size: 0.68rem; }
.cls-glyph {
  width: 22px; height: 22px;
  display: flex; align-items: center; justify-content: center;
  border-radius: 5px;
  font-size: 0.8rem;
}
.tone-violet { color: #7c3aed; background: rgba(124, 58, 237, 0.10); }
.tone-rose   { color: #e11d48; background: rgba(225, 29, 72, 0.10); }
.tone-teal   { color: #0891b2; background: rgba(8, 145, 178, 0.10); }
.tone-blue   { color: #2563eb; background: rgba(37, 99, 235, 0.10); }
.cls { color: var(--ink-primary); font-weight: 500; }
.tag-id { color: var(--ink-muted); margin-left: 4px; }
.cam { color: var(--ink-tertiary); font-size: 0.68rem; text-align: right; }
.conf { display: flex; align-items: center; gap: 8px; color: var(--ink-secondary); font-size: 0.7rem; }
.bar { width: 60px; height: 4px; background: rgba(15, 23, 42, 0.08); border-radius: 2px; overflow: hidden; }
.bar-fill { display: block; height: 100%; background: linear-gradient(90deg, #2563eb, #7c3aed); border-radius: 2px; }

/* transition */
.log-enter-from { opacity: 0; transform: translateY(-6px); }
.log-enter-active { transition: opacity 0.24s var(--ease-out-quint), transform 0.24s var(--ease-out-quint); }
.log-leave-to { opacity: 0; }
.log-leave-active { transition: opacity 0.18s ease; position: absolute; }
.log-move { transition: transform 0.24s var(--ease-out-quint); }

/* scrollbar */
.log-body::-webkit-scrollbar { width: 6px; }
.log-body::-webkit-scrollbar-thumb { background: rgba(15, 23, 42, 0.12); border-radius: 3px; }
</style>
