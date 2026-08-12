<script setup lang="ts">
// 递归行为树节点 — 编辑预览 + 运行态着色 (第 3 期 #1)
// nodeStates 来自 /api/missions/status 的 node_states (path → {status, result})
import { computed } from 'vue'

export interface BtNodeData {
  type: string
  params?: Record<string, unknown>
  children?: BtNodeData[]
}

const props = defineProps<{
  node: BtNodeData
  path: string
  nodeStates: Record<string, { status?: string; result?: Record<string, unknown> }>
}>()

const COMPOSITE = new Set(['sequence', 'fallback', 'retry', 'repeat'])

const NODE_META: Record<string, { icon: string; label: string; hue: string }> = {
  sequence:    { icon: '⮕', label: '顺序', hue: 'blue' },
  fallback:    { icon: '⑂', label: '备选', hue: 'violet' },
  retry:       { icon: '↻', label: '重试', hue: 'amber' },
  repeat:      { icon: '⟳', label: '重复', hue: 'amber' },
  goto:        { icon: '🧭', label: '导航', hue: 'blue' },
  forward:     { icon: '⬆', label: '前进', hue: 'teal' },
  spin:        { icon: '↺', label: '旋转', hue: 'teal' },
  twist:       { icon: '🕹', label: '速度', hue: 'teal' },
  wait:        { icon: '⏲', label: '等待', hue: 'idle' },
  speak:       { icon: '🔊', label: '播报', hue: 'emerald' },
  photo:       { icon: '📷', label: '拍照', hue: 'emerald' },
  vlm:         { icon: '👁', label: 'VLM 问视觉', hue: 'violet' },
  read_furnace:{ icon: '🌡', label: '读炉温', hue: 'rose' },
  detect_wait: { icon: '🎯', label: '等检测', hue: 'rose' },
}

const meta = computed(() => NODE_META[props.node.type] ?? { icon: '❓', label: props.node.type, hue: 'idle' })
const isComposite = computed(() => COMPOSITE.has(props.node.type))
const state = computed(() => props.nodeStates[props.path]?.status ?? '')

const paramsText = computed(() => {
  const p = props.node.params
  if (!p || Object.keys(p).length === 0) return ''
  return Object.entries(p).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(' · ')
})

const resultErr = computed(() => {
  const r = props.nodeStates[props.path]?.result
  return r && r.ok === false ? String(r.error ?? '') : ''
})
</script>

<template>
  <div class="bt-node" :class="[`hue-${meta.hue}`, state ? `st-${state}` : '']">
    <div class="bt-head">
      <span class="bt-icon">{{ meta.icon }}</span>
      <span class="bt-type">{{ meta.label }}</span>
      <span class="bt-typecode mono">{{ node.type }}</span>
      <span v-if="paramsText" class="bt-params mono">{{ paramsText }}</span>
      <span v-if="state" class="bt-state mono" :class="`bs-${state}`">
        {{ state === 'running' ? '● running' : state === 'done' ? '✓ done' : state === 'failed' ? '✗ failed' : state }}
      </span>
    </div>
    <div v-if="resultErr" class="bt-err mono">{{ resultErr }}</div>
    <div v-if="isComposite && node.children?.length" class="bt-children">
      <BtNode
        v-for="(c, i) in node.children"
        :key="i"
        :node="c"
        :path="`${path}.${i}`"
        :node-states="nodeStates"
      />
    </div>
  </div>
</template>

<style scoped>
.bt-node {
  border: 1px solid var(--line-border);
  border-left: 3px solid var(--ink-muted);
  border-radius: 10px;
  padding: 8px 10px;
  margin-bottom: 6px;
  background: rgba(255, 255, 255, 0.72);
  transition: border-color 0.2s, box-shadow 0.2s;
}
.hue-blue    { border-left-color: #2563eb; }
.hue-teal    { border-left-color: #0891b2; }
.hue-emerald { border-left-color: #059669; }
.hue-violet  { border-left-color: #7c3aed; }
.hue-amber   { border-left-color: #d97706; }
.hue-rose    { border-left-color: #e11d48; }
.hue-idle    { border-left-color: #94a3b8; }

.st-running { box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.25); animation: btPulse 1.4s ease-in-out infinite; }
.st-done    { background: rgba(5, 150, 105, 0.06); }
.st-failed  { background: rgba(225, 29, 72, 0.07); }
.st-aborted { opacity: 0.55; }
@keyframes btPulse {
  0%, 100% { box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.25); }
  50%      { box-shadow: 0 0 0 4px rgba(37, 99, 235, 0.10); }
}

.bt-head { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.bt-icon { font-size: 0.95rem; }
.bt-type { font-size: 0.8rem; font-weight: 600; color: var(--ink-primary); }
.bt-typecode { font-size: 0.66rem; color: var(--ink-muted); }
.bt-params {
  font-size: 0.68rem; color: var(--ink-tertiary);
  background: rgba(15, 23, 42, 0.04); border-radius: 5px; padding: 1px 6px;
  max-width: 360px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.bt-state { margin-left: auto; font-size: 0.68rem; font-weight: 700; }
.bs-running { color: #2563eb; }
.bs-done    { color: #059669; }
.bs-failed  { color: #e11d48; }
.bs-aborted { color: #94a3b8; }

.bt-err {
  margin-top: 4px; font-size: 0.66rem; color: #e11d48;
  background: rgba(225, 29, 72, 0.06); border-radius: 5px; padding: 3px 7px;
}
.bt-children { margin-top: 8px; padding-left: 16px; border-left: 1px dashed var(--line-divider); }
</style>
