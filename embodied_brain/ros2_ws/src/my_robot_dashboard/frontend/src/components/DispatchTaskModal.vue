<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useToastStore } from '@/stores/toast'
import { useTelemetryStore } from '@/stores/telemetry'
import { useAudioFeedback } from '@/composables/useAudioFeedback'
import type {
  PickupFlowCommandPayload,
  PickupFlowCommandResult,
  PickupFlowStatus,
  Tone,
} from '@/types/telemetry'

interface Props {
  open: boolean
}
const props = defineProps<Props>()
const emit = defineEmits<{ (e: 'close'): void }>()

const toasts = useToastStore()
const telemetry = useTelemetryStore()
const audio = useAudioFeedback()

type TaskKind = 'fetch_sample' | 'patrol' | 'monitor_furnace' | 'return_home'

const TASK_DEFS: Array<{
  kind: TaskKind
  label: string
  desc: string
  icon: string
  needsBottle?: boolean
  needsTarget?: boolean
}> = [
  {
    kind: 'fetch_sample',
    label: 'Fetch sample',
    desc: '取瓶 → 烧结炉',
    icon: '🜨',
    needsBottle: true,
    needsTarget: true,
  },
  { kind: 'patrol', label: 'Patrol', desc: '巡航指定区域', icon: '◆', needsTarget: true },
  {
    kind: 'monitor_furnace',
    label: 'Monitor furnace',
    desc: '对焦+OCR 监控数显',
    icon: '🌡',
    needsTarget: true,
  },
  { kind: 'return_home', label: 'Return home', desc: '回到充电桩', icon: '⌂' },
]

const kind = ref<TaskKind>('fetch_sample')
const bottle = ref('SYGO-1')
const source = ref('shelf_1_slot_1')
const target = ref('furnace_1')
const priority = ref<'low' | 'normal' | 'high'>('normal')
const timeoutS = ref(90)
const taskId = ref(createTaskId())
const dispatchingPickup = ref(false)
const lastPickupResult = ref<PickupFlowCommandResult | null>(null)

const BOTTLES = ['SYGO-1', 'SYGO-2', 'SYGO-3', 'YCAS-1', 'YCAS-2', 'YAG:Cr-A', 'GAGG:Cr-B']
const SOURCES = ['shelf_1_slot_1', 'shelf_1_slot_2', 'shelf_2_slot_1', 'workstation', 'bay_2']
const TARGETS = ['furnace_1', 'furnace_2', 'shelf_1', 'shelf_2', 'charge_station', 'bay_2']
const PRIORITY_VALUE = { low: 1, normal: 2, high: 3 } as const

const PICKUP_STATE_LABEL: Record<PickupFlowStatus, string> = {
  idle: 'Idle',
  unknown: 'Unknown',
  waiting_dispatch: 'Waiting for DispatchTask',
  sent: 'Goal sent',
  accepted: 'Goal accepted',
  running: 'Running',
  simulated: 'Simulation complete',
  reported_completed: 'F407 sequence reported',
  completed: 'Completed (unclassified)',
  rejected: 'Rejected',
  failed: 'Failed',
  timeout: 'Timed out',
}

const currentDef = computed(() => TASK_DEFS.find((t) => t.kind === kind.value)!)
const bridgeAlive = computed(() => telemetry.packet?.bridge?.alive ?? false)
const pickup = computed(() => telemetry.packet?.bridge?.pickup_flow ?? null)

function createTaskId(): string {
  return `dispatch-${Date.now().toString(36)}-${Math.floor(Math.random() * 1296)
    .toString(36)
    .padStart(2, '0')}`
}

const pickupPayload = computed<PickupFlowCommandPayload>(() => ({
  task_id: taskId.value,
  task_type: 'fetch_sample',
  bottle_id: bottle.value,
  from_location: source.value,
  to_location: target.value,
  priority: PRIORITY_VALUE[priority.value],
  timeout_s: Math.min(150, Math.max(10, Number(timeoutS.value) || 90)),
}))

const payloadPreview = computed(() => {
  if (kind.value === 'fetch_sample') return pickupPayload.value
  const def = currentDef.value
  const args: Record<string, string> = {}
  if (def.needsBottle) args.bottle_id = bottle.value
  if (def.needsTarget) {
    if (def.kind === 'fetch_sample') {
      args.from = 'shelf_1'
      args.to = target.value
    } else args.target = target.value
  }
  return {
    task_id: taskId.value,
    type: kind.value,
    args,
    priority: priority.value,
  }
})

const matchingPickupResult = computed(() => {
  const result = lastPickupResult.value
  if (!result) return null
  const bridgeFlowId = pickup.value?.flow_id
  if (bridgeFlowId && result.flow_id && bridgeFlowId !== result.flow_id) return null
  return result
})

const pickupStateLabel = computed(() => PICKUP_STATE_LABEL[pickup.value?.state ?? 'unknown'])
const pickupTone = computed<Tone>(() => {
  const state = pickup.value?.state ?? 'unknown'
  if (state === 'reported_completed') return 'warn'
  if (state === 'simulated' || state === 'completed') return 'info'
  if (state === 'failed' || state === 'rejected' || state === 'timeout') return 'err'
  if (pickup.value?.active) return 'info'
  return state === 'idle' ? 'idle' : 'warn'
})
const simulatedOnly = computed(
  () =>
    pickup.value?.state === 'simulated' ||
    matchingPickupResult.value?.completion_class === 'simulated',
)
const f407Reported = computed(
  () =>
    pickup.value?.state === 'reported_completed' ||
    matchingPickupResult.value?.completion_class === 'f407_reported' ||
    matchingPickupResult.value?.completion_class === 'reported_completed' ||
    matchingPickupResult.value?.actuator_sequence_completed === true,
)
const physicalCompleted = computed(
  () =>
    pickup.value?.physical_completed === true ||
    matchingPickupResult.value?.physical_completed === true,
)
const pickupBlocked = computed(
  () =>
    dispatchingPickup.value ||
    pickup.value?.active === true ||
    !bridgeAlive.value ||
    telemetry.packet?.bridge?.estop === true,
)

async function dispatchPickupFlow() {
  if (pickupBlocked.value) return
  dispatchingPickup.value = true
  lastPickupResult.value = null
  try {
    const response = await fetch('/api/pickup_flow', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(pickupPayload.value),
    })
    const result = (await response.json()) as PickupFlowCommandResult
    lastPickupResult.value = result
    if (!response.ok || !result.ok) throw new Error(result.error || `HTTP ${response.status}`)

    const tone: Tone = result.physical_completed
      ? 'ok'
      : result.completion_class === 'f407_reported' ||
          result.completion_class === 'reported_completed'
        ? 'warn'
        : 'info'
    const title = result.physical_completed
      ? 'Pickup physically confirmed'
      : result.completion_class === 'f407_reported' ||
          result.completion_class === 'reported_completed'
        ? 'F407 sequence reported; physical pickup unconfirmed'
        : result.completion_class === 'simulated'
          ? 'Pickup simulation complete'
          : 'Pickup flow completed; physical pickup unconfirmed'
    toasts.push({ tone, title, detail: result.message, durationMs: 6000 })
    audio.play(result.physical_completed ? 'success' : 'ping')
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error)
    toasts.push({ tone: 'err', title: 'Pickup flow failed', detail, durationMs: 6000 })
    audio.play('err')
  } finally {
    dispatchingPickup.value = false
  }
}

async function dispatch() {
  if (kind.value === 'fetch_sample') {
    await dispatchPickupFlow()
    return
  }
  const p = payloadPreview.value
  toasts.push({
    tone: 'info',
    title: `Preview only · ${p.task_id}`,
    detail: `${kind.value} · priority ${p.priority}`,
    durationMs: 4000,
  })
  audio.play('click')
  emit('close')
}

function backdropClick(evt: MouseEvent) {
  if ((evt.target as HTMLElement).classList.contains('dt-backdrop')) emit('close')
}

watch(
  () => props.open,
  (o) => {
    if (o && !dispatchingPickup.value) {
      taskId.value = createTaskId()
      lastPickupResult.value = null
    }
  },
)
</script>

<template>
  <transition name="dt">
    <div v-if="open" class="dt-backdrop" @click="backdropClick">
      <div class="dt-card glass-strong" role="dialog" aria-label="Dispatch task">
        <div class="dt-head">
          <div>
            <div class="dt-title">Dispatch Task</div>
            <div class="dt-sub mono">
              {{
                kind === 'fetch_sample'
                  ? '/api/pickup_flow → DispatchTask safety path'
                  : 'Preview only · route not wired'
              }}
            </div>
          </div>
          <button type="button" class="dt-close" aria-label="Close dispatch" @click="emit('close')">
            ×
          </button>
        </div>

        <div class="dt-body">
          <div class="field">
            <label class="field-label">Task Kind</label>
            <div class="kind-grid">
              <button
                v-for="def in TASK_DEFS"
                :key="def.kind"
                type="button"
                class="kind-card"
                :class="{ active: kind === def.kind }"
                @click="kind = def.kind"
              >
                <span class="kind-icon">{{ def.icon }}</span>
                <div class="kind-info">
                  <div class="kind-label">{{ def.label }}</div>
                  <div class="kind-desc">{{ def.desc }}</div>
                </div>
              </button>
            </div>
          </div>

          <div class="field-row">
            <div v-if="kind === 'fetch_sample'" class="field">
              <label class="field-label">Source</label>
              <select v-model="source" class="input">
                <option v-for="s in SOURCES" :key="s" :value="s">{{ s }}</option>
              </select>
            </div>
            <div v-if="currentDef.needsBottle" class="field">
              <label class="field-label">Bottle</label>
              <select v-model="bottle" class="input">
                <option v-for="b in BOTTLES" :key="b" :value="b">{{ b }}</option>
              </select>
            </div>
            <div v-if="currentDef.needsTarget" class="field">
              <label class="field-label">
                {{ kind === 'fetch_sample' ? 'Destination' : 'Target' }}
              </label>
              <select v-model="target" class="input">
                <option v-for="t in TARGETS" :key="t" :value="t">{{ t }}</option>
              </select>
            </div>
            <div class="field">
              <label class="field-label">Priority</label>
              <div class="seg-tiny">
                <button
                  type="button"
                  class="seg-tiny-btn"
                  :class="{ active: priority === 'low' }"
                  @click="priority = 'low'"
                >
                  low
                </button>
                <button
                  type="button"
                  class="seg-tiny-btn"
                  :class="{ active: priority === 'normal' }"
                  @click="priority = 'normal'"
                >
                  normal
                </button>
                <button
                  type="button"
                  class="seg-tiny-btn"
                  :class="{ active: priority === 'high' }"
                  @click="priority = 'high'"
                >
                  high
                </button>
              </div>
            </div>
            <div v-if="kind === 'fetch_sample'" class="field">
              <label class="field-label">Timeout (s)</label>
              <input
                v-model.number="timeoutS"
                class="input"
                type="number"
                min="10"
                max="150"
                step="5"
              />
            </div>
          </div>

          <div v-if="kind === 'fetch_sample'" class="pickup-panel" aria-live="polite">
            <div class="pickup-head">
              <div>
                <div class="field-label">Bridge pickup state</div>
                <div class="pickup-task mono">{{ pickup?.task_id || 'no flow recorded' }}</div>
              </div>
              <span class="chip" :class="`chip-${pickupTone}`">{{ pickupStateLabel }}</span>
            </div>

            <div v-if="pickup?.active || (pickup?.progress_pct ?? 0) > 0" class="pickup-progress">
              <div
                class="pickup-progress-fill"
                :style="{ width: `${Math.min(100, Math.max(0, pickup?.progress_pct ?? 0))}%` }"
              ></div>
            </div>
            <div class="pickup-message">
              {{
                pickup?.stage_message ||
                pickup?.message ||
                pickup?.error ||
                (bridgeAlive ? 'Bridge ready' : 'Bridge offline')
              }}
              <span v-if="pickup?.elapsed_s !== undefined" class="mono">
                · {{ pickup.elapsed_s.toFixed(1) }}s
              </span>
            </div>

            <div class="truth-grid">
              <div class="truth-item" :class="simulatedOnly ? 'truth-info' : ''">
                <span class="truth-key">Simulation</span>
                <strong class="mono">
                  {{ simulatedOnly ? 'SIMULATED_ONLY' : 'not simulated' }}
                </strong>
              </div>
              <div class="truth-item" :class="f407Reported ? 'truth-warn' : ''">
                <span class="truth-key">Actuator report</span>
                <strong class="mono">
                  {{ f407Reported ? 'F407_REPORTED_COMPLETED' : 'not reported' }}
                </strong>
              </div>
              <div class="truth-item" :class="physicalCompleted ? 'truth-ok' : 'truth-negative'">
                <span class="truth-key">Physical truth</span>
                <strong class="mono">physical_completed={{ physicalCompleted }}</strong>
              </div>
            </div>
            <div v-if="!physicalCompleted" class="physical-note">
              No independent encoder, limit, or object-presence confirmation.
            </div>
          </div>

          <div class="field">
            <label class="field-label">Payload preview</label>
            <pre class="payload mono">{{ JSON.stringify(payloadPreview, null, 2) }}</pre>
          </div>
        </div>

        <div class="dt-foot">
          <span class="foot-hint mono">
            {{
              kind === 'fetch_sample'
                ? 'Live bridge state · reported execution is not physical proof'
                : 'Preview only · no command is sent'
            }}
          </span>
          <span class="foot-spacer"></span>
          <button type="button" class="btn" @click="emit('close')">Close</button>
          <button
            type="button"
            class="btn btn-primary"
            :disabled="kind === 'fetch_sample' && pickupBlocked"
            @click="dispatch"
          >
            {{
              kind === 'fetch_sample'
                ? dispatchingPickup
                  ? 'Running…'
                  : '▶ Start Pickup Flow'
                : 'Preview Dispatch'
            }}
          </button>
        </div>
      </div>
    </div>
  </transition>
</template>

<style scoped>
.dt-backdrop {
  position: fixed;
  inset: 0;
  background: rgba(11, 18, 32, 0.4);
  backdrop-filter: blur(10px) saturate(160%);
  -webkit-backdrop-filter: blur(10px) saturate(160%);
  z-index: 100;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 24px;
}
.dt-card {
  width: 580px;
  max-width: 100%;
  max-height: 90vh;
  border-radius: 18px;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.dt-head {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  padding: 18px 22px 14px;
  border-bottom: 1px solid var(--line-divider);
}
.dt-title {
  font-size: 1.1rem;
  font-weight: 700;
  color: var(--ink-primary);
  letter-spacing: -0.01em;
}
.dt-sub {
  font-size: 0.7rem;
  color: var(--ink-tertiary);
  margin-top: 3px;
}
.dt-close {
  background: transparent;
  border: none;
  width: 32px;
  height: 32px;
  font-size: 1.4rem;
  color: var(--ink-tertiary);
  border-radius: 8px;
  cursor: pointer;
}
.dt-close:hover {
  background: rgba(15, 23, 42, 0.06);
  color: var(--ink-primary);
}

.dt-body {
  flex: 1;
  overflow-y: auto;
  padding: 16px 22px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.field {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.field-label {
  font-size: 0.7rem;
  font-weight: 600;
  color: var(--ink-tertiary);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.field-row {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}
@media (max-width: 640px) {
  .field-row {
    grid-template-columns: 1fr;
  }
}

.kind-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 8px;
}
.kind-card {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  padding: 10px 12px;
  background: var(--bg-elevated);
  border: 1px solid var(--line-divider);
  border-radius: 10px;
  cursor: pointer;
  text-align: left;
  transition: all 0.18s var(--ease-out-quint);
  color: var(--ink-secondary);
}
.kind-card:hover {
  transform: translateY(-1px);
  border-color: var(--accent-blue);
}
.kind-card.active {
  background: linear-gradient(135deg, rgba(37, 99, 235, 0.1), rgba(8, 145, 178, 0.06));
  border-color: var(--accent-blue);
  color: var(--ink-primary);
  box-shadow: inset 0 0 0 1px rgba(37, 99, 235, 0.2);
}
.kind-icon {
  font-size: 1.1rem;
}
.kind-label {
  font-size: 0.82rem;
  font-weight: 600;
}
.kind-desc {
  font-size: 0.68rem;
  color: var(--ink-muted);
  margin-top: 1px;
}

.input {
  padding: 7px 10px;
  border-radius: 8px;
  border: 1px solid var(--line-border);
  background: var(--bg-card);
  color: var(--ink-primary);
  font-size: 0.8rem;
  font-family: inherit;
  outline: none;
  transition:
    border-color 0.18s var(--ease-out-quint),
    box-shadow 0.18s var(--ease-out-quint);
}
.input:focus {
  border-color: var(--accent-blue);
  box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15);
}

.seg-tiny {
  display: inline-flex;
  background: var(--bg-elevated);
  border: 1px solid var(--line-border);
  border-radius: 8px;
  overflow: hidden;
}
.seg-tiny-btn {
  background: transparent;
  border: none;
  padding: 6px 10px;
  font-size: 0.74rem;
  color: var(--ink-tertiary);
  cursor: pointer;
}
.seg-tiny-btn:hover {
  color: var(--ink-primary);
}
.seg-tiny-btn.active {
  background: linear-gradient(135deg, var(--accent-blue), var(--accent-teal));
  color: white;
}

.payload {
  background: var(--bg-elevated);
  border: 1px solid var(--line-divider);
  border-radius: 8px;
  padding: 10px 12px;
  font-size: 0.72rem;
  color: var(--ink-secondary);
  margin: 0;
  white-space: pre;
  overflow-x: auto;
  line-height: 1.5;
}

.pickup-panel {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 12px;
  border: 1px solid var(--line-divider);
  border-radius: 8px;
  background: color-mix(in srgb, var(--bg-elevated) 82%, transparent);
}
.pickup-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
}
.pickup-task {
  margin-top: 3px;
  font-size: 0.7rem;
  color: var(--ink-secondary);
}
.pickup-progress {
  height: 5px;
  overflow: hidden;
  border-radius: 999px;
  background: var(--line-hairline);
}
.pickup-progress-fill {
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, var(--accent-blue), var(--accent-teal));
  transition: width 0.28s var(--ease-out-quint);
}
.pickup-message {
  min-height: 1.2em;
  font-size: 0.72rem;
  color: var(--ink-secondary);
  line-height: 1.45;
}
.truth-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 8px;
}
.truth-item {
  min-width: 0;
  padding: 8px;
  border: 1px solid var(--line-divider);
  border-radius: 8px;
  background: var(--bg-card);
  color: var(--ink-tertiary);
}
.truth-item strong {
  display: block;
  margin-top: 4px;
  overflow-wrap: anywhere;
  color: var(--ink-secondary);
  font-size: 0.64rem;
  line-height: 1.35;
}
.truth-key {
  font-size: 0.66rem;
  font-weight: 700;
}
.truth-info {
  border-color: rgba(37, 99, 235, 0.35);
  background: rgba(37, 99, 235, 0.05);
}
.truth-warn {
  border-color: rgba(217, 119, 6, 0.38);
  background: rgba(217, 119, 6, 0.06);
}
.truth-ok {
  border-color: rgba(5, 150, 105, 0.38);
  background: rgba(5, 150, 105, 0.06);
}
.truth-negative {
  border-color: rgba(225, 29, 72, 0.28);
  background: rgba(225, 29, 72, 0.04);
}
.physical-note {
  font-size: 0.68rem;
  color: #be123c;
}
@media (max-width: 640px) {
  .truth-grid {
    grid-template-columns: 1fr;
  }
}

.dt-foot {
  display: flex;
  gap: 8px;
  align-items: center;
  padding: 12px 22px;
  border-top: 1px solid var(--line-divider);
}
.foot-hint {
  font-size: 0.66rem;
  color: var(--ink-muted);
}
.foot-spacer {
  flex: 1;
}
.dt-foot .btn {
  flex-shrink: 0;
  white-space: nowrap;
}
.dt-foot .btn:disabled {
  opacity: 0.45;
  cursor: not-allowed;
  transform: none;
}
@media (max-width: 640px) {
  .dt-foot {
    flex-wrap: wrap;
  }
  .foot-hint {
    flex: 1 0 100%;
  }
  .foot-spacer {
    display: none;
  }
  .dt-foot .btn {
    flex: 1;
  }
}

.dt-enter-from {
  opacity: 0;
}
.dt-enter-from .dt-card {
  transform: translateY(-8px) scale(0.96);
}
.dt-leave-to {
  opacity: 0;
}
.dt-leave-to .dt-card {
  transform: translateY(-4px) scale(0.98);
}
.dt-enter-active,
.dt-leave-active {
  transition: opacity 0.22s var(--ease-out-quint);
}
.dt-enter-active .dt-card,
.dt-leave-active .dt-card {
  transition: transform 0.22s var(--ease-out-quint);
}
</style>
