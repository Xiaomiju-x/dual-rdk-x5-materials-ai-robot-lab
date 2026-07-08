import { defineStore } from 'pinia'
import { computed, ref, shallowRef, triggerRef } from 'vue'
import type {
  ArmId, JointPacket, StatusPacket, CoopEvent, CoopThroughput, HealthPacket,
} from '@/types/telemetry'

type ConnectionState = 'idle' | 'connecting' | 'open' | 'closed' | 'error'

const POLL_MS = 100              // 10 Hz joint/status poll
const COOP_MS = 1500             // co-op events + throughput poll
const HISTORY_LEN = 600          // 60 s at 10 Hz

export interface HistorySample {
  t: number
  v: number
}

const ZERO_JOINTS = (arm: ArmId): JointPacket => ({
  arm, ts: 0, angles: [0, 0, 0, 0, 0, 0], gripper: 50, online: false,
})

export const useTelemetryStore = defineStore('telemetry', () => {
  const arm01 = shallowRef<JointPacket>(ZERO_JOINTS('arm01'))
  const arm02 = shallowRef<JointPacket>(ZERO_JOINTS('arm02'))
  const status = shallowRef<StatusPacket | null>(null)
  const coopEvents = shallowRef<CoopEvent[]>([])
  const coopThroughput = shallowRef<CoopThroughput | null>(null)
  const health = shallowRef<HealthPacket | null>(null)

  const state = ref<ConnectionState>('idle')
  const lastFrameAt = ref(0)
  const frameCount = ref(0)
  const observedHz = ref(0)
  const lastError = ref('')
  const paused = ref(false)

  // ring buffers for sparklines (status metrics over time)
  const history = shallowRef<Record<string, HistorySample[]>>({
    arm01_temp: [], arm02_temp: [], cam01_fps: [], cam02_fps: [],
    ai_ms: [], car_ms: [],
  })

  let pollTimer: number | null = null
  let coopTimer: number | null = null
  let hzTimer: number | null = null
  let lastHzCount = 0
  let lastHzTick = 0

  const isConnected = computed(() => state.value === 'open')
  const isStale = computed(() =>
    state.value === 'open' && lastFrameAt.value > 0 && Date.now() - lastFrameAt.value > 1200)
  const mode = computed(() => health.value?.mode ?? 'mock')

  const aiOnline = computed(() => (status.value?.ai_brain_ms ?? null) != null)
  const carOnline = computed(() => (status.value?.car_brain_ms ?? null) != null)

  function jointsOf(arm: ArmId) { return arm === 'arm01' ? arm01.value : arm02.value }

  function pushRing(key: string, sample: HistorySample) {
    const buf = history.value[key]
    if (!buf) return
    buf.push(sample)
    if (buf.length > HISTORY_LEN) buf.splice(0, buf.length - HISTORY_LEN)
  }

  async function getJSON<T>(url: string): Promise<T> {
    const r = await fetch(url, { cache: 'no-store' })
    if (!r.ok) throw new Error(`${url} → ${r.status}`)
    return r.json() as Promise<T>
  }

  async function pollOnce() {
    if (paused.value) { lastFrameAt.value = Date.now(); frameCount.value++; return }
    try {
      const [j1, j2, st] = await Promise.all([
        getJSON<JointPacket>('/api/joints/arm01'),
        getJSON<JointPacket>('/api/joints/arm02'),
        getJSON<StatusPacket>('/api/status'),
      ])
      arm01.value = j1
      arm02.value = j2
      status.value = st
      const t = Date.now()
      pushRing('arm01_temp', { t, v: st.arm01?.temp_c ?? 0 })
      pushRing('arm02_temp', { t, v: st.arm02?.temp_c ?? 0 })
      pushRing('cam01_fps', { t, v: st.cam01_fps ?? 0 })
      pushRing('cam02_fps', { t, v: st.cam02_fps ?? 0 })
      pushRing('ai_ms', { t, v: st.ai_brain_ms ?? 0 })
      pushRing('car_ms', { t, v: st.car_brain_ms ?? 0 })
      triggerRef(history)
      state.value = 'open'
      lastError.value = ''
      lastFrameAt.value = t
      frameCount.value++
    } catch (e) {
      state.value = 'error'
      lastError.value = (e as Error).message
    }
  }

  async function pollCoop() {
    if (paused.value) return
    try {
      const [ev, tp] = await Promise.all([
        getJSON<CoopEvent[]>('/api/coop/events'),
        getJSON<CoopThroughput>('/api/coop/throughput'),
      ])
      coopEvents.value = ev
      coopThroughput.value = tp
    } catch (_e) { /* coop is best-effort */ }
  }

  async function connect() {
    if (pollTimer !== null) return
    state.value = 'connecting'
    try { health.value = await getJSON<HealthPacket>('/api/health') } catch (_e) { /* ignore */ }
    lastHzCount = 0
    lastHzTick = performance.now()

    await pollOnce()
    pollTimer = window.setInterval(pollOnce, POLL_MS)
    pollCoop()
    coopTimer = window.setInterval(pollCoop, COOP_MS)
    hzTimer = window.setInterval(() => {
      const now = performance.now()
      const dt = (now - lastHzTick) / 1000
      if (dt > 0) observedHz.value = Math.round(((frameCount.value - lastHzCount) / dt) * 10) / 10
      lastHzCount = frameCount.value
      lastHzTick = now
    }, 2000)
  }

  function disconnect() {
    for (const t of [pollTimer, coopTimer, hzTimer]) if (t !== null) window.clearInterval(t)
    pollTimer = coopTimer = hzTimer = null
    state.value = 'idle'
  }

  function setPaused(p: boolean) { paused.value = p }
  function togglePaused() { paused.value = !paused.value }

  function buffer(key: string): HistorySample[] { return history.value[key] ?? [] }

  return {
    arm01, arm02, status, coopEvents, coopThroughput, health,
    state, isConnected, isStale, mode, aiOnline, carOnline,
    lastFrameAt, frameCount, observedHz, lastError, paused, history,
    jointsOf, buffer, connect, disconnect, setPaused, togglePaused,
  }
})
