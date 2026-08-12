import { defineStore } from 'pinia'
import { computed, ref, shallowRef, triggerRef } from 'vue'
import type { Alarm, TelemetryPacket, WsEnvelope } from '@/types/telemetry'

type ConnectionState = 'idle' | 'connecting' | 'open' | 'closed' | 'error'

const RECONNECT_DELAYS = [500, 1000, 2000, 4000, 8000] as const
const SENSOR_HISTORY_LEN = 600 // 60 s at 10 Hz
const HOST_HISTORY_LEN = 600

export interface HistorySample {
  t: number
  v: number
}

export const useTelemetryStore = defineStore('telemetry', () => {
  const packet = shallowRef<TelemetryPacket | null>(null)
  const state = ref<ConnectionState>('idle')
  const lastFrameAt = ref(0)
  const frameCount = ref(0)
  const observedHz = ref(0)
  const lastError = ref('')

  // ring buffers — kept as shallowRef Maps. We mutate the inner array in place
  // (push + shift) for speed and trigger reactivity manually.
  const sensorHistory = shallowRef<Map<string, HistorySample[]>>(new Map())
  const hostHistory = shallowRef<{
    cpu: HistorySample[]
    bpu: HistorySample[]
    ram: HistorySample[]
    cma: HistorySample[]
  }>({ cpu: [], bpu: [], ram: [], cma: [] })
  const bpuSlotHistory = shallowRef<Map<string, HistorySample[]>>(new Map())
  const alarmHistory = shallowRef<Alarm[]>([])
  const alarmIds = new Set<string>()
  const ALARM_HISTORY_LEN = 100

  let ws: WebSocket | null = null
  let attempt = 0
  let reconnectTimer: number | null = null
  let hzTimer: number | null = null
  let lastHzCount = 0
  let lastHzTick = 0

  const isConnected = computed(() => state.value === 'open')
  const isStale = computed(() => {
    if (state.value !== 'open' || lastFrameAt.value === 0) return false
    return Date.now() - lastFrameAt.value > 1500
  })

  function wsUrl() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:'
    return `${proto}//${location.host}/ws/telemetry`
  }

  function pushRing(buf: HistorySample[], sample: HistorySample, max: number): void {
    buf.push(sample)
    if (buf.length > max) buf.splice(0, buf.length - max)
  }

  // when paused, we keep counting frames (for Hz / liveness) but stop updating
  // the visible `packet` so the UI freezes on a single snapshot. The WS stays open.
  const paused = ref(false)
  function setPaused(p: boolean) { paused.value = p }

  function ingest(env: WsEnvelope) {
    if (env.type === 'pong') return
    if (!env.payload) return
    if (!paused.value) packet.value = env.payload
    lastFrameAt.value = Date.now()
    frameCount.value += 1
    if (paused.value) return

    const t = env.payload.heartbeat.epoch_ms

    // sensor ring buffers
    const sm = sensorHistory.value
    for (const s of env.payload.sensors) {
      let buf = sm.get(s.id)
      if (!buf) {
        buf = []
        sm.set(s.id, buf)
      }
      pushRing(buf, { t, v: s.hz }, SENSOR_HISTORY_LEN)
    }
    triggerRef(sensorHistory)

    // host ring buffers
    const hh = hostHistory.value
    pushRing(hh.cpu, { t, v: env.payload.host.cpu_pct }, HOST_HISTORY_LEN)
    pushRing(hh.bpu, { t, v: env.payload.host.bpu_pct }, HOST_HISTORY_LEN)
    pushRing(hh.ram, { t, v: env.payload.host.ram_gb }, HOST_HISTORY_LEN)
    pushRing(hh.cma, { t, v: env.payload.host.cma_mb }, HOST_HISTORY_LEN)
    triggerRef(hostHistory)

    // bpu slot ring buffers
    const bm = bpuSlotHistory.value
    for (const slot of env.payload.bpu_slots) {
      let buf = bm.get(slot.id)
      if (!buf) {
        buf = []
        bm.set(slot.id, buf)
      }
      pushRing(buf, { t, v: slot.util_pct }, SENSOR_HISTORY_LEN)
    }
    triggerRef(bpuSlotHistory)

    // alarm history — dedupe on id, cap at ALARM_HISTORY_LEN
    let alarmsDirty = false
    for (const a of env.payload.alarms ?? []) {
      if (alarmIds.has(a.id)) continue
      alarmIds.add(a.id)
      alarmHistory.value.push(a)
      alarmsDirty = true
    }
    if (alarmHistory.value.length > ALARM_HISTORY_LEN) {
      const trimmed = alarmHistory.value.slice(-ALARM_HISTORY_LEN)
      for (const a of alarmHistory.value.slice(0, alarmHistory.value.length - ALARM_HISTORY_LEN)) {
        alarmIds.delete(a.id)
      }
      alarmHistory.value = trimmed
      alarmsDirty = true
    }
    if (alarmsDirty) triggerRef(alarmHistory)
  }

  function connect() {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
      return
    }
    state.value = 'connecting'
    lastError.value = ''
    const sock = new WebSocket(wsUrl())
    ws = sock

    sock.addEventListener('open', () => {
      state.value = 'open'
      attempt = 0
      lastHzCount = frameCount.value
      lastHzTick = performance.now()
    })

    sock.addEventListener('message', (evt) => {
      try {
        const env = JSON.parse(evt.data as string) as WsEnvelope
        ingest(env)
      } catch (e) {
        lastError.value = `parse: ${(e as Error).message}`
      }
    })

    sock.addEventListener('close', () => {
      if (state.value !== 'error') state.value = 'closed'
      ws = null
      scheduleReconnect()
    })

    sock.addEventListener('error', () => {
      state.value = 'error'
      lastError.value = 'websocket error'
    })

    if (hzTimer === null) {
      hzTimer = window.setInterval(() => {
        const now = performance.now()
        const dt = (now - lastHzTick) / 1000
        if (dt > 0) {
          observedHz.value = Math.round(((frameCount.value - lastHzCount) / dt) * 10) / 10
        }
        lastHzCount = frameCount.value
        lastHzTick = now
      }, 2000)
    }
  }

  function scheduleReconnect() {
    if (reconnectTimer !== null) return
    const delay = RECONNECT_DELAYS[Math.min(attempt, RECONNECT_DELAYS.length - 1)]
    attempt += 1
    reconnectTimer = window.setTimeout(() => {
      reconnectTimer = null
      connect()
    }, delay)
  }

  function disconnect() {
    if (reconnectTimer !== null) {
      window.clearTimeout(reconnectTimer)
      reconnectTimer = null
    }
    if (hzTimer !== null) {
      window.clearInterval(hzTimer)
      hzTimer = null
    }
    if (ws) {
      ws.close()
      ws = null
    }
    state.value = 'idle'
  }

  function sensorBuffer(id: string): HistorySample[] {
    return sensorHistory.value.get(id) ?? []
  }

  function bpuSlotBuffer(id: string): HistorySample[] {
    return bpuSlotHistory.value.get(id) ?? []
  }

  return {
    packet,
    state,
    isConnected,
    isStale,
    paused,
    lastFrameAt,
    frameCount,
    observedHz,
    lastError,
    sensorHistory,
    hostHistory,
    bpuSlotHistory,
    alarmHistory,
    sensorBuffer,
    bpuSlotBuffer,
    setPaused,
    connect,
    disconnect,
  }
})
