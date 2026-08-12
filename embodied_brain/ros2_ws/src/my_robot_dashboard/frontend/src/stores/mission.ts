import { defineStore } from 'pinia'
import { computed, ref, watch } from 'vue'
import { useTelemetryStore } from './telemetry'
import type { Tone } from '@/types/telemetry'

export type MissionPhase = 'IDLE' | 'TRANSIT' | 'PICKUP' | 'RETURN' | 'DOCK' | 'FAULT'

export interface Subsystem {
  id: string
  label: string
  status: 'GO' | 'NO-GO' | 'WARN'
  /** for tooltip drill */
  detail: string
}

/**
 * Mission store — derives a flight-director-style summary from raw
 * telemetry, drives MissionClock / MissionPhase / GoNoGoGrid / MasterAlarm.
 *
 *  - MET (mission elapsed time) ticks from packet.heartbeat.uptime_s
 *  - phase inferred from packet.tasks[0] + pose distance to origin
 *  - subsystems derived from packet.sensors / cameras / bpu_slots / ai_link
 *  - master alarm latches when any err alarm is in the recent window
 *    until user silences it explicitly
 */
export const useMissionStore = defineStore('mission', () => {
  const telemetry = useTelemetryStore()

  // ----- alarm latch / silence -----
  const masterAlarmSilenced = ref(false)
  const lastAcknowledgedAlarmId = ref<string | null>(null)

  function silenceMasterAlarm() {
    const recent = telemetry.alarmHistory.slice(-1)[0]
    lastAcknowledgedAlarmId.value = recent?.id ?? null
    masterAlarmSilenced.value = true
  }
  function rearmMasterAlarm() {
    masterAlarmSilenced.value = false
    lastAcknowledgedAlarmId.value = null
  }

  // ----- MET (mission elapsed time) -----
  // packet.heartbeat.uptime_s is the backend's clock; we treat it as
  // "since last service restart" which is good enough for demos.
  const metSeconds = computed(() => {
    const u = telemetry.packet?.heartbeat?.uptime_s ?? 0
    return Math.max(0, Math.floor(u))
  })

  const metFormatted = computed(() => {
    const s = metSeconds.value
    const hh = Math.floor(s / 3600).toString().padStart(2, '0')
    const mm = Math.floor((s % 3600) / 60).toString().padStart(2, '0')
    const ss = (s % 60).toString().padStart(2, '0')
    return `${hh}:${mm}:${ss}`
  })

  // ----- phase inference -----
  // priority: explicit override > faulted > task-driven > pose-driven > idle
  const phaseOverride = ref<MissionPhase | null>(null)
  function setPhaseOverride(p: MissionPhase | null) { phaseOverride.value = p }

  const phaseInferred = computed<MissionPhase>(() => {
    if (phaseOverride.value) return phaseOverride.value
    const pkt = telemetry.packet
    if (!pkt) return 'IDLE'
    const recentErr = telemetry.alarmHistory.slice(-3).some((a) => a.severity === 'err')
    if (recentErr) return 'FAULT'
    const task = pkt.tasks?.find((t) => t.status === 'running')
    if (task) {
      const name = task.name.toLowerCase()
      if (name.includes('pickup') || name.includes('fetch')) return 'PICKUP'
      if (name.includes('return') || name.includes('home')) return 'RETURN'
      if (name.includes('dock') || name.includes('charge')) return 'DOCK'
      return 'TRANSIT'
    }
    const v = pkt.velocity?.linear ?? 0
    if (v > 0.05) return 'TRANSIT'
    return 'IDLE'
  })

  const PHASE_TONE: Record<MissionPhase, Tone> = {
    IDLE: 'idle', TRANSIT: 'info', PICKUP: 'warn', RETURN: 'info', DOCK: 'ok', FAULT: 'err',
  }
  const phaseTone = computed<Tone>(() => PHASE_TONE[phaseInferred.value])

  // ----- subsystem GO/NO-GO grid -----
  const subsystems = computed<Subsystem[]>(() => {
    const pkt = telemetry.packet
    const out: Subsystem[] = []
    const unavailable = new Set(pkt?.provenance?.unavailable_fields ?? [])
    // LIDAR — health of ld14 sensor
    const ld = pkt?.sensors?.find((s) => s.id === 'ld14')
    out.push({ id: 'LIDAR', label: 'LIDAR', status: ld?.health === 'ok' ? 'GO' : ld ? 'WARN' : 'NO-GO', detail: ld ? `${ld.hz.toFixed(1)} Hz` : 'no signal' })
    // CAM
    const camOnline = (pkt?.cameras ?? []).filter((c) => c.online).length
    const liveDepth = pkt?.sensors?.some((s) => s.id === 'astra_depth' && s.health === 'ok') ?? false
    out.push({ id: 'CAM', label: 'CAM', status: camOnline > 0 || liveDepth ? 'GO' : 'NO-GO', detail: camOnline > 0 ? `${camOnline}/${pkt?.cameras?.length ?? 0} online` : liveDepth ? 'depth source live' : 'no confirmed source' })
    // IMU
    const imu = pkt?.sensors?.find((s) => s.id === 'imu')
    out.push({ id: 'IMU', label: 'IMU', status: imu?.health === 'ok' ? 'GO' : imu ? 'WARN' : 'NO-GO', detail: imu ? `${imu.hz.toFixed(0)} Hz` : 'no signal' })
    // NAV
    const odom = pkt?.sensors?.find((s) => s.id === 'odom')
    out.push({ id: 'NAV', label: 'NAV', status: odom?.health === 'ok' ? 'GO' : odom ? 'WARN' : 'NO-GO', detail: pkt?.pose ? `(${pkt.pose.x.toFixed(1)},${pkt.pose.y.toFixed(1)})` : 'no pose' })
    // BPU
    const bpuActive = (pkt?.bpu_slots ?? []).length
    out.push({ id: 'BPU', label: 'BPU', status: bpuActive > 0 ? 'GO' : 'NO-GO', detail: bpuActive > 0 ? `${bpuActive} verified slot${bpuActive !== 1 ? 's' : ''}` : 'no verified runtime' })
    // AI link
    const ai = pkt?.ai_link
    out.push({ id: 'AI', label: 'AI', status: ai?.online ? 'GO' : 'NO-GO', detail: ai?.online ? `${ai.rtt_ms.toFixed(0)}ms RTT` : 'offline' })
    // STM32 (proxied from magnet sensor if present, otherwise speculative)
    const f407 = pkt?.sensors?.find((s) => s.id === 'f407')
    out.push({ id: 'STM32', label: 'STM32', status: f407 ? (f407.health === 'ok' ? 'GO' : 'WARN') : 'NO-GO', detail: f407 ? f407.detail : 'firmware identity unavailable' })
    // POWER (battery)
    const pct = pkt?.battery?.pct ?? 0
    const batteryUnavailable = unavailable.has('battery')
    out.push({ id: 'POWER', label: 'POWER', status: batteryUnavailable ? 'WARN' : pct > 20 ? 'GO' : pct > 10 ? 'WARN' : 'NO-GO', detail: batteryUnavailable ? 'battery telemetry unavailable' : `${pct.toFixed(0)}%` })
    return out
  })

  const subsystemsGo = computed(() => subsystems.value.filter((s) => s.status === 'GO').length)
  const subsystemsAll = computed(() => subsystems.value.length)
  const allGo = computed(() => subsystemsGo.value === subsystemsAll.value)

  // ----- master alarm: any unread err alarm latches on -----
  const masterAlarmActive = computed(() => {
    if (masterAlarmSilenced.value) return false
    const recent = telemetry.alarmHistory.slice(-10)
    const errs = recent.filter((a) => a.severity === 'err')
    if (errs.length === 0) return false
    // active if the latest err is past the last acked id
    if (!lastAcknowledgedAlarmId.value) return true
    const latest = errs[errs.length - 1]
    return latest.id !== lastAcknowledgedAlarmId.value
  })

  // when a NEW err arrives after silence, automatically rearm
  // (a fresh err deserves attention even after silence)
  watch(() => telemetry.alarmHistory.slice(-1)[0], (recent) => {
    if (!masterAlarmSilenced.value) return null
    if (recent && recent.severity === 'err' && recent.id !== lastAcknowledgedAlarmId.value) {
      // new err — auto re-arm
      queueMicrotask(() => { masterAlarmSilenced.value = false })
    }
    return null
  })

  return {
    metSeconds,
    metFormatted,
    phaseOverride,
    phaseInferred,
    phaseTone,
    setPhaseOverride,
    subsystems,
    subsystemsGo,
    subsystemsAll,
    allGo,
    masterAlarmActive,
    masterAlarmSilenced,
    silenceMasterAlarm,
    rearmMasterAlarm,
  }
})
