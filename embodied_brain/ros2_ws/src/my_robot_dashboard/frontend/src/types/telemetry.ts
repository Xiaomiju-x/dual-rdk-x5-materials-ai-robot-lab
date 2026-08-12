// Wire-side types that mirror backend/api/models.py.
// Kept in sync by hand — only the field names matter, the rest is plain JSON.

export type Tone = 'ok' | 'warn' | 'err' | 'info' | 'idle'
export type Accent = 'blue' | 'teal' | 'emerald' | 'violet' | 'amber' | 'rose'

export interface Heartbeat {
  epoch_ms: number
  uptime_s: number
  sequence: number
}

export interface Pose2D {
  x: number
  y: number
  yaw: number
}

export interface Velocity {
  linear: number
  angular: number
}

export interface Battery {
  pct: number
  voltage: number
  current: number
  temp_c: number
}

export interface HostResources {
  cpu_pct: number
  ram_gb: number
  ram_total_gb: number
  cma_mb: number
  cma_total_mb: number
  bpu_pct: number
}

export interface Sensor {
  id: string
  label: string
  kind: 'lidar' | 'depth_cam' | 'rgb_cam' | 'imu' | 'odom' | 'mic_array' | 'temp' | 'magnet'
  health: Tone
  hz: number
  detail: string
}

export interface Camera {
  id: string
  label: string
  width: number
  height: number
  fps: number
  online: boolean
  stream_url: string | null
  provenance?: string | null
}

export interface BpuSlot {
  id: string
  label: string
  model: string
  size_mb: number
  last_ms: number
  util_pct: number
  state?: string | null
  runtime?: string | null
  used?: boolean | null
  util_available?: boolean
}

export interface TaskState {
  id: string
  name: string
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled'
  progress_pct: number
  eta_s: number | null
  started_at_ms: number | null
}

export interface Alarm {
  id: string
  severity: Tone
  title: string
  detail: string
  at_ms: number
  source: string
}

export interface FurnaceReading {
  id: string
  pv: number
  sv: number
  mv: number
  state: 'heating' | 'holding' | 'cooling' | 'idle' | 'fault'
}

export interface KpiCard {
  key: string
  label: string
  value: string
  unit: string
  tone: Tone
  trend: string
  accent: Accent
}

export interface TimelineEvent {
  id: string
  track: 'dispatch' | 'nav' | 'perception' | 'ai_brain' | 'system'
  label: string
  start_ms: number
  end_ms: number
  status: Tone
  detail: string
}

export interface Waypoint {
  id: string
  x: number
  y: number
  label: string
  kind: 'start' | 'pickup' | 'dropoff' | 'patrol' | 'home'
  eta_s: number | null
}

export interface AiLinkState {
  online: boolean
  rtt_ms: number
  last_seen_ms: number
  dispatches_24h: number
  last_dispatch_label: string
  endpoint: string
}

// ---- 第 3 期: cockpit_bridge 真数据覆盖 (backend/bridge_state.overlay) ----

export interface BridgeDetection {
  label: string
  conf: number
}

export interface BridgeSafety {
  speed_cap: number
  fence: Array<[number, number]> | null
  fence_enabled: boolean
}

export type PickupFlowStatus =
  | 'idle'
  | 'unknown'
  | 'waiting_dispatch'
  | 'sent'
  | 'accepted'
  | 'running'
  | 'simulated'
  | 'reported_completed'
  | 'completed'
  | 'rejected'
  | 'failed'
  | 'timeout'

export type PickupFlowCompletionClass =
  | 'simulated'
  | 'f407_reported'
  | 'reported_completed'
  | 'physical'
  | 'completed'
  | 'unverified'
  | 'rejected'
  | 'timeout'
  | 'failed'

export interface PickupFlowState {
  active: boolean
  state: PickupFlowStatus
  flow_id?: string
  task_id?: string
  task_type?: string
  bottle_id?: string
  from_location?: string
  to_location?: string
  message?: string
  error?: string
  stage?: number
  progress_pct?: number
  stage_message?: string
  elapsed_s?: number
  updated_at?: number
  completion_class?: PickupFlowCompletionClass
  actuator_sequence_completed?: boolean
  /** True only when the bridge supplies independent physical evidence. */
  physical_completed?: boolean
  /** Canonical evidence JSON; empty unless physical_completed is true. */
  physical_confirmation?: string
  /** True only when this dispatch requested Nav2/base motion. */
  base_motion_requested?: boolean | null
}

export interface PickupFlowCommandPayload {
  task_id: string
  task_type: 'fetch_sample'
  bottle_id: string
  from_location: string
  to_location: string
  priority: 1 | 2 | 3
  timeout_s: number
}

export interface PickupFlowCommandResult {
  ok: boolean
  error?: string
  message?: string
  flow_id?: string
  task_id?: string
  task_type?: string
  bottle_id?: string
  from_location?: string
  to_location?: string
  completion_class?: PickupFlowCompletionClass
  actuator_sequence_completed?: boolean
  physical_completed?: boolean
  physical_confirmation?: string
  base_motion_requested?: boolean | null
  elapsed_s?: number
  note?: string
}

export interface BridgeInfo {
  alive: boolean
  estop: boolean
  motion_busy: boolean
  pickup_flow: PickupFlowState
  safety: BridgeSafety | null
  detections: BridgeDetection[]
  /** 降采样雷达点 (x,y) base frame, 0.5Hz 刷新 */
  scan: Array<[number, number]> | null
  sys_extra: {
    cpu_temp_c: number | null
    ai_brain_reachable: boolean | null
    ai_brain_latency_ms: number | null
    slam_active: boolean | null
    nav2_active: boolean | null
    nav2_state: string | null
    battery_pct: number | null
    distance_m: number | null
  } | null
  lab_fsd: Record<string, unknown> | null
  f407: Record<string, unknown> | null
  map_etag: number
}

export interface TelemetryProvenance {
  mode: 'live_partial' | 'fixture_only'
  live_fields: string[]
  fixture_fields: string[]
  unavailable_fields: string[]
  note: string
}

export interface TelemetryPacket {
  heartbeat: Heartbeat
  pose: Pose2D
  velocity: Velocity
  battery: Battery
  host: HostResources
  kpis: KpiCard[]
  sensors: Sensor[]
  cameras: Camera[]
  bpu_slots: BpuSlot[]
  tasks: TaskState[]
  alarms: Alarm[]
  furnaces: FurnaceReading[]
  timeline: TimelineEvent[]
  waypoints: Waypoint[]
  ai_link: AiLinkState | null
  /** 哪些字段被真车数据覆盖了 (诚实标注, 桥离线时为空对象) */
  real?: Record<string, boolean>
  /** cockpit_bridge 桥接元数据 (estop/安全围栏/检测/雷达点) */
  bridge?: BridgeInfo
  /** Explicit field-level truth source. Live mode never backfills missing values from fixtures. */
  provenance?: TelemetryProvenance | null
}

export interface WsEnvelope {
  type: 'hello' | 'telemetry' | 'pong'
  payload?: TelemetryPacket
}
