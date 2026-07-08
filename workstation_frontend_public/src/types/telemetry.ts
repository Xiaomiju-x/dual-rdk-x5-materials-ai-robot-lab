// Wire-side types — mirror the Flask API in workstation/web (mock.py / real.py).
// Only field names matter; the rest is plain JSON.

export type Tone = 'ok' | 'warn' | 'err' | 'info' | 'idle'
export type Accent = 'blue' | 'teal' | 'emerald' | 'violet' | 'amber' | 'rose'
export type ArmId = 'arm01' | 'arm02'

/** GET /api/joints/<arm> */
export interface JointPacket {
  arm: ArmId
  ts: number
  angles: number[]      // 6 joint angles, deg
  gripper: number       // 0..100 (jaw closed %)
  online?: boolean
}

/** GET /api/status */
export interface StatusPacket {
  arm01: { online: boolean; temp_c: number | null }
  arm02: { online: boolean; temp_c: number | null }
  cam01_fps: number | null
  cam02_fps: number | null
  ai_brain_ms: number | null
  car_brain_ms: number | null
}

/** GET /api/coop/events item */
export interface CoopEvent {
  ts: number
  kind: string
  src: string
  dst: string
  endpoint: string
  rtt_ms: number
  bytes: number
  ok: boolean
}

/** GET /api/coop/throughput */
export interface CoopThroughput {
  now: number
  arm_to_ai: number[]
  arm_to_car: number[]
  car_to_ai: number[]
}

export interface HealthPacket {
  status: string
  mode: 'mock' | 'real'
  host: string
  service: string
}
