import * as THREE from 'three'

// Real myCobot 280 standard DH forward kinematics — for the "真实物理写照"
// skill trajectory preview. This is the SAME table as workstation/web/interlock.py
// `_DH` (the互锁 ghost FK), so the tip path drawn in the cockpit matches what the
// real arm's end-effector traces when replaying a recorded skill.
//
//   Source: Elephant Robotics myCobot 280 URDF (mycobot_ros) / GitBook DH table.
//   Units: mm / rad.  (alpha, a, d, theta_offset)
//
// NOTE: distinct from kinematics.ts, which uses a down-scaled *studio* chain that
// matches the stylised buildArm visual. Here we want true dimensions.
const DH: Array<[number, number, number, number]> = [
  [Math.PI / 2, 0.0, 131.22, 0.0],
  [0.0, -110.4, 0.0, -Math.PI / 2],
  [0.0, -96.0, 0.0, 0.0],
  [Math.PI / 2, 0.0, 63.4, -Math.PI / 2],
  [-Math.PI / 2, 0.0, 75.05, Math.PI / 2],
  [0.0, 0.0, 45.6, 0.0],
]

const DEG = Math.PI / 180

/** 6 joint angles (deg) → 7 origin points (base + 6 joint ends), mm, arm frame. */
export function mycobotFkPoints(anglesDeg: number[]): Array<[number, number, number]> {
  let m = new THREE.Matrix4() // identity
  const pts: Array<[number, number, number]> = [[0, 0, 0]]
  for (let i = 0; i < 6; i++) {
    const [alpha, a, d, off] = DH[i]
    const theta = (anglesDeg[i] ?? 0) * DEG + off
    const ct = Math.cos(theta), st = Math.sin(theta)
    const ca = Math.cos(alpha), sa = Math.sin(alpha)
    // standard DH transform (row-major args to Matrix4.set)
    const t = new THREE.Matrix4().set(
      ct, -st * ca, st * sa, a * ct,
      st, ct * ca, -ct * sa, a * st,
      0, sa, ca, d,
      0, 0, 0, 1,
    )
    m = m.multiply(t)
    const e = m.elements // column-major: translation at 12,13,14
    pts.push([e[12], e[13], e[14]])
  }
  return pts
}

/** End-effector position only (mm). */
export function mycobotTip(anglesDeg: number[]): [number, number, number] {
  const p = mycobotFkPoints(anglesDeg)
  return p[p.length - 1]
}

export interface TipPath {
  dense: Array<[number, number, number]>  // smooth interpolated tip path, metres
  marks: Array<[number, number, number]>  // waypoint tips, metres
  reachMm: number                          // max tip distance from base, for autoscale
}

/**
 * Build a smooth end-effector trajectory from sparse joint waypoints.
 * Lerps joint angles between consecutive waypoints (`steps` substeps) and runs
 * real FK at each — the tip follows the genuine curved path through joint space.
 * Output positions are converted mm → m and re-centred to the base.
 */
export function tipPathFromWaypoints(waypoints: number[][], steps = 12): TipPath {
  const dense: Array<[number, number, number]> = []
  const marks: Array<[number, number, number]> = []
  let reach = 0
  const push = (p: [number, number, number], arr: Array<[number, number, number]>) => {
    arr.push([p[0] * 0.001, p[2] * 0.001, -p[1] * 0.001]) // arm(x,y,z mm) → scene(x,z,-y m)
    reach = Math.max(reach, Math.hypot(p[0], p[1], p[2]))
  }
  if (!waypoints.length) return { dense, marks, reachMm: 1 }
  for (const w of waypoints) push(mycobotTip(w), marks)
  for (let k = 0; k < waypoints.length - 1; k++) {
    const a = waypoints[k], b = waypoints[k + 1]
    for (let s = 0; s < steps; s++) {
      const u = s / steps
      const j = a.map((v, i) => v + (b[i] - v) * u)
      push(mycobotTip(j), dense)
    }
  }
  push(mycobotTip(waypoints[waypoints.length - 1]), dense)
  return { dense, marks, reachMm: reach || 1 }
}
