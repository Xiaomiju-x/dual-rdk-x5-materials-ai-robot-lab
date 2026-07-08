import * as THREE from 'three'
import { fk, AXES, maxReach } from './kinematics'

// Damped Least-Squares (DLS) numerical IK for a 6-DoF chain.
// Position-only: minimises ||fk(q) - target||² subject to joint limits.
// Reads jacobian via central difference (cheap and robust enough at 30Hz UI).
// Output is in degrees, same domain as the RingSlider targets.

const DEG2RAD = Math.PI / 180
const RAD2DEG = 180 / Math.PI

export interface IKOptions {
  maxIter?: number
  tolerance?: number     // metres
  damping?: number       // λ for DLS
  stepClamp?: number     // deg per joint per iter
  limits?: [number, number][]   // deg, length 6
}

export interface IKResult {
  angles: number[]         // 6 deg
  reachable: boolean
  finalErrorM: number
  iterations: number
}

const DEFAULT_LIMITS: [number, number][] = [
  [-168, 168], [-135, 135], [-150, 150], [-145, 145], [-165, 165], [-180, 180],
]

function clampLimits(q: number[], limits: [number, number][]): void {
  for (let i = 0; i < q.length; i++) {
    const [lo, hi] = limits[i]
    if (q[i] < lo) q[i] = lo
    else if (q[i] > hi) q[i] = hi
  }
}

/** Numerical jacobian via central difference (deg → m). Returns 3×6 row-major. */
function jacobian(q: number[]): number[][] {
  const J: number[][] = [[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]
  const eps = 0.5  // half-degree probe
  for (let j = 0; j < 6; j++) {
    const qp = q.slice(); qp[j] += eps
    const qm = q.slice(); qm[j] -= eps
    const fp = fk(qp)
    const fm = fk(qm)
    J[0][j] = (fp.x - fm.x) / (2 * eps * DEG2RAD)
    J[1][j] = (fp.y - fm.y) / (2 * eps * DEG2RAD)
    J[2][j] = (fp.z - fm.z) / (2 * eps * DEG2RAD)
  }
  return J
}

/** Solve (J Jᵀ + λ²I) x = e for x in ℝ³, then Δθ = Jᵀ x (DLS step). */
function dlsStep(J: number[][], e: THREE.Vector3, lambda: number): number[] {
  // A = J Jᵀ (3×3 symmetric)
  const a00 = J[0][0]*J[0][0] + J[0][1]*J[0][1] + J[0][2]*J[0][2] + J[0][3]*J[0][3] + J[0][4]*J[0][4] + J[0][5]*J[0][5]
  const a01 = J[0][0]*J[1][0] + J[0][1]*J[1][1] + J[0][2]*J[1][2] + J[0][3]*J[1][3] + J[0][4]*J[1][4] + J[0][5]*J[1][5]
  const a02 = J[0][0]*J[2][0] + J[0][1]*J[2][1] + J[0][2]*J[2][2] + J[0][3]*J[2][3] + J[0][4]*J[2][4] + J[0][5]*J[2][5]
  const a11 = J[1][0]*J[1][0] + J[1][1]*J[1][1] + J[1][2]*J[1][2] + J[1][3]*J[1][3] + J[1][4]*J[1][4] + J[1][5]*J[1][5]
  const a12 = J[1][0]*J[2][0] + J[1][1]*J[2][1] + J[1][2]*J[2][2] + J[1][3]*J[2][3] + J[1][4]*J[2][4] + J[1][5]*J[2][5]
  const a22 = J[2][0]*J[2][0] + J[2][1]*J[2][1] + J[2][2]*J[2][2] + J[2][3]*J[2][3] + J[2][4]*J[2][4] + J[2][5]*J[2][5]
  const l2 = lambda * lambda
  // A + λ²I
  const m00 = a00 + l2, m01 = a01, m02 = a02
  const m11 = a11 + l2, m12 = a12
  const m22 = a22 + l2
  // invert 3×3 via cofactors
  const c00 = m11 * m22 - m12 * m12
  const c01 = -(m01 * m22 - m12 * m02)
  const c02 = m01 * m12 - m11 * m02
  const det = m00 * c00 + m01 * c01 + m02 * c02
  if (Math.abs(det) < 1e-14) return [0,0,0,0,0,0]
  const inv = 1 / det
  const c11 = m00 * m22 - m02 * m02
  const c12 = -(m00 * m12 - m02 * m01)
  const c22 = m00 * m11 - m01 * m01
  // A⁻¹ e
  const ex = e.x, ey = e.y, ez = e.z
  const x0 = (c00 * ex + c01 * ey + c02 * ez) * inv
  const x1 = (c01 * ex + c11 * ey + c12 * ez) * inv
  const x2 = (c02 * ex + c12 * ey + c22 * ez) * inv
  // Δθ_rad = Jᵀ x
  const dq: number[] = []
  for (let j = 0; j < 6; j++) {
    dq.push(J[0][j] * x0 + J[1][j] * x1 + J[2][j] * x2)
  }
  return dq
}

export function solveIK(
  target: THREE.Vector3,
  seed: number[],
  opts: IKOptions = {}
): IKResult {
  const maxIter = opts.maxIter ?? 30
  const tol = opts.tolerance ?? 0.002
  const lambda = opts.damping ?? 0.06
  const stepClamp = opts.stepClamp ?? 8
  const limits = opts.limits ?? DEFAULT_LIMITS
  const reach = maxReach()

  const q = seed.slice(0, 6)
  while (q.length < 6) q.push(0)

  // If target is way outside reach, project onto reach sphere to give DLS a hint
  const r = target.length()
  const probe = target.clone()
  if (r > reach * 0.98) probe.multiplyScalar((reach * 0.98) / r)

  let err = 0
  let iter = 0
  for (; iter < maxIter; iter++) {
    const tip = fk(q)
    const e = new THREE.Vector3().subVectors(probe, tip)
    err = e.length()
    if (err < tol) break
    const J = jacobian(q)
    const dq = dlsStep(J, e, lambda)
    // dq is in radians — convert to degrees and clamp
    for (let j = 0; j < 6; j++) {
      let stepDeg = dq[j] * RAD2DEG
      if (stepDeg > stepClamp) stepDeg = stepClamp
      else if (stepDeg < -stepClamp) stepDeg = -stepClamp
      q[j] += stepDeg
    }
    clampLimits(q, limits)
  }

  // measure error against TRUE target (not probe)
  const finalTip = fk(q)
  const finalErr = finalTip.distanceTo(target)
  return {
    angles: q,
    reachable: finalErr < 0.02,   // 20mm tolerance for "reachable"
    finalErrorM: finalErr,
    iterations: iter,
  }
}

// re-export so callers don't need to import kinematics directly
export { AXES }
