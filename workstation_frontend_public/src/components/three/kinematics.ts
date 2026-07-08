import * as THREE from 'three'

// Forward kinematics matching the procedural arm built in buildArm.ts.
// Geometry constants live alongside the visual chain so the IK target the user
// drags maps exactly to the tip rendered in the scene. We don't try to mirror
// the real myCobot 280-Pi DH parameters (mm scale) — the studio arm uses a
// down-scaled chain to fit the workbench frame; matching the visual is what
// matters for cockpit interaction.

export const AXES: Array<'x' | 'y' | 'z'> = ['y', 'x', 'x', 'x', 'z', 'x']
export const LINKS = [0.055, 0.085, 0.075, 0.05, 0.04, 0.03]
export const BASE_Y = 0.07
export const LINK_GAP = 0.012     // small gap between knuckle and next pivot
export const TIP_OFFSET = 0.06    // gripper palm + finger reach beyond last pivot

const DEG2RAD = Math.PI / 180

/** Forward kinematics — returns world position of tip given 6 joint angles (deg). */
export function fk(anglesDeg: number[]): THREE.Vector3 {
  const m = new THREE.Matrix4().identity()

  // base lift
  m.multiply(new THREE.Matrix4().makeTranslation(0, BASE_Y, 0))

  for (let i = 0; i < 6; i++) {
    const rad = (anglesDeg[i] ?? 0) * DEG2RAD
    const ax = AXES[i]
    const rot = new THREE.Matrix4()
    if (ax === 'x') rot.makeRotationX(rad)
    else if (ax === 'y') rot.makeRotationY(rad)
    else rot.makeRotationZ(rad)
    m.multiply(rot)

    const len = LINKS[i] + LINK_GAP
    m.multiply(new THREE.Matrix4().makeTranslation(0, len, 0))
  }

  // tip extends a bit further in the final local frame
  m.multiply(new THREE.Matrix4().makeTranslation(0, TIP_OFFSET, 0))

  const v = new THREE.Vector3()
  v.setFromMatrixPosition(m)
  return v
}

/** Conservative reach: sum of link lengths + base + tip — used to cap IK search. */
export function maxReach(): number {
  let r = BASE_Y + TIP_OFFSET
  for (const l of LINKS) r += l + LINK_GAP
  return r
}

/** Full tip pose — position (m) + intrinsic Euler XYZ (rad). */
export function fkPose(anglesDeg: number[]): { position: THREE.Vector3; eulerXYZ: THREE.Euler } {
  const m = new THREE.Matrix4().identity()
  m.multiply(new THREE.Matrix4().makeTranslation(0, BASE_Y, 0))
  for (let i = 0; i < 6; i++) {
    const rad = (anglesDeg[i] ?? 0) * DEG2RAD
    const ax = AXES[i]
    const rot = new THREE.Matrix4()
    if (ax === 'x') rot.makeRotationX(rad)
    else if (ax === 'y') rot.makeRotationY(rad)
    else rot.makeRotationZ(rad)
    m.multiply(rot)
    m.multiply(new THREE.Matrix4().makeTranslation(0, LINKS[i] + LINK_GAP, 0))
  }
  m.multiply(new THREE.Matrix4().makeTranslation(0, TIP_OFFSET, 0))
  const position = new THREE.Vector3()
  const quat = new THREE.Quaternion()
  const scale = new THREE.Vector3()
  m.decompose(position, quat, scale)
  const eulerXYZ = new THREE.Euler().setFromQuaternion(quat, 'XYZ')
  return { position, eulerXYZ }
}

/** Same as fk() but also returns intermediate pivot positions — useful for skeleton debug. */
export function fkChain(anglesDeg: number[]): THREE.Vector3[] {
  const m = new THREE.Matrix4().identity()
  m.multiply(new THREE.Matrix4().makeTranslation(0, BASE_Y, 0))
  const pts: THREE.Vector3[] = []
  pts.push(new THREE.Vector3(0, 0, 0))
  pts.push(new THREE.Vector3(0, BASE_Y, 0))

  for (let i = 0; i < 6; i++) {
    const rad = (anglesDeg[i] ?? 0) * DEG2RAD
    const ax = AXES[i]
    const rot = new THREE.Matrix4()
    if (ax === 'x') rot.makeRotationX(rad)
    else if (ax === 'y') rot.makeRotationY(rad)
    else rot.makeRotationZ(rad)
    m.multiply(rot)
    m.multiply(new THREE.Matrix4().makeTranslation(0, LINKS[i] + LINK_GAP, 0))
    const v = new THREE.Vector3()
    v.setFromMatrixPosition(m)
    pts.push(v)
  }
  m.multiply(new THREE.Matrix4().makeTranslation(0, TIP_OFFSET, 0))
  const tip = new THREE.Vector3()
  tip.setFromMatrixPosition(m)
  pts.push(tip)
  return pts
}
