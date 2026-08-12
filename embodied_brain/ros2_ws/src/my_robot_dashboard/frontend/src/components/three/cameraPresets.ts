import * as THREE from 'three'
import type { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'

/**
 * cameraPresets — 5 named camera framings with smooth lerp.
 * Use during /immersive: keys 1-5 trigger presetByKey.
 */
export interface PresetPose {
  pos: THREE.Vector3
  target: THREE.Vector3
  fov?: number
}

export const PRESETS: Record<string, PresetPose> = {
  TOP:    { pos: new THREE.Vector3( 0,  5.5,  0.001), target: new THREE.Vector3(0, 0, 0),    fov: 45 },
  FRONT:  { pos: new THREE.Vector3( 0,  1.0,  4.0),   target: new THREE.Vector3(0, 0.3, 0),  fov: 45 },
  ISO:    { pos: new THREE.Vector3( 2.4, 2.0, 2.4),   target: new THREE.Vector3(0, 0.3, 0),  fov: 45 },
  CHASE:  { pos: new THREE.Vector3(-1.4, 0.9, -1.4),  target: new THREE.Vector3(0, 0.3, 0),  fov: 55 },
  ORBIT:  { pos: new THREE.Vector3( 3.2, 1.6, 3.2),   target: new THREE.Vector3(0, 0.3, 0),  fov: 42 },
}

export const PRESET_KEYS: Record<string, keyof typeof PRESETS> = {
  '1': 'TOP', '2': 'FRONT', '3': 'ISO', '4': 'CHASE', '5': 'ORBIT',
}

/**
 * Animate camera + controls.target toward a preset over `durationMs`.
 * Returns a cancel function to abort mid-flight.
 */
export function flyToPreset(
  camera: THREE.PerspectiveCamera,
  controls: OrbitControls | null,
  preset: PresetPose,
  durationMs = 900,
): () => void {
  const startPos = camera.position.clone()
  const startTarget = controls ? controls.target.clone() : new THREE.Vector3(0, 0.3, 0)
  const startFov = camera.fov
  const endPos = preset.pos.clone()
  const endTarget = preset.target.clone()
  const endFov = preset.fov ?? camera.fov
  const t0 = performance.now()
  let raf = 0
  let cancelled = false

  const tick = (now: number) => {
    if (cancelled) return
    const t = Math.min(1, (now - t0) / durationMs)
    const eased = 1 - Math.pow(1 - t, 4)
    camera.position.lerpVectors(startPos, endPos, eased)
    if (controls) controls.target.lerpVectors(startTarget, endTarget, eased)
    camera.fov = startFov + (endFov - startFov) * eased
    camera.updateProjectionMatrix()
    camera.lookAt(controls ? controls.target : endTarget)
    if (t < 1) raf = requestAnimationFrame(tick)
  }
  raf = requestAnimationFrame(tick)

  return () => { cancelled = true; cancelAnimationFrame(raf) }
}
