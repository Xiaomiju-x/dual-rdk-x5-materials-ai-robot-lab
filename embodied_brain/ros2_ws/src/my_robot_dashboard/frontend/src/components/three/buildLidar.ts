import * as THREE from 'three'

/**
 * LiDAR rotating sweep — a fan of fading rays + an outer detection arc.
 * Animated in updateSweep(t).
 */
export interface LidarSweep {
  group: THREE.Group
  setRange: (r: number) => void
  update: (t: number) => void
}

const RAY_COUNT = 96
const SWEEP_SPAN = Math.PI * 0.45 // fan width, ~80°

export function buildLidarSweep(): LidarSweep {
  const group = new THREE.Group()
  group.name = 'lidarSweep'
  group.position.y = 0.32 // sit on top of lidar puck

  let baseRange = 2.4

  // Build a TRIANGLE FAN approximating the sweep cone — one center vertex + 96 edge points.
  const geom = new THREE.BufferGeometry()
  const positions = new Float32Array((RAY_COUNT + 2) * 3)
  const colors = new Float32Array((RAY_COUNT + 2) * 3)
  const indices: number[] = []
  for (let i = 1; i <= RAY_COUNT; i++) indices.push(0, i, i + 1)
  geom.setIndex(indices)
  geom.setAttribute('position', new THREE.BufferAttribute(positions, 3).setUsage(THREE.DynamicDrawUsage))
  geom.setAttribute('color', new THREE.BufferAttribute(colors, 3))

  const mat = new THREE.MeshBasicMaterial({
    transparent: true,
    opacity: 0.45,
    vertexColors: true,
    side: THREE.DoubleSide,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  })
  const fan = new THREE.Mesh(geom, mat)
  group.add(fan)

  // Outer scanning arc (bright leading edge)
  const arcGeom = new THREE.BufferGeometry()
  const arcPositions = new Float32Array((RAY_COUNT + 1) * 3)
  arcGeom.setAttribute('position', new THREE.BufferAttribute(arcPositions, 3).setUsage(THREE.DynamicDrawUsage))
  const arcMat = new THREE.LineBasicMaterial({ color: 0x06b6d4, transparent: true, opacity: 0.85 })
  const arc = new THREE.Line(arcGeom, arcMat)
  group.add(arc)

  // Center pulsing dot
  const dotMat = new THREE.MeshBasicMaterial({ color: 0x06b6d4, transparent: true, opacity: 0.9 })
  const dot = new THREE.Mesh(new THREE.SphereGeometry(0.025, 12, 8), dotMat)
  group.add(dot)

  function setRange(r: number): void {
    baseRange = r
  }

  function update(t: number): void {
    const heading = t * 0.85
    const range = baseRange + 0.18 * Math.sin(t * 1.4)
    const colInner = new THREE.Color(0x06b6d4)
    const colOuter = new THREE.Color(0x7c3aed)

    // center vertex
    positions[0] = 0
    positions[1] = 0
    positions[2] = 0
    colors[0] = colInner.r
    colors[1] = colInner.g
    colors[2] = colInner.b

    for (let i = 0; i <= RAY_COUNT; i++) {
      const u = i / RAY_COUNT
      const a = heading - SWEEP_SPAN / 2 + u * SWEEP_SPAN
      const wobble = 0.08 * Math.sin(t * 9 + u * 18)
      const r = range * (0.92 + 0.08 * Math.sin(t * 4 + u * 12)) + wobble
      const idx = (i + 1) * 3
      positions[idx] = Math.cos(a) * r
      positions[idx + 1] = 0
      positions[idx + 2] = Math.sin(a) * r
      const fade = 1 - Math.abs(u - 0.5) * 1.5
      const blend = new THREE.Color().lerpColors(colOuter, colInner, Math.max(0, fade))
      colors[idx] = blend.r * fade
      colors[idx + 1] = blend.g * fade
      colors[idx + 2] = blend.b * fade

      // arc outer
      const arcIdx = i * 3
      arcPositions[arcIdx] = Math.cos(a) * r
      arcPositions[arcIdx + 1] = 0.005
      arcPositions[arcIdx + 2] = Math.sin(a) * r
    }
    geom.attributes['position']!.needsUpdate = true
    geom.attributes['color']!.needsUpdate = true
    arcGeom.attributes['position']!.needsUpdate = true

    dot.scale.setScalar(1 + 0.2 * Math.sin(t * 4))
    dotMat.opacity = 0.75 + 0.25 * Math.sin(t * 4)
  }

  return { group, setRange, update }
}
