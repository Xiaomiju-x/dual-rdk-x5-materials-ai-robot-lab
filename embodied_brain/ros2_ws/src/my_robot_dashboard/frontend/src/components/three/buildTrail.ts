import * as THREE from 'three'

/**
 * buildTrail — geometry-based ribbon trail behind a moving object.
 * Stores last N positions in a circular buffer, rebuilds a tube on
 * each update.
 *
 *   const trail = buildTrail({ maxPoints: 60 })
 *   scene.add(trail.group)
 *   trail.push(robot.position)
 *   trail.update()
 */
export interface TrailBundle {
  group: THREE.Group
  push: (p: THREE.Vector3) => void
  update: () => void
  clear: () => void
  setVisible: (v: boolean) => void
  dispose: () => void
}

interface Opts {
  maxPoints?: number
  color?: number
  width?: number
  fadeStart?: number   // 0..1
}

export function buildTrail(opts: Opts = {}): TrailBundle {
  const maxPoints = opts.maxPoints ?? 60
  const color = new THREE.Color(opts.color ?? 0x22d3ee)
  const group = new THREE.Group()
  group.name = 'trail'

  const positions = new Float32Array(maxPoints * 3)
  const alphas = new Float32Array(maxPoints)
  let head = 0
  let written = 0

  const geom = new THREE.BufferGeometry()
  geom.setAttribute('position', new THREE.BufferAttribute(positions, 3).setUsage(THREE.DynamicDrawUsage))
  geom.setAttribute('aAlpha',   new THREE.BufferAttribute(alphas, 1).setUsage(THREE.DynamicDrawUsage))

  const mat = new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    uniforms: {
      uColor: { value: color },
    },
    vertexShader: /* glsl */`
      attribute float aAlpha;
      varying float vA;
      void main() {
        vA = aAlpha;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: /* glsl */`
      varying float vA;
      uniform vec3 uColor;
      void main() {
        gl_FragColor = vec4(uColor * (0.8 + 0.4 * vA), vA * 0.85);
      }
    `,
  })

  const line = new THREE.Line(geom, mat)
  group.add(line)

  let lastPushed: THREE.Vector3 | null = null
  function push(p: THREE.Vector3): void {
    if (lastPushed && p.distanceToSquared(lastPushed) < 0.0004) return
    positions[head * 3]     = p.x
    positions[head * 3 + 1] = p.y + 0.04
    positions[head * 3 + 2] = p.z
    head = (head + 1) % maxPoints
    written = Math.min(written + 1, maxPoints)
    lastPushed = p.clone()
  }
  function update(): void {
    // build linear array oldest→newest, set alpha gradient
    const reordered = new Float32Array(maxPoints * 3)
    const aArr = alphas
    for (let i = 0; i < written; i++) {
      const idx = (head - written + i + maxPoints) % maxPoints
      reordered[i * 3]     = positions[idx * 3]
      reordered[i * 3 + 1] = positions[idx * 3 + 1]
      reordered[i * 3 + 2] = positions[idx * 3 + 2]
      aArr[i] = i / Math.max(1, written - 1)
    }
    ;(geom.getAttribute('position') as THREE.BufferAttribute).set(reordered)
    ;(geom.getAttribute('position') as THREE.BufferAttribute).needsUpdate = true
    ;(geom.getAttribute('aAlpha') as THREE.BufferAttribute).needsUpdate = true
    geom.setDrawRange(0, written)
  }
  function clear(): void { written = 0; head = 0; lastPushed = null; geom.setDrawRange(0, 0) }
  function setVisible(v: boolean): void { group.visible = v }
  function dispose(): void { geom.dispose(); mat.dispose() }

  return { group, push, update, clear, setVisible, dispose }
}
