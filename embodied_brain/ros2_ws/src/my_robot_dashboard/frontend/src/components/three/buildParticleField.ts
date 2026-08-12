import * as THREE from 'three'

/**
 * buildParticleField — N drifting points lit by emissive shader.
 * Cheap (BufferGeometry, single draw call) but adds tons of depth.
 *
 *   const pf = buildParticleField({ count: 1200 })
 *   scene.add(pf.group)
 *   pf.update(t)
 *
 * Colors blend between two palette stops per particle, drift on
 * gentle sine flow, fade in/out near a dome envelope.
 */
export interface ParticleFieldBundle {
  group: THREE.Group
  update: (t: number) => void
  setIntensity: (k: number) => void
  setColors: (a: THREE.ColorRepresentation, b: THREE.ColorRepresentation) => void
  dispose: () => void
}

interface Opts {
  count?: number
  radius?: number
  size?: number
  colorA?: THREE.ColorRepresentation
  colorB?: THREE.ColorRepresentation
}

export function buildParticleField(opts: Opts = {}): ParticleFieldBundle {
  const count = opts.count ?? 1200
  const radius = opts.radius ?? 5
  const baseSize = opts.size ?? 0.018
  const group = new THREE.Group()
  group.name = 'particleField'

  const positions = new Float32Array(count * 3)
  const phases = new Float32Array(count)
  const speeds = new Float32Array(count)
  const colors = new Float32Array(count * 3)
  const sizes = new Float32Array(count)

  const cA = new THREE.Color(opts.colorA ?? 0x2563eb)
  const cB = new THREE.Color(opts.colorB ?? 0x06b6d4)

  for (let i = 0; i < count; i++) {
    // dome distribution (more on the floor, dome upward)
    const u = Math.random()
    const v = Math.random()
    const theta = u * Math.PI * 2
    const phi = Math.acos(2 * v - 1) * 0.55  // squash dome
    const r = radius * Math.pow(Math.random(), 0.35)
    positions[i * 3]     = r * Math.sin(phi) * Math.cos(theta)
    positions[i * 3 + 1] = 0.05 + 1.4 * Math.cos(phi) * Math.random()
    positions[i * 3 + 2] = r * Math.sin(phi) * Math.sin(theta)
    phases[i] = Math.random() * Math.PI * 2
    speeds[i] = 0.2 + Math.random() * 0.6
    sizes[i] = baseSize * (0.4 + Math.random() * 1.6)
    const mix = Math.random()
    const c = new THREE.Color().lerpColors(cA, cB, mix)
    colors[i * 3]     = c.r
    colors[i * 3 + 1] = c.g
    colors[i * 3 + 2] = c.b
  }

  const geom = new THREE.BufferGeometry()
  geom.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  geom.setAttribute('aPhase',   new THREE.BufferAttribute(phases, 1))
  geom.setAttribute('aSpeed',   new THREE.BufferAttribute(speeds, 1))
  geom.setAttribute('aSize',    new THREE.BufferAttribute(sizes, 1))
  geom.setAttribute('color',    new THREE.BufferAttribute(colors, 3))

  const mat = new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    vertexColors: true,
    uniforms: {
      uTime: { value: 0 },
      uIntensity: { value: 1 },
      uPxRatio: { value: Math.min(window.devicePixelRatio, 2) },
    },
    vertexShader: /* glsl */`
      attribute float aPhase;
      attribute float aSpeed;
      attribute float aSize;
      varying float vAlpha;
      uniform float uTime;
      uniform float uIntensity;
      uniform float uPxRatio;
      void main() {
        vec3 pos = position;
        // gentle vertical drift
        pos.y += sin(uTime * aSpeed * 0.6 + aPhase) * 0.12;
        // tiny horizontal sway
        pos.x += cos(uTime * aSpeed * 0.4 + aPhase * 1.3) * 0.06;
        vec4 mv = modelViewMatrix * vec4(pos, 1.0);
        gl_Position = projectionMatrix * mv;
        // size attenuates with depth + breathing
        float breath = 0.6 + 0.4 * sin(uTime * aSpeed + aPhase);
        gl_PointSize = aSize * uIntensity * breath * (300.0 / -mv.z) * uPxRatio;
        vAlpha = clamp(0.18 + 0.55 * breath, 0.0, 0.9);
      }
    `,
    fragmentShader: /* glsl */`
      varying float vAlpha;
      varying vec3 vColor;
      // re-declare since 'color' attribute uses default varying
      void main() {
        vec2 c = gl_PointCoord - 0.5;
        float d = length(c);
        float a = smoothstep(0.5, 0.0, d) * vAlpha;
        gl_FragColor = vec4(gl_FragColor.rgb, 1.0) * vec4(a);
        // pull color from built-in vColor (via vertexColors=true → THREE injects)
      }
    `,
  })

  // three injects vColor varying automatically when vertexColors=true; but the
  // shader above uses a custom material so we have to wire it manually:
  mat.vertexShader = mat.vertexShader.replace(
    'varying float vAlpha;',
    'varying float vAlpha;\nvarying vec3 vColor;',
  ).replace(
    'vAlpha = clamp',
    'vColor = color;\n        vAlpha = clamp',
  )
  mat.fragmentShader = mat.fragmentShader.replace(
    'void main() {',
    'void main() {\n  gl_FragColor = vec4(vColor, 1.0);',
  )

  const points = new THREE.Points(geom, mat)
  points.frustumCulled = false
  group.add(points)

  function update(t: number): void { mat.uniforms.uTime.value = t }
  function setIntensity(k: number): void { mat.uniforms.uIntensity.value = k }
  function setColors(a: THREE.ColorRepresentation, b: THREE.ColorRepresentation): void {
    const ca = new THREE.Color(a)
    const cb = new THREE.Color(b)
    const arr = colors
    for (let i = 0; i < count; i++) {
      const mix = (arr[i * 3] - ca.r) / Math.max(0.001, cb.r - ca.r) || Math.random()
      const c = new THREE.Color().lerpColors(ca, cb, mix)
      arr[i * 3] = c.r; arr[i * 3 + 1] = c.g; arr[i * 3 + 2] = c.b
    }
    ;(geom.getAttribute('color') as THREE.BufferAttribute).needsUpdate = true
  }
  function dispose(): void {
    geom.dispose()
    mat.dispose()
  }

  return { group, update, setIntensity, setColors, dispose }
}
