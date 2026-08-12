import * as THREE from 'three'

/**
 * buildVolumetricCone — cheap screen-space "god rays" via a backside
 * additive cone mesh that fades from key-light position. Not a true
 * volumetric raymarcher; just enough to add cinematic light shafts.
 *
 *   const vol = buildVolumetricCone()
 *   vol.group.position.copy(keyLight.position)
 *   scene.add(vol.group)
 *   vol.update(t)
 */
export interface VolumetricBundle {
  group: THREE.Group
  update: (t: number) => void
  setVisible: (v: boolean) => void
  dispose: () => void
}

interface Opts {
  color?: number
  height?: number       // cone length
  radius?: number       // cone bottom radius
  intensity?: number
}

export function buildVolumetricCone(opts: Opts = {}): VolumetricBundle {
  const color = new THREE.Color(opts.color ?? 0xffffff)
  const height = opts.height ?? 4.5
  const radius = opts.radius ?? 1.6

  const group = new THREE.Group()
  group.name = 'volumetricCone'

  const geom = new THREE.ConeGeometry(radius, height, 36, 24, true)
  geom.translate(0, -height / 2, 0)  // tip at origin (light source), opening downward

  const mat = new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    side: THREE.BackSide,
    uniforms: {
      uTime:      { value: 0 },
      uColor:     { value: color },
      uIntensity: { value: opts.intensity ?? 0.45 },
      uHeight:    { value: height },
    },
    vertexShader: /* glsl */`
      varying vec3 vPos;
      varying vec3 vNormal;
      varying vec3 vViewDir;
      void main() {
        vPos = position;
        vNormal = normalize(normalMatrix * normal);
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        vViewDir = normalize(-mv.xyz);
        gl_Position = projectionMatrix * mv;
      }
    `,
    fragmentShader: /* glsl */`
      varying vec3 vPos;
      varying vec3 vNormal;
      varying vec3 vViewDir;
      uniform float uTime;
      uniform vec3  uColor;
      uniform float uIntensity;
      uniform float uHeight;
      void main() {
        // fade by depth (top of cone bright, base soft)
        float depth = clamp(1.0 - (-vPos.y / uHeight), 0.0, 1.0);
        // soft edge (face away from camera = darker — Fresnel-ish)
        float facing = pow(abs(dot(vNormal, vViewDir)), 0.6);
        // subtle noise shimmer
        float shim = 0.85 + 0.15 * sin(uTime * 1.4 + vPos.y * 3.0);
        float a = depth * (1.0 - facing) * uIntensity * shim;
        gl_FragColor = vec4(uColor * (0.6 + 0.4 * shim), a * 0.85);
      }
    `,
  })

  const mesh = new THREE.Mesh(geom, mat)
  mesh.frustumCulled = false
  group.add(mesh)

  function update(t: number) { mat.uniforms.uTime.value = t }
  function setVisible(v: boolean) { group.visible = v }
  function dispose() { geom.dispose(); mat.dispose() }

  return { group, update, setVisible, dispose }
}
