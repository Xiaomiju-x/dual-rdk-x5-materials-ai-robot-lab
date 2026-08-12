import * as THREE from 'three'

/**
 * createHoloMaterial — Fresnel-edge holographic material for the
 * "X-ray hologram" robot toggle. Cyan emissive with scan lines.
 */
export function createHoloMaterial(opts: { color?: number; scanSpeed?: number } = {}): THREE.ShaderMaterial {
  const color = new THREE.Color(opts.color ?? 0x22d3ee)
  const mat = new THREE.ShaderMaterial({
    uniforms: {
      uTime: { value: 0 },
      uColor: { value: color },
      uScanSpeed: { value: opts.scanSpeed ?? 1.0 },
    },
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    blending: THREE.AdditiveBlending,
    vertexShader: /* glsl */`
      varying vec3 vWorldNormal;
      varying vec3 vViewDir;
      varying vec3 vWorldPos;
      void main() {
        vec4 wp = modelMatrix * vec4(position, 1.0);
        vWorldPos = wp.xyz;
        vWorldNormal = normalize(mat3(modelMatrix) * normal);
        vViewDir = normalize(cameraPosition - wp.xyz);
        gl_Position = projectionMatrix * viewMatrix * wp;
      }
    `,
    fragmentShader: /* glsl */`
      varying vec3 vWorldNormal;
      varying vec3 vViewDir;
      varying vec3 vWorldPos;
      uniform float uTime;
      uniform vec3 uColor;
      uniform float uScanSpeed;
      void main() {
        // Fresnel — strongest along grazing edges
        float fres = 1.0 - max(dot(vWorldNormal, vViewDir), 0.0);
        fres = pow(fres, 1.8);
        // moving scan lines
        float scan = sin(vWorldPos.y * 80.0 - uTime * uScanSpeed * 6.0) * 0.5 + 0.5;
        scan = smoothstep(0.55, 1.0, scan);
        float a = fres * 0.85 + scan * 0.18;
        vec3 c = uColor * (1.2 + scan * 0.6);
        gl_FragColor = vec4(c, a);
      }
    `,
  })
  return mat
}
