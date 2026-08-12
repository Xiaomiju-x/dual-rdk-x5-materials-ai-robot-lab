import * as THREE from 'three'

/**
 * buildStarfield — large inverted sphere with shader-generated stars
 * + horizon glow gradient. Used only in /immersive (cinematic mode).
 */
export interface SkyboxBundle {
  mesh: THREE.Mesh
  setVisible: (v: boolean) => void
  dispose: () => void
}

export function buildStarfield(opts: { themeTone?: 'dark' | 'light' } = {}): SkyboxBundle {
  const themeTone = opts.themeTone ?? 'dark'
  const geom = new THREE.SphereGeometry(40, 32, 24)

  const mat = new THREE.ShaderMaterial({
    side: THREE.BackSide,
    depthWrite: false,
    uniforms: {
      uTheme: { value: themeTone === 'dark' ? 1.0 : 0.0 },
    },
    vertexShader: /* glsl */`
      varying vec3 vDir;
      void main() {
        vDir = normalize(position);
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: /* glsl */`
      varying vec3 vDir;
      uniform float uTheme;

      // hash for star positions
      float hash(vec3 p) {
        p = fract(p * 0.3183099 + vec3(0.71, 0.113, 0.419));
        p *= 17.0;
        return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
      }

      // density based starfield
      float stars(vec3 dir, float scale, float threshold) {
        vec3 p = floor(dir * scale);
        float h = hash(p);
        if (h < threshold) return 0.0;
        float intensity = (h - threshold) / (1.0 - threshold);
        // jitter per cell
        vec3 cellCenter = (p + 0.5) / scale;
        float d = distance(normalize(dir), normalize(cellCenter)) * scale;
        return smoothstep(0.30, 0.0, d) * intensity;
      }

      void main() {
        vec3 dir = normalize(vDir);
        // horizon gradient — dark mode: deep navy → black; light mode: pale gray
        vec3 base;
        if (uTheme > 0.5) {
          base = mix(vec3(0.020, 0.035, 0.07), vec3(0.005, 0.008, 0.018), smoothstep(0.0, 0.6, dir.y + 0.2));
        } else {
          base = mix(vec3(0.88, 0.93, 0.97), vec3(0.95, 0.97, 0.99), smoothstep(0.0, 0.6, dir.y + 0.2));
        }

        // stars only in dark mode
        float s = 0.0;
        if (uTheme > 0.5) {
          s += stars(dir, 180.0, 0.985) * 1.4;
          s += stars(dir, 360.0, 0.992) * 0.9;
          s += stars(dir, 720.0, 0.996) * 0.55;
        }

        // horizon glow ring
        float glow = exp(-abs(dir.y) * 6.0) * 0.18 * (uTheme > 0.5 ? 1.0 : 0.5);
        vec3 glowColor = uTheme > 0.5 ? vec3(0.18, 0.30, 0.55) : vec3(0.45, 0.65, 0.95);

        vec3 c = base + glowColor * glow + vec3(s);
        gl_FragColor = vec4(c, 1.0);
      }
    `,
  })

  const mesh = new THREE.Mesh(geom, mat)
  mesh.frustumCulled = false
  function setVisible(v: boolean) { mesh.visible = v }
  function dispose() { geom.dispose(); mat.dispose() }
  return { mesh, setVisible, dispose }
}
