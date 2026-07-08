import * as THREE from 'three'

/**
 * Workshop world props for DualArmScene — the things that make this look like
 * an actual bench rather than two arms floating in space.
 *
 *  • workbench surface (procedural carbon weave + steel edge)
 *  • AprilTag tag36h11 textures (visually plausible, hardcoded for id 0 and 3)
 *  • reagent bottle rack (glass cylinders with metal caps)
 *  • central hand-over hot zone ring
 *
 * Everything is procedural — zero external textures.
 */

// ---------- procedural carbon-weave bench texture ----------
export function benchTexture(theme: 'light' | 'dark'): THREE.CanvasTexture {
  const N = 512
  const cvs = document.createElement('canvas'); cvs.width = N; cvs.height = N
  const ctx = cvs.getContext('2d')!
  const base = theme === 'light' ? '#e7eaef' : '#1b2030'
  const dark = theme === 'light' ? '#cdd2da' : '#0e1118'
  // base
  ctx.fillStyle = base; ctx.fillRect(0, 0, N, N)
  // subtle carbon weave (5px cells)
  const cell = 6
  for (let y = 0; y < N; y += cell) {
    for (let x = 0; x < N; x += cell) {
      const isDiag = ((x / cell) + (y / cell)) % 2 === 0
      ctx.fillStyle = isDiag ? dark : base
      ctx.globalAlpha = 0.18
      ctx.fillRect(x, y, cell, cell)
    }
  }
  ctx.globalAlpha = 1
  // brand stripe along one edge
  ctx.fillStyle = theme === 'light' ? '#2563eb' : '#60a5fa'
  ctx.fillRect(0, N - 8, N, 8)
  ctx.fillStyle = theme === 'light' ? '#d97706' : '#f59e0b'
  ctx.fillRect(0, 0, N, 8)
  const tex = new THREE.CanvasTexture(cvs)
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping
  tex.anisotropy = 8
  return tex
}

// ---------- AprilTag tag36h11 procedural texture ----------
// 36 bits hand-picked to look like real tag36h11 patterns; not decodable but
// indistinguishable visually from id=0 / id=3 in the AprilTag family. We label
// them so the operator can map them back to the physical tag they target.
const TAG_BITS: Record<number, string> = {
  0: '111010100110100110111001011001011001',
  3: '101001110100110011010110101100101110',
}

export function aprilTagTexture(id: 0 | 3): THREE.CanvasTexture {
  const bits = TAG_BITS[id]
  const N = 256
  const cvs = document.createElement('canvas'); cvs.width = N; cvs.height = N
  const ctx = cvs.getContext('2d')!
  // white background
  ctx.fillStyle = '#ffffff'; ctx.fillRect(0, 0, N, N)
  // black 2-cell border, 6x6 inner = 10x10 grid total
  const cell = N / 10
  ctx.fillStyle = '#000000'
  // entire frame
  ctx.fillRect(0, 0, N, N)
  // white inner 8x8 (1 cell smaller border)
  ctx.fillStyle = '#ffffff'; ctx.fillRect(cell, cell, N - 2 * cell, N - 2 * cell)
  // black inner 6x6 region (extends 1 cell from outer for the proper 2-cell border)
  ctx.fillStyle = '#000000'; ctx.fillRect(2 * cell, 2 * cell, N - 4 * cell, N - 4 * cell)
  // re-fill inner cells based on bits
  for (let row = 0; row < 6; row++) {
    for (let col = 0; col < 6; col++) {
      const bit = bits[row * 6 + col]
      if (bit === '1') {
        ctx.fillStyle = '#ffffff'
        ctx.fillRect((2 + col) * cell, (2 + row) * cell, cell, cell)
      }
    }
  }
  // label outside the tag (we'll keep texture's id readable)
  ctx.fillStyle = '#2563eb'; ctx.font = `bold ${cell * 0.6}px "JetBrains Mono", monospace`
  ctx.textAlign = 'left'; ctx.textBaseline = 'top'
  ctx.fillText(`id=${id}`, cell * 0.2, cell * 0.1)
  const tex = new THREE.CanvasTexture(cvs)
  tex.anisotropy = 8
  tex.colorSpace = THREE.SRGBColorSpace
  return tex
}

// ---------- ribbon trail (triangle strip, fades by age) ----------
export class RibbonTrail {
  private maxLen: number
  private pos: Float32Array
  private col: Float32Array
  private geom: THREE.BufferGeometry
  private mat: THREE.ShaderMaterial
  mesh: THREE.Mesh
  private write = 0
  private last = new THREE.Vector3()
  private hasLast = false
  private color: THREE.Color
  constructor(color: number, maxLen = 80) {
    this.color = new THREE.Color(color)
    this.maxLen = maxLen
    this.pos = new Float32Array(maxLen * 2 * 3)
    this.col = new Float32Array(maxLen * 2 * 4)   // rgba
    this.geom = new THREE.BufferGeometry()
    this.geom.setAttribute('position', new THREE.BufferAttribute(this.pos, 3))
    this.geom.setAttribute('color', new THREE.BufferAttribute(this.col, 4))
    // triangle-strip index pattern across the 2 rails
    const idx: number[] = []
    for (let i = 0; i < maxLen - 1; i++) {
      const a = i * 2, b = i * 2 + 1, c = (i + 1) * 2, d = (i + 1) * 2 + 1
      idx.push(a, b, c)
      idx.push(b, d, c)
    }
    this.geom.setIndex(idx)
    this.geom.setDrawRange(0, 0)
    this.mat = new THREE.ShaderMaterial({
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
      vertexShader: `
        attribute vec4 color;
        varying vec4 vCol;
        void main(){
          vCol = color;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        varying vec4 vCol;
        void main(){
          if (vCol.a < 0.01) discard;
          gl_FragColor = vCol;
        }
      `,
    })
    this.mesh = new THREE.Mesh(this.geom, this.mat)
    this.mesh.frustumCulled = false
    this.mesh.renderOrder = 5
  }
  push(p: THREE.Vector3) {
    if (this.hasLast && this.last.distanceToSquared(p) < 1e-7) {
      this.decay()
      return
    }
    // ribbon width direction: perpendicular to motion projected onto horizontal-ish
    const dir = this.hasLast ? p.clone().sub(this.last) : new THREE.Vector3(1, 0, 0)
    if (dir.lengthSq() < 1e-8) dir.set(1, 0, 0)
    dir.normalize()
    const up = new THREE.Vector3(0, 1, 0)
    const side = new THREE.Vector3().crossVectors(dir, up)
    if (side.lengthSq() < 1e-6) side.set(1, 0, 0)
    side.normalize().multiplyScalar(0.008)
    const i = this.write
    const a = i * 2, b = a + 1
    this.pos[a * 3] = p.x + side.x; this.pos[a * 3 + 1] = p.y + side.y; this.pos[a * 3 + 2] = p.z + side.z
    this.pos[b * 3] = p.x - side.x; this.pos[b * 3 + 1] = p.y - side.y; this.pos[b * 3 + 2] = p.z - side.z
    // freshest = full alpha
    this.col[a * 4 + 0] = this.color.r; this.col[a * 4 + 1] = this.color.g; this.col[a * 4 + 2] = this.color.b; this.col[a * 4 + 3] = 1.0
    this.col[b * 4 + 0] = this.color.r; this.col[b * 4 + 1] = this.color.g; this.col[b * 4 + 2] = this.color.b; this.col[b * 4 + 3] = 1.0
    this.write = (this.write + 1) % this.maxLen
    this.hasLast = true; this.last.copy(p)
    this.decay()
    const used = Math.min(this.maxLen, Math.max(2, this.write + 1))
    this.geom.setDrawRange(0, (used - 1) * 6)
  }
  private decay() {
    for (let i = 0; i < this.maxLen * 2; i++) {
      this.col[i * 4 + 3] *= 0.965
    }
    (this.geom.attributes.position as THREE.BufferAttribute).needsUpdate = true
    ;(this.geom.attributes.color as THREE.BufferAttribute).needsUpdate = true
  }
  dispose() { this.geom.dispose(); this.mat.dispose() }
}

// ---------- reagent rack (3 glass bottles on a base plate) ----------
export function buildReagentRack(): {
  group: THREE.Group; dispose: () => void
} {
  const group = new THREE.Group(); group.name = 'reagent-rack'
  const disp: Array<{ dispose: () => void }> = []
  // base plate
  const plate = new THREE.Mesh(
    new THREE.BoxGeometry(0.18, 0.008, 0.06),
    new THREE.MeshPhysicalMaterial({ color: 0x1c2230, metalness: 0.7, roughness: 0.4 }),
  )
  plate.position.y = 0.004; plate.castShadow = true
  group.add(plate)
  disp.push(plate.geometry as unknown as { dispose: () => void }, plate.material as THREE.Material)
  // 3 bottles
  const bodyGeo = new THREE.CylinderGeometry(0.012, 0.014, 0.05, 24)
  const neckGeo = new THREE.CylinderGeometry(0.0075, 0.012, 0.012, 16)
  const capGeo = new THREE.CylinderGeometry(0.0095, 0.0095, 0.008, 16)
  disp.push(bodyGeo, neckGeo, capGeo)
  const colors = [0x7dd3fc, 0xfca5a5, 0xa7f3d0]
  const tints = ['Cr³⁺ NIR', 'Ni²⁺ host', 'YAG flux']
  for (let i = 0; i < 3; i++) {
    const glassMat = new THREE.MeshPhysicalMaterial({
      color: colors[i], metalness: 0, roughness: 0.04, transmission: 0.88,
      thickness: 0.4, ior: 1.5, transparent: true, opacity: 0.85,
    })
    const capMat = new THREE.MeshPhysicalMaterial({ color: 0x111418, metalness: 0.5, roughness: 0.45 })
    disp.push(glassMat, capMat)
    const x = (-1 + i) * 0.05
    const body = new THREE.Mesh(bodyGeo, glassMat); body.position.set(x, 0.008 + 0.025, 0); body.castShadow = true
    const neck = new THREE.Mesh(neckGeo, glassMat); neck.position.set(x, 0.008 + 0.05 + 0.006, 0)
    const cap = new THREE.Mesh(capGeo, capMat); cap.position.set(x, 0.008 + 0.05 + 0.016, 0)
    group.add(body); group.add(neck); group.add(cap)
    // tiny label
    const lblCvs = document.createElement('canvas'); lblCvs.width = 128; lblCvs.height = 36
    const lctx = lblCvs.getContext('2d')!
    lctx.fillStyle = '#0b1220'; lctx.fillRect(0, 0, 128, 36)
    lctx.font = 'bold 18px "JetBrains Mono"'; lctx.fillStyle = '#e5e7eb'; lctx.textAlign = 'center'; lctx.textBaseline = 'middle'
    lctx.fillText(tints[i], 64, 18)
    const lblTex = new THREE.CanvasTexture(lblCvs); lblTex.colorSpace = THREE.SRGBColorSpace
    const lblMat = new THREE.MeshBasicMaterial({ map: lblTex, transparent: true })
    disp.push(lblTex, lblMat)
    const lbl = new THREE.Mesh(new THREE.PlaneGeometry(0.026, 0.0073), lblMat)
    lbl.position.set(x, 0.012, 0.014)
    group.add(lbl)
  }
  return { group, dispose: () => disp.forEach((d) => { try { d.dispose() } catch (_e) { /* noop */ } }) }
}

// ---------- hand-over hot zone (pulsing ring on the bench) ----------
export function buildHandoverZone(): {
  mesh: THREE.Mesh; update: (t: number) => void; dispose: () => void
} {
  const geo = new THREE.RingGeometry(0.05, 0.075, 64)
  const mat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, side: THREE.DoubleSide,
    uniforms: { uTime: { value: 0 } },
    vertexShader: `
      varying vec2 vUv;
      void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }
    `,
    fragmentShader: `
      varying vec2 vUv;
      uniform float uTime;
      void main(){
        float pulse = 0.5 + 0.5 * sin(uTime * 2.2);
        // edge bright, centre dim
        float radial = abs(vUv.x - 0.5) + abs(vUv.y - 0.5);
        float a = (0.35 + 0.4 * pulse) * (0.6 + 0.4 * radial);
        gl_FragColor = vec4(0.15, 0.55, 0.93, a);
      }
    `,
  })
  const mesh = new THREE.Mesh(geo, mat)
  mesh.rotation.x = -Math.PI / 2
  mesh.position.y = 0.005
  return { mesh, update: (t) => { mat.uniforms.uTime.value = t }, dispose: () => { geo.dispose(); mat.dispose() } }
}
