import * as THREE from 'three'
import { AXES, LINKS, BASE_Y, LINK_GAP, TIP_OFFSET } from './kinematics'

/**
 * myCobot 280-Pi — procedural high-fidelity studio model.
 *
 * Chain matches kinematics.ts (AXES / LINKS / BASE_Y / LINK_GAP / TIP_OFFSET)
 * so FK and IK in the cockpit map exactly to the rendered tip world position.
 *
 * Each joint segment is built from:
 *   - rounded motor housing (RoundedBox shader-free, custom geometry)
 *   - brushed aluminum side faces (anisotropy + canvas normal)
 *   - emissive accent strip + LED ring on the rotation face
 *   - black silicone cable wrap connecting adjacent housings
 * The Atom end-effector head carries the status LED.
 *
 * Materials are reused across both arms; only the accent colour changes.
 */

export type ArmPalette = 'amber' | 'blue'

const ACCENTS: Record<ArmPalette, {
  accent: number; emissive: number; nameplate: string
}> = {
  amber: { accent: 0xd97706, emissive: 0xf59e0b, nameplate: 'arm01' },
  blue:  { accent: 0x2563eb, emissive: 0x3b82f6, nameplate: 'arm02' },
}

export interface ArmBundle {
  group: THREE.Group
  setAngles: (deg: number[], gripper: number) => void
  tipWorld: () => THREE.Vector3
  setVerb: (v: 'idle' | 'moving' | 'error') => void
  setReachVisible: (v: boolean) => void
  dispose: () => void
}

// ---------- brushed metal normal map (procedural) ----------
function brushedNormal(seed: number): THREE.CanvasTexture {
  const cvs = document.createElement('canvas')
  cvs.width = 256; cvs.height = 256
  const ctx = cvs.getContext('2d')!
  // base normal (pointing up = #8080ff)
  ctx.fillStyle = '#8080ff'; ctx.fillRect(0, 0, 256, 256)
  // brushed horizontal lines (vary green channel slightly)
  for (let y = 0; y < 256; y++) {
    const v = 128 + Math.round(((Math.sin(y * 0.7 + seed) + Math.sin(y * 1.9 + seed * 3.1)) * 6))
    ctx.fillStyle = `rgb(128, ${v}, 255)`
    ctx.fillRect(0, y, 256, 1)
  }
  const tex = new THREE.CanvasTexture(cvs)
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping
  tex.repeat.set(2, 2)
  tex.anisotropy = 4
  return tex
}

// ---------- nameplate ----------
function nameplateTex(text: string, accent: string): THREE.CanvasTexture {
  const cvs = document.createElement('canvas')
  cvs.width = 256; cvs.height = 96
  const ctx = cvs.getContext('2d')!
  // black anodised background
  const bg = ctx.createLinearGradient(0, 0, 0, 96)
  bg.addColorStop(0, '#1f2937'); bg.addColorStop(1, '#0f172a')
  ctx.fillStyle = bg; ctx.fillRect(0, 0, 256, 96)
  // accent stripe
  ctx.fillStyle = accent; ctx.fillRect(0, 0, 256, 4)
  // text
  ctx.font = 'bold 30px "JetBrains Mono", monospace'
  ctx.fillStyle = '#e5e7eb'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'
  ctx.fillText('myCobot 280-Pi', 128, 38)
  ctx.font = '600 22px "JetBrains Mono", monospace'
  ctx.fillStyle = accent
  ctx.fillText(text, 128, 70)
  const tex = new THREE.CanvasTexture(cvs)
  tex.anisotropy = 4
  return tex
}

// ---------- shared rounded-box helper (4 corner segments) ----------
function roundedBox(w: number, h: number, d: number, r: number): THREE.BufferGeometry {
  // Use ExtrudeGeometry from a rounded rect — gives nice top/bottom + chamfered sides
  const shape = new THREE.Shape()
  const x = -w / 2, y = -h / 2
  shape.moveTo(x + r, y)
  shape.lineTo(x + w - r, y)
  shape.quadraticCurveTo(x + w, y, x + w, y + r)
  shape.lineTo(x + w, y + h - r)
  shape.quadraticCurveTo(x + w, y + h, x + w - r, y + h)
  shape.lineTo(x + r, y + h)
  shape.quadraticCurveTo(x, y + h, x, y + h - r)
  shape.lineTo(x, y + r)
  shape.quadraticCurveTo(x, y, x + r, y)
  const geo = new THREE.ExtrudeGeometry(shape, {
    depth: d, bevelEnabled: true, bevelThickness: 0.0025, bevelSize: 0.0022, bevelSegments: 2, steps: 1, curveSegments: 6,
  })
  geo.translate(0, 0, -d / 2)
  geo.computeVertexNormals()
  return geo
}

// ---------- cable wrap between two y-stacked frames ----------
function cableWrap(length: number, accent: number): THREE.Mesh {
  // spiral CatmullRom along y axis from 0 to length
  const turns = 3
  const radius = 0.014
  const pts: THREE.Vector3[] = []
  const N = 28
  for (let i = 0; i <= N; i++) {
    const t = i / N
    const a = t * turns * Math.PI * 2
    pts.push(new THREE.Vector3(Math.cos(a) * radius, t * length, Math.sin(a) * radius))
  }
  const curve = new THREE.CatmullRomCurve3(pts)
  const tube = new THREE.TubeGeometry(curve, 28, 0.0035, 6, false)
  // black silicone — subtle hint of accent so eye picks it up against white body
  const mat = new THREE.MeshPhysicalMaterial({
    color: 0x141821, metalness: 0.2, roughness: 0.55, sheen: 0.4, sheenColor: new THREE.Color(accent),
  })
  const mesh = new THREE.Mesh(tube, mat)
  mesh.userData.dispose = () => { tube.dispose(); mat.dispose() }
  return mesh
}

// ---------- a single joint segment ----------
interface SegmentMats {
  body: THREE.MeshPhysicalMaterial
  bodyMetal: THREE.MeshPhysicalMaterial
  dark: THREE.MeshPhysicalMaterial
  accent: THREE.MeshPhysicalMaterial
  ledMat: THREE.MeshStandardMaterial
}

function buildSegment(
  index: number,
  linkLen: number,
  axis: 'x' | 'y' | 'z',
  mats: SegmentMats,
  disposables: Array<{ dispose: () => void }>,
): { pivot: THREE.Group; next: THREE.Group; ledRing: THREE.Mesh } {
  const pivot = new THREE.Group()
  pivot.name = `j${index + 1}`

  // motor housing — slightly larger at first joint
  const housingSize = index === 0 ? 0.044 : 0.038
  const hGeo = roundedBox(housingSize, housingSize, housingSize, 0.006)
  disposables.push(hGeo as unknown as { dispose: () => void })
  const housing = new THREE.Mesh(hGeo, mats.bodyMetal)
  housing.castShadow = true; housing.receiveShadow = true
  pivot.add(housing)

  // rotation-face accent ring (faces along rotation axis)
  const ringGeo = new THREE.RingGeometry(housingSize * 0.42, housingSize * 0.5, 32)
  const ringMat = mats.ledMat.clone()
  ringMat.color = new THREE.Color(mats.accent.color)
  ringMat.emissive = new THREE.Color(mats.accent.emissive)
  ringMat.emissiveIntensity = 0.35
  ringMat.side = THREE.DoubleSide
  disposables.push(ringGeo, ringMat)
  const ledRing = new THREE.Mesh(ringGeo, ringMat)
  // orient the ring on the face perpendicular to the rotation axis
  if (axis === 'x') ledRing.rotation.y = Math.PI / 2
  else if (axis === 'y') ledRing.rotation.x = Math.PI / 2
  // z: facing camera default
  ledRing.position.set(
    axis === 'x' ? housingSize / 2 + 0.001 : 0,
    axis === 'y' ? housingSize / 2 + 0.001 : 0,
    axis === 'z' ? housingSize / 2 + 0.001 : 0,
  )
  pivot.add(ledRing)

  // small inset disk in the centre of the ring
  const diskGeo = new THREE.CircleGeometry(housingSize * 0.4, 24)
  disposables.push(diskGeo)
  const disk = new THREE.Mesh(diskGeo, mats.dark)
  disk.rotation.copy(ledRing.rotation)
  disk.position.copy(ledRing.position)
  if (axis === 'x') disk.position.x -= 0.0005
  else if (axis === 'y') disk.position.y -= 0.0005
  else disk.position.z -= 0.0005
  pivot.add(disk)

  // a tiny screw cluster on the housing for detail
  const screwGeo = new THREE.CylinderGeometry(0.0018, 0.0018, 0.0015, 6)
  disposables.push(screwGeo)
  for (let s = 0; s < 4; s++) {
    const ang = (s / 4) * Math.PI * 2 + Math.PI / 4
    const sr = housingSize * 0.36
    const screw = new THREE.Mesh(screwGeo, mats.dark)
    screw.rotation.x = Math.PI / 2
    screw.position.set(Math.cos(ang) * sr, housingSize / 2 + 0.0008, Math.sin(ang) * sr)
    pivot.add(screw)
  }

  // link cylinder above the housing — slightly waisted via two stacked cyls
  const linkLower = new THREE.CylinderGeometry(0.018, 0.022, linkLen * 0.55, 20)
  const linkUpper = new THREE.CylinderGeometry(0.022, 0.018, linkLen * 0.45, 20)
  disposables.push(linkLower, linkUpper)
  const lL = new THREE.Mesh(linkLower, mats.body)
  const lU = new THREE.Mesh(linkUpper, mats.body)
  lL.position.y = (linkLen * 0.55) / 2 + housingSize / 2
  lU.position.y = linkLen * 0.55 + (linkLen * 0.45) / 2 + housingSize / 2
  lL.castShadow = true; lU.castShadow = true
  pivot.add(lL); pivot.add(lU)

  // cable wrap (only between joints 1..5)
  if (index < 5) {
    const cable = cableWrap(linkLen + LINK_GAP * 0.5, (mats.accent.emissive as THREE.Color).getHex())
    cable.position.y = housingSize / 2
    pivot.add(cable)
    disposables.push({ dispose: cable.userData.dispose as () => void })
  }

  // next frame sits at top of the link
  const next = new THREE.Group()
  next.position.y = linkLen + LINK_GAP
  pivot.add(next)

  return { pivot, next, ledRing }
}

// ---------- main builder ----------
export function buildArm(palette: ArmPalette): ArmBundle {
  const pal = ACCENTS[palette]
  const disposables: Array<{ dispose: () => void }> = []

  // shared materials
  const brushed = brushedNormal(palette === 'amber' ? 1.7 : 4.2)
  disposables.push(brushed)
  const body = new THREE.MeshPhysicalMaterial({
    color: 0xf3f6fb, metalness: 0.22, roughness: 0.36, clearcoat: 0.55, clearcoatRoughness: 0.28,
    normalMap: brushed, normalScale: new THREE.Vector2(0.18, 0.18),
  })
  const bodyMetal = new THREE.MeshPhysicalMaterial({
    color: 0xe5e9f0, metalness: 0.72, roughness: 0.34, clearcoat: 0.6, clearcoatRoughness: 0.22,
    normalMap: brushed, normalScale: new THREE.Vector2(0.32, 0.32),
    anisotropy: 0.7,
  })
  const dark = new THREE.MeshPhysicalMaterial({
    color: 0x1c2230, metalness: 0.6, roughness: 0.48,
  })
  const accent = new THREE.MeshPhysicalMaterial({
    color: pal.accent, metalness: 0.4, roughness: 0.32,
    emissive: pal.emissive, emissiveIntensity: 0.35,
  })
  const ledMat = new THREE.MeshStandardMaterial({
    color: pal.accent, emissive: pal.emissive, emissiveIntensity: 0.4,
  })
  disposables.push(body, bodyMetal, dark, accent, ledMat)

  const group = new THREE.Group()
  group.name = `arm-${palette}`

  // ---------- base ----------
  // wide foot plate
  const footGeo = new THREE.CylinderGeometry(0.085, 0.095, 0.012, 64)
  disposables.push(footGeo)
  const foot = new THREE.Mesh(footGeo, dark); foot.position.y = 0.006
  foot.receiveShadow = true; group.add(foot)
  // accent ring inset on foot
  const footRingGeo = new THREE.TorusGeometry(0.07, 0.0028, 8, 64)
  disposables.push(footRingGeo)
  const footRing = new THREE.Mesh(footRingGeo, accent); footRing.rotation.x = Math.PI / 2
  footRing.position.y = 0.0125; group.add(footRing)
  // mid section
  const midGeo = new THREE.CylinderGeometry(0.052, 0.064, 0.024, 32)
  disposables.push(midGeo)
  const mid = new THREE.Mesh(midGeo, bodyMetal); mid.position.y = 0.024; group.add(mid)
  // nameplate panel (small box stuck on mid-section facing camera +z)
  const npGeo = new THREE.PlaneGeometry(0.05, 0.018)
  disposables.push(npGeo)
  const npTex = nameplateTex(pal.nameplate, palette === 'amber' ? '#f59e0b' : '#60a5fa')
  disposables.push(npTex)
  const npMat = new THREE.MeshBasicMaterial({ map: npTex })
  disposables.push(npMat)
  const namePlate = new THREE.Mesh(npGeo, npMat)
  namePlate.position.set(0, 0.024, 0.0645)
  group.add(namePlate)
  // top base under pivot
  const topGeo = new THREE.CylinderGeometry(0.04, 0.046, 0.026, 32)
  disposables.push(topGeo)
  const top = new THREE.Mesh(topGeo, body); top.position.y = 0.049; group.add(top)

  // ---------- chain ----------
  const pivots: THREE.Group[] = []
  const ledRings: THREE.Mesh[] = []
  let parent: THREE.Object3D = group
  let firstY = BASE_Y
  for (let i = 0; i < 6; i++) {
    const seg = buildSegment(i, LINKS[i], AXES[i], {
      body, bodyMetal, dark, accent, ledMat,
    }, disposables)
    seg.pivot.position.y = i === 0 ? firstY : 0
    parent.add(seg.pivot)
    pivots.push(seg.pivot)
    ledRings.push(seg.ledRing)
    parent = seg.next
    firstY = 0
  }

  // ---------- gripper (parallel-jaw) ----------
  const gripper = new THREE.Group()
  parent.add(gripper)
  // wrist plate
  const wristGeo = roundedBox(0.052, 0.014, 0.038, 0.005)
  disposables.push(wristGeo as unknown as { dispose: () => void })
  const wrist = new THREE.Mesh(wristGeo, dark)
  wrist.position.y = 0.007; gripper.add(wrist)
  // atom head — small cube with status LED on top
  const atomGeo = roundedBox(0.036, 0.024, 0.03, 0.004)
  disposables.push(atomGeo as unknown as { dispose: () => void })
  const atom = new THREE.Mesh(atomGeo, bodyMetal)
  atom.position.y = 0.026; gripper.add(atom)
  // status LED (idle blue / moving green / error red)
  const ledGeo = new THREE.SphereGeometry(0.0035, 12, 8)
  disposables.push(ledGeo)
  const ledStatusMat = new THREE.MeshStandardMaterial({
    color: 0x60a5fa, emissive: 0x60a5fa, emissiveIntensity: 1.2,
  })
  disposables.push(ledStatusMat)
  const led = new THREE.Mesh(ledGeo, ledStatusMat)
  led.position.set(0.011, 0.04, 0); gripper.add(led)

  // gripper rail housing
  const railGeo = roundedBox(0.05, 0.008, 0.024, 0.003)
  disposables.push(railGeo as unknown as { dispose: () => void })
  const rail = new THREE.Mesh(railGeo, dark); rail.position.y = 0.04; gripper.add(rail)

  // L-shaped fingers (made of two boxes each)
  const fingerLong = roundedBox(0.008, 0.044, 0.022, 0.002)
  const fingerHook = roundedBox(0.014, 0.008, 0.022, 0.002)
  disposables.push(fingerLong as unknown as { dispose: () => void }, fingerHook as unknown as { dispose: () => void })

  const fingerLg = new THREE.Group()
  const flA = new THREE.Mesh(fingerLong, accent)
  const flB = new THREE.Mesh(fingerHook, accent)
  flA.position.y = 0.022 + 0.044 / 2; flB.position.y = 0.022 + 0.044; flB.position.x = 0.003
  fingerLg.add(flA); fingerLg.add(flB); gripper.add(fingerLg)
  const fingerRg = new THREE.Group()
  const frA = new THREE.Mesh(fingerLong, accent)
  const frB = new THREE.Mesh(fingerHook, accent)
  frA.position.y = 0.022 + 0.044 / 2; frB.position.y = 0.022 + 0.044; frB.position.x = -0.003
  fingerRg.add(frA); fingerRg.add(frB); gripper.add(fingerRg)

  // silicone pads (inner side)
  const padGeo = new THREE.BoxGeometry(0.002, 0.04, 0.020)
  disposables.push(padGeo)
  const padMat = new THREE.MeshPhysicalMaterial({ color: 0x111418, roughness: 0.85, sheen: 0.5 })
  disposables.push(padMat)
  const padL = new THREE.Mesh(padGeo, padMat); padL.position.y = 0.022 + 0.044 / 2
  const padR = new THREE.Mesh(padGeo, padMat); padR.position.y = 0.022 + 0.044 / 2
  fingerLg.add(padL); fingerRg.add(padR)

  // tip marker (invisible) at the centre between fingers, matches TIP_OFFSET
  const tip = new THREE.Object3D()
  tip.position.y = TIP_OFFSET
  gripper.add(tip)

  // ---------- reach envelope (toggleable) ----------
  // soft translucent sphere fresnel
  const reachMat = new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    side: THREE.BackSide,
    uniforms: { uColor: { value: new THREE.Color(pal.accent) }, uTime: { value: 0 } },
    vertexShader: `
      varying vec3 vNormal;
      varying vec3 vView;
      void main() {
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        vNormal = normalize(normalMatrix * normal);
        vView = normalize(-mv.xyz);
        gl_Position = projectionMatrix * mv;
      }
    `,
    fragmentShader: `
      uniform vec3 uColor;
      uniform float uTime;
      varying vec3 vNormal;
      varying vec3 vView;
      void main() {
        float f = 1.0 - abs(dot(vNormal, vView));
        f = pow(f, 2.0);
        float pulse = 0.5 + 0.5 * sin(uTime * 1.5);
        float a = f * (0.08 + 0.04 * pulse);
        gl_FragColor = vec4(uColor, a);
      }
    `,
  })
  disposables.push(reachMat)
  const reachGeo = new THREE.SphereGeometry(0.35, 32, 24)
  disposables.push(reachGeo)
  const reachMesh = new THREE.Mesh(reachGeo, reachMat)
  reachMesh.position.y = BASE_Y
  reachMesh.visible = false
  group.add(reachMesh)

  // ---------- API ----------
  let lastAngles: number[] = [0,0,0,0,0,0]
  let lastSetAt = performance.now()
  let lastAngleVel = 0
  let verbState: 'idle' | 'moving' | 'error' = 'idle'

  function setAngles(deg: number[], gripperPct: number) {
    const now = performance.now()
    const dt = Math.max(0.001, (now - lastSetAt) / 1000)
    let sum = 0
    for (let i = 0; i < 6; i++) {
      const rad = ((deg[i] ?? 0) * Math.PI) / 180
      const ax = AXES[i]
      pivots[i].rotation.x = ax === 'x' ? rad : 0
      pivots[i].rotation.y = ax === 'y' ? rad : 0
      pivots[i].rotation.z = ax === 'z' ? rad : 0
      sum += Math.abs((deg[i] ?? 0) - lastAngles[i])
    }
    lastAngleVel = sum / dt
    lastAngles = deg.slice()
    lastSetAt = now

    // gripper: 0 open → fingers apart, 100 closed → together
    const open = 1 - Math.max(0, Math.min(100, gripperPct)) / 100
    const spread = 0.008 + open * 0.014
    fingerLg.position.x = -spread
    fingerRg.position.x =  spread

    // LED rings — emissive intensity driven by joint velocity
    const intensity = 0.25 + Math.min(1.5, lastAngleVel * 0.04)
    for (const ring of ledRings) {
      (ring.material as THREE.MeshStandardMaterial).emissiveIntensity = intensity
    }

    // reach envelope pulse
    if (reachMesh.visible) {
      reachMat.uniforms.uTime.value = now / 1000
    }

    // status LED auto-bias toward "moving" if velocity > threshold (verb override wins)
    if (verbState === 'idle') {
      if (lastAngleVel > 25) {
        ledStatusMat.color.setHex(0x10b981); ledStatusMat.emissive.setHex(0x10b981)
      } else {
        ledStatusMat.color.setHex(0x60a5fa); ledStatusMat.emissive.setHex(0x60a5fa)
      }
    }
  }

  function tipWorld(): THREE.Vector3 {
    const v = new THREE.Vector3()
    tip.getWorldPosition(v)
    return v
  }

  function setVerb(v: 'idle' | 'moving' | 'error') {
    verbState = v
    if (v === 'error') {
      ledStatusMat.color.setHex(0xef4444); ledStatusMat.emissive.setHex(0xef4444)
    } else if (v === 'moving') {
      ledStatusMat.color.setHex(0x10b981); ledStatusMat.emissive.setHex(0x10b981)
    } else {
      ledStatusMat.color.setHex(0x60a5fa); ledStatusMat.emissive.setHex(0x60a5fa)
    }
  }

  function setReachVisible(v: boolean) { reachMesh.visible = v }

  function dispose() {
    for (const d of disposables) {
      try { d.dispose() } catch (_e) { /* noop */ }
    }
  }

  setAngles([0, 0, 0, 0, 0, 0], 50)
  return { group, setAngles, tipWorld, setVerb, setReachVisible, dispose }
}
