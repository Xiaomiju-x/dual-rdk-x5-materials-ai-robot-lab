import * as THREE from 'three'

/**
 * Studio scene setup — premium stage look with light/dark palette swap.
 *  - ambient + key + rim lighting
 *  - radial grid floor
 *  - faint hemisphere bounce
 *  - subtle radial fog vignetting (handled in CSS overlay, not three.fog, so PMREM still works)
 */
export interface StageBundle {
  group: THREE.Group
  setGridFade: (t: number) => void
  applyTheme: (theme: 'light' | 'dark') => void
}

const PALETTES = {
  light: {
    floor: 0xf4f7fb,
    floorRoughness: 0.6,
    grid: 0x2563eb,
    gridBase: 0.12,
    core: 0x06b6d4,
    halo: 0x7c3aed,
    haloBase: 0.14,
    ambient: 0.55,
    keyColor: 0xffffff,
    keyIntensity: 1.1,
    fill: 0xdfe9f7,
    rim: 0xa5f3fc,
    hemiSky: 0xeaf2ff,
    hemiGround: 0xd1d5db,
  },
  dark: {
    floor: 0x1f2638,
    floorRoughness: 0.55,
    grid: 0x60a5fa,
    gridBase: 0.28,
    core: 0x22d3ee,
    halo: 0xa78bfa,
    haloBase: 0.30,
    ambient: 0.65,
    keyColor: 0xeef2ff,
    keyIntensity: 1.15,
    fill: 0x3b82f6,
    rim: 0x67e8f9,
    hemiSky: 0x334155,
    hemiGround: 0x1f2638,
  },
} as const

export function buildStage(initialTheme: 'light' | 'dark' = 'light'): StageBundle {
  const group = new THREE.Group()
  group.name = 'stage'

  let pal = PALETTES[initialTheme]

  // Lighting — three-point
  const ambient = new THREE.AmbientLight(0xffffff, pal.ambient)
  group.add(ambient)
  const key = new THREE.DirectionalLight(pal.keyColor, pal.keyIntensity)
  key.position.set(3, 4.5, 2.5)
  key.castShadow = true
  key.shadow.mapSize.set(1024, 1024)
  key.shadow.camera.near = 0.5
  key.shadow.camera.far = 12
  key.shadow.camera.left = -3
  key.shadow.camera.right = 3
  key.shadow.camera.top = 3
  key.shadow.camera.bottom = -3
  key.shadow.bias = -0.0008
  group.add(key)
  const fill = new THREE.DirectionalLight(pal.fill, 0.35)
  fill.position.set(-3, 2, -2)
  group.add(fill)
  const rim = new THREE.DirectionalLight(pal.rim, 0.4)
  rim.position.set(-1, 1.5, 4)
  group.add(rim)
  const hemi = new THREE.HemisphereLight(pal.hemiSky, pal.hemiGround, 0.3)
  group.add(hemi)

  // Floor — premium matte with subtle reflections (theme-tinted)
  const floorMat = new THREE.MeshPhysicalMaterial({
    color: pal.floor,
    metalness: 0.05,
    roughness: pal.floorRoughness,
    clearcoat: 0.3,
    clearcoatRoughness: 0.45,
  })
  const floor = new THREE.Mesh(new THREE.CircleGeometry(6, 64), floorMat)
  floor.rotation.x = -Math.PI / 2
  floor.position.y = -0.001
  floor.receiveShadow = true
  group.add(floor)

  // Radial grid using thin lines
  const gridRadius = 5.5
  const gridDivisions = 22
  const gridGeom = new THREE.BufferGeometry()
  const verts: number[] = []
  for (let i = 1; i <= 10; i++) {
    const r = (i / 10) * gridRadius
    const segs = 64
    for (let j = 0; j < segs; j++) {
      const a0 = (j / segs) * Math.PI * 2
      const a1 = ((j + 1) / segs) * Math.PI * 2
      verts.push(Math.cos(a0) * r, 0.001, Math.sin(a0) * r)
      verts.push(Math.cos(a1) * r, 0.001, Math.sin(a1) * r)
    }
  }
  for (let i = 0; i < gridDivisions; i++) {
    const a = (i / gridDivisions) * Math.PI * 2
    verts.push(0, 0.001, 0)
    verts.push(Math.cos(a) * gridRadius, 0.001, Math.sin(a) * gridRadius)
  }
  gridGeom.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3))
  const gridMat = new THREE.LineBasicMaterial({
    color: pal.grid,
    transparent: true,
    opacity: pal.gridBase,
  })
  const grid = new THREE.LineSegments(gridGeom, gridMat)
  group.add(grid)

  // bright inner core ring
  const coreRingGeom = new THREE.RingGeometry(0.5, 0.52, 64)
  const coreRingMat = new THREE.MeshBasicMaterial({
    color: pal.core,
    transparent: true,
    opacity: 0.6,
    side: THREE.DoubleSide,
  })
  const coreRing = new THREE.Mesh(coreRingGeom, coreRingMat)
  coreRing.rotation.x = -Math.PI / 2
  coreRing.position.y = 0.002
  group.add(coreRing)

  // outer glow ring
  const haloGeom = new THREE.RingGeometry(2.0, 2.05, 96)
  const haloMat = new THREE.MeshBasicMaterial({
    color: pal.halo,
    transparent: true,
    opacity: pal.haloBase,
    side: THREE.DoubleSide,
  })
  const halo = new THREE.Mesh(haloGeom, haloMat)
  halo.rotation.x = -Math.PI / 2
  halo.position.y = 0.0015
  group.add(halo)

  function setGridFade(t: number): void {
    gridMat.opacity = pal.gridBase + 0.05 * Math.sin(t * 0.8)
    haloMat.opacity = pal.haloBase + 0.06 * Math.sin(t * 0.6 + Math.PI / 3)
    coreRingMat.opacity = 0.5 + 0.15 * Math.sin(t * 1.6)
  }

  function applyTheme(theme: 'light' | 'dark'): void {
    pal = PALETTES[theme]
    ambient.intensity = pal.ambient
    key.color.set(pal.keyColor)
    key.intensity = pal.keyIntensity
    fill.color.set(pal.fill)
    rim.color.set(pal.rim)
    hemi.color.set(pal.hemiSky)
    hemi.groundColor.set(pal.hemiGround)
    floorMat.color.set(pal.floor)
    floorMat.roughness = pal.floorRoughness
    gridMat.color.set(pal.grid)
    coreRingMat.color.set(pal.core)
    haloMat.color.set(pal.halo)
  }

  return { group, setGridFade, applyTheme }
}
