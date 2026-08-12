import * as THREE from 'three'

/**
 * Procedural robot mesh — chassis + 4 wheels + LiDAR mast + lift column + cameras.
 * No URDF, no glTF: built from primitives so it ships in <1 KB and stays portable.
 */
export interface RobotGroup extends THREE.Group {
  userData: {
    lidarTop: THREE.Mesh
    cameraSpot: THREE.Mesh
    wheels: THREE.Mesh[]
  }
}

export function buildRobot(): RobotGroup {
  const group = new THREE.Group() as RobotGroup
  group.name = 'robot'

  const matChassis = new THREE.MeshPhysicalMaterial({
    color: 0xf3f6fb,
    metalness: 0.15,
    roughness: 0.35,
    clearcoat: 0.6,
    clearcoatRoughness: 0.25,
  })
  const matChassisAccent = new THREE.MeshPhysicalMaterial({
    color: 0x2563eb,
    metalness: 0.25,
    roughness: 0.4,
    emissive: 0x1e40af,
    emissiveIntensity: 0.15,
  })
  const matWheel = new THREE.MeshPhysicalMaterial({ color: 0x111827, metalness: 0.6, roughness: 0.45 })
  const matLidar = new THREE.MeshPhysicalMaterial({
    color: 0x0891b2,
    metalness: 0.5,
    roughness: 0.2,
    emissive: 0x06b6d4,
    emissiveIntensity: 0.45,
  })
  const matGlass = new THREE.MeshPhysicalMaterial({
    color: 0xffffff,
    metalness: 0.05,
    roughness: 0.05,
    transparent: true,
    opacity: 0.32,
    transmission: 0.85,
    thickness: 0.3,
  })

  // Chassis — main rounded box body
  const chassis = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.16, 0.42), matChassis)
  chassis.position.y = 0.16
  chassis.castShadow = true
  chassis.receiveShadow = true
  group.add(chassis)

  // Accent strip on top of chassis
  const stripe = new THREE.Mesh(new THREE.BoxGeometry(0.46, 0.02, 0.38), matChassisAccent)
  stripe.position.y = 0.25
  group.add(stripe)

  // 4 wheels (driven)
  const wheels: THREE.Mesh[] = []
  const wheelGeom = new THREE.CylinderGeometry(0.08, 0.08, 0.04, 24)
  wheelGeom.rotateZ(Math.PI / 2)
  const offsetsX = [-0.22, 0.22]
  const offsetsZ = [-0.16, 0.16]
  for (const ox of offsetsX) {
    for (const oz of offsetsZ) {
      const wheel = new THREE.Mesh(wheelGeom, matWheel)
      wheel.position.set(ox, 0.08, oz)
      wheel.castShadow = true
      group.add(wheel)
      wheels.push(wheel)
    }
  }

  // LiDAR puck (top, glowing)
  const lidarBase = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, 0.04, 24), matChassis)
  lidarBase.position.y = 0.28
  group.add(lidarBase)
  const lidarTop = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.055, 0.04, 24), matLidar)
  lidarTop.position.y = 0.32
  group.add(lidarTop)
  group.userData.lidarTop = lidarTop

  // Lift column (tall slim, with translucent collar)
  const column = new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.55, 0.06), matChassis)
  column.position.set(0.18, 0.43, 0)
  column.castShadow = true
  group.add(column)
  const collar = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.12, 0.08), matGlass)
  collar.position.set(0.18, 0.58, 0)
  group.add(collar)

  // Camera spot (front)
  const cameraSpot = new THREE.Mesh(new THREE.SphereGeometry(0.025, 16, 12), matChassisAccent)
  cameraSpot.position.set(0.25, 0.2, 0)
  group.add(cameraSpot)
  group.userData.cameraSpot = cameraSpot

  // Astra Pro depth — front-top wedge
  const astra = new THREE.Mesh(new THREE.BoxGeometry(0.15, 0.045, 0.045), matChassis)
  astra.position.set(0.2, 0.27, 0)
  group.add(astra)
  const astraLens1 = new THREE.Mesh(new THREE.SphereGeometry(0.012, 12, 8), matLidar)
  astraLens1.position.set(0.27, 0.27, -0.018)
  group.add(astraLens1)
  const astraLens2 = astraLens1.clone()
  astraLens2.position.z = 0.018
  group.add(astraLens2)

  group.userData.wheels = wheels
  return group
}
