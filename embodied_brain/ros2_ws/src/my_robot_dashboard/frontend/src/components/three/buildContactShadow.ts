import * as THREE from 'three'

/**
 * buildContactShadow — soft round contact shadow under a target,
 * using a radial-gradient transparent material on a flat disc.
 *
 *   const cs = buildContactShadow()
 *   robot.add(cs.group)
 */
export interface ContactShadowBundle {
  group: THREE.Group
  setStrength: (k: number) => void
  dispose: () => void
}

export function buildContactShadow(radius = 0.6, strength = 0.45): ContactShadowBundle {
  const group = new THREE.Group()
  group.name = 'contactShadow'

  // Build a radial gradient texture procedurally (no asset needed)
  const size = 256
  const canvas = document.createElement('canvas')
  canvas.width = size; canvas.height = size
  const ctx = canvas.getContext('2d')!
  const grad = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2)
  grad.addColorStop(0,    'rgba(15, 23, 42, 0.55)')
  grad.addColorStop(0.45, 'rgba(15, 23, 42, 0.22)')
  grad.addColorStop(0.85, 'rgba(15, 23, 42, 0.04)')
  grad.addColorStop(1,    'rgba(15, 23, 42, 0)')
  ctx.fillStyle = grad
  ctx.fillRect(0, 0, size, size)
  const tex = new THREE.CanvasTexture(canvas)
  tex.colorSpace = THREE.SRGBColorSpace

  const geom = new THREE.PlaneGeometry(radius * 2, radius * 2)
  const mat = new THREE.MeshBasicMaterial({
    map: tex,
    transparent: true,
    depthWrite: false,
    opacity: strength,
  })
  const mesh = new THREE.Mesh(geom, mat)
  mesh.rotation.x = -Math.PI / 2
  mesh.position.y = 0.001
  group.add(mesh)

  return {
    group,
    setStrength(k: number) { mat.opacity = k },
    dispose() { geom.dispose(); mat.dispose(); tex.dispose() },
  }
}
