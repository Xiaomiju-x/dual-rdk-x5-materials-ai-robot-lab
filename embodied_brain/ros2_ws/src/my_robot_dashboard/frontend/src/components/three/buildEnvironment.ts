import * as THREE from 'three'
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js'
import { RGBELoader } from 'three/examples/jsm/loaders/RGBELoader.js'

/**
 * buildEnvironment — produce a PMREM env map for premium reflections.
 *
 * Strategy:
 *  1. Eagerly install RoomEnvironment-baked PMREM so the scene NEVER
 *     looks flat, even on first paint and offline.
 *  2. Asynchronously try to upgrade to a real HDR (polyhaven CDN) and
 *     swap in once it lands. Falls back silently if the network refuses.
 *
 * Returns { envTexture, replaceWithHdr, dispose }.
 */
export interface EnvironmentBundle {
  envTexture: THREE.Texture
  /** request a higher-fidelity HDR upgrade in the background */
  upgradeToHdr: (url?: string) => void
  dispose: () => void
}

const DEFAULT_HDR_URL = 'https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/studio_small_03_1k.hdr'

export function buildEnvironment(renderer: THREE.WebGLRenderer): EnvironmentBundle {
  const pmrem = new THREE.PMREMGenerator(renderer)
  pmrem.compileEquirectangularShader()

  // Step 1 — instant procedural environment
  const room = new RoomEnvironment()
  const roomTarget = pmrem.fromScene(room, 0.04)
  let active: THREE.Texture = roomTarget.texture
  let activeRT: THREE.WebGLRenderTarget | null = roomTarget

  function upgradeToHdr(url: string = DEFAULT_HDR_URL): void {
    new RGBELoader().setDataType(THREE.HalfFloatType).load(
      url,
      (hdrTex) => {
        try {
          const hdrTarget = pmrem.fromEquirectangular(hdrTex)
          // swap reference; callers reading envTexture once at init still get
          // the original (it's a Texture), so we mutate by replacing image data:
          // simplest is just to assign hdrTarget.texture into the bundle field
          // and rely on the caller to scene.environment = bundle.envTexture again.
          // We expose `_swap` via the bundle:
          ;(bundle as { envTexture: THREE.Texture }).envTexture = hdrTarget.texture
          if (activeRT) activeRT.dispose()
          activeRT = hdrTarget
          active = hdrTarget.texture
          hdrTex.dispose()
        } catch (e) {
          // swallow — RoomEnvironment is good enough
          console.warn('[buildEnvironment] HDR upgrade failed:', e)
        }
      },
      undefined,
      () => {
        // network error — silent
      },
    )
  }

  function dispose(): void {
    if (activeRT) activeRT.dispose()
    pmrem.dispose()
  }

  const bundle: EnvironmentBundle = {
    envTexture: active,
    upgradeToHdr,
    dispose,
  }
  return bundle
}
