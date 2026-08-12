import * as THREE from 'three'
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js'
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js'
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js'
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js'

/**
 * buildPostFX — premium bloom-driven composer that wraps the scene/camera
 * renderer. Cheap-but-pretty: 1× RenderPass → UnrealBloomPass → OutputPass.
 *
 *   const fx = buildPostFX(renderer, scene, camera, w, h)
 *   fx.composer.render()   // call each frame instead of renderer.render()
 */
export interface PostFXBundle {
  composer: EffectComposer
  bloom: UnrealBloomPass
  setSize: (w: number, h: number) => void
  setBloom: (enabled: boolean) => void
  dispose: () => void
}

export function buildPostFX(
  renderer: THREE.WebGLRenderer,
  scene: THREE.Scene,
  camera: THREE.Camera,
  w: number,
  h: number,
): PostFXBundle {
  const composer = new EffectComposer(renderer)
  composer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
  composer.setSize(w, h)

  const renderPass = new RenderPass(scene, camera)
  composer.addPass(renderPass)

  // (strength, radius, threshold) — restrained but premium
  const bloom = new UnrealBloomPass(new THREE.Vector2(w, h), 0.42, 0.55, 0.88)
  composer.addPass(bloom)

  const output = new OutputPass()
  composer.addPass(output)

  function setSize(nw: number, nh: number): void {
    composer.setSize(nw, nh)
    bloom.setSize(nw, nh)
  }
  function setBloom(enabled: boolean): void {
    bloom.enabled = enabled
  }
  function dispose(): void {
    composer.dispose()
  }

  return { composer, bloom, setSize, setBloom, dispose }
}
