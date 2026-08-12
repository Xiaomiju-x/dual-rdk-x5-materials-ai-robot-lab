<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from 'vue'
import SlamScene from '@/components/three/SlamScene.vue'
import { useTelemetryStore } from '@/stores/telemetry'
import KineticTitle from '@/components/premium/KineticTitle.vue'
import HolographicHUD from '@/components/premium/HolographicHUD.vue'
import BorderBeam from '@/components/premium/BorderBeam.vue'
import MagneticBtn from '@/components/premium/MagneticBtn.vue'

const telemetry = useTelemetryStore()
const mode = ref<'orbit-auto' | 'free'>('free')
const holoMode = ref(false)
const cinematic = ref(true)
const photoMode = ref(false)
const activePreset = ref<string>('ISO')
const sceneRef = ref<InstanceType<typeof SlamScene> | null>(null)

const PRESETS = ['TOP', 'FRONT', 'ISO', 'CHASE', 'ORBIT'] as const
type Preset = typeof PRESETS[number]
function pickPreset(p: Preset) {
  activePreset.value = p
  ;(sceneRef.value as unknown as { applyPreset?: (n: Preset) => void } | null)?.applyPreset?.(p)
}

function onSceneKey(e: KeyboardEvent) {
  if (e.key === 'p' || e.key === 'P') { photoMode.value = !photoMode.value; e.preventDefault() }
}
onMounted(() => window.addEventListener('keydown', onSceneKey))
onBeforeUnmount(() => window.removeEventListener('keydown', onSceneKey))

async function takeScreenshot() {
  const canvas = document.querySelector('.immersive-stage canvas') as HTMLCanvasElement | null
  if (!canvas) return
  await new Promise((r) => requestAnimationFrame(r))
  const url = canvas.toDataURL('image/png')
  const a = document.createElement('a')
  a.href = url
  a.download = `navcockpit-${new Date().toISOString().replace(/[:.]/g, '-')}.png`
  a.click()
}
</script>

<template>
  <section class="page">
    <header class="page-header">
      <div>
        <h1 class="page-title">
          <KineticTitle text="Immersive · 沉浸大图" gradient="aurora" />
        </h1>
        <p class="page-subtitle">
          全屏 3D SLAM · HDR PMREM + bloom + 1.2K 粒子 + 末端拖尾 · 8s 闲置自动 dolly 巡航
        </p>
      </div>
      <div class="header-actions">
        <button class="btn" :class="{ 'btn-primary': cinematic }" @click="cinematic = !cinematic">
          {{ cinematic ? '✦' : '◌' }} Cinematic
        </button>
        <button class="btn" :class="{ 'btn-primary': holoMode }" @click="holoMode = !holoMode">
          {{ holoMode ? '◉' : '◎' }} Hologram
        </button>
        <button class="btn" :class="{ 'btn-primary': mode === 'free' }" @click="mode = 'free'">🖱 Manual</button>
        <MagneticBtn :strength="0.30" :radius="100">
          <button class="btn" :class="{ 'btn-primary': mode === 'orbit-auto' }" @click="mode = 'orbit-auto'">↻ Auto-Orbit</button>
        </MagneticBtn>
        <button class="btn" :class="{ 'btn-primary': photoMode }" @click="photoMode = !photoMode" title="Photo mode (P)">
          📷 Photo
        </button>
      </div>
    </header>

    <div class="immersive-stage card-floating" :class="{ photo: photoMode }">
      <BorderBeam :duration="14" :size="280" :radius="22" :colorFrom="'rgba(34, 211, 238, 0.85)'" :colorTo="'rgba(124, 58, 237, 0.85)'" />
      <HolographicHUD v-if="!photoMode" :tone="telemetry.isConnected ? 'ok' : 'warn'" :crosshair="true" :scanline="true" :cornerSize="28" :inset="14">
        <SlamScene ref="sceneRef" :embed="mode === 'orbit-auto'" :mode="mode" :cinematic="cinematic" :holoMode="holoMode" :hdr="true" :cinemaPlus="cinematic" :photoMode="photoMode" :key="`${mode}-${holoMode}-${cinematic}`">
          <template #hud>
            <div class="hud-corner hud-tl">
              <div class="hud-line section-label">SLAM Stage · {{ mode }}{{ holoMode ? ' · holo' : '' }}{{ cinematic ? ' · cinematic' : '' }}</div>
              <div class="hud-line mono">three.js r168 · webgl2 · bloom · HDR PMREM</div>
            </div>
            <div class="hud-corner hud-tr" v-if="telemetry.packet">
              <div class="hud-line section-label">Robot Pose</div>
              <div class="hud-line mono">
                x {{ telemetry.packet.pose.x.toFixed(3) }} m ·
                y {{ telemetry.packet.pose.y.toFixed(3) }} m ·
                θ {{ ((telemetry.packet.pose.yaw * 180) / Math.PI).toFixed(1) }}°
              </div>
              <div class="hud-line mono">
                v {{ telemetry.packet.velocity.linear.toFixed(2) }} m/s ·
                ω {{ telemetry.packet.velocity.angular.toFixed(2) }} rad/s
              </div>
            </div>
            <div class="hud-corner hud-bl">
              <div class="hud-line section-label">Sensors</div>
              <div class="hud-line mono">LD14 270° · Astra 30Hz · Lift 1280×720 · IMU 200Hz</div>
            </div>
            <div class="hud-corner hud-br" v-if="telemetry.packet">
              <div class="hud-line section-label">Stream</div>
              <div class="hud-line mono">
                {{ telemetry.observedHz.toFixed(1) }} Hz · seq {{ telemetry.packet.heartbeat.sequence }} ·
                uptime {{ telemetry.packet.heartbeat.uptime_s.toFixed(0) }}s
              </div>
            </div>
          </template>
        </SlamScene>
      </HolographicHUD>

      <!-- photo mode: no HUD wrapper, raw scene with watermark -->
      <SlamScene
        v-else
        ref="sceneRef"
        :embed="false"
        :mode="'free'"
        :cinematic="true"
        :holoMode="holoMode"
        :hdr="true"
        :cinemaPlus="true"
        :photoMode="true"
        :key="`photo-${holoMode}`"
      >
        <template #hud>
          <span></span>
        </template>
      </SlamScene>

      <!-- preset switcher overlay (bottom-left) -->
      <div class="preset-bar" v-if="!photoMode">
        <button
          v-for="(p, i) in PRESETS"
          :key="p"
          class="preset-btn mono"
          :class="{ active: activePreset === p }"
          @click="pickPreset(p)"
          :title="`Camera preset · key ${i + 1}`"
        >
          <span class="preset-key">{{ i + 1 }}</span>
          <span>{{ p }}</span>
        </button>
      </div>

      <!-- photo mode toolbar -->
      <div v-if="photoMode" class="photo-bar">
        <span class="photo-watermark mono">NavCockpit · {{ new Date().toLocaleString() }}</span>
        <div class="photo-actions">
          <button class="btn btn-primary" @click="takeScreenshot">📷 Capture</button>
          <button class="btn" @click="photoMode = false">✕ Exit Photo</button>
        </div>
      </div>
    </div>
  </section>
</template>

<style scoped>
.page {
  max-width: 1680px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  gap: 18px;
  animation: fadeUp 0.42s var(--ease-out-quint) both;
  height: 100%;
}
.page-header { display: flex; justify-content: space-between; align-items: flex-end; }
.header-actions { display: flex; gap: 8px; }

.immersive-stage {
  flex: 1;
  min-height: 0;
  position: relative;
  overflow: hidden;
  border-radius: 22px;
  padding: 0;
}
.immersive-stage.photo { background: #000; }

.preset-bar {
  position: absolute;
  bottom: 16px;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  gap: 6px;
  padding: 6px;
  background: rgba(255, 255, 255, 0.60);
  backdrop-filter: blur(20px) saturate(160%);
  -webkit-backdrop-filter: blur(20px) saturate(160%);
  border: 1px solid var(--line-divider);
  border-radius: 12px;
  z-index: 8;
}
[data-theme='dark'] .preset-bar { background: rgba(20, 25, 38, 0.65); }
.preset-btn {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 6px 10px;
  font-size: 0.66rem;
  font-weight: 600;
  letter-spacing: 0.08em;
  color: var(--ink-tertiary);
  background: transparent;
  border: 1px solid transparent;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.18s var(--ease-out-quint);
}
.preset-btn:hover { color: var(--ink-primary); background: rgba(241, 245, 249, 0.6); }
[data-theme='dark'] .preset-btn:hover { background: rgba(255, 255, 255, 0.05); }
.preset-btn.active {
  color: var(--accent-blue);
  background: linear-gradient(135deg, rgba(37,99,235,0.10), rgba(8,145,178,0.06));
  border-color: rgba(37,99,235,0.20);
}
.preset-key {
  display: inline-flex; align-items: center; justify-content: center;
  width: 16px; height: 16px;
  background: var(--bg-elevated);
  border-radius: 3px;
  font-size: 0.6rem;
  font-weight: 700;
  color: var(--ink-muted);
}
.preset-btn.active .preset-key { background: var(--accent-blue); color: white; }

.photo-bar {
  position: absolute;
  left: 0; right: 0; bottom: 0;
  padding: 16px 22px;
  display: flex; justify-content: space-between; align-items: center;
  background: linear-gradient(to top, rgba(0,0,0,0.55), transparent);
  z-index: 10;
  color: white;
}
.photo-watermark { font-size: 0.72rem; color: rgba(255,255,255,0.55); letter-spacing: 0.06em; }
.photo-actions { display: flex; gap: 8px; }

.hud-corner {
  position: absolute;
  padding: 10px 14px;
  background: rgba(255, 255, 255, 0.62);
  backdrop-filter: blur(18px) saturate(170%);
  -webkit-backdrop-filter: blur(18px) saturate(170%);
  border: 1px solid var(--line-divider);
  border-radius: 12px;
  display: flex;
  flex-direction: column;
  gap: 3px;
  box-shadow: var(--shadow-soft);
  z-index: 5;
}
.hud-tl { top: 16px; left: 16px; }
.hud-tr { top: 16px; right: 16px; align-items: flex-end; }
.hud-bl { bottom: 16px; left: 16px; }
.hud-br { bottom: 16px; right: 16px; align-items: flex-end; }
.hud-line { font-size: 0.72rem; color: var(--ink-secondary); white-space: nowrap; }
.hud-line.section-label { color: var(--ink-muted); font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; }
</style>
