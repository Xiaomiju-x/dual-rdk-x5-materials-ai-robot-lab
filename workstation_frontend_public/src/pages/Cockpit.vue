<script setup lang="ts">
import { computed } from 'vue'
import { useTelemetryStore, type HistorySample } from '@/stores/telemetry'
import type { Accent, Tone } from '@/types/telemetry'
import DualArmScene from '@/components/three/DualArmScene.vue'
import CameraTile from '@/components/CameraTile.vue'
import StatusCard from '@/components/StatusCard.vue'
import Odometer from '@/components/fx/Odometer.vue'
import GlowCard from '@/components/fx/GlowCard.vue'
import BorderBeam from '@/components/fx/BorderBeam.vue'
import RingChart from '@/components/charts/RingChart.vue'
import RadarStatus from '@/components/charts/RadarStatus.vue'
import EventMarquee from '@/components/EventMarquee.vue'

const telemetry = useTelemetryStore()
const online = (b: boolean): Tone => (b ? 'ok' : 'idle')

// hero metrics
const heroJointsHz = computed(() => telemetry.observedHz)
const arm01On = computed(() => telemetry.arm01.online ?? false)
const arm02On = computed(() => telemetry.arm02.online ?? false)
const armsOnline = computed(() => (arm01On.value ? 1 : 0) + (arm02On.value ? 1 : 0))
const camFps = computed(() => ((telemetry.status?.cam01_fps ?? 0) + (telemetry.status?.cam02_fps ?? 0)) / 2)
const aiMs = computed(() => telemetry.status?.ai_brain_ms ?? 0)
const carMs = computed(() => telemetry.status?.car_brain_ms ?? 0)
const linkScore = computed(() => {
  const arms = armsOnline.value / 2
  const cam = Math.min(1, camFps.value / 30)
  const brains = (telemetry.aiOnline ? 0.5 : 0) + (telemetry.carOnline ? 0.5 : 0)
  return arms * 0.6 + cam * 0.2 + brains * 0.2
})
const hzScore = computed(() => Math.min(1, telemetry.observedHz / 12))

const radarAxes = computed(() => [
  { label: 'arm01', value: arm01On.value ? 1 : 0.05 },
  { label: 'arm02', value: arm02On.value ? 1 : 0.05 },
  { label: 'cam01', value: Math.min(1, (telemetry.status?.cam01_fps ?? 0) / 30) },
  { label: 'cam02', value: Math.min(1, (telemetry.status?.cam02_fps ?? 0) / 30) },
  { label: 'AI 脑', value: telemetry.aiOnline ? Math.max(0.2, 1 - aiMs.value / 200) : 0.05 },
  { label: '车载脑', value: telemetry.carOnline ? Math.max(0.2, 1 - carMs.value / 200) : 0.05 },
])

interface CardData {
  label: string; icon: string; accent: Accent; value: string; unit: string; tone: Tone; samples: HistorySample[]
}
const cards = computed<CardData[]>(() => [
  { label: 'arm01 温度', icon: '🌡', accent: 'amber',
    value: telemetry.status?.arm01.temp_c != null ? telemetry.status.arm01.temp_c.toFixed(1) : '—',
    unit: '°C', tone: online(arm01On.value), samples: telemetry.buffer('arm01_temp') },
  { label: 'arm02 温度', icon: '🌡', accent: 'blue',
    value: telemetry.status?.arm02.temp_c != null ? telemetry.status.arm02.temp_c.toFixed(1) : '—',
    unit: '°C', tone: online(arm02On.value), samples: telemetry.buffer('arm02_temp') },
  { label: 'cam01 帧率', icon: '🎥', accent: 'teal',
    value: (telemetry.status?.cam01_fps ?? 0).toFixed(1), unit: 'fps',
    tone: 'ok', samples: telemetry.buffer('cam01_fps') },
  { label: 'cam02 帧率', icon: '🎥', accent: 'teal',
    value: (telemetry.status?.cam02_fps ?? 0).toFixed(1), unit: 'fps',
    tone: 'ok', samples: telemetry.buffer('cam02_fps') },
  { label: 'AI 脑链路', icon: '🧠', accent: 'violet',
    value: telemetry.status?.ai_brain_ms != null ? telemetry.status.ai_brain_ms.toFixed(0) : '—',
    unit: 'ms', tone: online(telemetry.aiOnline), samples: telemetry.buffer('ai_ms') },
  { label: '车载脑链路', icon: '🚗', accent: 'amber',
    value: telemetry.status?.car_brain_ms != null ? telemetry.status.car_brain_ms.toFixed(0) : '—',
    unit: 'ms', tone: online(telemetry.carOnline), samples: telemetry.buffer('car_ms') },
])
</script>

<template>
  <div class="cockpit-pro">
    <!-- HERO mission-control band -->
    <section class="ck-hero card-elevated">
      <BorderBeam :duration="14" />
      <div class="ck-hero-row">
        <div class="ck-hero-stat">
          <div class="metric-label">同步频率</div>
          <div class="metric-hero">
            <Odometer :value="heroJointsHz" :precision="1" />
          </div>
          <div class="metric-unit">Hz · observed</div>
        </div>
        <div class="ck-hero-stat">
          <div class="metric-label">机械臂在线</div>
          <div class="metric-hero">
            <Odometer :value="armsOnline" /><span class="hero-frac">/ 2</span>
          </div>
          <div class="metric-unit">{{ arm01On && arm02On ? 'dual-arm ready' : armsOnline === 0 ? 'offline' : 'partial' }}</div>
        </div>
        <div class="ck-hero-stat">
          <div class="metric-label">平均帧率</div>
          <div class="metric-hero">
            <Odometer :value="camFps" :precision="1" />
          </div>
          <div class="metric-unit">fps · dual cam</div>
        </div>
        <div class="ck-hero-stat">
          <div class="metric-label">推理 RTT</div>
          <div class="metric-hero">
            <Odometer :value="aiMs" :precision="0" />
            <span class="hero-frac">ms</span>
          </div>
          <div class="metric-unit">AI 脑 · {{ telemetry.aiOnline ? 'live' : 'idle' }}</div>
        </div>
        <div class="ck-hero-ring">
          <RingChart :value="linkScore" :inner="hzScore" accent="blue" inner-accent="teal"
                     :label="(linkScore*100).toFixed(0) + '%'" caption="link score" :size="128" />
        </div>
      </div>
      <div class="ck-hero-row hero-strip">
        <span class="dot dot-ok dot-live"></span>
        <span class="kv-mono">arm01 @ 192.0.2.64 · arm02 @ 192.0.2.136 · K70 DHCP</span>
        <span class="hairline-v"></span>
        <span class="chip" :class="telemetry.mode === 'real' ? 'chip-ok' : 'chip-info'">mode · {{ telemetry.mode }}</span>
        <span class="chip chip-info">SPA · WorkCockpit v2</span>
        <span class="chip chip-info">zero-CDN · local-first</span>
      </div>
    </section>

    <!-- main grid: scene + radar + cams -->
    <div class="ck-main">
      <GlowCard accent="blue" class="ck-scene-card">
        <div class="ck-card-head">
          <div>
            <div class="ck-card-title">实时双臂姿态</div>
            <div class="ck-card-sub">Three.js · 6-DoF · HDR · bloom · 接触阴影</div>
          </div>
          <div class="ck-card-tags">
            <span class="chip" :class="arm01On ? 'chip-ok' : 'chip-idle'">arm01</span>
            <span class="chip" :class="arm02On ? 'chip-ok' : 'chip-idle'">arm02</span>
          </div>
        </div>
        <div class="ck-scene-mount">
          <DualArmScene />
        </div>
      </GlowCard>

      <div class="ck-side">
        <GlowCard accent="teal" class="ck-radar-card">
          <div class="ck-card-head">
            <div>
              <div class="ck-card-title">健康雷达</div>
              <div class="ck-card-sub">6 维实时遥测</div>
            </div>
            <span class="chip chip-info kv-mono">6 axes</span>
          </div>
          <div class="ck-radar-mount">
            <RadarStatus :axes="radarAxes" accent="blue" :size="230" />
          </div>
        </GlowCard>

        <GlowCard accent="violet" class="ck-cam-card">
          <CameraTile arm="arm01" label="arm01 cam" tag-label="id=0 · d=0.142m" />
        </GlowCard>
        <GlowCard accent="amber" class="ck-cam-card">
          <CameraTile arm="arm02" label="arm02 cam" tag-label="id=3 · d=0.118m" />
        </GlowCard>
      </div>
    </div>

    <!-- bottom strip: cards + marquee -->
    <section class="ck-cards">
      <StatusCard v-for="c in cards" :key="c.label" v-bind="c" />
    </section>
    <EventMarquee />
  </div>
</template>

<style scoped>
.cockpit-pro {
  display: flex; flex-direction: column; gap: 14px;
  height: 100%; min-height: 0;
}

/* HERO */
.ck-hero {
  position: relative;
  padding: 18px 22px 14px;
  border-radius: 18px;
  overflow: hidden;
}
.ck-hero-row {
  display: flex; align-items: center; gap: 28px;
  flex-wrap: wrap;
}
.ck-hero-stat { display: flex; flex-direction: column; gap: 4px; min-width: 132px; }
.ck-hero-stat .metric-hero { display: flex; align-items: baseline; }
.hero-frac { font-family: 'JetBrains Mono Variable', monospace; font-size: 0.86rem; color: var(--ink-tertiary); margin-left: 6px; font-weight: 500; letter-spacing: 0; }
.ck-hero-ring { margin-left: auto; }

.hero-strip {
  margin-top: 14px;
  padding-top: 12px;
  border-top: 1px dashed var(--line-divider);
  gap: 12px;
  font-size: 0.78rem;
}
.hairline-v { display: inline-block; width: 1px; height: 14px; background: var(--line-divider); }

/* MAIN grid */
.ck-main {
  display: grid; gap: 14px;
  grid-template-columns: 1fr 360px;
  flex: 1;
  min-height: 0;
}
.ck-scene-card { display: flex; flex-direction: column; padding: 14px; }
.ck-scene-mount { flex: 1; min-height: 360px; border-radius: 12px; overflow: hidden; }
.ck-side {
  display: grid;
  grid-template-rows: minmax(240px, 280px) 1fr 1fr;
  gap: 14px;
  min-height: 0;
}
.ck-radar-card { display: flex; flex-direction: column; padding: 14px; }
.ck-radar-mount { display: flex; flex: 1; align-items: center; justify-content: center; }
.ck-cam-card { padding: 0; min-height: 0; overflow: hidden; }

.ck-card-head {
  display: flex; align-items: flex-start; justify-content: space-between;
  margin-bottom: 8px;
}
.ck-card-title { font-weight: 700; font-size: 0.95rem; letter-spacing: -0.01em; color: var(--ink-primary); }
.ck-card-sub { font-size: 0.7rem; color: var(--ink-tertiary); margin-top: 2px; }
.ck-card-tags { display: flex; gap: 6px; }

/* bottom strip cards */
.ck-cards {
  display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px;
  flex-shrink: 0;
}

@media (max-width: 1280px) {
  .ck-main { grid-template-columns: 1fr; }
  .ck-side { grid-template-rows: none; grid-template-columns: 1fr 1fr 1fr; }
  .ck-cards { grid-template-columns: repeat(3, 1fr); }
}
@media (max-width: 760px) {
  .ck-side { grid-template-columns: 1fr; }
  .ck-cards { grid-template-columns: repeat(2, 1fr); }
  .ck-hero-row { gap: 16px; }
  .ck-hero-ring { margin-left: 0; }
}
</style>
