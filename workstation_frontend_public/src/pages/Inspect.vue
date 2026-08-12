<script setup lang="ts">
// Inspect — single-arm deep inspector. The arm in the middle is exactly the
// same procedural model as the cockpit, but at large scale with slow auto
// orbit. Six joint cards on the right show target / actual / delta / 30s
// sparkline / estimated torque %. Bottom strip shows the 6D TCP pose.
import { computed, onBeforeUnmount, onMounted, reactive, ref } from 'vue'
import { useTelemetryStore } from '@/stores/telemetry'
import type { ArmId } from '@/types/telemetry'
import type { HistorySample } from '@/stores/telemetry'
import DualArmScene from '@/components/three/DualArmScene.vue'
import GlowCard from '@/components/fx/GlowCard.vue'
import Odometer from '@/components/fx/Odometer.vue'
import Sparkline from '@/components/charts/Sparkline.vue'
import BorderBeam from '@/components/fx/BorderBeam.vue'
import { fkPose } from '@/components/three/kinematics'

const telemetry = useTelemetryStore()
const arm = ref<ArmId>('arm01')

const JOINT_NAMES = ['J1 底座', 'J2 大臂', 'J3 小臂', 'J4 腕1', 'J5 腕2', 'J6 腕3']
const LIMITS: [number, number][] = [[-168, 168], [-135, 135], [-150, 150], [-145, 145], [-165, 165], [-180, 180]]

interface RingBuf { samples: HistorySample[] }
const HIST = 300   // 30s at 10Hz
const histories = reactive<Record<number, RingBuf>>({
  0: { samples: [] }, 1: { samples: [] }, 2: { samples: [] },
  3: { samples: [] }, 4: { samples: [] }, 5: { samples: [] },
})
const encoderCounts = reactive<Record<number, number>>({ 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0 })
let lastAngles: number[] | null = null
let pollTimer: number | null = null

function tick() {
  const a = telemetry.jointsOf(arm.value).angles
  const t = Date.now()
  for (let i = 0; i < 6; i++) {
    const buf = histories[i].samples
    buf.push({ t, v: a[i] ?? 0 })
    if (buf.length > HIST) buf.splice(0, buf.length - HIST)
    if (lastAngles) {
      const d = Math.abs((a[i] ?? 0) - (lastAngles[i] ?? 0))
      // 1 deg = ~1 encoder tick (placeholder until real hardware reports)
      encoderCounts[i] += Math.round(d * 4)
    }
  }
  lastAngles = a.slice()
}
onMounted(() => { tick(); pollTimer = window.setInterval(tick, 100) })
onBeforeUnmount(() => { if (pollTimer !== null) window.clearInterval(pollTimer) })

// joint cards data
const targets = ref<number[]>([0,0,0,0,0,0])
const cards = computed(() => {
  const actuals = telemetry.jointsOf(arm.value).angles
  return JOINT_NAMES.map((name, i) => {
    const actual = actuals[i] ?? 0
    const target = targets.value[i] ?? actual
    const diff = target - actual
    // estimated torque: |diff| × speed surrogate (10% per degree, capped at 100%)
    const torque = Math.min(100, Math.abs(diff) * 6)
    return {
      i, name, actual, target, diff, torque, limits: LIMITS[i],
      samples: histories[i].samples,
      encoder: encoderCounts[i],
    }
  })
})

// when arm switches, reset target seed & clear hists
function switchArm(a: ArmId) {
  if (arm.value === a) return
  arm.value = a
  targets.value = telemetry.jointsOf(a).angles.slice()
  for (let i = 0; i < 6; i++) { histories[i].samples = []; encoderCounts[i] = 0 }
  lastAngles = null
}
function syncTargetsToActual() {
  targets.value = telemetry.jointsOf(arm.value).angles.slice()
}

// 6D pose from FK
const pose = computed(() => {
  const a = telemetry.jointsOf(arm.value).angles
  const { position, eulerXYZ } = fkPose(a)
  return {
    x: position.x, y: position.y, z: position.z,
    rx: (eulerXYZ.x * 180) / Math.PI,
    ry: (eulerXYZ.y * 180) / Math.PI,
    rz: (eulerXYZ.z * 180) / Math.PI,
  }
})

const accentColor = computed(() => (arm.value === 'arm01' ? '#d97706' : '#2563eb'))
const accent = computed<'amber' | 'blue'>(() => (arm.value === 'arm01' ? 'amber' : 'blue'))
const sparkAccent = computed<'amber' | 'blue'>(() => (arm.value === 'arm01' ? 'amber' : 'blue'))
</script>

<template>
  <div class="inspect-pro">
    <!-- header -->
    <section class="in-hero card-elevated">
      <BorderBeam :duration="14" />
      <div class="in-hero-row">
        <div class="in-arm-toggle">
          <button class="in-tab" :class="{ active: arm === 'arm01' }" @click="switchArm('arm01')">
            <span class="dot dot-amber"></span>arm01
            <span class="mono in-tab-sub">@192.0.2.64</span>
          </button>
          <button class="in-tab" :class="{ active: arm === 'arm02' }" @click="switchArm('arm02')">
            <span class="dot dot-blue"></span>arm02
            <span class="mono in-tab-sub">@192.0.2.136</span>
          </button>
        </div>
        <div class="in-stat">
          <div class="metric-label">链路</div>
          <div class="metric-hero">
            <span :style="{ color: telemetry.jointsOf(arm).online ? '#10b981' : '#94a3b8' }">
              {{ telemetry.jointsOf(arm).online ? 'live' : 'idle' }}
            </span>
          </div>
          <div class="metric-unit">pymycobot · 1Mbps</div>
        </div>
        <div class="in-stat">
          <div class="metric-label">温度</div>
          <div class="metric-hero">
            <Odometer :value="(arm === 'arm01' ? telemetry.status?.arm01.temp_c : telemetry.status?.arm02.temp_c) ?? 0" :precision="1" />
            <span class="frac">°C</span>
          </div>
          <div class="metric-unit">basis</div>
        </div>
        <div class="in-stat">
          <div class="metric-label">同步频率</div>
          <div class="metric-hero">
            <Odometer :value="telemetry.observedHz" :precision="1" />
            <span class="frac">Hz</span>
          </div>
          <div class="metric-unit">observed</div>
        </div>
        <button class="in-sync-btn" @click="syncTargetsToActual" :style="{ borderColor: accentColor, color: accentColor }">
          ⟲ 同步 target 到实际
        </button>
      </div>
    </section>

    <!-- main: arm scene + joint cards -->
    <div class="in-main">
      <GlowCard :accent="accent" class="in-scene-card">
        <div class="ck-card-head">
          <div>
            <div class="ck-card-title">{{ arm }} · 巨幅检视</div>
            <div class="ck-card-sub">PRO+ 模型 · 工作站世界 · 双击对面臂可聚焦</div>
          </div>
          <span class="chip chip-info kv-mono">myCobot 280-Pi</span>
        </div>
        <div class="in-scene-mount">
          <DualArmScene :focus="arm" />
        </div>
      </GlowCard>

      <div class="in-cards">
        <div v-for="c in cards" :key="c.i" class="jt-card">
          <div class="jt-head">
            <span class="jt-name">{{ c.name }}</span>
            <span class="jt-range mono">{{ c.limits[0] }} ~ {{ c.limits[1] }}°</span>
          </div>
          <div class="jt-row">
            <div class="jt-col">
              <div class="jt-label">target</div>
              <div class="jt-val" :style="{ color: accentColor }"><Odometer :value="c.target" :precision="1" />°</div>
            </div>
            <div class="jt-col">
              <div class="jt-label">actual</div>
              <div class="jt-val"><Odometer :value="c.actual" :precision="1" />°</div>
            </div>
            <div class="jt-col">
              <div class="jt-label">Δ</div>
              <div class="jt-val" :class="{ warn: Math.abs(c.diff) > 5, err: Math.abs(c.diff) > 30 }">
                {{ c.diff >= 0 ? '+' : '' }}{{ c.diff.toFixed(1) }}°
              </div>
            </div>
          </div>
          <div class="jt-spark">
            <Sparkline :samples="c.samples" :accent="sparkAccent" :y-range="c.limits" />
          </div>
          <div class="jt-foot">
            <div class="torque-bar">
              <div class="tb-track">
                <div class="tb-fill"
                     :style="{
                       width: c.torque + '%',
                       background: c.torque > 70 ? 'linear-gradient(90deg,#d97706,#e11d48)' : c.torque > 35 ? 'linear-gradient(90deg,#10b981,#d97706)' : 'linear-gradient(90deg,#0891b2,#10b981)',
                     }"></div>
              </div>
              <div class="tb-label mono">torque {{ c.torque.toFixed(0) }}% <span class="placeholder">(est.)</span></div>
            </div>
            <div class="jt-meta">
              <div class="jt-meta-row">
                <span class="jt-meta-label">温度</span>
                <span class="jt-meta-val placeholder">—</span>
              </div>
              <div class="jt-meta-row">
                <span class="jt-meta-label">编码器</span>
                <span class="jt-meta-val mono"><Odometer :value="c.encoder" /></span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- bottom: TCP pose -->
    <GlowCard accent="violet" class="in-pose-card">
      <div class="ck-card-head">
        <div>
          <div class="ck-card-title">末端 TCP 6D Pose</div>
          <div class="ck-card-sub">FK 计算 · position (m) · intrinsic XYZ Euler (deg)</div>
        </div>
      </div>
      <div class="pose-grid">
        <div class="pose-cell">
          <div class="pose-key">X</div>
          <div class="pose-val mono"><Odometer :value="pose.x" :precision="3" /></div>
          <div class="pose-unit">m</div>
        </div>
        <div class="pose-cell">
          <div class="pose-key">Y</div>
          <div class="pose-val mono"><Odometer :value="pose.y" :precision="3" /></div>
          <div class="pose-unit">m</div>
        </div>
        <div class="pose-cell">
          <div class="pose-key">Z</div>
          <div class="pose-val mono"><Odometer :value="pose.z" :precision="3" /></div>
          <div class="pose-unit">m</div>
        </div>
        <div class="pose-cell">
          <div class="pose-key">RX</div>
          <div class="pose-val mono"><Odometer :value="pose.rx" :precision="1" /></div>
          <div class="pose-unit">°</div>
        </div>
        <div class="pose-cell">
          <div class="pose-key">RY</div>
          <div class="pose-val mono"><Odometer :value="pose.ry" :precision="1" /></div>
          <div class="pose-unit">°</div>
        </div>
        <div class="pose-cell">
          <div class="pose-key">RZ</div>
          <div class="pose-val mono"><Odometer :value="pose.rz" :precision="1" /></div>
          <div class="pose-unit">°</div>
        </div>
      </div>
    </GlowCard>
  </div>
</template>

<style scoped>
.inspect-pro { display: flex; flex-direction: column; gap: 14px; }

.in-hero { position: relative; padding: 14px 22px; border-radius: 18px; overflow: hidden; }
.in-hero-row { display: flex; align-items: center; gap: 22px; flex-wrap: wrap; }

.in-arm-toggle { display: flex; gap: 4px; padding: 4px; border-radius: 12px; background: var(--bg-elevated); border: 1px solid var(--line-border); }
.in-tab { display: flex; align-items: center; gap: 8px; padding: 8px 14px; border-radius: 10px;
  background: transparent; border: none; color: var(--ink-secondary); cursor: pointer;
  font-size: 0.86rem; font-weight: 700; font-family: inherit; transition: all .18s; }
.in-tab.active { background: white; color: var(--ink-primary); box-shadow: var(--shadow-soft); }
.in-tab .dot-amber { background: var(--accent-amber); box-shadow: 0 0 0 3px rgba(217,119,6,.18); }
.in-tab .dot-blue { background: var(--accent-blue); box-shadow: 0 0 0 3px rgba(37,99,235,.18); }
.in-tab-sub { font-size: 0.62rem; color: var(--ink-tertiary); margin-left: 2px; }
[data-theme='dark'] .in-tab.active { background: rgba(255,255,255,.08); }

.in-stat { display: flex; flex-direction: column; gap: 4px; min-width: 110px; }
.in-stat .metric-hero { display: flex; align-items: baseline; }
.frac { font-family: 'JetBrains Mono Variable', monospace; font-size: 0.86rem; color: var(--ink-tertiary); margin-left: 4px; font-weight: 500; }

.in-sync-btn { margin-left: auto; padding: 8px 16px; border-radius: 10px;
  background: rgba(255,255,255,.5); border: 1px solid; cursor: pointer; font-weight: 700;
  font-size: 0.78rem; font-family: inherit; transition: all .15s; }
.in-sync-btn:hover { transform: translateY(-1px); box-shadow: var(--shadow-soft); }

.in-main { display: grid; gap: 14px; grid-template-columns: minmax(0, 1fr) 480px; min-height: 480px; }
@media (max-width: 1280px) { .in-main { grid-template-columns: 1fr; } }

.in-scene-card { padding: 14px; display: flex; flex-direction: column; }
.in-scene-mount { flex: 1; min-height: 460px; border-radius: 12px; overflow: hidden; }

.in-cards { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.jt-card { padding: 10px 12px; border-radius: 14px;
  background: color-mix(in srgb, var(--bg-card) 86%, transparent);
  border: 1px solid var(--line-divider);
  display: flex; flex-direction: column; gap: 6px; }
.jt-head { display: flex; justify-content: space-between; align-items: center; }
.jt-name { font-weight: 700; font-size: 0.84rem; color: var(--ink-primary); }
.jt-range { font-size: 0.6rem; color: var(--ink-muted); }
.jt-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; }
.jt-col { display: flex; flex-direction: column; gap: 1px; }
.jt-label { font-size: 0.6rem; color: var(--ink-muted); text-transform: uppercase; letter-spacing: 0.1em; font-weight: 700; }
.jt-val { font-size: 0.94rem; font-weight: 700; font-family: 'JetBrains Mono Variable', monospace; color: var(--ink-primary); display: flex; align-items: baseline; }
.jt-val.warn { color: var(--accent-amber); }
.jt-val.err { color: var(--accent-rose); }
.jt-spark { height: 38px; }

.jt-foot { display: grid; grid-template-columns: minmax(0, 1.4fr) minmax(0, 1fr); gap: 8px; align-items: center; }
.torque-bar { display: flex; flex-direction: column; gap: 2px; }
.tb-track { height: 6px; background: var(--line-divider); border-radius: 999px; overflow: hidden; }
.tb-fill { height: 100%; border-radius: 999px; transition: width .25s; }
.tb-label { font-size: 0.62rem; color: var(--ink-tertiary); }
.placeholder { color: var(--ink-muted); font-style: italic; }

.jt-meta { display: flex; flex-direction: column; gap: 2px; }
.jt-meta-row { display: flex; justify-content: space-between; font-size: 0.62rem; }
.jt-meta-label { color: var(--ink-muted); }
.jt-meta-val { color: var(--ink-secondary); font-weight: 600; }

.in-pose-card { padding: 12px 16px; }
.pose-grid { display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px; margin-top: 6px; }
.pose-cell { padding: 10px; border-radius: 12px; background: color-mix(in srgb, var(--bg-elevated) 70%, transparent);
  display: flex; flex-direction: column; align-items: center; gap: 2px; border: 1px solid var(--line-divider); }
.pose-key { font-size: 0.62rem; color: var(--ink-muted); font-weight: 700; letter-spacing: 0.16em; }
.pose-val { font-size: 1.2rem; font-weight: 800; color: var(--ink-primary); }
.pose-unit { font-size: 0.6rem; color: var(--ink-tertiary); }

@media (max-width: 1000px) { .in-cards { grid-template-columns: 1fr; } .pose-grid { grid-template-columns: repeat(3, 1fr); } }
</style>
