<script setup lang="ts">
// Calibration — hand-eye workshop. The operator:
//  1. parks an AprilTag in front of the camera (we mock the detection),
//  2. moves the arm tip to touch the tag and "Capture pair",
//  3. repeats ≥5 times,
//  4. hits "Solve" → produces a 4×4 hand-eye transform (mocked Tsai-Lenz-style),
//  5. exports / saves the matrix.
//
// No backend CV here; the page is fully client-side. Production would feed
// detection pose + arm pose pairs to an OpenCV cv::calibrateHandEye() service.
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { useTelemetryStore } from '@/stores/telemetry'
import type { ArmId } from '@/types/telemetry'
import GlowCard from '@/components/fx/GlowCard.vue'
import Odometer from '@/components/fx/Odometer.vue'
import BorderBeam from '@/components/fx/BorderBeam.vue'
import MagneticBtn from '@/components/fx/MagneticBtn.vue'
import { fkPose } from '@/components/three/kinematics'

const telemetry = useTelemetryStore()
const arm = ref<ArmId>('arm01')

// mock detected tag pose — slowly drifts so the operator sees something happen
const tagPose = ref({ x: 0.12, y: 0.18, z: 0.32, rx: 0, ry: 30, rz: 0, id: 0 })
const detected = ref(true)
const distM = ref(0.34)
let tickHandle: number | null = null
onMounted(() => {
  let t = 0
  tickHandle = window.setInterval(() => {
    t += 0.1
    tagPose.value = {
      x: 0.12 + 0.02 * Math.sin(t * 0.7),
      y: 0.18 + 0.015 * Math.sin(t * 0.5),
      z: 0.32 + 0.04 * Math.sin(t * 0.4),
      rx: 1.8 * Math.sin(t * 0.6),
      ry: 30 + 2.2 * Math.sin(t * 0.8),
      rz: 0.6 * Math.sin(t * 0.5),
      id: arm.value === 'arm01' ? 0 : 3,
    }
    distM.value = Math.sqrt(tagPose.value.x ** 2 + tagPose.value.y ** 2 + tagPose.value.z ** 2)
  }, 100)
})
onBeforeUnmount(() => {
  if (tickHandle !== null) window.clearInterval(tickHandle)
})

// SVG bounding box that follows the simulated tag pose
const tagBox = computed(() => {
  // map tagPose to a viewport rectangle for the 480×270 overlay
  const cx = 240 + tagPose.value.x * 600
  const cy = 135 + tagPose.value.y * 280
  const w = 80 - tagPose.value.z * 50
  const h = w * 0.95
  return { cx, cy, w: Math.max(36, w), h: Math.max(36, h), rot: tagPose.value.ry }
})

interface Pair {
  id: string
  tagPose: { x: number; y: number; z: number; rx: number; ry: number; rz: number }
  armTipPose: { x: number; y: number; z: number; rx: number; ry: number; rz: number }
  ts: number
}
const pairs = ref<Pair[]>([])
const armAngles = computed(() => telemetry.jointsOf(arm.value).angles)
function capture() {
  const fk = fkPose(armAngles.value)
  pairs.value = [...pairs.value, {
    id: `cap-${Date.now()}`,
    tagPose: { ...tagPose.value },
    armTipPose: {
      x: fk.position.x, y: fk.position.y, z: fk.position.z,
      rx: (fk.eulerXYZ.x * 180) / Math.PI,
      ry: (fk.eulerXYZ.y * 180) / Math.PI,
      rz: (fk.eulerXYZ.z * 180) / Math.PI,
    },
    ts: Date.now(),
  }]
}
function clearPairs() { pairs.value = [] }
function dropPair(id: string) { pairs.value = pairs.value.filter((p) => p.id !== id) }

// solved matrix — mock Tsai-Lenz: identity + small bias scaled by captured count
const solvedMatrix = ref<number[][] | null>(null)
const lastResidualMm = ref<number | null>(null)
function solve() {
  if (pairs.value.length < 4) return
  // For demo we average tag-tip translation as a fake T_hand_eye and a small
  // rotation around Y as R. Real implementation would solve AX=XB with the
  // captured pairs via OpenCV.
  let sx = 0, sy = 0, sz = 0
  for (const p of pairs.value) {
    sx += p.armTipPose.x - p.tagPose.x
    sy += p.armTipPose.y - p.tagPose.y
    sz += p.armTipPose.z - p.tagPose.z
  }
  const n = pairs.value.length
  sx /= n; sy /= n; sz /= n
  const ang = (10 * Math.PI) / 180   // mock rotation
  const c = Math.cos(ang), s = Math.sin(ang)
  solvedMatrix.value = [
    [c, 0, s, sx],
    [0, 1, 0, sy],
    [-s, 0, c, sz],
    [0, 0, 0, 1],
  ]
  // residual: spread of tip points
  let var2 = 0
  for (const p of pairs.value) {
    var2 += (p.armTipPose.x - sx) ** 2 + (p.armTipPose.y - sy) ** 2 + (p.armTipPose.z - sz) ** 2
  }
  lastResidualMm.value = Math.round(Math.sqrt(var2 / n) * 1000)
}
function exportJson() {
  if (!solvedMatrix.value) return
  const payload = {
    arm: arm.value, capturedAt: new Date().toISOString(),
    n: pairs.value.length, residual_mm: lastResidualMm.value,
    T_hand_eye: solvedMatrix.value,
  }
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url; a.download = `calib_${arm.value}_${Date.now()}.json`; a.click()
  URL.revokeObjectURL(url)
}

// saved calibrations
interface Saved {
  id: string; arm: ArmId; matrix: number[][]; n: number; residual_mm: number | null; ts: number
}
const SAVED_KEY = 'workstation.calibrations.v1'
const saved = ref<Saved[]>([])
function loadSaved() {
  try { const s = localStorage.getItem(SAVED_KEY); if (s) saved.value = JSON.parse(s) }
  catch { /* noop */ }
}
function persistSaved() { try { localStorage.setItem(SAVED_KEY, JSON.stringify(saved.value)) } catch {} }
loadSaved()
function saveCurrent() {
  if (!solvedMatrix.value) return
  saved.value = [{
    id: `s-${Date.now()}`, arm: arm.value, matrix: solvedMatrix.value,
    n: pairs.value.length, residual_mm: lastResidualMm.value, ts: Date.now(),
  }, ...saved.value].slice(0, 12)
  persistSaved()
}
function deleteSaved(id: string) { saved.value = saved.value.filter((s) => s.id !== id); persistSaved() }
function loadSavedItem(s: Saved) {
  solvedMatrix.value = s.matrix; lastResidualMm.value = s.residual_mm
}

const minPairs = 5
const ready = computed(() => pairs.value.length >= 4)
const progressPct = computed(() => Math.min(100, (pairs.value.length / minPairs) * 100))
const accentColor = computed(() => (arm.value === 'arm01' ? '#d97706' : '#2563eb'))
</script>

<template>
  <div class="cal-pro">
    <!-- header -->
    <section class="cal-hero card-elevated">
      <BorderBeam :duration="14" />
      <div class="cal-hero-row">
        <div>
          <h1 class="page-title">手眼标定 · Hand-Eye Workshop</h1>
          <p class="page-subtitle">摆 AprilTag → 移臂触碰 → Capture × ≥{{ minPairs }} → Solve → Export · 离线本地存档</p>
        </div>
        <div class="cal-arm-toggle">
          <button class="cal-tab" :class="{ active: arm === 'arm01' }" @click="arm = 'arm01'">
            <span class="dot dot-amber"></span>arm01
          </button>
          <button class="cal-tab" :class="{ active: arm === 'arm02' }" @click="arm = 'arm02'">
            <span class="dot dot-blue"></span>arm02
          </button>
        </div>
        <div class="cal-stat">
          <div class="metric-label">已采集</div>
          <div class="metric-hero"><Odometer :value="pairs.length" /> / <span class="frac">{{ minPairs }}</span></div>
          <div class="metric-unit">pairs · {{ ready ? 'ready to solve' : 'gather more' }}</div>
        </div>
        <div class="cal-prog">
          <div class="cal-prog-track">
            <div class="cal-prog-fill" :style="{ width: progressPct + '%', background: accentColor }"></div>
          </div>
          <div class="cal-prog-pcts mono">{{ progressPct.toFixed(0) }}%</div>
        </div>
      </div>
    </section>

    <!-- main: cam + capture panel -->
    <div class="cal-main">
      <GlowCard accent="violet" class="cal-cam-card">
        <div class="ck-card-head">
          <div>
            <div class="ck-card-title">{{ arm }} eye · AprilTag tag36h11</div>
            <div class="ck-card-sub">检测框跟随模拟 pose · 真后端走 cv2.aruco</div>
          </div>
          <span class="chip" :class="detected ? 'chip-ok' : 'chip-idle'">{{ detected ? `detected · id=${tagPose.id}` : 'no tag' }}</span>
        </div>
        <div class="cal-cam-mount">
          <div class="cam-stage">
            <!-- mock camera background -->
            <div class="cam-bg">
              <div class="scan-bar"></div>
              <div class="cam-grid"></div>
            </div>
            <!-- SVG detection overlay -->
            <svg class="cam-svg" viewBox="0 0 480 270" preserveAspectRatio="xMidYMid meet">
              <g :transform="`translate(${tagBox.cx}, ${tagBox.cy}) rotate(${tagBox.rot})`">
                <rect :x="-tagBox.w/2" :y="-tagBox.h/2" :width="tagBox.w" :height="tagBox.h"
                      fill="none" :stroke="accentColor" stroke-width="2.5" stroke-dasharray="4 4" rx="3" />
                <circle cx="0" cy="0" r="3" :fill="accentColor" />
                <text x="0" :y="-tagBox.h/2 - 8" text-anchor="middle"
                      font-size="11" font-family="JetBrains Mono Variable, monospace" :fill="accentColor" font-weight="700">
                  id={{ tagPose.id }} · {{ (distM*100).toFixed(1) }}cm
                </text>
                <!-- axes -->
                <line x1="0" y1="0" :x2="20" y2="0" stroke="#ef4444" stroke-width="2" />
                <line x1="0" y1="0" x2="0" :y2="-20" stroke="#10b981" stroke-width="2" />
              </g>
            </svg>
            <div class="cam-meta mono">公开相机摘要 · {{ telemetry.mode }}</div>
          </div>
        </div>
        <div class="cal-tag-pose mono">
          <span>tag pose</span>
          <span>x {{ tagPose.x.toFixed(3) }}</span>
          <span>y {{ tagPose.y.toFixed(3) }}</span>
          <span>z {{ tagPose.z.toFixed(3) }}</span>
          <span>rx {{ tagPose.rx.toFixed(1) }}°</span>
          <span>ry {{ tagPose.ry.toFixed(1) }}°</span>
          <span>rz {{ tagPose.rz.toFixed(1) }}°</span>
        </div>
      </GlowCard>

      <GlowCard accent="amber" class="cal-cap-card">
        <div class="ck-card-head">
          <div>
            <div class="ck-card-title">Capture · Solve · Export</div>
            <div class="ck-card-sub">每个 pair = (tag pose, arm tip pose) · 至少 4 对</div>
          </div>
        </div>
        <div class="cal-actions">
          <MagneticBtn @click="capture">＋ Capture pair</MagneticBtn>
          <MagneticBtn @click="clearPairs">⌫ 清空</MagneticBtn>
          <MagneticBtn class="primary" @click="solve">▶ Solve hand-eye</MagneticBtn>
        </div>
        <div class="cal-pairs">
          <div v-if="!pairs.length" class="cal-empty">还没有采集 · 移动 arm 触碰 tag 并点 Capture</div>
          <div v-for="(p, i) in pairs" :key="p.id" class="cal-pair-row">
            <span class="cp-i mono">#{{ i + 1 }}</span>
            <span class="cp-meta mono">tag x={{ p.tagPose.x.toFixed(2) }} y={{ p.tagPose.y.toFixed(2) }} z={{ p.tagPose.z.toFixed(2) }}</span>
            <span class="cp-meta mono">tip x={{ p.armTipPose.x.toFixed(2) }} y={{ p.armTipPose.y.toFixed(2) }} z={{ p.armTipPose.z.toFixed(2) }}</span>
            <button class="cp-x" @click="dropPair(p.id)">×</button>
          </div>
        </div>
      </GlowCard>
    </div>

    <!-- bottom: matrix + saved list -->
    <div class="cal-bottom">
      <GlowCard accent="blue" class="cal-matrix-card">
        <div class="ck-card-head">
          <div>
            <div class="ck-card-title">T<sub>hand_eye</sub> · 4×4</div>
            <div class="ck-card-sub">residual {{ lastResidualMm != null ? `${lastResidualMm}mm rms` : '—' }} · {{ pairs.length }} pairs</div>
          </div>
          <div class="cal-matrix-actions">
            <button class="cm-btn" :disabled="!solvedMatrix" @click="saveCurrent">＋ 保存</button>
            <button class="cm-btn" :disabled="!solvedMatrix" @click="exportJson">⤓ Export JSON</button>
          </div>
        </div>
        <div class="cal-matrix">
          <template v-if="solvedMatrix">
            <div v-for="(row, ri) in solvedMatrix" :key="ri" class="cm-row">
              <div v-for="(v, ci) in row" :key="ci" class="cm-cell mono"
                   :class="{ diag: ri === ci, last: ci === 3 }">
                {{ Number(v).toFixed(3) }}
              </div>
            </div>
          </template>
          <div v-else class="cal-empty matrix-empty">▶ Solve 后这里显示 4×4 矩阵</div>
        </div>
      </GlowCard>

      <GlowCard accent="emerald" class="cal-saved-card">
        <div class="ck-card-head">
          <div>
            <div class="ck-card-title">已保存标定 · 本地</div>
            <div class="ck-card-sub">localStorage · 最近 12 条</div>
          </div>
        </div>
        <div v-if="!saved.length" class="cal-empty">还没有保存的标定</div>
        <div v-else class="cal-saved-list">
          <div v-for="s in saved" :key="s.id" class="cal-saved-row" @click="loadSavedItem(s)">
            <span class="ss-arm dot" :class="s.arm === 'arm01' ? 'dot-amber' : 'dot-blue'"></span>
            <span class="ss-arm-name mono">{{ s.arm }}</span>
            <span class="ss-time mono">{{ new Date(s.ts).toLocaleString().slice(0, 16) }}</span>
            <span class="ss-meta mono">{{ s.n }}pr · {{ s.residual_mm ?? '—' }}mm</span>
            <button class="ss-x" @click.stop="deleteSaved(s.id)">×</button>
          </div>
        </div>
      </GlowCard>
    </div>
  </div>
</template>

<style scoped>
.cal-pro { display: flex; flex-direction: column; gap: 14px; }

.cal-hero { position: relative; padding: 14px 22px; border-radius: 18px; overflow: hidden; }
.cal-hero-row { display: flex; align-items: center; gap: 22px; flex-wrap: wrap; }

.cal-arm-toggle { display: flex; gap: 4px; padding: 4px; border-radius: 12px; background: var(--bg-elevated); border: 1px solid var(--line-border); }
.cal-tab { display: flex; align-items: center; gap: 8px; padding: 7px 14px; border-radius: 10px;
  background: transparent; border: none; color: var(--ink-secondary); cursor: pointer;
  font-size: 0.82rem; font-weight: 700; font-family: inherit; transition: all .18s; }
.cal-tab.active { background: white; color: var(--ink-primary); box-shadow: var(--shadow-soft); }
[data-theme='dark'] .cal-tab.active { background: rgba(255,255,255,.08); }
.cal-tab .dot-amber { background: var(--accent-amber); box-shadow: 0 0 0 3px rgba(217,119,6,.18); }
.cal-tab .dot-blue { background: var(--accent-blue); box-shadow: 0 0 0 3px rgba(37,99,235,.18); }

.cal-stat { display: flex; flex-direction: column; gap: 4px; min-width: 120px; }
.cal-stat .metric-hero { display: flex; align-items: baseline; }
.frac { font-family: 'JetBrains Mono Variable', monospace; font-size: 0.86rem; color: var(--ink-tertiary); margin-left: 4px; }

.cal-prog { display: flex; flex-direction: column; gap: 4px; margin-left: auto; min-width: 180px; }
.cal-prog-track { height: 8px; background: var(--line-divider); border-radius: 999px; overflow: hidden; }
.cal-prog-fill { height: 100%; border-radius: 999px; transition: width .25s; }
.cal-prog-pcts { font-size: 0.66rem; color: var(--ink-tertiary); text-align: right; }

.cal-main { display: grid; gap: 14px; grid-template-columns: 1fr 420px; }
@media (max-width: 1180px) { .cal-main { grid-template-columns: 1fr; } }

.cal-cam-card { padding: 14px; }
.cal-cam-mount { margin-bottom: 8px; border-radius: 14px; overflow: hidden; }
.cam-stage { position: relative; aspect-ratio: 16 / 9; background: #0a0e1a; }
.cam-bg {
  position: absolute; inset: 0;
  background:
    radial-gradient(circle at 32% 40%, rgba(217,119,6,0.14), transparent 32%),
    radial-gradient(circle at 70% 62%, rgba(37,99,235,0.12), transparent 36%),
    #0a0e1a;
}
.cam-grid { position: absolute; inset: 0;
  background-image: repeating-linear-gradient(0deg, rgba(255,255,255,0.03) 0 1px, transparent 1px 36px),
                    repeating-linear-gradient(90deg, rgba(255,255,255,0.03) 0 1px, transparent 1px 36px);
}
.scan-bar {
  position: absolute; left: 0; right: 0; height: 1px; top: 0;
  background: linear-gradient(90deg, transparent, rgba(125,211,252,0.6), transparent);
  box-shadow: 0 0 12px rgba(125,211,252,0.45);
  animation: scanY 4s linear infinite;
}
@keyframes scanY { 0% { top: 2%; } 100% { top: 98%; } }
.cam-svg { position: absolute; inset: 0; width: 100%; height: 100%; }
.cam-meta { position: absolute; bottom: 8px; left: 10px; font-size: 0.66rem; color: rgba(255,255,255,.55); }

.cal-tag-pose { display: flex; gap: 12px; flex-wrap: wrap; font-size: 0.7rem; padding: 8px 12px;
  background: var(--bg-elevated); border-radius: 10px; color: var(--ink-secondary); border: 1px solid var(--line-divider); }
.cal-tag-pose span:first-child { color: var(--ink-muted); text-transform: uppercase; letter-spacing: 0.1em; font-weight: 700; }

.cal-cap-card { padding: 14px 16px; display: flex; flex-direction: column; gap: 10px; }
.cal-actions { display: grid; grid-template-columns: 1fr 1fr 1.4fr; gap: 8px; }
.cal-actions :deep(.magnetic-btn) { width: 100%; font-size: 0.8rem; padding: 9px 8px; }
.cal-actions .primary :deep(.magnetic-btn) {
  background: linear-gradient(135deg, var(--accent-blue), color-mix(in srgb, var(--accent-blue) 70%, #0891b2));
  color: white; border-color: transparent;
}
.cal-pairs { max-height: 300px; overflow-y: auto; display: flex; flex-direction: column; gap: 4px; }
.cal-empty { padding: 18px 4px; color: var(--ink-muted); font-size: 0.82rem; text-align: center; }
.cal-pair-row { display: grid; grid-template-columns: 28px 1fr 1fr 22px; gap: 8px; align-items: center;
  padding: 6px 8px; border-radius: 8px; background: color-mix(in srgb, var(--bg-elevated) 80%, transparent); font-size: 0.68rem; }
.cp-i { color: var(--ink-muted); font-weight: 700; }
.cp-meta { color: var(--ink-tertiary); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.cp-x { width: 20px; height: 20px; border-radius: 50%; border: none; background: rgba(225,29,72,.1); color: var(--accent-rose);
  font-weight: 800; cursor: pointer; font-size: 0.85rem; line-height: 1; }
.cp-x:hover { background: rgba(225,29,72,.22); }

.cal-bottom { display: grid; gap: 14px; grid-template-columns: 1fr 420px; }
@media (max-width: 1180px) { .cal-bottom { grid-template-columns: 1fr; } }

.cal-matrix-card { padding: 14px 16px; }
.cal-matrix-actions { display: flex; gap: 6px; }
.cm-btn { padding: 6px 12px; border-radius: 8px; border: 1px solid var(--line-border); background: var(--bg-card);
  color: var(--ink-secondary); font-size: 0.72rem; font-weight: 700; cursor: pointer; transition: all .15s; }
.cm-btn:hover:not(:disabled) { border-color: var(--accent-blue); color: var(--accent-blue); }
.cm-btn:disabled { opacity: .4; cursor: not-allowed; }
.cal-matrix { padding: 14px 16px; border-radius: 12px; background: color-mix(in srgb, var(--bg-elevated) 80%, transparent); border: 1px dashed var(--line-divider); margin-top: 8px; }
.matrix-empty { padding: 26px 4px; }
.cm-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }
.cm-cell { padding: 8px 10px; text-align: right; font-family: 'JetBrains Mono Variable', monospace;
  font-size: 0.95rem; font-weight: 600; color: var(--ink-secondary); border-radius: 6px;
  background: color-mix(in srgb, var(--bg-card) 70%, transparent); }
.cm-cell.diag { color: var(--accent-blue); font-weight: 800; }
.cm-cell.last { color: var(--accent-amber); font-weight: 800; }

.cal-saved-card { padding: 14px 16px; }
.cal-saved-list { display: flex; flex-direction: column; gap: 4px; max-height: 280px; overflow-y: auto; }
.cal-saved-row { display: grid; grid-template-columns: 14px auto 1fr auto 22px; gap: 8px; align-items: center;
  padding: 7px 10px; border-radius: 8px; background: color-mix(in srgb, var(--bg-elevated) 78%, transparent);
  cursor: pointer; transition: all .15s; font-size: 0.72rem; border: 1px solid transparent; }
.cal-saved-row:hover { transform: translateY(-1px); border-color: var(--accent-blue); }
.ss-arm-name { font-weight: 700; color: var(--ink-primary); }
.ss-time { color: var(--ink-tertiary); font-size: 0.66rem; }
.ss-meta { color: var(--ink-muted); font-size: 0.66rem; }
.ss-x { width: 20px; height: 20px; border-radius: 50%; border: none; background: rgba(225,29,72,.1); color: var(--accent-rose);
  font-weight: 800; cursor: pointer; font-size: 0.9rem; line-height: 1; }
.ss-x:hover { background: rgba(225,29,72,.22); }
</style>
