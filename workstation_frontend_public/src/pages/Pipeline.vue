<script setup lang="ts">
// Pipeline · v4 十幕剧本 — S1-S10 时间轴执行器 + 5 故障注入 + 防互撞互锁 + NL 任务
// (第 4 期 #2 #3 #5). 后端: /api/pipeline* /api/interlock* /api/nl_task (script_v4.py 等)
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import GlowCard from '@/components/fx/GlowCard.vue'
import MagneticBtn from '@/components/fx/MagneticBtn.vue'

// ---------------- 剧本状态 ----------------
interface Step { actor: string; action: string; params?: Record<string, unknown>; status: string; dur_s?: number }
interface Stage { id: string; name: string; status: string; steps: Step[] }
interface PipeState {
  state: string; stage_idx: number; step_idx: number; stages: Stage[]
  log: Array<{ t: number; kind: string; msg: string }>
  active_faults: string[]; mode: string; degraded: string[]
}
const pipe = ref<PipeState | null>(null)
const faults = ref<Array<{ mode: string; label: string }>>([])
let timer: number | null = null

async function poll() {
  try {
    pipe.value = await fetch('/api/pipeline').then((x) => x.json())
  } catch { /* offline */ }
}
const running = computed(() => pipe.value?.state === 'running' || pipe.value?.state === 'paused')

async function start() {
  await fetch('/api/pipeline/start', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' })
  poll()
}
async function ctl(action: string) {
  await fetch('/api/pipeline/control', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action }),
  })
  poll()
}
async function inject(mode: string) {
  await fetch('/api/pipeline/fault', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ mode }),
  })
}

const ACTOR_ICON: Record<string, string> = {
  arm01: '🦾', arm02: '🦿', car: '🚗', ai: '🧠', operator: '👤', sync: '⛓',
}

// ---------------- 快拆爪库 (v4 4 爪, 磁吸快拆) ----------------
// 数据源: workstation/docs/v4_cad_brief.md + CLAUDE.md §7 v4 硬件清单. 状态诚实标注 —
// MG996R×6 / SG90×2 / 磁铁+pogo 快拆件未到货, 爪体待 3D 打印, 现为软件层接口就绪。
interface Gripper {
  key: string; name: string; icon: string; actuator: string
  stages: string; payload: string; status: 'pending' | 'ready'
}
const GRIPPERS: Gripper[] = [
  { key: 'rod', name: '棒爪', icon: '🥢', actuator: 'MG996R 25T ×2 (开合)',
    stages: 'S5 研磨', payload: '研磨棒 (玛瑙杵)', status: 'pending' },
  { key: 'handle', name: '扶手爪', icon: '🤲', actuator: 'MG996R 25T ×2 (钩持)',
    stages: 'S5 / S9 端碗', payload: '承碗盘 D 把手 (凹槽承重 0g, 绕 atom 250g 上限)', status: 'pending' },
  { key: 'bag', name: '袋爪', icon: '👝', actuator: 'SG90 9g ×2 (省 atom 5V/500mA)',
    stages: 'S7 灌装', payload: '称量袋 (软袋夹口)', status: 'pending' },
  { key: 'bottle', name: '瓶+漏斗+相机', icon: '🍼', actuator: 'MG996R 25T ×2 + 模块',
    stages: 'S3 接瓶 / S4 倒粉', payload: '料瓶 + 小漏斗 + USB 相机模块', status: 'pending' },
]
const showGrippers = ref(true)

// ---------------- 互锁 ghost 顶视图 ----------------
interface Ghost { points: { arm01: number[][]; arm02: number[][] }; tick: { min_dist_mm: number | null; level: string } }
const ghost = ref<Ghost | null>(null)
const ilConfig = ref({ base_dx_mm: 400, base_yaw_deg: 180, enabled: true })
const cvRef = ref<HTMLCanvasElement | null>(null)
let ghostTimer: number | null = null

async function pollGhost() {
  try {
    const g = await fetch('/api/interlock/ghost').then((x) => x.json())
    ghost.value = g
    const st = await fetch('/api/interlock').then((x) => x.json())
    if (st.config) ilConfig.value = { ...ilConfig.value, ...st.config }
    drawGhost()
  } catch { /* offline */ }
}
function drawGhost() {
  const cv = cvRef.value
  const g = ghost.value
  if (!cv || !g) return
  const ctx = cv.getContext('2d')!
  ctx.clearRect(0, 0, cv.width, cv.height)
  // 顶视图: 世界 mm → 画布. 视野自适应两臂所有点
  const all = [...g.points.arm01, ...g.points.arm02]
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity
  for (const p of all) {
    minX = Math.min(minX, p[0]); maxX = Math.max(maxX, p[0])
    minY = Math.min(minY, p[1]); maxY = Math.max(maxY, p[1])
  }
  const pad = 80
  const sc = Math.min((cv.width - 30) / (maxX - minX + pad * 2), (cv.height - 30) / (maxY - minY + pad * 2))
  const px = (x: number) => 15 + (x - minX + pad) * sc
  const py = (y: number) => cv.height - 15 - (y - minY + pad) * sc
  const drawArm = (pts: number[][], color: string) => {
    ctx.beginPath()
    pts.forEach((p, i) => (i === 0 ? ctx.moveTo(px(p[0]), py(p[1])) : ctx.lineTo(px(p[0]), py(p[1]))))
    ctx.strokeStyle = color
    ctx.lineWidth = 7
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
    ctx.globalAlpha = 0.8
    ctx.stroke()
    ctx.globalAlpha = 1
    for (const p of pts) {
      ctx.beginPath(); ctx.arc(px(p[0]), py(p[1]), 4, 0, Math.PI * 2)
      ctx.fillStyle = color; ctx.fill()
    }
  }
  drawArm(g.points.arm01, '#2563eb')
  drawArm(g.points.arm02, '#7c3aed')
  // 基座标注
  ctx.font = '11px sans-serif'
  ctx.fillStyle = '#2563eb'; ctx.fillText('arm01', px(g.points.arm01[0][0]) - 16, py(g.points.arm01[0][1]) + 18)
  ctx.fillStyle = '#7c3aed'; ctx.fillText('arm02', px(g.points.arm02[0][0]) - 16, py(g.points.arm02[0][1]) + 18)
}
async function saveIlConfig() {
  await fetch('/api/interlock/config', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(ilConfig.value),
  })
}
const ilLevel = computed(() => ghost.value?.tick?.level ?? 'unknown')
const ilDist = computed(() => ghost.value?.tick?.min_dist_mm)

// ---------------- NL 任务 ----------------
const nlText = ref('')
const nlBusy = ref(false)
const nlPlan = ref<Array<{ actor: string; action: string; params?: Record<string, unknown> }> | null>(null)
const nlNote = ref('')
async function nlAsk() {
  if (!nlText.value.trim() || nlBusy.value) return
  nlBusy.value = true
  nlPlan.value = null
  nlNote.value = '🧠 arm_planner 规划中 (冷启动可能 ~21s)…'
  try {
    const r = await fetch('/api/nl_task', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: nlText.value.trim() }),
    }).then((x) => x.json())
    if (r.ok) {
      nlPlan.value = r.plan
      nlNote.value = `来源: ${r.source === 'llm' ? '🧠 ' : '📐 '}${r.note}`
    } else {
      nlNote.value = `✗ ${r.error}`
    }
  } catch (e) {
    nlNote.value = `✗ ${(e as Error).message}`
  } finally {
    nlBusy.value = false
  }
}
async function nlRun() {
  if (!nlPlan.value) return
  const r = await fetch('/api/nl_task/execute', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ plan: nlPlan.value, text: nlText.value }),
  }).then((x) => x.json())
  nlNote.value = r.ok ? '▶ 已进引擎执行 (上方时间轴)' : `✗ ${r.error}`
  poll()
}

onMounted(async () => {
  poll()
  pollGhost()
  timer = window.setInterval(poll, 1000)
  ghostTimer = window.setInterval(pollGhost, 1500)
  try {
    const f = await fetch('/api/pipeline/faults').then((x) => x.json())
    faults.value = f.faults ?? []
  } catch { /* noop */ }
  const cv = cvRef.value
  if (cv) { cv.width = cv.clientWidth || 420; cv.height = 240 }
})
onUnmounted(() => {
  if (timer !== null) window.clearInterval(timer)
  if (ghostTimer !== null) window.clearInterval(ghostTimer)
})
watch(ghost, drawGhost)
</script>

<template>
  <section class="page">
    <header class="head">
      <div>
        <h1 class="page-title">🎬 Pipeline · v4 十幕剧本</h1>
        <p class="sub">
          S1 配方下发 → S10 XRD/PL 回填 端到端时间轴 · 同步栅栏 · 5 故障注入 ·
          双臂 capsule 防互撞互锁 · 自然语言 → 任务序列 (arm_planner :9103)
          <span class="chip" :class="pipe?.mode === 'real' ? 'chip-ok' : 'chip-info'" style="margin-left:6px;">
            {{ pipe?.mode === 'real' ? '真机' : 'SIM 仿真' }}
          </span>
        </p>
      </div>
      <div class="head-ctl">
        <MagneticBtn v-if="!running" @click="start">▶ 启动剧本</MagneticBtn>
        <template v-else>
          <button v-if="pipe?.state === 'running'" class="btn" @click="ctl('pause')">⏸ 暂停</button>
          <button v-else class="btn" @click="ctl('resume')">▶ 恢复</button>
          <button class="btn btn-danger" @click="ctl('abort')">🟥 中止</button>
        </template>
      </div>
    </header>

    <!-- 十幕时间轴 -->
    <GlowCard class="panel">
      <div class="stage-rail">
        <div v-for="(s, i) in pipe?.stages ?? []" :key="s.id" class="stage" :class="`st-${s.status}`">
          <div class="stage-id mono">{{ s.id }}</div>
          <div class="stage-name">{{ s.name }}</div>
          <div class="stage-steps">
            <span v-for="(st, j) in s.steps" :key="j" class="step-pip" :class="`sp-${st.status}`"
                  :title="`${st.actor} ${st.action} ${JSON.stringify(st.params ?? {})}`">
              {{ ACTOR_ICON[st.actor] ?? '·' }}
            </span>
          </div>
          <div v-if="i < (pipe?.stages.length ?? 0) - 1" class="stage-arrow">→</div>
        </div>
        <div v-if="!pipe?.stages?.length" class="empty mono">点"启动剧本"开跑 (sim 模式全功能可演示)</div>
      </div>
      <div v-if="pipe?.degraded?.length" class="degraded mono">
        ⚠ 降级步骤 (诚实标注): {{ pipe.degraded.join(' · ') }}
      </div>
    </GlowCard>

    <div class="grid">
      <!-- 故障注入 + 日志 -->
      <GlowCard class="panel">
        <div class="panel-title">💉 故障注入 <span class="hint-inline">(运行中可点, S6 是专属演示窗口)</span></div>
        <div class="fault-row">
          <button v-for="f in faults" :key="f.mode" class="fault-btn"
                  :class="{ active: pipe?.active_faults?.includes(f.mode) }"
                  :disabled="!running" @click="inject(f.mode)">
            {{ f.label }}<span class="fault-code mono">{{ f.mode }}</span>
          </button>
        </div>
        <div class="panel-title" style="margin-top:14px;">执行日志</div>
        <div class="log-box mono">
          <div v-for="(l, i) in (pipe?.log ?? []).slice().reverse()" :key="i"
               class="log-line" :class="`lk-${l.kind}`">
            <span class="log-t">{{ new Date(l.t * 1000).toLocaleTimeString() }}</span>
            <span>{{ l.msg }}</span>
          </div>
          <div v-if="!pipe?.log?.length" class="empty mono">无日志</div>
        </div>
      </GlowCard>

      <!-- 互锁顶视图 -->
      <GlowCard class="panel">
        <div class="panel-title">
          🛡 防互撞互锁
          <span class="chip" :class="{
            'chip-ok': ilLevel === 'ok', 'chip-warn': ilLevel === 'warn',
            'chip-err': ilLevel === 'danger', 'chip-info': ilLevel === 'unknown' || ilLevel === 'off',
          }">
            {{ ilLevel === 'ok' ? '安全' : ilLevel === 'warn' ? '预警' : ilLevel === 'danger' ? '危险'
               : ilLevel === 'off' ? '已停用' : '—' }}
            <template v-if="ilDist != null"> · {{ ilDist }}mm</template>
          </span>
        </div>
        <canvas ref="cvRef" class="ghost-canvas"></canvas>
        <div class="il-cfg">
          <span class="lbl">基座间距</span>
          <input v-model.number="ilConfig.base_dx_mm" type="range" min="200" max="800" step="10" class="slider" @change="saveIlConfig" />
          <span class="mono">{{ ilConfig.base_dx_mm }}mm</span>
          <label class="lbl" style="margin-left:10px;">
            <input v-model="ilConfig.enabled" type="checkbox" @change="saveIlConfig" /> 启用
          </label>
        </div>
        <p class="hint">
          myCobot 280 DH 正运动学 (Elephant Robotics URDF) → 5 段 capsule (r=40mm) 两两最小距;
          &lt;60mm 拒发 move/回放 waypoint。move API 与技能回放都过这道硬闸。
        </p>
      </GlowCard>
    </div>

    <!-- 快拆爪库 (v4 4 爪) -->
    <GlowCard class="panel">
      <div class="panel-title">
        🧰 快拆爪库 <span class="hint-inline">v4 一臂 4 爪 · 磁吸快拆 (φ8×3 钕磁铁 ×8/爪 + pogo 供 5V/GND/PWM + 定位销防呆)</span>
        <span class="chip chip-warn" style="margin-left:auto;">待硬件</span>
      </div>
      <div class="grip-grid">
        <div v-for="g in GRIPPERS" :key="g.key" class="grip-card" :class="`gc-${g.status}`">
          <div class="grip-top">
            <span class="grip-icon">{{ g.icon }}</span>
            <span class="grip-name">{{ g.name }}</span>
            <span class="chip" :class="g.status === 'ready' ? 'chip-ok' : 'chip-warn'">
              {{ g.status === 'ready' ? '就绪' : '待装' }}
            </span>
          </div>
          <div class="grip-rows">
            <div class="grip-row"><span class="gk">幕</span><span class="gv">{{ g.stages }}</span></div>
            <div class="grip-row"><span class="gk">驱动</span><span class="gv mono">{{ g.actuator }}</span></div>
            <div class="grip-row"><span class="gk">夹持</span><span class="gv">{{ g.payload }}</span></div>
          </div>
        </div>
      </div>
      <p class="hint">
        4 爪复用 6× MG996R + 2× SG90, 磁吸接口 atom 法兰一拔一插换爪 (橙线 G22 PWM, 红 5V0, 棕 GND)。
        承碗盘 D 把手用凹槽承重 (0g 拉力) 绕开 atom 250g 负载上限 — 软件层接口已留, 等舵机/磁铁/3D 爪体到货验。
      </p>
    </GlowCard>

    <!-- NL 任务 -->
    <GlowCard class="panel">
      <div class="panel-title">🗣 自然语言 → 双臂任务 <span class="hint-inline">车载脑 arm_planner :9103 (1.7B LoRA), 不通时规则降级</span></div>
      <div class="nl-row">
        <input v-model="nlText" class="inp" placeholder="比如: 双臂研磨一下 / 倒粉 / 接瓶 / 都归位" :disabled="nlBusy" @keyup.enter="nlAsk" />
        <button class="btn" :disabled="nlBusy || !nlText.trim()" @click="nlAsk">{{ nlBusy ? '…' : '规划' }}</button>
        <button class="btn btn-go" :disabled="!nlPlan || running" @click="nlRun">▶ 执行</button>
      </div>
      <div v-if="nlNote" class="msg mono">{{ nlNote }}</div>
      <div v-if="nlPlan" class="plan-row">
        <span v-for="(s, i) in nlPlan" :key="i" class="plan-chip mono">
          {{ ACTOR_ICON[s.actor] }} {{ s.actor }}.{{ s.action }}{{ s.params?.name ? `(${s.params.name})` : '' }}
        </span>
      </div>
    </GlowCard>
  </section>
</template>

<style scoped>
.page { display: flex; flex-direction: column; gap: 16px; }
.head { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; }
.page-title { font-size: 1.5rem; font-weight: 800; margin: 0 0 4px; }
.sub { font-size: 0.78rem; color: var(--ink-tertiary, #64748b); margin: 0; line-height: 1.6; max-width: 780px; }
.head-ctl { display: flex; gap: 8px; flex-shrink: 0; }

.panel { padding: 16px; }
.panel-title { font-size: 0.82rem; font-weight: 700; margin-bottom: 12px; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.hint-inline { font-size: 0.64rem; font-weight: 400; color: var(--ink-muted, #94a3b8); }

.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
@media (max-width: 1080px) { .grid { grid-template-columns: 1fr; } }

/* 十幕时间轴 */
.stage-rail { display: flex; flex-wrap: wrap; gap: 8px; align-items: stretch; }
.stage {
  position: relative; min-width: 96px; flex: 1;
  border: 1px solid var(--line-border, #e2e8f0); border-radius: 10px;
  padding: 8px 10px; background: rgba(255,255,255,0.7);
  transition: all 0.25s;
}
.st-running { border-color: #2563eb; box-shadow: 0 0 0 2px rgba(37,99,235,0.18); animation: stPulse 1.4s infinite; }
.st-done    { background: rgba(5,150,105,0.06); border-color: rgba(5,150,105,0.3); }
.st-aborted { opacity: 0.5; }
@keyframes stPulse { 50% { box-shadow: 0 0 0 5px rgba(37,99,235,0.07); } }
.stage-id { font-size: 0.62rem; color: var(--ink-muted, #94a3b8); font-weight: 700; }
.stage-name { font-size: 0.76rem; font-weight: 700; margin: 2px 0 6px; }
.stage-steps { display: flex; gap: 3px; flex-wrap: wrap; }
.step-pip {
  width: 22px; height: 22px; display: inline-flex; align-items: center; justify-content: center;
  border-radius: 6px; font-size: 0.7rem; background: rgba(15,23,42,0.04);
}
.sp-running  { background: rgba(37,99,235,0.18); animation: stPulse 1s infinite; }
.sp-done     { background: rgba(5,150,105,0.16); }
.sp-degraded { background: rgba(217,119,6,0.18); }
.sp-skipped  { background: rgba(148,163,184,0.2); opacity: 0.6; }
.stage-arrow { position: absolute; right: -10px; top: 40%; color: var(--ink-muted, #cbd5e1); font-size: 0.8rem; z-index: 1; }

.degraded { margin-top: 10px; font-size: 0.68rem; color: #d97706; }

/* 故障注入 */
.fault-row { display: flex; flex-wrap: wrap; gap: 8px; }
.fault-btn {
  border: 1px solid rgba(225,29,72,0.3); background: rgba(225,29,72,0.04);
  border-radius: 9px; padding: 7px 12px; cursor: pointer;
  font-size: 0.74rem; font-weight: 700; color: #be123c; font-family: inherit;
  display: flex; flex-direction: column; align-items: center; gap: 2px;
  transition: all 0.15s;
}
.fault-btn:hover:not(:disabled) { transform: translateY(-1px); background: rgba(225,29,72,0.09); }
.fault-btn:disabled { opacity: 0.4; cursor: not-allowed; }
.fault-btn.active { background: #e11d48; color: white; animation: stPulse 0.8s infinite; }
.fault-code { font-size: 0.56rem; opacity: 0.7; font-weight: 400; }

.log-box { max-height: 230px; overflow-y: auto; font-size: 0.7rem; }
.log-line { display: flex; gap: 8px; padding: 2px 0; color: var(--ink-secondary, #475569); }
.log-t { color: var(--ink-muted, #94a3b8); flex-shrink: 0; }
.lk-alarm { color: #e11d48; }
.lk-llm   { color: #7c3aed; }
.lk-ok    { color: #059669; }
.lk-warn  { color: #d97706; }

/* 互锁 */
.ghost-canvas { display: block; width: 100%; height: 240px; border-radius: 10px; background: linear-gradient(160deg, #fdfefe, #f4f7fb); border: 1px solid var(--line-divider, #f1f5f9); }
.il-cfg { display: flex; align-items: center; gap: 8px; margin-top: 10px; font-size: 0.72rem; }
.lbl { color: var(--ink-secondary, #475569); }
.slider { flex: 1; accent-color: #2563eb; }
.hint { font-size: 0.66rem; color: var(--ink-muted, #94a3b8); line-height: 1.6; margin: 10px 0 0; }

/* 快拆爪库 */
.grip-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
@media (max-width: 1080px) { .grip-grid { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 560px) { .grip-grid { grid-template-columns: 1fr; } }
.grip-card {
  border: 1px solid var(--line-border, #e2e8f0); border-radius: 12px; padding: 12px;
  background: linear-gradient(165deg, rgba(255,255,255,0.9), rgba(248,250,252,0.7));
  transition: transform 0.18s, box-shadow 0.18s;
}
.grip-card:hover { transform: translateY(-2px); box-shadow: 0 8px 22px rgba(15,23,42,0.08); }
.gc-pending { border-style: dashed; }
.grip-top { display: flex; align-items: center; gap: 7px; margin-bottom: 9px; }
.grip-icon { font-size: 1.3rem; }
.grip-name { font-size: 0.86rem; font-weight: 800; flex: 1; }
.grip-rows { display: flex; flex-direction: column; gap: 5px; }
.grip-row { display: flex; gap: 8px; font-size: 0.7rem; line-height: 1.4; }
.gk { flex: 0 0 30px; color: var(--ink-muted, #94a3b8); font-weight: 700; }
.gv { flex: 1; color: var(--ink-secondary, #475569); }

/* NL */
.nl-row { display: flex; gap: 8px; }
.inp { flex: 1; border: 1px solid var(--line-border, #e2e8f0); border-radius: 9px; padding: 9px 12px; font-size: 0.82rem; font-family: inherit; background: rgba(255,255,255,0.85); }
.msg { margin-top: 8px; font-size: 0.72rem; color: #2563eb; }
.plan-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
.plan-chip { background: rgba(124,58,237,0.08); color: #6d28d9; border-radius: 999px; padding: 4px 11px; font-size: 0.68rem; font-weight: 600; }

.btn { border: 1px solid var(--line-border, #e2e8f0); background: white; border-radius: 9px; padding: 8px 14px; cursor: pointer; font-size: 0.78rem; font-weight: 700; font-family: inherit; }
.btn:hover:not(:disabled) { transform: translateY(-1px); border-color: rgba(37,99,235,0.4); }
.btn:disabled { opacity: 0.45; cursor: not-allowed; }
.btn-danger { border-color: rgba(225,29,72,0.4); color: #e11d48; }
.btn-go { background: linear-gradient(135deg, #2563eb, #0891b2); color: white; border-color: transparent; }

.empty { color: var(--ink-muted, #94a3b8); font-size: 0.74rem; padding: 8px 4px; }
.chip { display: inline-flex; align-items: center; gap: 4px; border-radius: 999px; padding: 2px 9px; font-size: 0.64rem; font-weight: 700; }
.chip-ok { background: rgba(5,150,105,0.1); color: #059669; }
.chip-warn { background: rgba(217,119,6,0.1); color: #d97706; }
.chip-err { background: rgba(225,29,72,0.1); color: #e11d48; }
.chip-info { background: rgba(37,99,235,0.1); color: #2563eb; }
</style>
