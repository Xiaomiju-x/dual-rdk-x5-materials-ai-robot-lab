<script setup lang="ts">
// Public review build: synthetic recording, replay preview, and lerobot-shaped export.
// No physical robot command is present in this repository.
import { computed, onMounted, onUnmounted, ref } from 'vue'
import GlowCard from '@/components/fx/GlowCard.vue'
import MagneticBtn from '@/components/fx/MagneticBtn.vue'
import SkillTrajectory3D from '@/components/three/SkillTrajectory3D.vue'
import { tipPathFromWaypoints, type TipPath } from '@/components/three/realFK'

interface SkillMeta { name: string; arm: string; hz: number; created_at: string; n_frames: number; note: string }
interface Session { state: string; skill: string | null; arm: string | null; n: number; progress_pct: number; error: string; mock: boolean }

const skills = ref<SkillMeta[]>([])
const session = ref<Session | null>(null)
const recName = ref('')
const recArm = ref<'arm01' | 'arm02'>('arm01')
const replaySpeed = ref(30)
const msg = ref('')
let timer: number | null = null

async function poll() {
  try {
    const r = await fetch('/api/skills').then((x) => x.json())
    skills.value = r.skills ?? []
    session.value = r.session ?? null
  } catch { /* offline */ }
}

const busy = computed(() => session.value?.state !== 'idle')
const recording = computed(() => session.value?.state === 'recording')
const replaying = computed(() => session.value?.state === 'replaying')

async function startRecord() {
  if (!recName.value.trim()) { msg.value = '先给技能起个名'; return }
  const r = await fetch('/api/skills/record', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name: recName.value.trim(), arm: recArm.value, mock: true }),
  }).then((x) => x.json())
  msg.value = r.ok ? '🔴 正在生成公开合成轨迹，完成后点停止' : `✗ ${r.error}`
  poll()
}
async function stopRecord() {
  await fetch('/api/skills/record/stop', { method: 'POST' })
  msg.value = '⏹ 已停止, 落库中…'
  setTimeout(poll, 600)
}
async function replay(name: string) {
  const r = await fetch(`/api/skills/${encodeURIComponent(name)}/replay`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ speed: replaySpeed.value, mock: true }),
  }).then((x) => x.json())
  msg.value = r.ok ? `▶ 回放 ${name} (${r.n_waypoints} waypoint, 每点过互锁预演)` : `✗ ${r.error}`
}
async function stopReplay() {
  await fetch('/api/skills/replay/stop', { method: 'POST' })
}
async function exportSkill(name: string) {
  const r = await fetch(`/api/skills/${encodeURIComponent(name)}/export`, { method: 'POST' }).then((x) => x.json())
  msg.value = r.ok ? `📦 已导出 lerobot 原始 episode → ${r.path}` : `✗ ${r.error}`
}
async function delSkill(name: string) {
  await fetch(`/api/skills/${encodeURIComponent(name)}`, { method: 'DELETE' })
  poll()
}

// ---- 末端轨迹 3D 预览 (G4) ----
const preview = ref<{ name: string; arm: string; note: string; nWp: number } | null>(null)
const tip = ref<TipPath | null>(null)
const previewPlay = ref(true)
const previewErr = ref('')
async function openPreview(s: SkillMeta) {
  preview.value = { name: s.name, arm: s.arm, note: s.note, nWp: 0 }
  tip.value = null; previewErr.value = ''; previewPlay.value = true
  try {
    const r = await fetch(`/api/skills/${encodeURIComponent(s.name)}/waypoints`).then((x) => x.json())
    if (!r.ok) { previewErr.value = r.error || '读取失败'; return }
    const wps: number[][] = r.waypoints ?? []
    if (wps.length < 2) { previewErr.value = `waypoint 太少 (${wps.length}), 无法成轨迹`; return }
    preview.value = { name: s.name, arm: r.arm || s.arm, note: r.note || s.note, nWp: wps.length }
    tip.value = tipPathFromWaypoints(wps)
  } catch { previewErr.value = '网络错误' }
}
function closePreview() { preview.value = null; tip.value = null }

onMounted(() => { poll(); timer = window.setInterval(poll, 1200) })
onUnmounted(() => { if (timer !== null) window.clearInterval(timer) })
</script>

<template>
  <section class="page">
    <header class="head">
      <div>
        <h1 class="page-title">🎓 Skills · 技能库</h1>
        <p class="sub">
          公开合成轨迹 → 10Hz 关节序列 → 命名入库 →
          waypoint 可视化回放 → 导出 lerobot 形状的离线样例
        </p>
      </div>
      <span v-if="session?.error" class="chip chip-err">{{ session.error }}</span>
    </header>

    <div class="grid">
      <!-- 录制台 -->
      <GlowCard class="panel">
        <div class="panel-title">示教录制台</div>
        <div class="rec-row">
          <input v-model="recName" class="inp" placeholder="技能名 (如: 研磨 / 倒粉 / 灌装)" :disabled="busy" />
          <select v-model="recArm" class="inp sel" :disabled="busy">
            <option value="arm01">arm01</option>
            <option value="arm02">arm02</option>
          </select>
        </div>
        <div class="rec-ctl">
          <MagneticBtn v-if="!recording" :disabled="busy" @click="startRecord">🔴 开始录制</MagneticBtn>
          <MagneticBtn v-else @click="stopRecord">⏹ 停止并保存</MagneticBtn>
          <div v-if="recording" class="rec-live mono">
            <span class="rec-dot"></span> {{ session?.n ?? 0 }} 帧 @10Hz · {{ session?.arm }}
          </div>
        </div>
        <p class="hint">
          公开仓库只运行 mock：轨迹标记 synthetic，不连接串口、设备或机械臂。
          复赛真机结果只作为已验证证据叙述，不由本页面复现控制。
        </p>
        <div v-if="msg" class="msg mono">{{ msg }}</div>
      </GlowCard>

      <!-- 回放控制 -->
      <GlowCard class="panel">
        <div class="panel-title">回放参数</div>
        <div class="rep-row">
          <span class="lbl">速度档</span>
          <input v-model.number="replaySpeed" type="range" min="5" max="80" class="slider" />
          <span class="mono">{{ replaySpeed }}</span>
        </div>
        <div v-if="replaying" class="rep-live">
          <div class="bar"><div class="bar-fill" :style="{ width: `${session?.progress_pct ?? 0}%` }"></div></div>
          <div class="mono rep-stat">
            ▶ {{ session?.skill }} · waypoint {{ session?.n }} · {{ session?.progress_pct }}%
            <button class="mini-btn" @click="stopReplay">⏹ 停</button>
          </div>
        </div>
        <p class="hint">
          公开回放仅驱动浏览器预览和进度条，不下发 waypoint。
          生产互锁与真实执行代码不在公开仓库内。
        </p>
      </GlowCard>
    </div>

    <!-- 技能列表 -->
    <GlowCard class="panel list-panel">
      <div class="panel-title">已录技能 <span class="chip chip-info">{{ skills.length }}</span></div>
      <div v-if="!skills.length" class="empty mono">空 — 录一条试试 (mock 模式也能录合成轨迹)</div>
      <table v-else class="tbl">
        <thead>
          <tr><th>名称</th><th>臂</th><th>帧数</th><th>录制时间</th><th>备注</th><th style="text-align:right">操作</th></tr>
        </thead>
        <tbody>
          <tr v-for="s in skills" :key="s.name">
            <td class="t-name">{{ s.name }}</td>
            <td><span class="chip" :class="s.arm === 'arm01' ? 'chip-info' : 'chip-violet'">{{ s.arm }}</span></td>
            <td class="mono">{{ s.n_frames }}</td>
            <td class="mono t-dim">{{ s.created_at }}</td>
            <td class="t-dim t-note">{{ s.note }}</td>
            <td class="t-ops">
              <button class="mini-btn" @click="openPreview(s)">👁 预览</button>
              <button class="mini-btn" :disabled="busy" @click="replay(s.name)">▶ 回放</button>
              <button class="mini-btn" @click="exportSkill(s.name)">📦 lerobot</button>
              <button class="mini-btn mini-del" @click="delSkill(s.name)">🗑</button>
            </td>
          </tr>
        </tbody>
      </table>
    </GlowCard>

    <!-- 末端轨迹 3D 预览 overlay (G4) -->
    <div v-if="preview" class="pv-mask" @click.self="closePreview">
      <div class="pv-card">
        <div class="pv-head">
          <div>
            <div class="pv-title">🌀 末端轨迹预览 · {{ preview.name }}</div>
            <div class="pv-sub mono">
              {{ preview.arm }} · {{ preview.nWp }} waypoint ·
              myCobot 280 DH 几何预览 · 单位 mm→m · synthetic
            </div>
          </div>
          <button class="pv-x" @click="closePreview">✕</button>
        </div>

        <div class="pv-body">
          <div v-if="previewErr" class="pv-err mono">⚠ {{ previewErr }}</div>
          <div v-else-if="!tip" class="pv-load mono">算运动学中…</div>
          <SkillTrajectory3D
            v-else
            :dense="tip.dense"
            :marks="tip.marks"
            :playing="previewPlay"
            class="pv-scene"
          />
        </div>

        <div class="pv-foot">
          <div class="pv-legend">
            <span><i class="dot dot-g"></i>起点</span>
            <span><i class="dot dot-p"></i>waypoint</span>
            <span><i class="dot dot-r"></i>终点</span>
            <span><i class="dot dot-a"></i>动点</span>
          </div>
          <div class="pv-ctl">
            <button class="mini-btn" @click="previewPlay = !previewPlay">
              {{ previewPlay ? '⏸ 暂停' : '▶ 播放' }}
            </button>
            <span class="pv-hint mono">拖动旋转 · 滚轮缩放</span>
          </div>
        </div>
        <p class="pv-note">
          轨迹由合成关节角 waypoint 经 DH 正运动学逐段插值生成，仅用于公开可视化复核。
          note 始终标记 synthetic，不代表真机遥测或可执行命令。
        </p>
      </div>
    </div>
  </section>
</template>

<style scoped>
.page { display: flex; flex-direction: column; gap: 16px; }
.head { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; }
.page-title { font-size: 1.5rem; font-weight: 800; margin: 0 0 4px; }
.sub { font-size: 0.78rem; color: var(--ink-tertiary, #64748b); margin: 0; line-height: 1.6; max-width: 760px; }

.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
@media (max-width: 980px) { .grid { grid-template-columns: 1fr; } }

.panel { padding: 16px; }
.panel-title { font-size: 0.82rem; font-weight: 700; margin-bottom: 12px; display: flex; align-items: center; gap: 8px; }

.rec-row { display: flex; gap: 8px; margin-bottom: 12px; }
.inp {
  flex: 1; border: 1px solid var(--line-border, #e2e8f0); border-radius: 9px;
  padding: 9px 12px; font-size: 0.82rem; font-family: inherit; background: rgba(255,255,255,0.85);
}
.sel { flex: 0 0 100px; }
.rec-ctl { display: flex; align-items: center; gap: 14px; }
.rec-live { display: flex; align-items: center; gap: 8px; font-size: 0.76rem; color: #e11d48; font-weight: 700; }
.rec-dot { width: 10px; height: 10px; border-radius: 50%; background: #e11d48; animation: recPulse 1s infinite; }
@keyframes recPulse { 50% { opacity: 0.3; } }

.rep-row { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
.lbl { font-size: 0.76rem; color: var(--ink-secondary, #475569); }
.slider { flex: 1; accent-color: #7c3aed; }
.bar { height: 8px; border-radius: 999px; background: rgba(124,58,237,0.12); overflow: hidden; margin-bottom: 6px; }
.bar-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #7c3aed, #2563eb); transition: width 0.3s; }
.rep-stat { font-size: 0.72rem; color: var(--ink-secondary, #475569); display: flex; align-items: center; gap: 10px; }

.hint { font-size: 0.68rem; color: var(--ink-muted, #94a3b8); line-height: 1.6; margin: 10px 0 0; }
.msg { margin-top: 10px; font-size: 0.72rem; color: #2563eb; word-break: break-all; }

.list-panel { overflow-x: auto; }
.tbl { width: 100%; border-collapse: collapse; font-size: 0.78rem; }
.tbl th { text-align: left; font-size: 0.66rem; color: var(--ink-muted, #94a3b8); font-weight: 700; padding: 6px 8px; border-bottom: 1px solid var(--line-border, #e2e8f0); }
.tbl td { padding: 8px; border-bottom: 1px dashed var(--line-divider, #f1f5f9); }
.t-name { font-weight: 700; }
.t-dim { color: var(--ink-muted, #94a3b8); font-size: 0.7rem; }
.t-note { max-width: 220px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.t-ops { text-align: right; white-space: nowrap; }

.mini-btn {
  border: 1px solid var(--line-border, #e2e8f0); background: white; border-radius: 7px;
  padding: 4px 10px; cursor: pointer; font-size: 0.7rem; font-family: inherit; margin-left: 4px;
}
.mini-btn:disabled { opacity: 0.4; cursor: not-allowed; }
.mini-btn:hover:not(:disabled) { border-color: rgba(37,99,235,0.4); }
.mini-del:hover { border-color: rgba(225,29,72,0.5); }

.empty { color: var(--ink-muted, #94a3b8); font-size: 0.74rem; padding: 12px 4px; }
.chip { display: inline-flex; border-radius: 999px; padding: 2px 9px; font-size: 0.64rem; font-weight: 700; }
.chip-info { background: rgba(37,99,235,0.1); color: #2563eb; }
.chip-violet { background: rgba(124,58,237,0.1); color: #7c3aed; }
.chip-err { background: rgba(225,29,72,0.1); color: #e11d48; }

/* 末端轨迹预览 overlay */
.pv-mask {
  position: fixed; inset: 0; z-index: 60; display: flex; align-items: center; justify-content: center;
  background: rgba(15, 23, 42, 0.42); backdrop-filter: blur(4px); padding: 20px;
  animation: pvIn 0.2s ease;
}
@keyframes pvIn { from { opacity: 0; } to { opacity: 1; } }
.pv-card {
  width: min(720px, 96vw); background: #fff; border-radius: 16px; overflow: hidden;
  border: 1px solid var(--line-border, #e2e8f0); box-shadow: 0 24px 60px rgba(15,23,42,0.28);
  display: flex; flex-direction: column;
}
.pv-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 10px; padding: 14px 16px; border-bottom: 1px solid var(--line-divider, #f1f5f9); }
.pv-title { font-size: 0.96rem; font-weight: 800; }
.pv-sub { font-size: 0.66rem; color: var(--ink-muted, #94a3b8); margin-top: 3px; }
.pv-x { border: none; background: rgba(241,245,249,0.8); border-radius: 8px; width: 30px; height: 30px; cursor: pointer; font-size: 0.9rem; color: #475569; }
.pv-x:hover { background: rgba(226,232,240,0.95); }
.pv-body { position: relative; height: 380px; background: linear-gradient(160deg, #f8fafc, #eef2ff); }
.pv-scene { width: 100%; height: 100%; }
.pv-err, .pv-load { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; font-size: 0.8rem; }
.pv-err { color: #e11d48; }
.pv-load { color: #64748b; }
.pv-foot { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 10px 16px; flex-wrap: wrap; }
.pv-legend { display: flex; gap: 14px; font-size: 0.7rem; color: var(--ink-secondary, #475569); }
.pv-legend span { display: inline-flex; align-items: center; gap: 5px; }
.dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
.dot-g { background: #10b981; } .dot-p { background: #7c3aed; } .dot-r { background: #ef4444; } .dot-a { background: #f59e0b; }
.pv-ctl { display: flex; align-items: center; gap: 10px; }
.pv-hint { font-size: 0.66rem; color: var(--ink-muted, #94a3b8); }
.pv-note { font-size: 0.66rem; color: var(--ink-muted, #94a3b8); line-height: 1.55; margin: 0; padding: 0 16px 14px; }
</style>
