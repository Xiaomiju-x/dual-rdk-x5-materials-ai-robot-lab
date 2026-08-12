<script setup lang="ts">
// Teleop Pro — joint-level (6 RingSliders) + cartesian-level (3D IK drag) +
// timeline-level (motion clip scrub) + pose-level (Waypoint teach + replay)
// for both arms. Magnetic action bar, mirror, recording, e-stop.
import { onBeforeUnmount, reactive, ref, computed } from 'vue'
import { useTelemetryStore } from '@/stores/telemetry'
import { useToastStore } from '@/stores/toast'
import { useWaypointStore, type Waypoint } from '@/stores/waypoints'
import { useAudioFeedback } from '@/composables/useAudioFeedback'
import type { ArmId } from '@/types/telemetry'
import RingSlider from '@/components/fx/RingSlider.vue'
import GlowCard from '@/components/fx/GlowCard.vue'
import MagneticBtn from '@/components/fx/MagneticBtn.vue'
import WaveformTimeline from '@/components/fx/WaveformTimeline.vue'
import Odometer from '@/components/fx/Odometer.vue'
import IkTargetScene from '@/components/three/IkTargetScene.vue'

const telemetry = useTelemetryStore()
const toasts = useToastStore()
const audio = useAudioFeedback()
const waypoints = useWaypointStore()

const LIMITS: [number, number][] = [[-168, 168], [-135, 135], [-150, 150], [-145, 145], [-165, 165], [-180, 180]]
const JOINT_NAMES = ['J1 底座', 'J2 大臂', 'J3 小臂', 'J4 腕1', 'J5 腕2', 'J6 腕3']

interface ArmCtl {
  targets: number[]
  gripper: number
  speed: number
  recording: boolean
  recBuf: { t: number; angles: number[]; gripper: number }[]
}
function freshCtl(): ArmCtl { return { targets: [0,0,0,0,0,0], gripper: 50, speed: 40, recording: false, recBuf: [] } }
const ctl = reactive<Record<ArmId, ArmCtl>>({ arm01: freshCtl(), arm02: freshCtl() })

interface Clip { id: string; arm: ArmId; frames: number; durMs: number; data: { angles: number[]; gripper: number }[] }
const clips = ref<Clip[]>([])
let clipSeq = 0

const ARMS: { id: ArmId; name: string; accent: 'amber' | 'blue'; color: string }[] = [
  { id: 'arm01', name: 'arm01', accent: 'amber', color: 'var(--accent-amber)' },
  { id: 'arm02', name: 'arm02', accent: 'blue',  color: 'var(--accent-blue)' },
]

function actual(arm: ArmId): number[] { return telemetry.jointsOf(arm).angles }

async function apply(arm: ArmId) {
  try {
    await fetch(`/api/move/${arm}`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ angles: ctl[arm].targets, speed: ctl[arm].speed, gripper: ctl[arm].gripper }),
    })
    toasts.push({ tone: 'ok', title: `${arm} 已下发`, detail: `speed ${ctl[arm].speed} · 夹爪 ${ctl[arm].gripper}%` })
    audio.play('success')
  } catch (e) {
    toasts.push({ tone: 'err', title: `${arm} 下发失败`, detail: (e as Error).message })
    audio.play('err')
  }
}
function home(arm: ArmId) { ctl[arm].targets = [0,0,0,0,0,0]; ctl[arm].gripper = 50; audio.play('click') }
function sync(arm: ArmId) { ctl[arm].targets = [...actual(arm)]; ctl[arm].gripper = telemetry.jointsOf(arm).gripper; audio.play('click') }
function mirror(from: ArmId) {
  const to: ArmId = from === 'arm01' ? 'arm02' : 'arm01'
  ctl[to].targets = ctl[from].targets.map((v, i) => (i === 0 ? -v : v))
  ctl[to].gripper = ctl[from].gripper
  toasts.push({ tone: 'info', title: `已镜像到 ${to}`, detail: 'J1 取反, 其余同步' })
  audio.play('ping')
}

// IK scene callback — replace the 6 targets with the solved pose
function onIkSolved(arm: ArmId, solved: number[]) {
  for (let i = 0; i < 6; i++) ctl[arm].targets[i] = Math.round(solved[i])
}

// Waypoints
function teachWaypoint(arm: ArmId, name?: string) {
  const wp = waypoints.add(arm, ctl[arm].targets, ctl[arm].gripper, name)
  toasts.push({ tone: 'ok', title: `已记录 ${wp.name}`, detail: arm })
  audio.play('ping')
}
function applyWaypoint(wp: Waypoint) {
  ctl[wp.arm].targets = wp.angles.slice()
  ctl[wp.arm].gripper = wp.gripper
  audio.play('click')
}
async function playSequence(arm: ArmId) {
  const seq = waypoints.list.filter((w) => w.arm === arm)
  if (seq.length < 1) {
    toasts.push({ tone: 'warn', title: '没有 waypoint', detail: `先示教 ${arm} 至少 1 个 waypoint` })
    return
  }
  toasts.push({ tone: 'info', title: `回放 ${arm} 序列`, detail: `${seq.length} 个 waypoint` })
  for (const wp of seq) {
    applyWaypoint(wp)
    await apply(arm)
    await new Promise((r) => setTimeout(r, 1200))
  }
  toasts.push({ tone: 'ok', title: `${arm} 序列完成`, detail: '' })
}

const recTimers: Record<ArmId, number | null> = { arm01: null, arm02: null }
function toggleRecord(arm: ArmId) {
  const c = ctl[arm]
  if (c.recording) {
    c.recording = false
    if (recTimers[arm] !== null) { window.clearInterval(recTimers[arm]!); recTimers[arm] = null }
    if (c.recBuf.length > 1) {
      const first = c.recBuf[0].t
      clips.value = [...clips.value, {
        id: `clip-${++clipSeq}`, arm, frames: c.recBuf.length,
        durMs: c.recBuf[c.recBuf.length - 1].t - first,
        data: c.recBuf.map((f) => ({ angles: [...f.angles], gripper: f.gripper })),
      }]
      toasts.push({ tone: 'ok', title: `录制完成 · ${c.recBuf.length} 帧`, detail: arm })
    }
    c.recBuf = []
    audio.play('success')
  } else {
    c.recording = true; c.recBuf = []
    recTimers[arm] = window.setInterval(() => {
      c.recBuf.push({ t: Date.now(), angles: [...c.targets], gripper: c.gripper })
    }, 100)
    audio.play('ping')
  }
}

const replaying = ref<string | null>(null)
const scrubPos = ref<Record<string, number>>({})
let replayTimer: number | null = null
function replay(clip: Clip) {
  if (replaying.value) return
  replaying.value = clip.id
  let i = 0
  replayTimer = window.setInterval(() => {
    if (i >= clip.data.length) { stopReplay(); return }
    ctl[clip.arm].targets = [...clip.data[i].angles]
    ctl[clip.arm].gripper = clip.data[i].gripper
    scrubPos.value = { ...scrubPos.value, [clip.id]: i / Math.max(1, clip.data.length - 1) }
    i++
  }, 100)
  audio.play('navigate')
}
function stopReplay() {
  if (replayTimer !== null) { window.clearInterval(replayTimer); replayTimer = null }
  replaying.value = null
}
function delClip(id: string) { clips.value = clips.value.filter((c) => c.id !== id); audio.play('click') }
function seekClip(clip: Clip, frameIdx: number) {
  if (clip.data[frameIdx]) {
    ctl[clip.arm].targets = [...clip.data[frameIdx].angles]
    ctl[clip.arm].gripper = clip.data[frameIdx].gripper
  }
}

const estopped = ref(false)
async function estop() {
  estopped.value = true
  for (const a of ['arm01','arm02'] as ArmId[]) {
    ctl[a].targets = [...actual(a)]
    if (ctl[a].recording) toggleRecord(a)
  }
  stopReplay()
  audio.play('err')
  toasts.push({ tone: 'err', title: '急停已触发', detail: '目标已锁定到当前实际姿态', durationMs: 6000 })
  setTimeout(() => (estopped.value = false), 1200)
}

const dirtyCount = computed(() => {
  let n = 0
  for (const a of ['arm01','arm02'] as ArmId[]) {
    for (let i = 0; i < 6; i++) if (Math.abs(ctl[a].targets[i] - (actual(a)[i] ?? 0)) > 1) n++
  }
  return n
})

const wpA = computed(() => waypoints.list.filter((w) => w.arm === 'arm01'))
const wpB = computed(() => waypoints.list.filter((w) => w.arm === 'arm02'))

onBeforeUnmount(() => {
  for (const a of ['arm01','arm02'] as ArmId[]) if (recTimers[a] !== null) window.clearInterval(recTimers[a]!)
  stopReplay()
})
</script>

<template>
  <div class="teleop-pro">
    <div class="tl-head">
      <div>
        <h1 class="page-title">触控操控 · 关节 + IK + Waypoint</h1>
        <p class="page-subtitle">6 RingSlider 关节级 · 拖球笛卡尔级 · 时间轴 scrub · Waypoint 示教回放</p>
      </div>
      <div class="tl-head-meta">
        <div class="kv-mono">脏值 <Odometer :value="dirtyCount" /> · 片段 <Odometer :value="clips.length" /> · waypoints <Odometer :value="waypoints.list.length" /></div>
        <button class="estop" :class="{ shake: estopped }" @click="estop">
          <span class="estop-glyph">⏹</span> 急停 ESTOP
        </button>
      </div>
    </div>

    <div class="tl-grid">
      <GlowCard v-for="a in ARMS" :key="a.id" :accent="a.accent" class="arm-panel">
        <header class="ap-head">
          <span class="ap-dot dot" :class="telemetry.jointsOf(a.id).online ? 'dot-ok' : 'dot-idle'"></span>
          <span class="ap-name">{{ a.name }}</span>
          <span v-if="ctl[a.id].recording" class="chip chip-err rec-chip">● REC {{ ctl[a.id].recBuf.length }}</span>
          <span class="ap-speed mono">speed {{ ctl[a.id].speed }} · 夹爪 {{ ctl[a.id].gripper }}%</span>
        </header>

        <div class="ap-body">
          <div class="rings-grid">
            <div v-for="(name, i) in JOINT_NAMES" :key="i" class="ring-cell">
              <RingSlider
                v-model="ctl[a.id].targets[i]"
                :min="LIMITS[i][0]" :max="LIMITS[i][1]" :step="1"
                :actual="actual(a.id)[i] ?? 0"
                :accent="a.color"
                :label="name"
                :size="86" :stroke="8"
              />
            </div>
          </div>

          <div class="ik-host">
            <IkTargetScene
              :arm="a.id"
              :angles="ctl[a.id].targets"
              :gripper="ctl[a.id].gripper"
              @solved="(s) => onIkSolved(a.id, s)"
            />
          </div>
        </div>

        <div class="aux-row">
          <div class="aux">
            <div class="aux-label">夹爪</div>
            <input type="range" class="jog jog-grip" min="0" max="100" step="1" v-model.number="ctl[a.id].gripper" />
            <div class="aux-val mono">{{ ctl[a.id].gripper }}%</div>
          </div>
          <div class="aux">
            <div class="aux-label">速度</div>
            <input type="range" class="jog jog-speed" min="5" max="100" step="5" v-model.number="ctl[a.id].speed" />
            <div class="aux-val mono">{{ ctl[a.id].speed }}</div>
          </div>
        </div>

        <div class="ap-actions">
          <MagneticBtn @click="home(a.id)">🏠 HOME</MagneticBtn>
          <MagneticBtn @click="sync(a.id)">⟲ 读取</MagneticBtn>
          <MagneticBtn @click="toggleRecord(a.id)">{{ ctl[a.id].recording ? '■ 停止' : '● 录制' }}</MagneticBtn>
          <MagneticBtn @click="teachWaypoint(a.id)">+ Waypoint</MagneticBtn>
          <MagneticBtn @click="mirror(a.id)">⇄ 镜像</MagneticBtn>
          <MagneticBtn class="primary" @click="apply(a.id)">▶ 下发</MagneticBtn>
        </div>

        <!-- waypoint chips for this arm -->
        <div class="wp-row" v-if="(a.id === 'arm01' ? wpA : wpB).length">
          <div class="wp-head">
            <span class="section-label">Waypoints · {{ a.name }}</span>
            <button class="wp-play" @click="playSequence(a.id)">▶ 顺序回放</button>
          </div>
          <div class="wp-list">
            <div v-for="wp in (a.id === 'arm01' ? wpA : wpB)" :key="wp.id" class="wp-chip" :class="`accent-${a.accent}`"
                 @click="applyWaypoint(wp)">
              <span class="wp-name">{{ wp.name }}</span>
              <span class="wp-mini mono">{{ wp.angles.map((v: number) => Math.round(v)).join(',') }}</span>
              <button class="wp-x" @click.stop="waypoints.remove(wp.id)" title="删除">×</button>
            </div>
          </div>
        </div>
      </GlowCard>
    </div>

    <GlowCard accent="violet" class="clips-card">
      <div class="clips-head">
        <span class="section-label">录制片段 · 拖拽时间轴可 scrub 关节</span>
        <span class="mono clips-count">{{ clips.length }} clips</span>
      </div>
      <div v-if="!clips.length" class="clips-empty">还没有录制片段 · 在某个臂上点"● 录制"开始采样关节轨迹</div>
      <div v-else class="clips-stack">
        <div v-for="c in clips" :key="c.id" class="clip-row">
          <div class="clip-meta">
            <span class="clip-arm" :class="`accent-${c.arm === 'arm01' ? 'amber' : 'blue'}`">{{ c.arm }}</span>
            <span class="kv-mono">{{ c.frames }}帧 · {{ (c.durMs/1000).toFixed(1) }}s</span>
          </div>
          <WaveformTimeline
            :frames="c.data"
            :position="scrubPos[c.id] ?? 0"
            :accent="c.arm === 'arm01' ? 'var(--accent-amber)' : 'var(--accent-blue)'"
            :width="380" :height="52"
            @update:position="(v: number) => scrubPos[c.id] = v"
            @seek="(i: number) => seekClip(c, i)"
          />
          <div class="clip-btns">
            <button class="cb cb-play" :disabled="!!replaying" @click="replay(c)">{{ replaying === c.id ? '▶ …' : '▶ 播放' }}</button>
            <button class="cb cb-del" @click="delClip(c.id)">×</button>
          </div>
        </div>
      </div>
    </GlowCard>
  </div>
</template>

<style scoped>
.teleop-pro { display: flex; flex-direction: column; gap: 14px; }

.tl-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; }
.tl-head-meta { display: flex; align-items: center; gap: 12px; }

.estop {
  display: inline-flex; align-items: center; gap: 8px; padding: 10px 18px; border-radius: 12px;
  background: linear-gradient(135deg, #e11d48, #b91c1c); color: white; border: none; cursor: pointer;
  font-weight: 700; font-size: 0.9rem; box-shadow: 0 6px 24px -6px rgba(225, 29, 72, 0.55);
  transition: transform .12s var(--ease-out-quint);
}
.estop:hover { transform: translateY(-1px); box-shadow: 0 10px 28px -8px rgba(225,29,72,.7); }
.estop:active { transform: scale(.97); }
.estop.shake { animation: shake .4s; }
.estop-glyph { font-size: 1.05rem; }
@keyframes shake { 0%,100%{transform:translateX(0)} 20%{transform:translateX(-5px)} 40%{transform:translateX(5px)} 60%{transform:translateX(-4px)} 80%{transform:translateX(4px)} }

.tl-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
@media (max-width: 1300px) { .tl-grid { grid-template-columns: 1fr; } }

.arm-panel { padding: 16px 18px; display: flex; flex-direction: column; gap: 12px; }
.ap-head { display: flex; align-items: center; gap: 10px; }
.ap-name { font-size: 1.05rem; font-weight: 700; color: var(--ink-primary); }
.ap-speed { margin-left: auto; font-size: 0.74rem; color: var(--ink-muted); }
.rec-chip { animation: pulseSoft 1s ease-in-out infinite; }

.ap-body { display: grid; grid-template-columns: minmax(0, 280px) minmax(0, 1fr); gap: 12px; align-items: stretch; }
@media (max-width: 900px) { .ap-body { grid-template-columns: 1fr; } }

.rings-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
.ring-cell { display: flex; justify-content: center; }

.ik-host { height: 280px; border-radius: 12px; overflow: hidden; border: 1px solid var(--line-divider); background: var(--bg-card); }

.aux-row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; align-items: center; padding: 10px 12px; border-radius: 12px; background: color-mix(in srgb, var(--bg-elevated) 70%, transparent); }
.aux { display: grid; grid-template-columns: 56px 1fr 56px; gap: 10px; align-items: center; }
.aux-label { font-size: 0.74rem; color: var(--ink-tertiary); font-weight: 600; }
.aux-val { font-size: 0.84rem; color: var(--ink-primary); font-weight: 700; text-align: right; }

.jog { -webkit-appearance: none; appearance: none; height: 10px; border-radius: 999px; background: var(--line-border); outline: none; cursor: pointer; }
.jog::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; width: 26px; height: 26px; border-radius: 50%; background: white; border: 3px solid var(--accent-blue); box-shadow: var(--shadow-soft); cursor: grab; transition: transform .1s; }
.jog::-webkit-slider-thumb:active { transform: scale(1.18); cursor: grabbing; }
.jog-grip { background: linear-gradient(90deg, var(--accent-emerald) 0%, var(--line-border) 50%, var(--accent-rose) 100%); }
.jog-speed::-webkit-slider-thumb { border-color: var(--ink-tertiary); }

.ap-actions { display: grid; grid-template-columns: repeat(6, 1fr); gap: 8px; }
.ap-actions :deep(.magnetic-btn) { width: 100%; padding: 8px 6px; font-size: 0.78rem; }
.ap-actions .primary :deep(.magnetic-btn), .ap-actions :deep(.magnetic-btn.primary) {
  background: linear-gradient(135deg, var(--accent-blue), color-mix(in srgb, var(--accent-blue) 70%, #0891b2));
  color: white; border-color: transparent;
}

.wp-row { display: flex; flex-direction: column; gap: 6px; }
.wp-head { display: flex; align-items: center; justify-content: space-between; }
.wp-play {
  padding: 4px 10px; border-radius: 8px; border: 1px solid var(--line-border);
  background: rgba(37, 99, 235, .08); color: var(--accent-blue); font-size: 0.72rem; font-weight: 700; cursor: pointer;
}
.wp-play:hover { background: rgba(37, 99, 235, .15); }
.wp-list { display: flex; flex-wrap: wrap; gap: 6px; }
.wp-chip {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 4px 6px 4px 10px; border-radius: 999px;
  background: var(--bg-elevated); border: 1px solid var(--line-border);
  font-size: 0.72rem; cursor: pointer; transition: all .15s;
}
.wp-chip:hover { transform: translateY(-1px); box-shadow: var(--shadow-soft); border-color: var(--accent-blue); }
.wp-chip.accent-amber { border-color: rgba(217, 119, 6, .35); }
.wp-chip.accent-blue  { border-color: rgba(37, 99, 235, .35); }
.wp-name { font-weight: 700; color: var(--ink-primary); }
.wp-mini { font-size: 0.62rem; color: var(--ink-muted); }
.wp-x { width: 18px; height: 18px; border-radius: 50%; border: none; background: rgba(225, 29, 72, .12); color: var(--accent-rose); cursor: pointer; font-weight: 800; line-height: 1; }
.wp-x:hover { background: rgba(225, 29, 72, .25); }

.clips-card { padding: 14px 16px; }
.clips-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px; }
.clips-count { font-size: 0.74rem; color: var(--ink-muted); }
.clips-empty { font-size: 0.84rem; color: var(--ink-muted); padding: 10px 2px; }
.clips-stack { display: flex; flex-direction: column; gap: 10px; }
.clip-row { display: grid; grid-template-columns: 180px 1fr auto; gap: 14px; align-items: center; padding: 8px 6px; border-radius: 12px; transition: background .2s; }
.clip-row:hover { background: color-mix(in srgb, var(--bg-elevated) 80%, transparent); }
.clip-meta { display: flex; flex-direction: column; gap: 2px; }
.clip-arm { font-weight: 700; font-size: 0.86rem; }
.clip-arm.accent-amber { color: var(--accent-amber); }
.clip-arm.accent-blue  { color: var(--accent-blue); }
.clip-btns { display: flex; gap: 6px; }
.cb { padding: 7px 12px; border-radius: 10px; border: 1px solid var(--line-border); background: var(--bg-card); color: var(--ink-secondary); cursor: pointer; font-size: 0.78rem; font-weight: 600; transition: all .15s; }
.cb:hover:not(:disabled) { border-color: var(--accent-blue); color: var(--accent-blue); transform: translateY(-1px); }
.cb-del { color: var(--accent-rose); border-color: color-mix(in srgb, var(--accent-rose) 30%, transparent); }
.cb-del:hover { background: rgba(225,29,72,.08); }
.cb:disabled { opacity: .45; cursor: not-allowed; }

@media (max-width: 900px) {
  .rings-grid { grid-template-columns: repeat(3, 1fr); }
  .ap-actions { grid-template-columns: repeat(3, 1fr); }
  .clip-row { grid-template-columns: 1fr; }
}
</style>
