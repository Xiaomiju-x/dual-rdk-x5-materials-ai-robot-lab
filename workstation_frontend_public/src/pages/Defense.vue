<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch, nextTick } from 'vue'
import gsap from 'gsap'
import DualArmScene from '@/components/three/DualArmScene.vue'
import Odometer from '@/components/fx/Odometer.vue'
import BorderBeam from '@/components/fx/BorderBeam.vue'
import GlowCard from '@/components/fx/GlowCard.vue'
import RingChart from '@/components/charts/RingChart.vue'
import PhaseWheel from '@/components/charts/PhaseWheel.vue'
import { useKeyboard } from '@/composables/useKeyboard'
import { useTelemetryStore } from '@/stores/telemetry'

const telemetry = useTelemetryStore()
const a01 = computed(() => telemetry.arm01.angles)
const a02 = computed(() => telemetry.arm02.angles)
const syncScore = computed(() => {
  let s = 0
  for (let i = 0; i < 6; i++) s += Math.abs((a01.value[i] ?? 0) - (a02.value[i] ?? 0))
  return Math.max(0, 1 - s / (90 * 6))
})

interface Scene { id: string; dur: number; kind: 'title' | 'arch' | 'hardware' | 'live' | 'caps' | 'closing'; label: string }
const SCENES: Scene[] = [
  { id: 's1', dur: 7,  kind: 'title',    label: '开场' },
  { id: 's2', dur: 10, kind: 'arch',     label: '三机异构' },
  { id: 's3', dur: 9,  kind: 'hardware', label: '硬件构成' },
  { id: 's4', dur: 12, kind: 'live',     label: '现场 LIVE' },
  { id: 's5', dur: 10, kind: 'caps',     label: '核心能力' },
  { id: 's6', dur: 8,  kind: 'closing',  label: '收尾' },
]

const idx = ref(0)
const playing = ref(true)
const elapsed = ref(0)
const stageEl = ref<HTMLElement | null>(null)
const fullscreen = ref(false)
let ticker: number | null = null

const cur = computed(() => SCENES[idx.value])
const progress = computed(() => Math.min(1, elapsed.value / cur.value.dur))
const totalDur = SCENES.reduce((a, s) => a + s.dur, 0)
const totalElapsed = computed(() => {
  let t = 0
  for (let i = 0; i < idx.value; i++) t += SCENES[i].dur
  return t + elapsed.value
})

function animateIn() {
  nextTick(() => {
    if (!stageEl.value) return
    const els = stageEl.value.querySelectorAll('[data-anim]')
    gsap.fromTo(els,
      { opacity: 0, y: 26, filter: 'blur(8px)' },
      { opacity: 1, y: 0, filter: 'blur(0px)', duration: 0.85, stagger: 0.1, ease: 'power3.out' })

    // big kinetic title — letter splits
    const heroes = stageEl.value.querySelectorAll('[data-kinetic]')
    heroes.forEach((h) => {
      const text = h.textContent || ''
      h.innerHTML = ''
      const frag = document.createDocumentFragment()
      ;[...text].forEach((ch) => {
        const s = document.createElement('span')
        s.textContent = ch === ' ' ? ' ' : ch
        s.style.display = 'inline-block'
        frag.appendChild(s)
      })
      h.appendChild(frag)
      gsap.fromTo(h.querySelectorAll('span'),
        { opacity: 0, y: 36, rotateX: -60 },
        { opacity: 1, y: 0, rotateX: 0, duration: 0.85, stagger: 0.04, ease: 'back.out(1.4)' })
    })

    // scale-in big-numbers
    const bigs = stageEl.value.querySelectorAll('[data-big]')
    gsap.fromTo(bigs, { scale: 0.7, opacity: 0 }, { scale: 1, opacity: 1, duration: 0.9, stagger: 0.15, ease: 'expo.out' })
  })
}
function goto(n: number) { idx.value = (n + SCENES.length) % SCENES.length; elapsed.value = 0; animateIn() }
function next() { goto(idx.value + 1) }
function prev() { goto(idx.value - 1) }
function toggle() { playing.value = !playing.value }
function restart() { idx.value = 0; elapsed.value = 0; playing.value = true; animateIn() }
async function toggleFs() {
  fullscreen.value = !fullscreen.value
  if (fullscreen.value && stageEl.value && document.fullscreenEnabled) {
    try { await stageEl.value.requestFullscreen() } catch {}
  } else if (!fullscreen.value && document.fullscreenElement) {
    try { await document.exitFullscreen() } catch {}
  }
}

onMounted(() => {
  animateIn()
  let last = performance.now()
  ticker = window.setInterval(() => {
    const now = performance.now()
    const dt = (now - last) / 1000; last = now
    if (!playing.value) return
    elapsed.value += dt
    if (elapsed.value >= cur.value.dur) next()
  }, 100)
  document.addEventListener('fullscreenchange', onFsChange)
})
function onFsChange() { fullscreen.value = !!document.fullscreenElement }
onBeforeUnmount(() => {
  if (ticker !== null) window.clearInterval(ticker)
  document.removeEventListener('fullscreenchange', onFsChange)
})
watch(idx, animateIn)

useKeyboard({
  ' ': () => toggle(),
  'arrowright': () => next(),
  'arrowleft': () => prev(),
  'r': () => restart(),
  'f': () => toggleFs(),
})

const CAPS = [
  { icon: '🦾', title: '双臂协同', desc: '双 myCobot 280-Pi 固定工位 · 6 DoF ×2 · 1Mbps 串口控制', accent: 'amber' as const },
  { icon: '👁', title: '双目视觉', desc: 'USB cam 1280×720 + AprilTag tag36h11 位姿估计', accent: 'teal' as const },
  { icon: '🧠', title: 'BPU 跨网', desc: '调 AI 脑 9 LLM + 车载脑 BPU 感知, HTTP 聚合', accent: 'violet' as const },
  { icon: '🎛', title: '触控操控', desc: 'Apple Watch 圆环 + 波形 scrub + 录制 + 镜像 + 急停', accent: 'blue' as const },
  { icon: '📦', title: 'PWA kiosk', desc: '小米 Pad 横屏全屏 · 零外部 CDN · 离线可用', accent: 'emerald' as const },
  { icon: '⚡', title: 'WebGL 渲染', desc: 'Vue3 + Three.js HDR/Bloom + ECharts + GSAP', accent: 'rose' as const },
]
const ARCH = [
  { name: 'AI 脑',     sub: '实验室端 RDK X5 8G', detail: '9 本地 LLM + 5 BPU slot · 配方预测 / 光谱 / R1 判决', accent: 'violet' as const },
  { name: '车载脑',    sub: '推车 RDK X5 8G',    detail: 'ROS2 + Nav2 + 8 BPU bin · SLAM / 取料 / 烧结监控',   accent: 'amber'  as const },
  { name: '双臂工位',  sub: 'Pi 4B 2GB ×2',     detail: 'dual myCobot 280-Pi · 固定精操作 · 与移动车协同',     accent: 'blue'   as const },
]
const HW_STATS = [
  { value: 2, suffix: '', label: 'myCobot 280-Pi', sub: '机械臂' },
  { value: 6, suffix: '', label: '自由度 / 臂', sub: 'DoF' },
  { value: 1000000, suffix: '', label: '串口速率', sub: 'bps · 1M baud' },
  { value: 720, suffix: 'p', label: '双目视频', sub: '1280×720 @ 30fps' },
]
</script>

<template>
  <div class="defense-pro" :class="{ fs: fullscreen }">
    <div ref="stageEl" class="stage">
      <!-- top chapter ticks -->
      <div class="ticks">
        <span v-for="(s, i) in SCENES" :key="s.id" class="tick"
              :class="{ done: i < idx, active: i === idx }"
              @click="goto(i)">
          <span class="tick-fill" :style="{ width: i < idx ? '100%' : i === idx ? progress * 100 + '%' : '0%' }"></span>
          <span class="tick-label">{{ s.label }}</span>
        </span>
      </div>

      <!-- scenes -->
      <Transition name="scene" mode="out-in">
        <div :key="cur.id" class="scene-wrap">
          <!-- TITLE -->
          <div v-if="cur.kind === 'title'" class="scene title-scene">
            <div class="bg-aurora-mask"></div>
            <span class="t-badge chip chip-info" data-anim>荧光具身智研 · 2026 嵌入式竞赛</span>
            <h1 class="t-title kinetic-title" data-kinetic>双臂工位 驾驶舱</h1>
            <p class="t-sub" data-anim>Dual myCobot 280-Pi · 固定工位精操作 · 三机异构协同</p>
            <div class="t-marks" data-anim>
              <span class="t-mark">🦾 6 DoF ×2</span>
              <span class="t-mark">👁 AprilTag pose</span>
              <span class="t-mark">🧠 BPU 跨网</span>
              <span class="t-mark">⚡ 实时 10Hz</span>
            </div>
            <div class="t-meter" data-anim>
              <span class="kv-mono">SPA · zero-CDN · PWA · Vue3 + Vite + Three.js</span>
            </div>
          </div>

          <!-- ARCHITECTURE — dual-arm cooperative theatre + 3 brain cards -->
          <div v-else-if="cur.kind === 'arch'" class="scene arch-scene">
            <h2 class="s-title kinetic-title" data-kinetic>三机异构 · 双臂为协同节点</h2>
            <div class="arch-grid">
              <div class="arch-3d">
                <DualArmScene />
                <div class="arch-3d-overlay">
                  <PhaseWheel :a01="a01" :a02="a02" :size="180" :soft-cap="90" />
                </div>
              </div>
              <div class="arch-cards">
                <GlowCard v-for="a in ARCH" :key="a.name" :accent="a.accent" class="arch-card" data-anim>
                  <div class="arch-name" :class="`accent-${a.accent}`">{{ a.name }}</div>
                  <div class="arch-sub mono">{{ a.sub }}</div>
                  <div class="arch-detail">{{ a.detail }}</div>
                </GlowCard>
              </div>
            </div>
          </div>

          <!-- HARDWARE — odometers -->
          <div v-else-if="cur.kind === 'hardware'" class="scene hw-scene">
            <h2 class="s-title kinetic-title" data-kinetic>硬件构成</h2>
            <div class="hw-grid">
              <div v-for="h in HW_STATS" :key="h.label" class="hw-big" data-big>
                <BorderBeam :duration="16" />
                <div class="hw-num">
                  <Odometer :value="h.value" :suffix="h.suffix" />
                </div>
                <div class="hw-lbl">{{ h.label }}</div>
                <div class="hw-sub kv-mono">{{ h.sub }}</div>
              </div>
            </div>
            <div class="hw-foot kv-mono" data-anim>
              arm01 @ 192.0.2.64 · arm02 @ 192.0.2.136 · K70 DHCP · USB 1280×720@30fps MJPG
            </div>
          </div>

          <!-- LIVE -->
          <div v-else-if="cur.kind === 'live'" class="scene live-scene">
            <DualArmScene :embed="true" />
            <div class="sync-hud" data-anim>
              <RingChart :value="syncScore" accent="emerald"
                         :label="(syncScore * 100).toFixed(0) + '%'" caption="sync" :size="92" />
            </div>
            <div class="live-caps">
              <div class="live-cap-tag" data-anim>实时双臂姿态 · 数据驱动 3D 孪生 · HDR + Bloom</div>
              <div class="live-cap-meta" data-anim>
                <span class="kv-mono">轮询 10 Hz</span>
                <span class="kv-mono">Three.js r168</span>
                <span class="kv-mono">UnrealBloom + PMREM</span>
              </div>
            </div>
          </div>

          <!-- CAPABILITIES -->
          <div v-else-if="cur.kind === 'caps'" class="scene caps-scene">
            <h2 class="s-title kinetic-title" data-kinetic>核心能力</h2>
            <div class="caps-grid">
              <GlowCard v-for="c in CAPS" :key="c.title" :accent="c.accent" class="cap-card" data-anim>
                <span class="cap-icon">{{ c.icon }}</span>
                <div class="cap-title">{{ c.title }}</div>
                <div class="cap-desc">{{ c.desc }}</div>
              </GlowCard>
            </div>
          </div>

          <!-- CLOSING -->
          <div v-else class="scene closing-scene">
            <div class="bg-aurora-mask"></div>
            <div class="cl-mark" data-anim>🦾</div>
            <h2 class="cl-title kinetic-title" data-kinetic>大脑 + 小脑 + 巧手</h2>
            <p class="cl-sub" data-anim>AI 脑出脑力 · 车载脑出执行力 · 双臂工位出精细操作</p>
            <div class="cl-tag chip chip-ok" data-anim>WorkCockpit · 谢谢观看</div>
          </div>
        </div>
      </Transition>

      <!-- HUD -->
      <div class="hud">
        <button class="hud-btn" @click="prev" title="prev">⏮</button>
        <button class="hud-btn hud-play" @click="toggle" :title="playing ? 'pause (space)' : 'play (space)'">{{ playing ? '⏸' : '▶' }}</button>
        <button class="hud-btn" @click="next" title="next">⏭</button>
        <button class="hud-btn" @click="restart" title="restart (r)">↺</button>
        <button class="hud-btn fs-btn" @click="toggleFs" :title="fullscreen ? 'exit fullscreen' : 'fullscreen (f)'">⛶</button>
        <div class="hud-meta mono">
          {{ idx + 1 }}/{{ SCENES.length }} · {{ Math.round(elapsed) }}s/{{ cur.dur }}s ·
          <span class="hud-total">{{ Math.round(totalElapsed) }}s / {{ totalDur }}s</span>
        </div>
        <div class="hud-hint mono">空格 播放/暂停 · ← → 切换 · r 重播 · f 全屏</div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.defense-pro { height: 100%; min-height: 0; display: flex; }
.defense-pro.fs .stage { border-radius: 0; }
.stage {
  position: relative; flex: 1; overflow: hidden;
  display: flex; flex-direction: column;
  border-radius: 20px;
  background: linear-gradient(180deg, color-mix(in srgb, var(--bg-elevated) 60%, transparent), color-mix(in srgb, var(--bg-base) 80%, transparent));
  border: 1px solid var(--line-divider);
  box-shadow: var(--shadow-elevated);
}

/* progress ticks */
.ticks { position: absolute; top: 0; left: 0; right: 0; display: flex; gap: 4px; padding: 10px 14px; z-index: 5; }
.tick { flex: 1; height: 26px; position: relative; cursor: pointer; display: flex; align-items: flex-end; }
.tick::before { content: ''; position: absolute; left: 0; right: 0; bottom: 0; height: 4px; border-radius: 999px; background: color-mix(in srgb, var(--ink-muted) 22%, transparent); }
.tick-fill { display: block; position: absolute; left: 0; bottom: 0; height: 4px; border-radius: 999px;
  background: linear-gradient(90deg, var(--accent-blue), var(--accent-teal));
  transition: width 0.1s linear;
  box-shadow: 0 0 8px color-mix(in srgb, var(--accent-blue) 65%, transparent);
}
.tick-label { position: absolute; top: -4px; left: 6px; font-size: 0.66rem; color: var(--ink-muted); font-family: 'JetBrains Mono Variable', monospace; opacity: 0; transition: opacity .2s; }
.tick.active .tick-label, .tick:hover .tick-label { opacity: 1; color: var(--accent-blue); }

.scene-wrap { flex: 1; position: relative; min-height: 0; }
.scene { position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 64px 48px; gap: 18px; text-align: center; perspective: 1200px; }

.scene-enter-from, .scene-leave-to { opacity: 0; transform: scale(.98); filter: blur(8px); }
.scene-enter-active, .scene-leave-active { transition: opacity .55s var(--ease-out-expo), transform .8s var(--ease-out-expo), filter .55s var(--ease-out-expo); }

/* aurora wash — adds a subtle vignette over the global aurora */
.bg-aurora-mask {
  position: absolute; inset: 0; pointer-events: none; z-index: 0;
  background:
    radial-gradient(circle at 50% 30%, rgba(37,99,235,.07), transparent 50%),
    radial-gradient(circle at 80% 80%, rgba(8,145,178,.06), transparent 55%),
    radial-gradient(circle at 20% 75%, rgba(124,58,237,.06), transparent 60%);
}
[data-theme='dark'] .bg-aurora-mask {
  background:
    radial-gradient(circle at 50% 30%, rgba(96,165,250,.14), transparent 50%),
    radial-gradient(circle at 80% 80%, rgba(34,211,238,.10), transparent 55%),
    radial-gradient(circle at 20% 75%, rgba(167,139,250,.10), transparent 60%);
}

/* TITLE */
.title-scene { gap: 22px; }
.t-title { font-size: clamp(3rem, 7vw, 6rem); font-weight: 800; letter-spacing: -0.04em; line-height: 1.02; perspective: 800px; }
.t-sub { font-size: clamp(1rem, 1.4vw, 1.2rem); color: var(--ink-tertiary); }
.t-marks { display: flex; gap: 16px; margin-top: 8px; flex-wrap: wrap; justify-content: center; }
.t-mark { font-size: 0.9rem; color: var(--ink-secondary); padding: 9px 18px; border-radius: 999px;
  background: var(--bg-glass); border: 1px solid var(--line-divider);
  backdrop-filter: blur(12px) saturate(170%);
}
.t-meter { margin-top: 16px; }

.s-title { font-size: clamp(2rem, 4vw, 3rem); font-weight: 800; letter-spacing: -0.03em; }

/* ARCHITECTURE */
.arch-scene { padding: 56px 36px; gap: 24px; }
.arch-grid { display: grid; grid-template-columns: 1fr 340px; gap: 22px; width: 100%; max-width: 1180px; flex: 1; align-items: stretch; min-height: 0; }
.arch-3d { position: relative; min-height: 360px; border-radius: 18px; overflow: hidden; border: 1px solid var(--line-divider); background: color-mix(in srgb, var(--bg-card) 80%, transparent); }
.arch-3d-overlay { position: absolute; bottom: 14px; right: 14px; z-index: 4; padding: 8px; border-radius: 14px;
  background: var(--bg-glass-strong); backdrop-filter: blur(16px) saturate(160%); border: 1px solid var(--line-divider);
  box-shadow: var(--shadow-soft); }
.arch-cards { display: flex; flex-direction: column; gap: 12px; text-align: left; }
.arch-card { padding: 16px 18px; }
.arch-name { font-size: 1.2rem; font-weight: 700; }
.arch-name.accent-violet { color: var(--accent-violet); }
.arch-name.accent-amber { color: var(--accent-amber); }
.arch-name.accent-blue { color: var(--accent-blue); }
.arch-sub { font-size: 0.72rem; color: var(--ink-muted); margin: 4px 0 8px; }
.arch-detail { font-size: 0.85rem; color: var(--ink-secondary); line-height: 1.55; }

/* HARDWARE */
.hw-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 18px; width: 100%; max-width: 1120px; }
.hw-big { position: relative; padding: 28px 24px; display: flex; flex-direction: column; align-items: center; gap: 6px;
  border-radius: 18px; background: var(--bg-card); border: 1px solid var(--line-divider);
  box-shadow: var(--shadow-elevated);
  overflow: hidden;
}
.hw-num { font-family: 'JetBrains Mono Variable', monospace; font-size: clamp(2.2rem, 4vw, 3.4rem); font-weight: 700; letter-spacing: -0.03em;
  background: linear-gradient(135deg, var(--accent-blue), var(--accent-teal) 50%, var(--accent-violet));
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hw-lbl { font-size: 0.92rem; color: var(--ink-primary); font-weight: 600; }
.hw-sub { font-size: 0.72rem; color: var(--ink-tertiary); }
.hw-foot { margin-top: 8px; color: var(--ink-tertiary); }

/* LIVE */
.live-scene { padding: 0; }
.live-scene > :first-child { position: absolute; inset: 0; }
.live-caps { position: absolute; bottom: 80px; left: 50%; transform: translateX(-50%); z-index: 3; display: flex; flex-direction: column; align-items: center; gap: 10px; }
.live-cap-tag { padding: 12px 26px; border-radius: 999px; background: var(--bg-glass-strong); border: 1px solid var(--line-divider); backdrop-filter: blur(20px) saturate(160%); font-size: 1rem; font-weight: 600; color: var(--ink-primary); box-shadow: var(--shadow-elevated); }
.live-cap-meta { display: flex; gap: 14px; flex-wrap: wrap; justify-content: center; }
.live-cap-meta .kv-mono { padding: 4px 10px; border-radius: 6px; background: var(--bg-elevated); border: 1px solid var(--line-divider); }
.sync-hud { position: absolute; top: 72px; right: 24px; z-index: 4; padding: 8px 10px; border-radius: 16px;
  background: var(--bg-glass-strong); backdrop-filter: blur(18px) saturate(180%); border: 1px solid var(--line-divider); box-shadow: var(--shadow-elevated); }

/* CAPS */
.caps-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; width: 100%; max-width: 1100px; }
.cap-card { padding: 22px 20px; text-align: left; min-height: 168px; }
.cap-icon { font-size: 2rem; }
.cap-title { font-size: 1.1rem; font-weight: 700; color: var(--ink-primary); margin: 10px 0 4px; letter-spacing: -0.01em; }
.cap-desc { font-size: 0.86rem; color: var(--ink-tertiary); line-height: 1.55; }

/* CLOSING */
.closing-scene { gap: 14px; }
.cl-mark { font-size: 5rem; filter: drop-shadow(0 6px 24px rgba(37,99,235,.4)); }
.cl-title { font-size: clamp(2.2rem, 5vw, 3.6rem); font-weight: 800; }
.cl-sub { font-size: 1.05rem; color: var(--ink-tertiary); }
.cl-tag { font-size: 0.95rem; padding: 12px 22px; margin-top: 8px; }

/* HUD */
.hud { position: absolute; bottom: 18px; left: 50%; transform: translateX(-50%); z-index: 6; display: flex; align-items: center; gap: 8px; padding: 8px 14px; border-radius: 999px; background: var(--bg-glass-strong); backdrop-filter: blur(20px) saturate(170%); -webkit-backdrop-filter: blur(20px) saturate(170%); border: 1px solid var(--line-divider); box-shadow: var(--shadow-elevated); }
.hud-btn { width: 38px; height: 38px; border-radius: 50%; border: 1px solid var(--line-border); background: var(--bg-card); color: var(--ink-secondary); cursor: pointer; font-size: 0.92rem; transition: all 0.15s var(--ease-out-quint); }
.hud-btn:hover { border-color: var(--accent-blue); color: var(--accent-blue); transform: translateY(-1px); box-shadow: 0 4px 12px -4px rgba(37,99,235,.5); }
.hud-play { background: linear-gradient(135deg, var(--accent-blue), var(--accent-teal)); color: white; border-color: transparent; }
.fs-btn { background: var(--bg-elevated); }
.hud-meta { font-size: 0.74rem; color: var(--ink-tertiary); margin-left: 6px; }
.hud-total { color: var(--ink-secondary); font-weight: 600; }
.hud-hint { font-size: 0.66rem; color: var(--ink-muted); margin-left: 4px; }

@media (max-width: 1080px) {
  .arch-grid { grid-template-columns: 1fr; }
  .arch-3d { min-height: 280px; }
  .hw-grid, .caps-grid { grid-template-columns: 1fr 1fr; }
}
@media (max-width: 720px) {
  .hud-hint, .tick-label { display: none; }
  .hw-grid, .caps-grid { grid-template-columns: 1fr; }
}
</style>
