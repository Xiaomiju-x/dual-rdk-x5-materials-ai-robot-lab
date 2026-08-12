<script setup lang="ts">
import { computed, ref } from 'vue'
import { useInputMode } from '@/composables/useInputMode'

interface Props {
  open: boolean
}
defineProps<Props>()
const emit = defineEmits<{ (e: 'close'): void }>()

const { isTouch } = useInputMode()
const mode = ref<'auto' | 'touch' | 'keyboard'>('auto')

const effectiveMode = computed<'touch' | 'keyboard'>(() => {
  if (mode.value !== 'auto') return mode.value
  return isTouch.value ? 'touch' : 'keyboard'
})

const KEYBOARD_SECTIONS = [
  {
    name: '导航',
    items: [
      { keys: ['⌘K', 'Ctrl+K'], desc: '打开命令面板' },
      { keys: ['/'],             desc: '打开命令面板 (alt)' },
      { keys: ['g', 'c'],        desc: '→ 驾驶舱',     chord: true },
      { keys: ['g', 'o'],        desc: '→ 触控操控',   chord: true },
      { keys: ['g', 'x'],        desc: '→ 三机协同',   chord: true },
      { keys: ['g', 'd'],        desc: '→ 答辩自演',   chord: true },
    ],
  },
  {
    name: '显示与操作',
    items: [
      { keys: ['t'], desc: '切换浅色 / 深色主题' },
      { keys: ['s'], desc: '切换音效反馈' },
      { keys: ['p'], desc: '冻结 / 恢复数据流' },
      { keys: ['?'], desc: '显示本帮助' },
      { keys: ['Esc'], desc: '关闭任意弹层' },
    ],
  },
  {
    name: '命令面板内',
    items: [
      { keys: ['↑', '↓'],   desc: '移动光标' },
      { keys: ['↵'],        desc: '执行' },
      { keys: ['Esc'],      desc: '关闭面板' },
    ],
  },
  {
    name: '答辩页',
    items: [
      { keys: ['Space'],         desc: '播放 / 暂停' },
      { keys: ['←', '→'],        desc: '上一幕 / 下一幕' },
      { keys: ['r'],             desc: '从头重播' },
    ],
  },
]

interface TouchCard {
  glyph: string
  title: string
  desc: string
  tone: 'blue' | 'teal' | 'emerald' | 'violet' | 'amber' | 'rose'
}

const TOUCH_SECTIONS: { name: string; cards: TouchCard[] }[] = [
  {
    name: '导航',
    cards: [
      { glyph: '☰', title: '左侧菜单', desc: '点 4 个标签直接切页, 当前页有渐变高亮 + 蓝色左轨', tone: 'blue' },
      { glyph: '🦾', title: '点 Logo', desc: '左上 WorkCockpit 标题 = 打开 About 信息卡', tone: 'violet' },
      { glyph: '⌘', title: 'Search 按钮', desc: '右上 ⌘K 按钮 = 命令面板, 搜页面 + 一键切主题/静音/冻结', tone: 'teal' },
    ],
  },
  {
    name: '操控 (Teleop 页)',
    cards: [
      { glyph: '🎚', title: '关节滑杆', desc: '拖动 6 个滑杆设目标角, 与实际偏差 >1° 行高亮橙底', tone: 'amber' },
      { glyph: '▶', title: '应用', desc: '蓝色"应用"= POST 下发目标角 + 速度 + 夹爪到该臂', tone: 'blue' },
      { glyph: '●', title: '录制 / 回放', desc: '"录制"按 100ms 采样轨迹, 停止生成片段, 底部可回放/删除', tone: 'emerald' },
      { glyph: '⇄', title: '镜像', desc: '把当前臂目标镜像到另一臂 (J1 取反), 双臂对称摆位', tone: 'teal' },
      { glyph: '⏹', title: '急停 ESTOP', desc: '右上红色大按钮 = 目标锁回实际姿态 + 停录制/回放', tone: 'rose' },
    ],
  },
  {
    name: '触控手势',
    cards: [
      { glyph: '✋', title: '驾驶舱 3D', desc: '单指拖 = 旋转 · 双指捏 = 缩放 · 双指拖 = 平移双臂舞台', tone: 'violet' },
      { glyph: '⏸', title: 'Pause Stream', desc: '顶栏"Pause"= 冻结当前数据画面 (适合截图/讲解)', tone: 'amber' },
      { glyph: '◉', title: '摄像头瓦片', desc: 'mock 下叠加 AprilTag 动效, 真实模式显示 MJPEG 直播', tone: 'emerald' },
      { glyph: '🎬', title: '答辩自演', desc: '6 幕自动播放, 底部 HUD 可上/下一幕/暂停/重播', tone: 'blue' },
    ],
  },
]

function backdropClick(evt: MouseEvent) {
  if ((evt.target as HTMLElement).classList.contains('hk-backdrop')) emit('close')
}
function onKey(evt: KeyboardEvent) {
  if (evt.key === 'Escape') emit('close')
}
</script>

<template>
  <transition name="hk">
    <div v-if="open" class="hk-backdrop" @click="backdropClick" @keydown="onKey" tabindex="-1">
      <div class="hk-panel glass-strong" role="dialog" aria-label="Help">
        <div class="hk-head">
          <div>
            <div class="hk-title">{{ effectiveMode === 'touch' ? 'Touch Gestures · 触控手册' : 'Keyboard Shortcuts' }}</div>
            <div class="hk-sub mono">{{ effectiveMode === 'touch' ? '检测到触控设备 · 平板/手机 kiosk 模式' : 'vim-style chord 支持 (g c / g i / …)' }}</div>
          </div>
          <div class="hk-head-actions">
            <div class="mode-seg">
              <button class="mode-btn" :class="{ active: effectiveMode === 'touch' }" @click="mode = 'touch'" aria-label="touch mode">
                ✋ Touch
              </button>
              <button class="mode-btn" :class="{ active: effectiveMode === 'keyboard' }" @click="mode = 'keyboard'" aria-label="keyboard mode">
                ⌨ Keyboard
              </button>
            </div>
            <button class="hk-close" @click="emit('close')" aria-label="close">×</button>
          </div>
        </div>

        <div v-if="effectiveMode === 'touch'" class="hk-body">
          <section v-for="s in TOUCH_SECTIONS" :key="s.name" class="touch-section">
            <div class="section-label">{{ s.name }}</div>
            <div class="touch-grid">
              <article v-for="(c, i) in s.cards" :key="i" class="touch-card" :class="`tone-${c.tone}`">
                <span class="touch-glyph">{{ c.glyph }}</span>
                <div class="touch-body">
                  <div class="touch-title">{{ c.title }}</div>
                  <div class="touch-desc">{{ c.desc }}</div>
                </div>
              </article>
            </div>
          </section>
        </div>

        <div v-else class="hk-body">
          <section v-for="s in KEYBOARD_SECTIONS" :key="s.name" class="hk-section">
            <div class="section-label">{{ s.name }}</div>
            <ul class="hk-list">
              <li v-for="(it, i) in s.items" :key="i" class="hk-row">
                <span class="hk-keys">
                  <template v-for="(k, ki) in it.keys" :key="ki">
                    <kbd class="kbd">{{ k }}</kbd>
                    <span v-if="ki < it.keys.length - 1" class="hk-sep">
                      {{ (it as any).chord ? 'then' : ' / ' }}
                    </span>
                  </template>
                </span>
                <span class="hk-desc">{{ it.desc }}</span>
              </li>
            </ul>
          </section>
        </div>

        <div class="hk-foot mono">
          <span v-if="effectiveMode === 'touch'">点空白处或右上 × 关闭 · 接键盘后可切换到 keyboard 视图</span>
          <span v-else>按 <kbd class="kbd kbd-inline">Esc</kbd> 或 <kbd class="kbd kbd-inline">?</kbd> 关闭</span>
        </div>
      </div>
    </div>
  </transition>
</template>

<style scoped>
.hk-backdrop {
  position: fixed; inset: 0;
  background: rgba(11, 18, 32, 0.40);
  backdrop-filter: blur(10px) saturate(160%);
  -webkit-backdrop-filter: blur(10px) saturate(160%);
  z-index: 100;
  display: flex; align-items: center; justify-content: center;
  padding: 24px;
  outline: none;
}
.hk-panel {
  width: 800px;
  max-width: 100%;
  max-height: 84vh;
  border-radius: 18px;
  display: flex; flex-direction: column;
  overflow: hidden;
}
.hk-head {
  display: flex; align-items: flex-start; justify-content: space-between;
  padding: 18px 22px 14px;
  border-bottom: 1px solid var(--line-divider);
  gap: 14px;
}
.hk-title { font-size: 1.1rem; font-weight: 700; color: var(--ink-primary); letter-spacing: -0.01em; }
.hk-sub { font-size: 0.72rem; color: var(--ink-tertiary); margin-top: 4px; }

.hk-head-actions { display: flex; align-items: center; gap: 10px; }
.mode-seg {
  display: inline-flex;
  background: var(--bg-elevated);
  border: 1px solid var(--line-border);
  border-radius: 8px;
  overflow: hidden;
}
.mode-btn {
  background: transparent; border: none;
  padding: 7px 12px;
  font-size: 0.74rem; color: var(--ink-tertiary);
  cursor: pointer; font-weight: 500;
}
.mode-btn:hover { color: var(--ink-primary); background: rgba(241, 245, 249, 0.6); }
.mode-btn.active {
  background: linear-gradient(135deg, var(--accent-blue), var(--accent-teal));
  color: white;
}
.hk-close {
  background: transparent; border: none;
  width: 36px; height: 36px;
  font-size: 1.4rem; color: var(--ink-tertiary);
  border-radius: 8px; cursor: pointer;
  display: flex; align-items: center; justify-content: center;
}
.hk-close:hover { background: rgba(15, 23, 42, 0.06); color: var(--ink-primary); }

.hk-body {
  flex: 1;
  overflow-y: auto;
  padding: 16px 22px;
  display: flex; flex-direction: column; gap: 18px;
}
.section-label { color: var(--ink-muted); padding-bottom: 6px; }

/* ---------- touch mode ---------- */
.touch-section { display: flex; flex-direction: column; gap: 8px; }
.touch-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 10px;
}
@media (max-width: 640px) { .touch-grid { grid-template-columns: 1fr; } }
.touch-card {
  display: grid;
  grid-template-columns: 40px 1fr;
  gap: 10px;
  padding: 10px 12px;
  background: var(--bg-elevated);
  border: 1px solid var(--line-divider);
  border-left: 3px solid;
  border-radius: 10px;
  align-items: flex-start;
}
.touch-card.tone-blue    { border-left-color: var(--accent-blue); }
.touch-card.tone-teal    { border-left-color: var(--accent-teal); }
.touch-card.tone-emerald { border-left-color: var(--accent-emerald); }
.touch-card.tone-violet  { border-left-color: var(--accent-violet); }
.touch-card.tone-amber   { border-left-color: var(--accent-amber); }
.touch-card.tone-rose    { border-left-color: var(--accent-rose); }
.touch-glyph {
  width: 36px; height: 36px;
  display: flex; align-items: center; justify-content: center;
  font-size: 1.25rem;
  border-radius: 9px;
}
.touch-card.tone-blue    .touch-glyph { color: var(--accent-blue);    background: rgba(37, 99, 235, 0.10); }
.touch-card.tone-teal    .touch-glyph { color: var(--accent-teal);    background: rgba(8, 145, 178, 0.10); }
.touch-card.tone-emerald .touch-glyph { color: var(--accent-emerald); background: rgba(5, 150, 105, 0.10); }
.touch-card.tone-violet  .touch-glyph { color: var(--accent-violet);  background: rgba(124, 58, 237, 0.10); }
.touch-card.tone-amber   .touch-glyph { color: var(--accent-amber);   background: rgba(217, 119, 6, 0.10); }
.touch-card.tone-rose    .touch-glyph { color: var(--accent-rose);    background: rgba(225, 29, 72, 0.10); }
.touch-title { font-size: 0.84rem; font-weight: 600; color: var(--ink-primary); }
.touch-desc { font-size: 0.72rem; color: var(--ink-tertiary); margin-top: 3px; line-height: 1.5; }

/* ---------- keyboard mode ---------- */
.hk-section { display: flex; flex-direction: column; gap: 6px; }
.hk-list { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 4px; }
.hk-row {
  display: grid;
  grid-template-columns: minmax(120px, auto) 1fr;
  gap: 16px;
  padding: 6px 0;
  align-items: center;
  border-bottom: 1px dashed var(--line-hairline);
  font-size: 0.8rem;
}
.hk-keys { display: inline-flex; align-items: center; gap: 4px; flex-wrap: wrap; }
.kbd {
  display: inline-flex; align-items: center; justify-content: center;
  min-width: 24px; height: 22px; padding: 0 7px;
  font-family: 'JetBrains Mono Variable', monospace;
  font-size: 0.72rem;
  background: var(--bg-elevated);
  border: 1px solid var(--line-border);
  border-bottom-width: 2px;
  border-radius: 5px;
  color: var(--ink-primary);
  font-weight: 600;
}
.kbd-inline { vertical-align: middle; margin: 0 2px; }
.hk-sep { color: var(--ink-muted); font-size: 0.7rem; }
.hk-desc { color: var(--ink-secondary); }

.hk-foot {
  padding: 12px 22px;
  border-top: 1px solid var(--line-divider);
  font-size: 0.72rem;
  color: var(--ink-tertiary);
  text-align: center;
}

.hk-enter-from { opacity: 0; }
.hk-enter-from .hk-panel { transform: translateY(-8px) scale(0.97); }
.hk-leave-to { opacity: 0; }
.hk-leave-to .hk-panel { transform: translateY(-4px) scale(0.98); }
.hk-enter-active, .hk-leave-active { transition: opacity 0.22s var(--ease-out-quint); }
.hk-enter-active .hk-panel, .hk-leave-active .hk-panel { transition: transform 0.22s var(--ease-out-quint); }
</style>
