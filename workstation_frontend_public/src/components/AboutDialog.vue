<script setup lang="ts">
interface Props { open: boolean }
defineProps<Props>()
const emit = defineEmits<{ (e: 'close'): void }>()

const FACTS = [
  { k: '硬件', v: '双 myCobot 280-Pi · Pi 4B 2GB ×2 · atom 固件' },
  { k: 'arm01', v: 'er@192.0.2.64 · 服务端 + arm02 聚合' },
  { k: 'arm02', v: 'er@192.0.2.136 · 轻量服务 :8891' },
  { k: '感知', v: 'USB cam 1280×720 · AprilTag tag36h11' },
  { k: '夹爪', v: 'MG996R PWM @ atom G22 · 50Hz' },
  { k: '协同', v: 'AI 脑 (9 LLM) + 车载脑 (BPU) + 工位' },
]
const STACK = ['Vue 3', 'Vite', 'TypeScript', 'Pinia', 'Three.js', 'ECharts', 'GSAP', 'Tailwind']
</script>

<template>
  <transition name="ab">
    <div v-if="open" class="ab-backdrop" @click.self="emit('close')">
      <div class="ab-panel glass-strong" role="dialog" aria-label="About">
        <div class="ab-head">
          <span class="ab-mark">🦾</span>
          <div>
            <div class="ab-title">WorkCockpit</div>
            <div class="ab-sub mono">双臂工位实时驾驶舱 · 荧光具身智研</div>
          </div>
          <button class="ab-close" @click="emit('close')">×</button>
        </div>
        <div class="ab-body">
          <p class="ab-lede text-pretty">
            myCobot 280-Pi 双臂固定工位的统一操作面板:3D 实时姿态、触控操控、三机协同与答辩自演,
            为小米 Pad 7S Pro 横屏 kiosk 设计。
          </p>
          <div class="ab-facts">
            <div v-for="f in FACTS" :key="f.k" class="ab-fact">
              <span class="ab-fact-k section-label">{{ f.k }}</span>
              <span class="ab-fact-v mono">{{ f.v }}</span>
            </div>
          </div>
          <div class="ab-stack">
            <span v-for="s in STACK" :key="s" class="chip">{{ s }}</span>
          </div>
        </div>
        <div class="ab-foot mono">点空白处或 × 关闭</div>
      </div>
    </div>
  </transition>
</template>

<style scoped>
.ab-backdrop {
  position: fixed; inset: 0; z-index: 100; display: flex; align-items: center; justify-content: center; padding: 24px;
  background: rgba(11, 18, 32, 0.40); backdrop-filter: blur(10px) saturate(160%); -webkit-backdrop-filter: blur(10px) saturate(160%);
}
.ab-panel { width: 560px; max-width: 100%; border-radius: 20px; overflow: hidden; display: flex; flex-direction: column; }
.ab-head { display: flex; align-items: center; gap: 12px; padding: 20px 22px 16px; border-bottom: 1px solid var(--line-divider); }
.ab-mark { font-size: 1.8rem; }
.ab-title { font-size: 1.15rem; font-weight: 700; color: var(--ink-primary); }
.ab-sub { font-size: 0.72rem; color: var(--ink-tertiary); margin-top: 2px; }
.ab-close { margin-left: auto; width: 34px; height: 34px; border: none; background: transparent; font-size: 1.4rem; color: var(--ink-tertiary); border-radius: 8px; cursor: pointer; }
.ab-close:hover { background: rgba(15,23,42,0.06); color: var(--ink-primary); }
.ab-body { padding: 18px 22px; display: flex; flex-direction: column; gap: 16px; }
.ab-lede { font-size: 0.84rem; color: var(--ink-secondary); line-height: 1.6; }
.ab-facts { display: grid; grid-template-columns: 1fr; gap: 8px; }
.ab-fact { display: grid; grid-template-columns: 64px 1fr; gap: 12px; align-items: baseline; padding: 6px 0; border-bottom: 1px dashed var(--line-hairline); }
.ab-fact-k { color: var(--ink-muted); }
.ab-fact-v { font-size: 0.76rem; color: var(--ink-secondary); }
.ab-stack { display: flex; flex-wrap: wrap; gap: 6px; }
.ab-foot { padding: 12px 22px; border-top: 1px solid var(--line-divider); text-align: center; font-size: 0.72rem; color: var(--ink-tertiary); }
.ab-enter-from, .ab-leave-to { opacity: 0; }
.ab-enter-from .ab-panel, .ab-leave-to .ab-panel { transform: translateY(-8px) scale(0.97); }
.ab-enter-active, .ab-leave-active { transition: opacity 0.22s var(--ease-out-quint); }
.ab-enter-active .ab-panel, .ab-leave-active .ab-panel { transition: transform 0.22s var(--ease-out-quint); }
</style>
