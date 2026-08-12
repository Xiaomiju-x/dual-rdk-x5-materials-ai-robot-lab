<script setup lang="ts">
import { useRouter } from 'vue-router'
import { useUiStore } from '@/stores/ui'
import { useInputMode } from '@/composables/useInputMode'
import KineticTitle from '@/components/premium/KineticTitle.vue'
import MagneticBtn from '@/components/premium/MagneticBtn.vue'

const router = useRouter()
const ui = useUiStore()
const { isTouch } = useInputMode()
</script>

<template>
  <section class="nf-wrap">
    <div class="nf-card glass-strong">
      <div class="nf-glyph">⬢</div>
      <div class="nf-code mono">404</div>
      <h1 class="nf-title">
        <KineticTitle text="该坐标不在地图上" gradient="aurora" />
      </h1>
      <p class="nf-sub">这条路径不在路由表里. 也许是手滑, 也许 phase 还没填实.</p>
      <div class="nf-actions">
        <MagneticBtn :strength="0.30" :radius="100">
          <button class="btn btn-primary" @click="router.push('/')">⌂ Back to Cockpit</button>
        </MagneticBtn>
        <button class="btn" @click="router.back()">← Back</button>
        <button class="btn" @click="ui.openPalette">⌕ Search pages</button>
      </div>
      <div class="nf-hint">
        {{ isTouch ? '点左侧菜单可直跳任一页 · 右上 ? 看完整触控手册' : '提示: 按 ⌘K 打开 palette · g+c 直跳 Cockpit · ? 全部快捷键' }}
      </div>
    </div>
  </section>
</template>

<style scoped>
.nf-wrap {
  display: flex; align-items: center; justify-content: center;
  height: 100%;
  padding: 40px 24px;
}
.nf-card {
  width: 480px; max-width: 100%;
  padding: 36px 36px 30px;
  border-radius: 20px;
  display: flex; flex-direction: column;
  align-items: center; text-align: center;
  gap: 12px;
}
.nf-glyph {
  font-size: 3.4rem;
  background: linear-gradient(135deg, var(--accent-blue), var(--accent-violet) 60%, var(--accent-rose));
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
  filter: drop-shadow(0 6px 16px rgba(124, 58, 237, 0.25));
  animation: breath 3.4s ease-in-out infinite;
  margin-bottom: 4px;
}
.nf-code {
  font-size: 0.78rem;
  color: var(--ink-tertiary);
  letter-spacing: 0.35em;
  text-transform: uppercase;
}
.nf-title {
  font-size: 1.4rem;
  font-weight: 700;
  margin: 6px 0 0;
  letter-spacing: -0.01em;
  color: var(--ink-primary);
}
.nf-sub {
  font-size: 0.86rem;
  color: var(--ink-tertiary);
  margin: 0;
  text-wrap: balance;
}
.nf-actions {
  display: flex; gap: 10px; margin-top: 12px;
}
.nf-hint {
  margin-top: 8px;
  font-size: 0.7rem;
  color: var(--ink-muted);
  padding-top: 10px;
  border-top: 1px solid var(--line-divider);
  width: 100%;
}
</style>
