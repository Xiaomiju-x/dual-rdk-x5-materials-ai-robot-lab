<script setup lang="ts">
interface Props {
  title: string
  subtitle: string
  phase: string
  icon?: string
}
defineProps<Props>()
</script>

<template>
  <section class="page">
    <header class="page-header">
      <div>
        <h1 class="page-title">
          <span v-if="icon" class="page-icon">{{ icon }}</span>
          {{ title }}
        </h1>
        <p class="page-subtitle">{{ subtitle }}</p>
      </div>
      <span class="chip chip-info">{{ phase }}</span>
    </header>

    <div class="placeholder card-floating">
      <div class="placeholder-bg grid-bg"></div>
      <div class="placeholder-content">
        <span class="placeholder-mark">⬢</span>
        <div class="placeholder-title">Coming in {{ phase }}</div>
        <div class="placeholder-sub mono">scaffold ready · waiting on real data wiring</div>
      </div>
    </div>
  </section>
</template>

<style scoped>
.page {
  max-width: 1680px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  gap: 22px;
  animation: fadeUp 0.42s var(--ease-out-quint) both;
}
.page-header { display: flex; justify-content: space-between; align-items: flex-end; }
.page-icon { margin-right: 8px; }

.placeholder {
  position: relative;
  height: 480px;
  overflow: hidden;
  border-radius: 22px;
}
.placeholder-bg {
  position: absolute; inset: 0;
  background-color: #fafbfc;
  mask-image: radial-gradient(ellipse at center, black 0%, transparent 70%);
  -webkit-mask-image: radial-gradient(ellipse at center, black 0%, transparent 70%);
  opacity: 0.45;
}
.placeholder-content {
  position: relative; z-index: 1;
  height: 100%;
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  gap: 14px;
}
.placeholder-mark {
  font-size: 3rem;
  background: linear-gradient(135deg, var(--accent-blue), var(--accent-teal) 50%, var(--accent-emerald));
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
  filter: drop-shadow(0 4px 16px rgba(37, 99, 235, 0.25));
  animation: breath 4s ease-in-out infinite;
}
.placeholder-title {
  font-size: 1.4rem; font-weight: 700; letter-spacing: -0.015em;
  background: linear-gradient(135deg, #0b1220, #334155);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}
.placeholder-sub { font-size: 0.78rem; color: var(--ink-muted); letter-spacing: 0.02em; }
</style>
