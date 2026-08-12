<script setup lang="ts">
import { useRouter } from 'vue-router'
const router = useRouter()
</script>

<template>
  <div class="nf">
    <div class="nf-mark">🦾</div>
    <div class="nf-code kpi-num">404</div>
    <p class="nf-text">这个工位还没建好</p>
    <button class="btn btn-primary" @click="router.push('/')">← 回驾驶舱</button>
  </div>
</template>

<style scoped>
.nf { height: 100%; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 14px; }
.nf-mark { font-size: 3rem; opacity: 0.7; }
.nf-code { font-size: 4rem; color: var(--ink-muted); }
.nf-text { color: var(--ink-tertiary); font-size: 1rem; }
</style>
