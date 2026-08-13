<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from 'vue'

// BeforeInstallPromptEvent — not in lib.dom typings, declare locally.
interface BeforeInstallPromptEvent extends Event {
  prompt: () => Promise<void>
  userChoice: Promise<{ outcome: 'accepted' | 'dismissed'; platform: string }>
}

const deferred = ref<BeforeInstallPromptEvent | null>(null)
const installed = ref(false)

function onPrompt(evt: Event) {
  evt.preventDefault()
  deferred.value = evt as BeforeInstallPromptEvent
}

function onInstalled() {
  installed.value = true
  deferred.value = null
}

async function install() {
  if (!deferred.value) return
  await deferred.value.prompt()
  const result = await deferred.value.userChoice
  if (result.outcome === 'accepted') installed.value = true
  deferred.value = null
}

onMounted(() => {
  window.addEventListener('beforeinstallprompt', onPrompt)
  window.addEventListener('appinstalled', onInstalled)
  // already installed (standalone display-mode)
  if (window.matchMedia('(display-mode: standalone)').matches) installed.value = true
})
onBeforeUnmount(() => {
  window.removeEventListener('beforeinstallprompt', onPrompt)
  window.removeEventListener('appinstalled', onInstalled)
})
</script>

<template>
  <button
    v-if="deferred && !installed"
    class="pwa-btn"
    title="安装为 PWA · 添加到主屏"
    @click="install"
  >
    <span class="pwa-glyph">⤓</span>
    <span class="pwa-label">Install</span>
  </button>
</template>

<style scoped>
.pwa-btn {
  display: inline-flex; align-items: center; gap: 5px;
  padding: 5px 10px;
  border-radius: 6px;
  border: 1px solid rgba(37, 99, 235, 0.30);
  background: linear-gradient(135deg, rgba(37, 99, 235, 0.08), rgba(8, 145, 178, 0.06));
  color: var(--accent-blue);
  font-size: 0.74rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.18s var(--ease-out-quint);
}
.pwa-btn:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px -4px rgba(37, 99, 235, 0.35);
  border-color: var(--accent-blue);
}
.pwa-glyph { font-size: 0.86rem; }
</style>
