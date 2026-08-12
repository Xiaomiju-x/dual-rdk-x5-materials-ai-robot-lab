import { defineStore } from 'pinia'
import { ref } from 'vue'
import type { Tone } from '@/types/telemetry'

export interface Toast {
  id: string
  tone: Tone
  title: string
  detail?: string
  durationMs?: number
}

export const useToastStore = defineStore('toast', () => {
  const toasts = ref<Toast[]>([])
  let nextId = 0

  function push(t: Omit<Toast, 'id'>): string {
    const id = `t-${++nextId}-${Date.now()}`
    const toast: Toast = { id, durationMs: 4200, ...t }
    toasts.value = [...toasts.value, toast]
    if ((toast.durationMs ?? 0) > 0) {
      window.setTimeout(() => dismiss(id), toast.durationMs)
    }
    return id
  }

  function dismiss(id: string) {
    toasts.value = toasts.value.filter((t) => t.id !== id)
  }

  function clear() { toasts.value = [] }

  return { toasts, push, dismiss, clear }
})
