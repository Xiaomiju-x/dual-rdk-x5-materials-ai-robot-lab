/**
 * Tiny UI store — coordinates global modals/popovers from any component.
 * (Avoids prop-drilling or event buses.)
 */
import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useUiStore = defineStore('ui', () => {
  const paletteOpen = ref(false)
  const hotkeysOpen = ref(false)
  const aboutOpen = ref(false)
  const dispatchOpen = ref(false)
  const streamPaused = ref(false)

  function openPalette()  { paletteOpen.value = true }
  function openHotkeys()  { hotkeysOpen.value = true }
  function openAbout()    { aboutOpen.value = true }
  function openDispatch() { dispatchOpen.value = true }

  function togglePaused() { streamPaused.value = !streamPaused.value }

  function closeAll() {
    paletteOpen.value = false
    hotkeysOpen.value = false
    aboutOpen.value = false
    dispatchOpen.value = false
  }

  return {
    paletteOpen, hotkeysOpen, aboutOpen, dispatchOpen, streamPaused,
    openPalette, openHotkeys, openAbout, openDispatch, togglePaused,
    closeAll,
  }
})
