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
  const activityStreamOpen = ref(false)
  const notifCenterOpen = ref(false)
  const kpiDrillKey = ref<string | null>(null)

  function openPalette()  { paletteOpen.value = true }
  function openHotkeys()  { hotkeysOpen.value = true }
  function openAbout()    { aboutOpen.value = true }
  function openDispatch() { dispatchOpen.value = true }
  function openActivityStream() { activityStreamOpen.value = true }
  function toggleActivityStream() { activityStreamOpen.value = !activityStreamOpen.value }
  function openNotifCenter() { notifCenterOpen.value = true }
  function toggleNotifCenter() { notifCenterOpen.value = !notifCenterOpen.value }
  function openKpiDrill(key: string) { kpiDrillKey.value = key }
  function closeKpiDrill() { kpiDrillKey.value = null }

  function togglePaused() { streamPaused.value = !streamPaused.value }

  function closeAll() {
    paletteOpen.value = false
    hotkeysOpen.value = false
    aboutOpen.value = false
    dispatchOpen.value = false
    activityStreamOpen.value = false
    notifCenterOpen.value = false
    kpiDrillKey.value = null
  }

  return {
    paletteOpen, hotkeysOpen, aboutOpen, dispatchOpen, streamPaused,
    activityStreamOpen, notifCenterOpen, kpiDrillKey,
    openPalette, openHotkeys, openAbout, openDispatch, togglePaused,
    openActivityStream, toggleActivityStream,
    openNotifCenter, toggleNotifCenter,
    openKpiDrill, closeKpiDrill,
    closeAll,
  }
})
