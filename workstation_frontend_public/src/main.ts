import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './App.vue'
import router from './router'
import { useTelemetryStore } from './stores/telemetry'
import { useSettingsStore } from './stores/settings'
import { vTilt } from './directives/tilt'
import './style.css'

const app = createApp(App)
const pinia = createPinia()
app.use(pinia)
app.use(router)
app.directive('tilt', vTilt)

// Apply persisted theme before mount → no light-mode flash.
useSettingsStore(pinia)

app.mount('#app')

const telemetry = useTelemetryStore(pinia)
telemetry.connect()
