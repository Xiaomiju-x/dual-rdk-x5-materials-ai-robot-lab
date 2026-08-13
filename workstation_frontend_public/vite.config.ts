import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { VitePWA } from 'vite-plugin-pwa'
import path from 'node:path'

// WorkCockpit Vite config.
// Dev:  npm run dev   → http://localhost:5174 (proxies /api + /video to Flask on 8896)
// Build: npm run build → dist/ (served by Flask app.py at / when present)
export default defineConfig({
  base: '/',
  plugins: [
    vue(),
    VitePWA({
      registerType: 'autoUpdate',
      includeAssets: ['favicon.svg'],
      manifest: {
        name: 'WorkCockpit · 双臂工位',
        short_name: 'WorkCockpit',
        description: 'myCobot 280-Pi 双臂工位实时驾驶舱',
        theme_color: '#2563eb',
        background_color: '#f6f8fb',
        display: 'standalone',
        orientation: 'landscape',
        icons: [],
      },
    }),
  ],
  resolve: {
    alias: {
      '@': path.resolve(import.meta.dirname, 'src'),
    },
  },
  server: {
    host: '0.0.0.0',
    port: 5174,
    strictPort: true,
    proxy: {
      '/api': { target: 'http://localhost:8896', changeOrigin: true },
      '/video': { target: 'http://localhost:8896', changeOrigin: true },
      '/socket.io': { target: 'http://localhost:8896', ws: true, changeOrigin: true },
    },
  },
  build: {
    outDir: 'dist',
    target: 'es2022',
    sourcemap: false,
    chunkSizeWarningLimit: 1200,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('/node_modules/three/')) return 'three'
          if (id.includes('/node_modules/echarts/') || id.includes('/node_modules/vue-echarts/')) return 'echarts'
          if (id.includes('/node_modules/gsap/')) return 'gsap'
        },
      },
    },
  },
})
