import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { VitePWA } from 'vite-plugin-pwa'
import path from 'node:path'

// NavCockpit Vite config — Phase 0 minimal
// Dev:  pnpm dev   → http://localhost:5173 (proxies /api + /ws to FastAPI on 8890)
// Build: pnpm build → dist/ (served by FastAPI as static)
export default defineConfig({
  plugins: [
    vue(),
    VitePWA({
      registerType: 'autoUpdate',
      includeAssets: ['favicon.svg'],
      manifest: {
        name: 'NavCockpit',
        short_name: 'NavCockpit',
        description: '具身脑实时驾驶舱',
        theme_color: '#2563eb',
        background_color: '#fafbfc',
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
    port: 5173,
    strictPort: true,
    proxy: {
      '/api': { target: 'http://localhost:8890', changeOrigin: true },
      '/ws': { target: 'ws://localhost:8890', ws: true, changeOrigin: true },
    },
  },
  build: {
    outDir: 'dist',
    target: 'es2022',
    sourcemap: true,
    chunkSizeWarningLimit: 1000,
    rollupOptions: {
      output: {
        manualChunks(id) {
          // Phase 5+ heavy libs split for parallel load
          if (id.includes('/node_modules/three/')) return 'three'
          if (id.includes('/node_modules/echarts/') || id.includes('/node_modules/vue-echarts/')) return 'echarts'
          if (id.includes('/node_modules/gsap/')) return 'gsap'
          if (id.includes('/node_modules/lottie-web/')) return 'lottie'
        },
      },
    },
  },
})
