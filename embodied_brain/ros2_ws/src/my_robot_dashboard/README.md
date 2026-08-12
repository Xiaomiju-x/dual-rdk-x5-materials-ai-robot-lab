# my_robot_dashboard — NavCockpit

具身脑实时驾驶舱前端 (Vue 3 + Three.js + ECharts) + FastAPI/rclpy 后端. 跑在车载 X5 (198.51.100.85:8890), 小米平板浏览器看.

公开规划与当前边界见[具身脑模块导航](../../../../docs/modules/EMBODIED_BRAIN.md)和[系统架构](../../../../docs/architecture/SYSTEM_ARCHITECTURE.md)。这里只记录该历史 Phase 进度。

---

## Phase 进度

- [x] **Phase 0** — 工具链 (本 commit). `pnpm install && pnpm dev` 跑得通空壳页.
- [ ] Phase 1 — 设计 token + 字体 + 玻璃工具类 + 主题切换
- [ ] Phase 2 — FastAPI 骨架 + mock_generator + WS hub
- [ ] Phase 3 — StatusBar + 路由 + 7 页空壳 + 页面转场
- [ ] Phase 4 — KPI 6 卡
- [ ] Phase 5 — Hero 3D SLAM Scene (Three.js)
- [ ] Phase 6 — 传感器 8 卡
- [ ] Phase 7 — AR 摄像头 + BPU 监控 + 烧结炉
- [ ] Phase 8 — 任务卡 + Gantt + 报警 + AI 脑链路
- [ ] Phase 9 — 音效 + 触感 + Lottie + PWA + GSAP 收尾
- [ ] Phase 10 — rclpy 接真 ROS2 topic (需 X5)

---

## 目录

```
my_robot_dashboard/
├── package.xml / setup.py / resource/      ROS2 Python 包定义
├── launch/dashboard.launch.py              Phase 2 起填实
├── my_robot_dashboard/                     Python 后端 (Phase 2 起)
└── frontend/                               Vue 3 + TS + Vite 工程
    ├── package.json
    ├── vite.config.ts
    ├── tsconfig.json
    ├── tailwind.config.ts
    ├── index.html
    └── src/
        ├── main.ts / App.vue / router.ts
        ├── pages/                          7 个 (Cockpit / Immersive / ...)
        ├── components/                     35+ Phase 4+ 填
        ├── composables/                    Phase 2+ 填
        ├── stores/                         Phase 2+ 填
        ├── styles/                         Phase 1 填
        └── assets/                         Phase 5+ 填
```

---

## Phase 0 本地验证

```bash
cd embodied_brain/ros2_ws/src/my_robot_dashboard/frontend
pnpm install                       # 或 npm install
pnpm dev                           # http://localhost:5173
```

预期: 空白浅色页, 顶部 "NavCockpit · empty", 左侧 7 个 router-link 可点 (各页只显示页名). 没数据流, 没动画, 没花哨 — 只是工具链验证.

---

## 依赖说明

**前端** (`frontend/package.json`):
- vue@3.5 + vue-router@4 + pinia@2
- typescript@5.5 + vite@5 + @vitejs/plugin-vue
- tailwindcss@4 + autoprefixer + postcss
- three@0.168 + @types/three (Phase 5)
- echarts@5.5 + vue-echarts (Phase 7)
- gsap@3.12 (Phase 4+ 动画)
- lottie-web@5 (Phase 9)
- @vueuse/core (composable 工具)
- vite-plugin-pwa + workbox (Phase 9)

**后端** (Python, Phase 2 起):
- fastapi + uvicorn + websockets + pydantic
- rclpy (Phase 10, X5 上有)
