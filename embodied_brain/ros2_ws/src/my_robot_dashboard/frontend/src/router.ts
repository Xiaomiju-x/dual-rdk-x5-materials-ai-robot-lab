import { createRouter, createWebHistory, type RouteRecordRaw } from 'vue-router'

// 7 pages — all placeholders in Phase 0, filled in later phases.
const routes: RouteRecordRaw[] = [
  {
    path: '/',
    name: 'cockpit',
    component: () => import('@/pages/Cockpit.vue'),
    meta: { title: 'Cockpit · 主驾驶舱', icon: '🛰' },
  },
  {
    path: '/immersive',
    name: 'immersive',
    component: () => import('@/pages/Immersive.vue'),
    meta: { title: 'Immersive 3D · 沉浸大图', icon: '🎬' },
  },
  {
    path: '/perception',
    name: 'perception',
    component: () => import('@/pages/Perception.vue'),
    meta: { title: 'AR Perception · 摄像头', icon: '📷' },
  },
  {
    path: '/inspector',
    name: 'inspector',
    component: () => import('@/pages/Inspector.vue'),
    meta: { title: 'Inspector · 传感器诊断', icon: '🔬' },
  },
  {
    path: '/timeline',
    name: 'timeline',
    component: () => import('@/pages/Timeline.vue'),
    meta: { title: 'Timeline · 时间轴', icon: '⏱' },
  },
  {
    path: '/twin',
    name: 'twin',
    component: () => import('@/pages/Twin.vue'),
    meta: { title: 'Digital Twin · 数字孪生', icon: '🏛' },
  },
  {
    path: '/planner',
    name: 'planner',
    component: () => import('@/pages/Planner.vue'),
    meta: { title: 'What-If Planner · 规划沙盒', icon: '🧭' },
  },
  {
    path: '/topology',
    name: 'topology',
    component: () => import('@/pages/Topology.vue'),
    meta: { title: 'Topology · 系统拓扑', icon: '🕸' },
  },
  // ---- 第 3 期 (2026-06-11): cockpit_bridge 真车数据 4 页 ----
  {
    path: '/missions',
    name: 'missions',
    component: () => import('@/pages/Missions.vue'),
    meta: { title: 'Missions · 任务编排', icon: '🌳' },
  },
  {
    path: '/livemap',
    name: 'livemap',
    component: () => import('@/pages/LiveMap.vue'),
    meta: { title: 'LiveMap · 实战地图', icon: '🗺' },
  },
  {
    path: '/chat',
    name: 'carchat',
    component: () => import('@/pages/CarChat.vue'),
    meta: { title: 'CarChat · 问车', icon: '💬' },
  },
  {
    path: '/blackbox',
    name: 'blackbox',
    component: () => import('@/pages/Blackbox.vue'),
    meta: { title: 'Blackbox · 黑匣子', icon: '📼' },
  },
  {
    path: '/:pathMatch(.*)*',
    name: 'not-found',
    component: () => import('@/pages/NotFound.vue'),
    meta: {},
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
  // View Transitions API on browsers that ship it (Chromium 111+).
  // Vue Router will call document.startViewTransition() around the
  // navigation if this flag is on and the API is present.
  // (Falls back silently on Safari/Firefox.)
})

router.afterEach((to) => {
  document.title = (to.meta.title as string | undefined) ?? 'NavCockpit'
})

// Programmatically wrap each navigation in startViewTransition where
// supported, so the page-swap gets a native crossfade on Chromium.
const _push = router.push.bind(router)
router.push = (to: Parameters<typeof _push>[0]) => {
  type WithVT = Document & { startViewTransition?: (cb: () => void | Promise<void>) => unknown }
  const doc = document as WithVT
  if (typeof doc.startViewTransition === 'function') {
    return new Promise((resolve, reject) => {
      doc.startViewTransition!(() => {
        _push(to).then(resolve, reject)
      })
    }) as ReturnType<typeof _push>
  }
  return _push(to)
}

export default router
