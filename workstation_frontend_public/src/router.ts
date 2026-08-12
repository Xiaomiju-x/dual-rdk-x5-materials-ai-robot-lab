import { createRouter, createWebHistory, type RouteRecordRaw } from 'vue-router'

const routes: RouteRecordRaw[] = [
  {
    path: '/',
    name: 'cockpit',
    component: () => import('@/pages/Cockpit.vue'),
    meta: { title: 'Cockpit · 驾驶舱', icon: '🦾' },
  },
  {
    path: '/teleop',
    name: 'teleop',
    component: () => import('@/pages/Teleop.vue'),
    meta: { title: 'Teleop · 触控操控', icon: '🎛' },
  },
  {
    path: '/handover',
    name: 'handover',
    component: () => import('@/pages/Handover.vue'),
    meta: { title: 'Handover · 协同剧场', icon: '🤝' },
  },
  {
    path: '/inspect',
    name: 'inspect',
    component: () => import('@/pages/Inspect.vue'),
    meta: { title: 'Inspect · 单臂深检', icon: '🔬' },
  },
  {
    path: '/calibration',
    name: 'calibration',
    component: () => import('@/pages/Calibration.vue'),
    meta: { title: 'Calibration · 手眼标定', icon: '🎯' },
  },
  {
    path: '/defense',
    name: 'defense',
    component: () => import('@/pages/Defense.vue'),
    meta: { title: 'Showcase · 答辩自演', icon: '🎬' },
  },
  // ---- 第 4 期 (2026-06-12): 技能库 + v4 十幕剧本 ----
  {
    path: '/skills',
    name: 'skills',
    component: () => import('@/pages/Skills.vue'),
    meta: { title: 'Skills · 技能库', icon: '🎓' },
  },
  {
    path: '/pipeline',
    name: 'pipeline',
    component: () => import('@/pages/Pipeline.vue'),
    meta: { title: 'Pipeline · 十幕剧本', icon: '🎞' },
  },
  // legacy /coop redirect — keep old shortcuts working
  { path: '/coop', redirect: '/handover' },
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
})

router.afterEach((to) => {
  document.title = (to.meta.title as string | undefined) ?? 'WorkCockpit'
})

export default router
