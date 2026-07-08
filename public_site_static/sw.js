/* 指挥中心门户 service worker — 装到主屏 + 离线降级壳 (H23).
 * 外壳静态资源: cache-first 回退; 导航/脚本: network-first (更新即时), 离线回落缓存。
 * 少量只读 API 留"最近一次成功"快照 → 离线时展示最近缓存而非空白 (仅离线时启用)。
 * 跨域 iframe (lab/car/arm 子域) 与写操作不接管。 */
const C = 'cmdcenter-shell-v73-site27-highlight-claims';
const API_C = 'cmdcenter-api-v73-site27-highlight-claims';
const SHELL = ['/', '/icon.svg', '/manifest.webmanifest', '/style.css', '/app.js', '/i18n.js', '/twin.js'];
// 离线时可回放最近快照的只读 API (实时性可降级的状态/聚合类)
const API_CACHEABLE = ['/api/public_status', '/api/openapi.json', '/api/public_manifest', '/api/global_benchmark', '/api/hardening', '/api/fleet', '/api/fleet_cockpit', '/api/tasks_cockpit', '/api/agent_studio', '/api/lab_fsd_console', '/api/experiment_replay', '/api/cloud_command_center', '/api/defense_mode', '/api/observability_cockpit', '/api/traces', '/api/ops', '/api/kpi', '/api/models', '/api/atlas', '/api/ai_brain/explain', '/api/twin', '/api/materials/explorer', '/api/materials/', '/api/predictions/',
  '/api/preflight', '/api/diagnose', '/api/assets', '/api/systems'];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(C).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});
self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys().then((ks) => Promise.all(
      ks.filter((k) => k !== C && k !== API_C).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});
self.addEventListener('fetch', (e) => {
  const u = new URL(e.request.url);
  if (e.request.method !== 'GET' || u.origin !== location.origin) return; // 跨域 iframe / 非 GET 不接管
  if (u.pathname.startsWith('/api/')) {
    // 只读白名单: network-first + 落最近快照, 离线回放; 其余 API 不接管 (实时)
    if (API_CACHEABLE.some((p) => u.pathname === p || u.pathname.startsWith(p))) {
      e.respondWith(
        fetch(e.request)
          .then((r) => { const cp = r.clone(); caches.open(API_C).then((c) => c.put(e.request, cp)); return r; })
          .catch(() => caches.match(e.request).then((m) => m || new Response(
            JSON.stringify({ offline: true, error: '离线 — 无缓存快照' }),
            { status: 503, headers: { 'Content-Type': 'application/json' } })))
      );
    }
    return;
  }
  e.respondWith(
    fetch(e.request)
      .then((r) => { const cp = r.clone(); caches.open(C).then((c) => c.put(e.request, cp)); return r; })
      .catch(() => caches.match(e.request).then((m) => m || caches.match('/')))
  );
});

