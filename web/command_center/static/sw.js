/* XRD Command Center Site32 service worker.
 * Navigation is network-first. Versioned same-origin assets are served from the
 * versioned shell cache immediately and revalidated without caching private traffic. */
const RELEASE = 'site32-global-commercial-v1.13-20260720';
const CACHE_NAME = 'cmdcenter-shell-v96-site32-global-commercial-v1.13';
const RELEASE_QUERY = `?v=${RELEASE}`;
const OFFLINE_URL = `/${RELEASE_QUERY}`;

const CORE_PATHS = new Set([
  '/',
  '/icon.svg',
  '/manifest.webmanifest',
  '/style.css',
  '/app.js',
  '/i18n.js',
  '/twin.js',
  '/r4.css',
  '/r4.js',
  '/r4-performance.js',
  '/r4-accessibility.js',
  '/site32.css',
  '/full-experience.css',
  '/full-experience.js',
  '/three.min.js',
  '/site32.js',
  '/src/site32/release.js',
  '/src/site32/runtime.js',
  '/src/site32/state.js',
  '/src/site32/appearance.js',
  '/src/site32/telemetry.js',
  '/src/site32/motion.js',
  '/src/site32/a11y.js',
  '/src/site32/router.js',
  '/src/site32/search.js',
  '/src/site32/theater.js'
]);
const PRECACHE_URLS = [...CORE_PATHS].map((path) => `${path}${RELEASE_QUERY}`);

const STATIC_EXTENSIONS = /\.(?:css|js|mjs|json|webmanifest|svg|png|jpe?g|gif|webp|avif|ico|woff2?|ttf|otf|wasm|mp4|webm)$/i;
const AUTH_PATH = /^\/(?:auth|login|logout|sso|oauth|callback|signin|signout|session)(?:\/|$)/i;

function isNavigation(request) {
  return request.mode === 'navigate' || request.destination === 'document';
}

function isPrivateRequest(request, url) {
  return url.pathname === '/api' ||
    url.pathname.startsWith('/api/') ||
    AUTH_PATH.test(url.pathname) ||
    request.headers.has('authorization');
}

function isStaticRequest(request, url) {
  return CORE_PATHS.has(url.pathname) ||
    ['style', 'script', 'image', 'font', 'manifest', 'video'].includes(request.destination) ||
    STATIC_EXTENSIONS.test(url.pathname);
}

function canonicalCacheRequest(request, url) {
  if (!CORE_PATHS.has(url.pathname)) return request;
  if (url.searchParams.has('v')) return request;
  return new Request(`${url.origin}${url.pathname}${RELEASE_QUERY}`, {
    method: 'GET',
    credentials: 'same-origin'
  });
}

function responseIsPrivate(response) {
  const cacheControl = response.headers.get('Cache-Control') || '';
  const vary = response.headers.get('Vary') || '';
  let finalUrl;
  try {
    finalUrl = response.url ? new URL(response.url) : null;
  } catch (_) {
    finalUrl = null;
  }
  return /(?:^|,)\s*(?:no-store|private)\b/i.test(cacheControl) ||
    /(?:^|,)\s*(?:cookie|authorization)\s*(?:,|$)/i.test(vary) ||
    (finalUrl !== null && AUTH_PATH.test(finalUrl.pathname));
}

function expectedContentType(pathname) {
  if (/\.css$/i.test(pathname)) return /^text\/css\b/i;
  if (/\.(?:js|mjs)$/i.test(pathname)) {
    return /^(?:text|application)\/(?:javascript|x-javascript|ecmascript)\b/i;
  }
  if (/\.webmanifest$/i.test(pathname)) return /^application\/(?:manifest\+json|json)\b/i;
  if (/\.json$/i.test(pathname)) return /^application\/(?:[\w.+-]+\+)?json\b/i;
  if (/\.(?:svg|png|jpe?g|gif|webp|avif|ico)$/i.test(pathname)) return /^image\//i;
  if (/\.(?:woff2?|ttf|otf)$/i.test(pathname)) return /^(?:font\/|application\/(?:font-|x-font-|octet-stream))/i;
  if (/\.wasm$/i.test(pathname)) return /^application\/wasm\b/i;
  if (/\.(?:mp4|webm)$/i.test(pathname)) return /^video\//i;
  return null;
}

function canCache(response, pathname, expectHtml = false) {
  if (response.status !== 200 || response.redirected || responseIsPrivate(response)) return false;

  const contentType = response.headers.get('Content-Type') || '';
  if (expectHtml) return /^text\/html\b/i.test(contentType);
  if (/^text\/html\b/i.test(contentType)) return false;

  const expected = expectedContentType(pathname);
  return expected === null || expected.test(contentType);
}

async function precache() {
  const cache = await caches.open(CACHE_NAME);
  await Promise.all(PRECACHE_URLS.map(async (url) => {
    const request = new Request(url, {
      cache: 'reload',
      credentials: 'same-origin'
    });
    const response = await fetch(request);
    const pathname = new URL(request.url).pathname;
    if (!canCache(response, pathname, pathname === '/')) {
      throw new Error(`Refusing invalid precache response for ${pathname}`);
    }
    await cache.put(request, response);
  }));
}

self.addEventListener('install', (event) => {
  event.waitUntil(precache().then(() => self.skipWaiting()));
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys
        .filter((key) => key.startsWith('cmdcenter-') && key !== CACHE_NAME)
        .map((key) => caches.delete(key))))
      .then(() => self.clients.claim())
  );
});

async function networkFirstNavigation(request) {
  const offlineRequest = new Request(new URL(OFFLINE_URL, self.location.origin), {
    credentials: 'same-origin'
  });
  try {
    return await fetch(request);
  } catch (_) {
    const cached = await caches.match(offlineRequest);
    return cached || new Response(
      '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>当前离线</title><body><main><h1>当前离线</h1><p>请恢复网络连接后刷新页面。</p></main></body></html>',
      { status: 503, headers: { 'Content-Type': 'text/html; charset=utf-8' } }
    );
  }
}

async function staleWhileRevalidate(event, request, url) {
  const cache = await caches.open(CACHE_NAME);
  const cacheRequest = canonicalCacheRequest(request, url);
  const cached = await cache.match(cacheRequest);
  const update = fetch(cacheRequest).then(async (response) => {
    if (canCache(response, url.pathname)) {
      await cache.put(cacheRequest, response.clone());
    }
    return response;
  });

  if (cached) {
    event.waitUntil(update.catch(() => undefined));
    return cached;
  }

  try {
    return await update;
  } catch (_) {
    return new Response('Offline and no cached asset is available.', {
      status: 503,
      headers: { 'Content-Type': 'text/plain; charset=utf-8' }
    });
  }
}

self.addEventListener('fetch', (event) => {
  const request = event.request;
  const url = new URL(request.url);
  if (request.method !== 'GET' || url.origin !== self.location.origin) return;
  if (isPrivateRequest(request, url)) return;

  if (isNavigation(request)) {
    event.respondWith(networkFirstNavigation(request));
    return;
  }

  if (isStaticRequest(request, url)) {
    const requestedVersion = url.searchParams.get('v');
    if (requestedVersion && requestedVersion !== RELEASE) {
      event.respondWith(fetch(request, { cache: 'no-store' }));
      return;
    }
    event.respondWith(staleWhileRevalidate(event, request, url));
  }
});
