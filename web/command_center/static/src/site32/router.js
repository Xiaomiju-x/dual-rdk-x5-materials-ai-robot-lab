function currentRoute(documentRef) {
  return (documentRef && documentRef.body && documentRef.body.dataset.view) || 'home';
}

function currentCluster(documentRef) {
  return (documentRef && documentRef.body && documentRef.body.dataset.routeCluster) || '';
}

function routeList(documentRef) {
  if (!documentRef || !documentRef.querySelectorAll) return [];
  const routes = new Set();
  documentRef.querySelectorAll('[data-k]').forEach((node) => {
    if (node.dataset && node.dataset.k) routes.add(node.dataset.k);
  });
  return Array.from(routes).sort();
}

function wrapGlobal(windowRef, name, around) {
  const original = windowRef && windowRef[name];
  if (typeof original !== 'function' || original.__site32Wrapped) return false;
  const wrapped = function site32WrappedGlobal(...args) {
    return around.call(this, original, args);
  };
  Object.defineProperty(wrapped, '__site32Wrapped', { value: true });
  Object.defineProperty(wrapped, '__site32Original', { value: original });
  windowRef[name] = wrapped;
  return true;
}

export function installSite32Router({ state, telemetry, windowRef = window, documentRef = document } = {}) {
  let previous = '';
  let last = '';

  function sync(source = 'sync', extra = {}) {
    const route = currentRoute(documentRef);
    const cluster = currentCluster(documentRef);
    if (route !== last || source !== 'mutation') {
      previous = last || previous;
      last = route;
      if (state && state.update) {
        state.update('router', {
          current: route,
          previous,
          cluster,
          source
        }, extra);
      }
      if (telemetry && telemetry.track) telemetry.track('route.sync', { route, previous, source });
      try {
        windowRef.dispatchEvent(new CustomEvent('site32:route', {
          detail: Object.freeze({ route, previous, cluster, source })
        }));
      } catch (error) {}
    }
    return route;
  }

  function go(route, options = {}) {
    if (typeof windowRef.go === 'function') return windowRef.go(route, options);
    return false;
  }

  wrapGlobal(windowRef, 'go', function aroundGo(original, args) {
    const before = currentRoute(documentRef);
    const result = original.apply(this, args);
    queueMicrotask(() => sync('global:go', { requested: args[0], before }));
    return result;
  });

  const observer = typeof MutationObserver === 'function'
    ? new MutationObserver(() => sync('mutation'))
    : { observe() {}, disconnect() {} };
  if (documentRef && documentRef.body) {
    observer.observe(documentRef.body, {
      attributes: true,
      attributeFilter: ['data-view', 'data-route-cluster']
    });
  }

  function handlePopstate() {
    setTimeout(() => sync('popstate'), 0);
  }

  if (windowRef && windowRef.addEventListener) {
    windowRef.addEventListener('popstate', handlePopstate);
  }

  sync('install');

  return Object.freeze({
    current: () => currentRoute(documentRef),
    cluster: () => currentCluster(documentRef),
    routes: () => routeList(documentRef),
    go,
    sync,
    dispose() {
      observer.disconnect();
      if (windowRef && windowRef.removeEventListener) {
        windowRef.removeEventListener('popstate', handlePopstate);
      }
    }
  });
}
