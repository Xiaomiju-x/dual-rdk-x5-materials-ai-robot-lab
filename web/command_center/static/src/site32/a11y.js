function count(documentRef, selector) {
  return documentRef && documentRef.querySelectorAll ? documentRef.querySelectorAll(selector).length : 0;
}

function currentRoute(documentRef) {
  return (documentRef && documentRef.body && documentRef.body.dataset.view) || 'home';
}

export function installSite32A11y({ state, telemetry, windowRef = window, documentRef = document } = {}) {
  function report(source = 'sync') {
    const live = !!(documentRef && documentRef.getElementById && documentRef.getElementById('routeAnnouncer'));
    const patch = {
      route: currentRoute(documentRef),
      liveRegion: live,
      currentNav: count(documentRef, '[aria-current="page"]'),
      semanticButtons: count(documentRef, '[role="button"][tabindex="0"]'),
      tabs: count(documentRef, '[role="tab"]'),
      source
    };
    if (state && state.update) state.update('a11y', patch, { source });
    return patch;
  }

  function announce(message, politeness = 'polite') {
    const live = documentRef && documentRef.getElementById ? documentRef.getElementById('routeAnnouncer') : null;
    if (!live) return false;
    live.setAttribute('aria-live', politeness);
    live.textContent = '';
    windowRef.requestAnimationFrame(() => {
      live.textContent = String(message || '');
    });
    if (telemetry && telemetry.track) telemetry.track('a11y.announce', { politeness });
    return true;
  }

  function handleRoute(event) {
    const patch = report('route');
    if (telemetry && telemetry.track) {
      telemetry.track('a11y.route', {
        route: event && event.detail ? event.detail.route : patch.route,
        currentNav: patch.currentNav
      });
    }
  }

  function handleKeydown(event) {
    if (event.key === 'Escape' || event.key === '?' || event.key === 'Tab') {
      if (telemetry && telemetry.track) telemetry.track('a11y.key', { key: event.key });
    }
  }

  if (windowRef && windowRef.addEventListener) windowRef.addEventListener('site32:route', handleRoute);
  if (documentRef && documentRef.addEventListener) documentRef.addEventListener('keydown', handleKeydown, { capture: true });

  report('install');

  return Object.freeze({
    report,
    announce,
    dispose() {
      if (windowRef && windowRef.removeEventListener) windowRef.removeEventListener('site32:route', handleRoute);
      if (documentRef && documentRef.removeEventListener) documentRef.removeEventListener('keydown', handleKeydown, { capture: true });
    }
  });
}
