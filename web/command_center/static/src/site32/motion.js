function matchReduced(windowRef) {
  return windowRef && windowRef.matchMedia
    ? windowRef.matchMedia('(prefers-reduced-motion: reduce)')
    : { matches: false };
}

function readMotion(documentRef, media) {
  const root = documentRef && documentRef.documentElement;
  const body = documentRef && documentRef.body;
  return {
    reduced: !!(media && media.matches) || !!(root && root.dataset.motionMode === 'reduce'),
    state: (body && body.dataset.motionState) || 'idle',
    mode: (root && root.dataset.motionMode) || '',
    tier: (body && body.dataset.r4Render) || '',
    source: 'site32-motion'
  };
}

export function installSite32Motion({ state, telemetry, windowRef = window, documentRef = document } = {}) {
  const media = matchReduced(windowRef);

  function snapshot(source = 'sync') {
    const patch = { ...readMotion(documentRef, media), source };
    if (state && state.update) state.update('motion', patch, { source });
    if (telemetry && telemetry.track) telemetry.track('motion.sync', { source, reduced: patch.reduced, state: patch.state });
    return patch;
  }

  function setMode(mode) {
    const root = documentRef && documentRef.documentElement;
    if (root) {
      if (mode) root.dataset.motionMode = String(mode);
      else delete root.dataset.motionMode;
    }
    const patch = snapshot('setMode');
    try {
      windowRef.dispatchEvent(new CustomEvent('site32:motion-mode', { detail: patch }));
    } catch (error) {}
    return patch;
  }

  if (typeof windowRef.motionReduced === 'function' && !windowRef.motionReduced.__site32Wrapped) {
    const original = windowRef.motionReduced;
    const wrapped = function site32MotionReduced() {
      const result = original.apply(this, arguments);
      snapshot('global:motionReduced');
      return result;
    };
    Object.defineProperty(wrapped, '__site32Wrapped', { value: true });
    Object.defineProperty(wrapped, '__site32Original', { value: original });
    windowRef.motionReduced = wrapped;
  }

  const observer = typeof MutationObserver === 'function'
    ? new MutationObserver(() => snapshot('mutation'))
    : { observe() {}, disconnect() {} };
  if (documentRef && documentRef.body) {
    observer.observe(documentRef.body, {
      attributes: true,
      attributeFilter: ['data-motion-state', 'data-r4-render']
    });
  }
  if (documentRef && documentRef.documentElement) {
    observer.observe(documentRef.documentElement, {
      attributes: true,
      attributeFilter: ['data-motion-mode']
    });
  }

  function handleMediaChange() {
    snapshot('media');
  }

  function handleRenderChange(event) {
    snapshot(event && event.type ? event.type : 'renderchange');
  }

  if (media && media.addEventListener) media.addEventListener('change', handleMediaChange);
  else if (media && media.addListener) media.addListener(handleMediaChange);
  if (windowRef && windowRef.addEventListener) windowRef.addEventListener('r4renderchange', handleRenderChange);

  snapshot('install');

  return Object.freeze({
    snapshot,
    reduced: () => snapshot('read').reduced,
    setMode,
    dispose() {
      observer.disconnect();
      if (media && media.removeEventListener) media.removeEventListener('change', handleMediaChange);
      else if (media && media.removeListener) media.removeListener(handleMediaChange);
      if (windowRef && windowRef.removeEventListener) windowRef.removeEventListener('r4renderchange', handleRenderChange);
    }
  });
}
