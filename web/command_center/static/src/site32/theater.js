function textOf(documentRef, id) {
  const node = documentRef && documentRef.getElementById ? documentRef.getElementById(id) : null;
  return node ? String(node.textContent || '').replace(/\s+/g, ' ').trim() : '';
}

function isActive(documentRef) {
  const node = documentRef && documentRef.getElementById ? documentRef.getElementById('theater') : null;
  return !!(node && node.classList.contains('show'));
}

function isPlaying(documentRef) {
  const node = documentRef && documentRef.getElementById ? documentRef.getElementById('thPlay') : null;
  return node ? node.getAttribute('aria-pressed') === 'true' : false;
}

function wrapGlobal(windowRef, name, after) {
  const original = windowRef && windowRef[name];
  if (typeof original !== 'function' || original.__site32Wrapped) return false;
  const wrapped = function site32WrappedGlobal(...args) {
    const result = original.apply(this, args);
    queueMicrotask(() => after(name));
    return result;
  };
  Object.defineProperty(wrapped, '__site32Wrapped', { value: true });
  Object.defineProperty(wrapped, '__site32Original', { value: original });
  windowRef[name] = wrapped;
  return true;
}

export function installSite32Theater({ state, telemetry, windowRef = window, documentRef = document } = {}) {
  function snapshot(source = 'sync') {
    const patch = {
      active: isActive(documentRef),
      label: textOf(documentRef, 'thNow'),
      title: textOf(documentRef, 'thT'),
      playing: isPlaying(documentRef),
      source
    };
    if (state && state.update) state.update('theater', patch, { source });
    if (telemetry && telemetry.track) telemetry.track('theater.sync', { source, active: patch.active, label: patch.label });
    return patch;
  }

  function call(name) {
    const fn = windowRef && windowRef[name];
    const result = typeof fn === 'function' ? fn() : false;
    snapshot(name);
    return result;
  }

  ['tourStart', 'theaterNext', 'theaterPrev', 'theaterPlayPause', 'tourExit'].forEach((name) => {
    wrapGlobal(windowRef, name, () => snapshot(`global:${name}`));
  });

  const observer = typeof MutationObserver === 'function'
    ? new MutationObserver(() => snapshot('mutation'))
    : { observe() {}, disconnect() {} };
  const theater = documentRef && documentRef.getElementById ? documentRef.getElementById('theater') : null;
  if (theater) {
    observer.observe(theater, {
      attributes: true,
      attributeFilter: ['class', 'aria-pressed'],
      childList: true,
      subtree: true
    });
  }

  snapshot('install');

  return Object.freeze({
    snapshot,
    start: () => call('tourStart'),
    next: () => call('theaterNext'),
    prev: () => call('theaterPrev'),
    togglePlay: () => call('theaterPlayPause'),
    exit: () => call('tourExit'),
    dispose() {
      observer.disconnect();
    }
  });
}
