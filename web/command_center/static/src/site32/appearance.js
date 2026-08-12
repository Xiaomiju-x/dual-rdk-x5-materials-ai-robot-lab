const STORAGE_KEY = 'cmdcenter.visualMode.v3';
const LEGACY_STORAGE_KEY = 'cmdcenter.visualMode.v2';
const VALID_MODES = new Set(['vivid', 'defense', 'minimal']);

function mediaQuery(windowRef, query) {
  return windowRef && windowRef.matchMedia
    ? windowRef.matchMedia(query)
    : { matches: false, addEventListener() {}, removeEventListener() {} };
}

function validMode(value) {
  return VALID_MODES.has(value) ? value : 'vivid';
}

function readStoredMode(windowRef) {
  try {
    const current = windowRef.localStorage.getItem(STORAGE_KEY);
    if (VALID_MODES.has(current)) return current;
    return windowRef.localStorage.getItem(LEGACY_STORAGE_KEY) === 'minimal' ? 'minimal' : 'vivid';
  } catch (error) {
    return 'vivid';
  }
}

function visualModeFor(mode) {
  return mode === 'minimal' ? 'minimal' : 'vivid';
}

export function installSite32Appearance({ state, windowRef = window, documentRef = document } = {}) {
  const root = documentRef.documentElement;
  const transparency = mediaQuery(windowRef, '(prefers-reduced-transparency: reduce)');
  const forcedColors = mediaQuery(windowRef, '(forced-colors: active)');
  const controls = Array.from(documentRef.querySelectorAll('input[name="site32-visual-mode"]'));
  let requested = validMode(root.dataset.site32PresentationMode || readStoredMode(windowRef));

  function syncState(source) {
    const patch = {
      requested,
      effective: forcedColors.matches ? 'forced-colors' : requested,
      transparencyReduced: !!transparency.matches,
      forcedColors: !!forcedColors.matches,
      source
    };
    if (state && state.update) state.update('appearance', patch, { source });
    return patch;
  }

  function syncControls() {
    controls.forEach((control) => {
      const active = control.value === requested;
      control.checked = active;
      if (control.parentElement) control.parentElement.dataset.active = active ? 'true' : 'false';
    });
  }

  function announce() {
    const region = documentRef.getElementById('appearanceAnnouncer');
    if (!region) return;
    const english = (root.lang || '').toLowerCase().startsWith('en');
    const messages = english
      ? { vivid: 'Full liquid-glass mode enabled.', defense: 'Defense static-glass mode enabled.', minimal: 'Efficient minimal mode enabled.' }
      : { vivid: '\u5df2\u5207\u6362\u5230\u5b8c\u6574\u6db2\u6001\u73bb\u7483\u6a21\u5f0f\u3002', defense: '\u5df2\u5207\u6362\u5230\u7b54\u8fa9\u9759\u6001\u73bb\u7483\u6a21\u5f0f\u3002', minimal: '\u5df2\u5207\u6362\u5230\u9ad8\u6548\u6781\u7b80\u6a21\u5f0f\u3002' };
    region.textContent = messages[requested];
  }

  function apply(mode, options = {}) {
    requested = validMode(mode);
    const visualMode = visualModeFor(requested);
    root.dataset.site32PresentationMode = requested;
    root.dataset.site32VisualMode = visualMode;
    if (documentRef.body) {
      documentRef.body.dataset.site32PresentationMode = requested;
      documentRef.body.dataset.site32VisualMode = visualMode;
    }
    if (options.persist) {
      try { windowRef.localStorage.setItem(STORAGE_KEY, requested); } catch (error) {}
    }
    syncControls();
    const performanceController = windowRef.R4Performance;
    if (performanceController && performanceController.setVisualMode) {
      performanceController.setVisualMode(requested, options.source || 'appearance');
    }
    const patch = syncState(options.source || 'apply');
    if (options.announce) announce();
    try {
      windowRef.dispatchEvent(new CustomEvent('site32:appearance-change', { detail: patch }));
    } catch (error) {}
    return patch;
  }

  function onControlChange(event) {
    if (event.target && event.target.checked) {
      apply(event.target.value, { persist: true, announce: true, source: 'user' });
      const menu = documentRef.getElementById('moreMenu');
      const trigger = documentRef.getElementById('btnMore');
      if (menu) menu.classList.remove('show');
      if (trigger) {
        trigger.classList.remove('on');
        trigger.setAttribute('aria-expanded', 'false');
      }
      if (documentRef.body) documentRef.body.classList.remove('more-open');
    }
  }

  function onStorage(event) {
    if (event.key === STORAGE_KEY) apply(validMode(event.newValue), { source: 'storage' });
  }

  function onPreferenceChange() {
    syncState('preference');
    if (windowRef.R4Performance && windowRef.R4Performance.refresh) windowRef.R4Performance.refresh();
  }

  controls.forEach((control) => control.addEventListener('change', onControlChange));
  windowRef.addEventListener('storage', onStorage);
  [transparency, forcedColors].forEach((query) => query.addEventListener('change', onPreferenceChange));
  apply(requested, { source: 'install' });

  return Object.freeze({
    storageKey: STORAGE_KEY,
    getMode: () => requested,
    setMode: (mode) => apply(mode, { persist: true, announce: true, source: 'api' }),
    dispose() {
      controls.forEach((control) => control.removeEventListener('change', onControlChange));
      windowRef.removeEventListener('storage', onStorage);
      [transparency, forcedColors].forEach((query) => query.removeEventListener('change', onPreferenceChange));
    }
  });
}
