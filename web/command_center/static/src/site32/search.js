function valueOf(documentRef, id) {
  const node = documentRef && documentRef.getElementById ? documentRef.getElementById(id) : null;
  return node && 'value' in node ? String(node.value || '').trim() : '';
}

function setValue(documentRef, id, value) {
  const node = documentRef && documentRef.getElementById ? documentRef.getElementById(id) : null;
  if (node && 'value' in node) node.value = String(value || '');
}

function wrapGlobal(windowRef, name, after) {
  const original = windowRef && windowRef[name];
  if (typeof original !== 'function' || original.__site32Wrapped) return false;
  const wrapped = function site32WrappedGlobal(...args) {
    const result = original.apply(this, args);
    queueMicrotask(() => after(name, args));
    return result;
  };
  Object.defineProperty(wrapped, '__site32Wrapped', { value: true });
  Object.defineProperty(wrapped, '__site32Original', { value: original });
  windowRef[name] = wrapped;
  return true;
}

export function installSite32Search({ state, telemetry, windowRef = window, documentRef = document } = {}) {
  function snapshot(source = 'sync', extra = {}) {
    const patch = {
      homeQuery: valueOf(documentRef, 'homeSearchInput'),
      atlasQuery: valueOf(documentRef, 'mxQ'),
      source,
      ...extra
    };
    if (state && state.update) state.update('search', patch, { source });
    if (telemetry && telemetry.track) {
      telemetry.track('search.sync', {
        source,
        homeLength: patch.homeQuery.length,
        atlasLength: patch.atlasQuery.length
      });
    }
    return patch;
  }

  function runHomeSearch(query) {
    if (query != null) setValue(documentRef, 'homeSearchInput', query);
    const fn = windowRef && windowRef.homeSearchGo;
    const result = typeof fn === 'function' ? fn() : false;
    snapshot('homeSearchGo', { homeQuery: valueOf(documentRef, 'homeSearchInput') });
    return result;
  }

  function applyAtlas(query) {
    if (query != null) setValue(documentRef, 'mxQ', query);
    const fn = windowRef && windowRef.mxApply;
    const result = typeof fn === 'function' ? fn() : false;
    snapshot('mxApply', { atlasQuery: valueOf(documentRef, 'mxQ') });
    return result;
  }

  function fetchAtlas() {
    const fn = windowRef && windowRef.mxFetch;
    const result = typeof fn === 'function' ? fn() : false;
    snapshot('mxFetch');
    return result;
  }

  function example(query) {
    const fn = windowRef && windowRef.mxExample;
    const result = typeof fn === 'function' ? fn(query) : applyAtlas(query);
    snapshot('mxExample', { atlasQuery: String(query || '') });
    return result;
  }

  function bindGlobalOwners() {
    let bound = 0;
    ['homeSearchGo', 'homeSearchExample', 'mxApply', 'mxFetch', 'mxExample', 'detailOpen', 'detailSetTab'].forEach((name) => {
      if (wrapGlobal(windowRef, name, (globalName, args) => {
        const extra = {};
        if (globalName === 'detailOpen') {
          extra.detailKind = String(args[0] || '');
          extra.detailId = String(args[1] || '');
        }
        snapshot(`global:${globalName}`, extra);
      })) bound += 1;
    });
    return bound;
  }

  function handleOwnerReady() {
    snapshot('owner-ready', { rebound: bindGlobalOwners() });
  }

  bindGlobalOwners();

  function handleInput(event) {
    const id = event.target && event.target.id;
    if (id === 'homeSearchInput' || id === 'mxQ') snapshot(`input:${id}`);
  }

  function handleKeydown(event) {
    const id = event.target && event.target.id;
    if ((id === 'homeSearchInput' || id === 'mxQ') && event.key === 'Enter') {
      snapshot(`enter:${id}`);
    }
  }

  if (documentRef && documentRef.addEventListener) {
    documentRef.addEventListener('input', handleInput, { capture: true, passive: true });
    documentRef.addEventListener('keydown', handleKeydown, { capture: true });
    documentRef.addEventListener('site32searchownerready', handleOwnerReady);
  }

  snapshot('install');

  return Object.freeze({
    snapshot,
    runHomeSearch,
    applyAtlas,
    fetchAtlas,
    example,
    dispose() {
      if (documentRef && documentRef.removeEventListener) {
        documentRef.removeEventListener('input', handleInput, { capture: true });
        documentRef.removeEventListener('keydown', handleKeydown, { capture: true });
        documentRef.removeEventListener('site32searchownerready', handleOwnerReady);
      }
    }
  });
}
