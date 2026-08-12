const MAX_EVENTS = 180;

function safeNow() {
  return (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
}

function safeText(value) {
  return String(value || '').replace(/\s+/g, ' ').trim().slice(0, 160);
}

function routeFromDocument(documentRef) {
  return (documentRef && documentRef.body && documentRef.body.dataset.view) || 'home';
}

export function installSite32Telemetry({ state, windowRef = window, documentRef = document } = {}) {
  const events = [];
  let disposed = false;

  function track(name, detail = {}) {
    if (disposed || !name) return null;
    const event = Object.freeze({
      name: String(name),
      detail: Object.freeze({ ...detail }),
      route: routeFromDocument(documentRef),
      t: Date.now(),
      monotonic: safeNow()
    });
    events.push(event);
    if (events.length > MAX_EVENTS) events.shift();
    if (state && state.update) {
      state.update('telemetry', {
        events: events.length,
        lastEvent: event.name,
        source: 'site32-telemetry'
      }, { event: event.name });
    }
    try {
      windowRef.dispatchEvent(new CustomEvent('site32:telemetry', { detail: event }));
    } catch (error) {}
    return event;
  }

  function flush(options = {}) {
    const batch = events.slice();
    if (options.clear !== false) events.length = 0;
    if (
      options.beaconUrl &&
      options.sendBeacon === true &&
      windowRef.navigator &&
      typeof windowRef.navigator.sendBeacon === 'function'
    ) {
      try {
        const payload = new Blob([JSON.stringify({ events: batch })], { type: 'application/json' });
        windowRef.navigator.sendBeacon(options.beaconUrl, payload);
      } catch (error) {}
    }
    return batch;
  }

  function handleClick(event) {
    const target = event.target && event.target.closest
      ? event.target.closest('[data-site32-action],[data-k],button,a,[role="button"],[role="tab"]')
      : null;
    if (!target) return;
    track('ui.action', {
      key: target.dataset ? safeText(target.dataset.site32Action || target.dataset.k) : '',
      label: safeText(target.getAttribute('aria-label') || target.textContent),
      tag: target.tagName || ''
    });
  }

  function handleVisibility() {
    track('page.visibility', { hidden: !!documentRef.hidden });
  }

  if (documentRef && documentRef.addEventListener) {
    documentRef.addEventListener('click', handleClick, { capture: true, passive: true });
    documentRef.addEventListener('visibilitychange', handleVisibility, { passive: true });
  }

  return Object.freeze({
    track,
    flush,
    events: () => events.slice(),
    dispose() {
      disposed = true;
      if (documentRef && documentRef.removeEventListener) {
        documentRef.removeEventListener('click', handleClick, { capture: true });
        documentRef.removeEventListener('visibilitychange', handleVisibility);
      }
    }
  });
}
