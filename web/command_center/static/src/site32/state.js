const EMPTY_OBJECT = Object.freeze({});

function freezeRecord(value) {
  if (!value || typeof value !== 'object' || Object.isFrozen(value)) return value;
  return Object.freeze(Array.isArray(value) ? value.slice() : { ...value });
}

function makeSnapshot(next) {
  const out = {};
  Object.entries(next).forEach(([key, value]) => {
    out[key] = freezeRecord(value);
  });
  return Object.freeze(out);
}

export function createSite32State(options = EMPTY_OBJECT) {
  const release = options.release || 'site32-dev';
  const subscribers = new Set();
  const history = [];
  let snapshot = makeSnapshot({
    release,
    router: {
      current: 'home',
      previous: '',
      cluster: '',
      source: 'initial'
    },
    search: {
      homeQuery: '',
      atlasQuery: '',
      detailKind: '',
      detailId: '',
      source: 'initial'
    },
    theater: {
      active: false,
      label: '',
      playing: false,
      source: 'initial'
    },
    motion: {
      reduced: false,
      state: 'idle',
      mode: '',
      tier: '',
      source: 'initial'
    },
    appearance: {
      requested: 'vivid',
      effective: 'vivid',
      transparencyReduced: false,
      forcedColors: false,
      source: 'initial'
    },
    a11y: {
      route: 'home',
      liveRegion: false,
      currentNav: 0,
      source: 'initial'
    },
    telemetry: {
      events: 0,
      lastEvent: '',
      source: 'initial'
    }
  });

  function notify(change) {
    subscribers.forEach((subscriber) => {
      try {
        subscriber(snapshot, change);
      } catch (error) {
        setTimeout(() => { throw error; }, 0);
      }
    });
  }

  function update(scope, patch = EMPTY_OBJECT, meta = EMPTY_OBJECT) {
    if (!scope || typeof scope !== 'string') return snapshot;
    const current = snapshot[scope] || EMPTY_OBJECT;
    const nextRecord = freezeRecord({
      ...(current && typeof current === 'object' ? current : EMPTY_OBJECT),
      ...(patch && typeof patch === 'object' ? patch : EMPTY_OBJECT),
      updatedAt: Date.now()
    });
    const previous = snapshot;
    snapshot = makeSnapshot({ ...snapshot, [scope]: nextRecord });
    const change = Object.freeze({
      scope,
      previous,
      current: snapshot,
      meta: freezeRecord(meta || EMPTY_OBJECT)
    });
    history.push(change);
    if (history.length > 80) history.shift();
    notify(change);
    return snapshot;
  }

  function select(scope) {
    return snapshot[scope] || EMPTY_OBJECT;
  }

  function subscribe(subscriber, options = EMPTY_OBJECT) {
    if (typeof subscriber !== 'function') return () => {};
    subscribers.add(subscriber);
    if (options.immediate) {
      subscriber(snapshot, Object.freeze({ scope: 'initial', current: snapshot }));
    }
    return () => subscribers.delete(subscriber);
  }

  return Object.freeze({
    release,
    getSnapshot: () => snapshot,
    select,
    update,
    subscribe,
    history: () => history.slice()
  });
}
