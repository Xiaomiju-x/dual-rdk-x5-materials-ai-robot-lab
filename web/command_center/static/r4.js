/* Site31 R4 front-end integration: accessible federated research search. */
(function () {
  'use strict';

  var RELEASE = 'site32-global-commercial-v1.13-20260720';
  var LIMIT = 12;
  var legacySearchGo = window.homeSearchGo;
  var state = {
    input: null,
    list: null,
    status: null,
    submit: null,
    kind: null,
    source: null,
    statusFilter: null,
    share: null,
    summary: null,
    items: [],
    active: -1,
    query: '',
    timer: 0,
    request: null,
    requestSerial: 0,
    composing: false,
    lastPayload: null
  };

  function text(zh, en) {
    return typeof window.uiText === 'function' ? window.uiText(zh, en) : zh;
  }

  function valueOf(value) {
    if (value == null) return '';
    if (typeof value === 'object') {
      return String(value.label || value.name || value.kind || value.status || value.source || '');
    }
    return String(value);
  }

  function currentQuery() {
    return state.input ? state.input.value.trim() : '';
  }

  function currentFilters() {
    return {
      kind: state.kind ? state.kind.value : '',
      status: state.statusFilter ? state.statusFilter.value : '',
      source: state.source ? state.source.value : ''
    };
  }

  function searchParams(query, includeLimit) {
    var params = new URLSearchParams();
    if (query) params.set('q', query);
    var filters = currentFilters();
    Object.keys(filters).forEach(function (key) {
      if (filters[key]) params.set(key, filters[key]);
    });
    if (includeLimit) params.set('limit', String(LIMIT));
    return params;
  }

  function sharePath(query) {
    var params = searchParams(query, false);
    return '/?' + params.toString();
  }

  function syncShareState(query) {
    if (state.share) state.share.disabled = !query;
    if (!query || !window.history || !window.history.replaceState) return;
    window.history.replaceState({ site32Search: true }, '', sharePath(query));
  }

  function setSummary(message) {
    if (state.summary) state.summary.textContent = message;
  }

  function announce(message) {
    if (!state.status) return;
    state.status.textContent = '';
    window.requestAnimationFrame(function () { state.status.textContent = message; });
  }

  function setExpanded(expanded) {
    if (!state.input || !state.list) return;
    state.input.setAttribute('aria-expanded', expanded ? 'true' : 'false');
    state.list.hidden = !expanded;
    if (!expanded) {
      state.input.removeAttribute('aria-activedescendant');
      state.active = -1;
    }
  }

  function closeList() {
    setExpanded(false);
  }

  function cancelRequest() {
    state.requestSerial += 1;
    if (state.request) state.request.abort();
    state.request = null;
    if (state.input) state.input.removeAttribute('aria-busy');
  }

  function kindLabel(kind) {
    var key = valueOf(kind).toLowerCase();
    var zh = {
      material: '材料', prediction: '预测', evidence: '证据', trace: '链路',
      work_order: '工单', workorder: '工单', page: '页面', asset: '资产',
      dataset: '数据集', fallback: '继续检索', retry: '重试'
    };
    var en = {
      material: 'Material', prediction: 'Prediction', evidence: 'Evidence', trace: 'Trace',
      work_order: 'Work order', workorder: 'Work order', page: 'Page', asset: 'Asset',
      dataset: 'Dataset', fallback: 'Continue', retry: 'Retry'
    };
    return (document.documentElement.lang === 'en' ? en[key] : zh[key]) || valueOf(kind) || text('科研对象', 'Research object');
  }

  function statusLabel(status) {
    var key = valueOf(status).toLowerCase();
    var zh = {
      live: '实时', mirror: '镜像', replay: '回放', curated: '策展', observed: '实测',
      computed: '计算', planned: '计划', stale: '陈旧', offline: '离线', unknown: '未知',
      available: '可用', public: '公开', mock: '模拟', error: '错误'
    };
    var en = {
      live: 'Live', mirror: 'Mirror', replay: 'Replay', curated: 'Curated', observed: 'Observed',
      computed: 'Computed', planned: 'Planned', stale: 'Stale', offline: 'Offline', unknown: 'Unknown',
      available: 'Available', public: 'Public', mock: 'Mock', error: 'Error'
    };
    return (document.documentElement.lang === 'en' ? en[key] : zh[key]) || valueOf(status) || text('来源未标注', 'Unlabelled');
  }

  function safeHref(href) {
    var raw = valueOf(href).trim();
    if (!raw) return '';
    try {
      var target = new URL(raw, window.location.origin);
      if (target.origin !== window.location.origin || !/^https?:$/.test(target.protocol)) return '';
      return target.pathname + target.search + target.hash;
    } catch (error) {
      return '';
    }
  }

  function normalizeItem(item, index) {
    item = item && typeof item === 'object' ? item : {};
    var id = valueOf(item.id || item.evidence_id || item.trace_id || item.object_id);
    var title = valueOf(item.title || item.formula || item.name || id);
    if (!title) return null;
    return {
      key: valueOf(item.kind) + ':' + id + ':' + index,
      kind: valueOf(item.kind) || 'evidence',
      id: id,
      title: title,
      subtitle: valueOf(item.subtitle || item.summary || item.description),
      href: safeHref(item.href),
      status: valueOf(item.status) || 'unknown',
      source: valueOf(item.source) || 'unknown',
      freshness: valueOf(item.freshness || (item.provenance && item.provenance.freshness)),
      preview: valueOf(item.preview || item.claim_boundary || item.limitation),
      matchedFields: Array.isArray(item.matched_fields) ? item.matched_fields.map(valueOf).filter(Boolean) : [],
      action: valueOf(item.action),
      fallback: false
    };
  }

  function fallbackItem(query) {
    return {
      key: 'fallback:' + query,
      kind: 'fallback',
      id: query,
      title: text('在材料图鉴中继续检索', 'Continue in Materials Explorer'),
      subtitle: text('联邦索引暂无直接匹配，将保留原有材料、工单和链路路由。', 'No direct federated match; keep the existing material, work-order and trace routing.'),
      href: '',
      status: 'available',
      source: 'local route fallback',
      fallback: true
    };
  }

  function retryItem(query, reason) {
    return {
      key: 'retry:' + query,
      kind: 'retry',
      id: query,
      title: text('重试联合检索', 'Retry federated search'),
      subtitle: reason,
      preview: text('保留当前查询与筛选条件，不会清空输入。', 'The current query and filters will be preserved.'),
      href: '',
      status: 'error',
      source: 'public-index',
      action: 'retry',
      fallback: false
    };
  }

  function optionNode(item, index) {
    var option = document.createElement('div');
    option.id = 'homeSearchOption-' + index;
    option.className = 'r4-search-option';
    option.dataset.kind = item.kind;
    option.setAttribute('role', 'option');
    option.setAttribute('aria-selected', index === state.active ? 'true' : 'false');
    option.setAttribute('aria-posinset', String(index + 1));
    option.setAttribute('aria-setsize', String(state.items.length));

    var kind = document.createElement('span');
    kind.className = 'r4-search-kind';
    kind.textContent = kindLabel(item.kind);

    var copy = document.createElement('span');
    copy.className = 'r4-search-copy';
    var title = document.createElement('b');
    title.textContent = item.title;
    var subtitle = document.createElement('span');
    subtitle.textContent = item.subtitle || item.id;
    var preview = document.createElement('small');
    preview.textContent = item.preview || (item.matchedFields && item.matchedFields.length
      ? text('匹配: ', 'Matched: ') + item.matchedFields.join(', ')
      : item.id);
    copy.appendChild(title);
    copy.appendChild(subtitle);
    copy.appendChild(preview);

    var meta = document.createElement('span');
    meta.className = 'r4-search-meta';
    var status = document.createElement('span');
    status.className = 'r4-search-status ' + valueOf(item.status).toLowerCase().replace(/[^a-z0-9_-]+/g, '-');
    status.textContent = statusLabel(item.status);
    var source = document.createElement('span');
    source.className = 'r4-search-source';
    source.textContent = item.source;
    meta.appendChild(status);
    meta.appendChild(source);
    if (item.freshness) {
      var freshness = document.createElement('span');
      freshness.className = 'r4-search-freshness';
      freshness.textContent = item.freshness;
      meta.appendChild(freshness);
    }

    option.appendChild(kind);
    option.appendChild(copy);
    option.appendChild(meta);
    option.setAttribute('aria-label', kind.textContent + ': ' + item.title + '. ' + subtitle.textContent + '. ' + preview.textContent + '. ' + status.textContent + ', ' + source.textContent);
    option.addEventListener('pointerdown', function (event) { event.preventDefault(); });
    option.addEventListener('mouseenter', function () { setActive(index, false); });
    option.addEventListener('click', function () { activate(item); });
    return option;
  }

  function renderItems(items, message) {
    state.items = items;
    state.active = items.length ? 0 : -1;
    state.list.replaceChildren();
    items.forEach(function (item, index) { state.list.appendChild(optionNode(item, index)); });
    setExpanded(items.length > 0);
    if (items.length) setActive(state.active, false);
    announce(message);
    setSummary(message);
  }

  function renderLoading() {
    state.items = [];
    state.active = -1;
    var loading = document.createElement('div');
    loading.className = 'r4-search-option';
    loading.setAttribute('role', 'option');
    loading.setAttribute('aria-disabled', 'true');
    loading.textContent = text('正在检索材料、预测、证据与链路...', 'Searching materials, predictions, evidence and traces...');
    state.list.replaceChildren(loading);
    setExpanded(true);
    announce(loading.textContent);
    setSummary(text('正在检索...', 'Searching...'));
  }

  function resultSummary(payload, shown) {
    var total = Number(payload && payload.total) || 0;
    var groups = payload && payload.facet_groups;
    var kindGroup = Array.isArray(groups) ? groups.find(function (group) { return group && group.key === 'kind'; }) : null;
    var facets = kindGroup ? kindGroup.options : groups && groups.kind;
    var parts = [];
    if (Array.isArray(facets)) {
      facets.slice(0, 3).forEach(function (facet) {
        if (facet && facet.count) parts.push(kindLabel(facet.value) + ' ' + facet.count);
      });
    }
    var prefix = text('找到 ', 'Found ') + total + text(' 条', ' results');
    if (shown && shown < total) prefix += text('，显示前 ', ', showing ') + shown;
    return parts.length ? prefix + ' · ' + parts.join(' · ') : prefix;
  }

  function setActive(next, scroll) {
    if (!state.items.length) return;
    state.active = (next + state.items.length) % state.items.length;
    Array.prototype.forEach.call(state.list.querySelectorAll('[role="option"]'), function (option, index) {
      option.setAttribute('aria-selected', index === state.active ? 'true' : 'false');
    });
    var current = document.getElementById('homeSearchOption-' + state.active);
    if (current) {
      state.input.setAttribute('aria-activedescendant', current.id);
      if (scroll !== false) current.scrollIntoView({ block: 'nearest' });
    }
  }

  function legacyFallback(query) {
    closeList();
    if (state.input) state.input.value = query;
    if (typeof legacySearchGo === 'function') legacySearchGo.call(window);
  }

  function activate(item) {
    if (!item) return;
    closeList();
    if (item.action === 'retry') {
      runSearch(true);
      return;
    }
    if (item.fallback) {
      legacyFallback(item.id || currentQuery());
      return;
    }
    if (item.href) {
      var material = item.href.match(/^\/materials\/([^?#]+)/);
      var prediction = item.href.match(/^\/predictions\/([^?#]+)/);
      var evidence = item.href.match(/^\/api\/evidence_objects\/([^/?#]+)/);
      if (material && typeof window.detailOpen === 'function') {
        window.detailOpen('material', decodeURIComponent(material[1]));
        return;
      }
      if (prediction && typeof window.detailOpen === 'function') {
        window.detailOpen('prediction', decodeURIComponent(prediction[1]));
        return;
      }
      if (evidence && typeof window.detailOpen === 'function') {
        window.detailOpen('evidence', decodeURIComponent(evidence[1]));
        return;
      }
      window.location.assign(item.href);
      return;
    }
    var kind = item.kind.toLowerCase();
    if (kind === 'material' && typeof window.detailOpen === 'function') {
      window.detailOpen('material', item.id);
      return;
    }
    if (kind === 'prediction' && typeof window.detailOpen === 'function') {
      window.detailOpen('prediction', item.id);
      return;
    }
    if (kind === 'evidence' && typeof window.detailOpen === 'function') {
      window.detailOpen('evidence', item.id);
      return;
    }
    if (kind === 'trace' && typeof window.go === 'function') {
      window.go('traces', { after: function () {
        if (typeof window.traceFill === 'function') window.traceFill(item.id || item.title);
      } });
      return;
    }
    legacyFallback(item.id || item.title || currentQuery());
  }

  async function runSearch(explicit) {
    var query = currentQuery();
    window.clearTimeout(state.timer);
    if (!query) {
      closeList();
      announce(text('请输入化学式、掺杂、trace_id、evidence_id 或关键词。', 'Enter a formula, dopant, trace_id, evidence_id, or keyword.'));
      setSummary(text('等待检索', 'Ready to search'));
      if (state.share) state.share.disabled = true;
      if (explicit && typeof window.toast === 'function') window.toast(text('请输入科研检索关键词', 'Enter a research search term'), 'warn');
      return;
    }

    if (state.request) state.request.abort();
    var controller = new AbortController();
    var serial = ++state.requestSerial;
    state.request = controller;
    state.query = query;
    state.input.setAttribute('aria-busy', 'true');
    renderLoading();
    var timeout = window.setTimeout(function () { controller.abort(); }, 6500);

    try {
      var response = await fetch('/api/search/federated?' + searchParams(query, true).toString(), {
        cache: 'no-store',
        headers: { Accept: 'application/json' },
        signal: controller.signal
      });
      if (!response.ok) {
        var httpError = new Error('search ' + response.status);
        httpError.status = response.status;
        throw httpError;
      }
      var payload = await response.json();
      if (!payload || !Array.isArray(payload.items)) throw new Error('invalid search payload');
      if (serial !== state.requestSerial || query !== currentQuery()) return;
      state.lastPayload = payload;
      syncShareState(query);
      var raw = payload.items;
      var items = raw.slice(0, LIMIT).map(normalizeItem).filter(Boolean);
      if (!items.length) {
        renderItems([fallbackItem(query)], text('联邦索引无直接匹配，已提供材料图鉴兜底入口。', 'No direct federated match; the Materials Explorer fallback is available.'));
      } else {
        var summary = resultSummary(payload, items.length);
        renderItems(items, summary + text('。使用上下方向键选择，回车打开。', '. Use arrow keys to choose and Enter to open.'));
      }
    } catch (error) {
      if (error && error.name === 'AbortError' && serial !== state.requestSerial) return;
      var reason;
      if (error && error.name === 'AbortError') reason = text('检索超时，请重试或进入材料图鉴。', 'Search timed out. Retry or continue in Materials Explorer.');
      else if (error && error.status === 429) reason = text('检索请求过于频繁，请稍后重试。', 'Search is rate limited. Retry shortly.');
      else if (navigator.onLine === false) reason = text('当前离线。恢复网络后可重试，材料图鉴仍可查看已缓存内容。', 'You are offline. Retry after reconnecting; cached Explorer content may remain available.');
      else reason = text('联合检索暂不可用，请重试或进入材料图鉴。', 'Federated search is unavailable. Retry or continue in Materials Explorer.');
      renderItems([retryItem(query, reason), fallbackItem(query)], reason);
    } finally {
      window.clearTimeout(timeout);
      if (serial === state.requestSerial) {
        state.input.removeAttribute('aria-busy');
        state.request = null;
      }
    }
  }

  function onInput() {
    if (state.composing) return;
    window.clearTimeout(state.timer);
    closeList();
    if (!currentQuery()) {
      announce(text('输入关键词后检索材料、预测、证据与链路。', 'Type to search materials, predictions, evidence and traces.'));
      setSummary(text('等待检索', 'Ready to search'));
      if (state.share) state.share.disabled = true;
      return;
    }
    state.timer = window.setTimeout(function () { runSearch(false); }, 220);
  }

  function onFilterChange() {
    closeList();
    if (currentQuery()) runSearch(false);
    else setSummary(text('输入关键词以应用筛选', 'Enter a query to apply filters'));
  }

  function restoreFromUrl() {
    if (window.location.pathname !== '/') return false;
    var params = new URLSearchParams(window.location.search || '');
    var query = valueOf(params.get('q')).slice(0, 120);
    var allowed = {
      kind: ['material', 'prediction', 'evidence', 'work_order', 'page'],
      status: ['live', 'mirror', 'replay', 'offline', 'planned'],
      source: ['curated', 'history', 'mirror', 'live']
    };
    if (state.input) state.input.value = query;
    [['kind', state.kind], ['status', state.statusFilter], ['source', state.source]].forEach(function (entry) {
      var raw = valueOf(params.get(entry[0]));
      if (entry[1] && allowed[entry[0]].indexOf(raw) >= 0) entry[1].value = raw;
    });
    if (state.share) state.share.disabled = !query;
    return Boolean(query);
  }

  async function copyShareLink() {
    var query = currentQuery();
    if (!query) return;
    var value = window.location.origin + sharePath(query);
    try {
      if (!navigator.clipboard || !navigator.clipboard.writeText) throw new Error('clipboard unavailable');
      await navigator.clipboard.writeText(value);
      if (typeof window.toast === 'function') window.toast(text('检索链接已复制', 'Search link copied'), 'ok');
      announce(text('可分享检索链接已复制。', 'Shareable search link copied.'));
    } catch (error) {
      var field = document.createElement('textarea');
      field.value = value;
      field.setAttribute('readonly', '');
      field.style.position = 'fixed';
      field.style.opacity = '0';
      document.body.appendChild(field);
      field.select();
      var copied = false;
      try { copied = document.execCommand('copy'); } catch (copyError) {}
      field.remove();
      if (typeof window.toast === 'function') window.toast(copied ? text('检索链接已复制', 'Search link copied') : value, copied ? 'ok' : 'info');
    }
  }

  function onKeydown(event) {
    if (event.key === 'ArrowDown') {
      event.preventDefault();
      if (state.items.length && !state.list.hidden) setActive(state.active + 1);
      else runSearch(false);
      return;
    }
    if (event.key === 'ArrowUp') {
      event.preventDefault();
      if (state.items.length && !state.list.hidden) setActive(state.active - 1);
      else runSearch(false);
      return;
    }
    if (event.key === 'Enter') {
      event.preventDefault();
      if (!state.list.hidden && state.active >= 0 && state.items[state.active]) activate(state.items[state.active]);
      else runSearch(true);
      return;
    }
    if (event.key === 'Escape' && !state.list.hidden) {
      event.preventDefault();
      cancelRequest();
      closeList();
      announce(text('检索结果已关闭。', 'Search results closed.'));
    }
  }

  function syncLanguage() {
    if (!state.input || !state.list) return;
    state.input.setAttribute('aria-label', text('统一科研检索输入', 'Unified research search input'));
    state.list.setAttribute('aria-label', text('科研检索结果', 'Research search results'));
    state.submit.setAttribute('aria-label', text('执行统一科研检索', 'Run unified research search'));
    if (state.items.length && !state.list.hidden) renderItems(state.items, text('检索语言已切换。', 'Search language changed.'));
  }

  function init() {
    state.input = document.getElementById('homeSearchInput');
    state.list = document.getElementById('homeSearchResults');
    state.status = document.getElementById('homeSearchStatus');
    state.submit = document.getElementById('homeSearchSubmit');
    state.kind = document.getElementById('homeSearchKind');
    state.statusFilter = document.getElementById('homeSearchStatusFilter');
    state.source = document.getElementById('homeSearchSource');
    state.share = document.getElementById('homeSearchShare');
    state.summary = document.getElementById('homeSearchSummary');
    if (!state.input || !state.list || !state.status || !state.submit) return;

    state.input.addEventListener('input', onInput);
    state.input.addEventListener('keydown', onKeydown);
    state.input.addEventListener('compositionstart', function () { state.composing = true; });
    state.input.addEventListener('compositionend', function () { state.composing = false; onInput(); });
    state.input.addEventListener('focus', function () {
      if (state.items.length && state.query === currentQuery()) setExpanded(true);
    });
    state.submit.addEventListener('click', function () { runSearch(true); });
    [state.kind, state.statusFilter, state.source].forEach(function (control) {
      if (control) control.addEventListener('change', onFilterChange);
    });
    if (state.share) state.share.addEventListener('click', copyShareLink);
    document.addEventListener('pointerdown', function (event) {
      if (!event.target.closest('.site31-hero-search')) { cancelRequest(); closeList(); }
    });
    new MutationObserver(syncLanguage).observe(document.documentElement, { attributes: true, attributeFilter: ['lang'] });

    window.homeSearchGo = function () { runSearch(true); };
    window.homeSearchKey = onKeydown;
    window.homeSearchExample = function (query) {
      state.input.value = valueOf(query);
      state.input.focus();
      runSearch(true);
    };
    window.R4FederatedSearch = Object.freeze({
      release: RELEASE,
      search: function (query) {
        state.input.value = valueOf(query);
        return runSearch(true);
      },
      close: closeList
    });
    document.dispatchEvent(new CustomEvent('site32searchownerready'));

    syncLanguage();
    announce(text('输入关键词后检索材料、预测、证据与链路。', 'Type to search materials, predictions, evidence and traces.'));
    if (restoreFromUrl()) window.setTimeout(function () { runSearch(false); }, 0);
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init, { once: true });
  else init();
})();
