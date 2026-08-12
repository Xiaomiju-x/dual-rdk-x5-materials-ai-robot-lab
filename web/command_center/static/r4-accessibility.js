(function () {
  'use strict';

  var SINGLE_KEY_STORAGE = 'xrd_a11y_single_key_shortcuts';
  var OVERLAY_ROOT_IDS = new Set([
    'pal', 'scanOv', 'kbdHelp', 'repModal', 'adminModal', 'woModal',
    'ncMask', 'cpPanel', 'iModal'
  ]);
  var FOCUSABLE = [
    'a[href]',
    'button:not([disabled])',
    'input:not([disabled]):not([type="hidden"])',
    'select:not([disabled])',
    'textarea:not([disabled])',
    '[contenteditable="true"]',
    '[tabindex]:not([tabindex="-1"])'
  ].join(',');

  var overlaySpecs = [
    { id: 'theater', label: '端到端证据演示', close: 'tourExit', initial: '#thPlay' },
    { id: 'pal', panel: '.palbox', label: 'Global command palette', close: 'palClose', initial: '#palIn' },
    { id: 'scanOv', panel: '.scan-box', label: 'Code scanner', close: 'scanClose', initial: '.scan-x' },
    { id: 'kbdHelp', panel: '.kbd-box', label: 'Keyboard shortcuts', close: 'kbdToggle', initial: '.scan-x' },
    { id: 'repModal', panel: '.wocard', label: 'Operations report', close: 'repClose', labelledBy: 'repTitle' },
    { id: 'adminModal', panel: '.wocard', label: 'Platform administration', close: 'adminClose' },
    { id: 'woModal', panel: '.wocard', label: 'Work order details', close: 'woClose', labelledBy: 'wodCode' },
    { id: 'ncMask', panel: '#ncDrawer', label: 'Notification center', close: 'ncToggle', initial: '.cp-x' },
    {
      id: 'iModal',
      panel: '.imodal-box',
      label: 'Details dialog',
      close: 'iModalClose',
      initial: 'input:not([type="hidden"]), select, textarea, button'
    }
  ];

  var states = new Map();
  var openSequence = 0;
  var pendingActivator = null;
  var pendingActivatorAt = 0;
  var lastExternalFocus = null;
  var observer = null;
  var syncFrame = 0;

  function byId(id) {
    return document.getElementById(id);
  }

  function setIfMissing(element, name, value) {
    if (element && !element.hasAttribute(name)) element.setAttribute(name, value);
  }

  function isSpecOpen(spec, root) {
    if (!root) return false;
    if (spec.id === 'iModal') return root.style.display !== 'none';
    return root.classList.contains('show');
  }

  function setInert(element, inert) {
    if (!element) return;
    if (inert) {
      element.setAttribute('inert', '');
      try { element.inert = true; } catch (error) {}
    } else {
      element.removeAttribute('inert');
      try { element.inert = false; } catch (error) {}
    }
  }

  function isVisible(element) {
    if (!element || !element.isConnected || element.hidden) return false;
    if (element.closest('[hidden],[inert],[aria-hidden="true"]')) return false;
    var style = window.getComputedStyle(element);
    return style.display !== 'none' && style.visibility !== 'hidden' &&
      style.visibility !== 'collapse' && element.getClientRects().length > 0;
  }

  function canRestoreFocus(element) {
    if (!isVisible(element)) return false;
    if (element.matches('[disabled],[aria-disabled="true"]')) return false;
    return typeof element.focus === 'function';
  }

  function focusElement(element) {
    if (!element || typeof element.focus !== 'function') return false;
    try {
      element.focus({ preventScroll: true });
    } catch (error) {
      try { element.focus(); } catch (focusError) { return false; }
    }
    return document.activeElement === element;
  }

  function panelFor(spec, root) {
    return spec.panel ? root.querySelector(spec.panel) : root;
  }

  function setupDialog(spec, root) {
    var panel = panelFor(spec, root);
    if (!panel) return;

    panel.setAttribute('role', 'dialog');
    panel.setAttribute('aria-modal', 'true');
    if (!panel.hasAttribute('tabindex')) panel.setAttribute('tabindex', '-1');

    var label = spec.labelledBy && byId(spec.labelledBy);
    if (label) {
      panel.setAttribute('aria-labelledby', spec.labelledBy);
      panel.removeAttribute('aria-label');
    } else {
      setIfMissing(panel, 'aria-label', spec.label);
    }

    panel.querySelectorAll('button').forEach(function (button) {
      var text = (button.textContent || '').trim();
      if (!button.hasAttribute('aria-label') && /^(x|\u2715)$/i.test(text)) {
        button.setAttribute('aria-label', 'Close dialog');
      }
    });
  }

  function focusableWithin(spec, root) {
    var panel = panelFor(spec, root);
    if (!panel) return [];
    return Array.prototype.filter.call(panel.querySelectorAll(FOCUSABLE), function (element) {
      return isVisible(element) && !element.matches('[disabled],[aria-disabled="true"]');
    });
  }

  function focusInto(spec, root) {
    if (!isSpecOpen(spec, root)) return;
    var initial = spec.initial ? root.querySelector(spec.initial) : null;
    if (initial && isVisible(initial) && focusElement(initial)) return;

    var focusables = focusableWithin(spec, root);
    if (focusables.length && focusElement(focusables[0])) return;
    focusElement(panelFor(spec, root));
  }

  function ownerOverlay(element) {
    if (!element) return null;
    for (var i = 0; i < overlaySpecs.length; i += 1) {
      var root = byId(overlaySpecs[i].id);
      if (root && root.contains(element)) return overlaySpecs[i];
    }
    return null;
  }

  function resolveReturnTarget(candidate) {
    var seen = new Set();
    var current = candidate;
    while (current && !seen.has(current)) {
      seen.add(current);
      if (canRestoreFocus(current)) return current;
      var owner = ownerOverlay(current);
      var state = owner && states.get(owner.id);
      current = state ? state.returnFocus : null;
    }
    return canRestoreFocus(lastExternalFocus) ? lastExternalFocus : null;
  }

  function restoreFocus(spec, root) {
    if (!root.contains(document.activeElement)) return;
    var state = states.get(spec.id) || {};
    var target = resolveReturnTarget(state.returnFocus);
    if (target && focusElement(target)) return;
    var main = byId('main');
    if (main) focusElement(main);
  }

  function syncOverlay(spec) {
    var root = byId(spec.id);
    if (!root) return;
    var state = states.get(spec.id) || { open: false, openedAt: 0, returnFocus: null };
    var open = isSpecOpen(spec, root);

    setupDialog(spec, root);
    if (open) {
      root.hidden = false;
      setInert(root, false);
      root.removeAttribute('aria-hidden');

      if (!state.open) {
        var recentActivator = Date.now() - pendingActivatorAt < 1500 ? pendingActivator : null;
        var active = document.activeElement;
        state.returnFocus = recentActivator && !root.contains(recentActivator)
          ? recentActivator
          : (active && active !== document.body && !root.contains(active) ? active : lastExternalFocus);
        state.openedAt = ++openSequence;
        window.requestAnimationFrame(function () { focusInto(spec, root); });
      }
    } else {
      if (state.open) restoreFocus(spec, root);
      root.setAttribute('aria-hidden', 'true');
      setInert(root, true);
      root.hidden = true;
    }

    state.open = open;
    states.set(spec.id, state);
  }

  function syncAllOverlays() {
    overlaySpecs.forEach(syncOverlay);
    syncDisclosures();
    syncCommandPalette();
  }

  function openOverlays() {
    return overlaySpecs.filter(function (spec) {
      var root = byId(spec.id);
      return root && isSpecOpen(spec, root);
    }).sort(function (left, right) {
      var leftState = states.get(left.id) || {};
      var rightState = states.get(right.id) || {};
      return (leftState.openedAt || 0) - (rightState.openedAt || 0);
    });
  }

  function topOverlay() {
    var open = openOverlays();
    return open.length ? open[open.length - 1] : null;
  }

  function closeOverlay(spec) {
    var root = byId(spec.id);
    if (!root || !isSpecOpen(spec, root)) return;
    var close = window[spec.close];
    try {
      if (typeof close === 'function') close();
      else if (spec.id === 'iModal') root.style.display = 'none';
      else root.classList.remove('show');
    } catch (error) {
      if (spec.id === 'iModal') root.style.display = 'none';
      else root.classList.remove('show');
    }
  }

  function trapTab(event, spec) {
    var root = byId(spec.id);
    if (!root) return;
    var focusables = focusableWithin(spec, root);
    if (!focusables.length) {
      event.preventDefault();
      focusElement(panelFor(spec, root));
      return;
    }

    var first = focusables[0];
    var last = focusables[focusables.length - 1];
    var active = document.activeElement;
    if (!root.contains(active)) {
      event.preventDefault();
      focusElement(event.shiftKey ? last : first);
    } else if (event.shiftKey && active === first) {
      event.preventDefault();
      focusElement(last);
    } else if (!event.shiftKey && active === last) {
      event.preventDefault();
      focusElement(first);
    }
  }

  function syncDisclosures() {
    var disclosures = [
      { trigger: 'bell', controlled: 'ncDrawer', openRoot: 'ncMask', kind: 'dialog' },
      { trigger: 'cpFab', controlled: 'cpPanel', openRoot: 'cpPanel', label: '运维副驾' },
      { trigger: 'btnMore', controlled: 'moreMenu', openRoot: 'moreMenu', label: '更多功能' }
    ];

    disclosures.forEach(function (item) {
      var trigger = byId(item.trigger);
      var controlled = byId(item.controlled);
      var openRoot = byId(item.openRoot);
      if (!trigger || !controlled || !openRoot) return;

      var open = openRoot.classList.contains('show');
      trigger.setAttribute('aria-controls', item.controlled);
      trigger.setAttribute('aria-expanded', open ? 'true' : 'false');
      if (item.kind) trigger.setAttribute('aria-haspopup', item.kind);
      else trigger.removeAttribute('aria-haspopup');

      if (!item.kind) {
        if (!open && controlled.contains(document.activeElement)) focusElement(trigger);
        controlled.setAttribute('role', 'region');
        setIfMissing(controlled, 'aria-label', item.label);
        controlled.hidden = !open;
        controlled.setAttribute('aria-hidden', open ? 'false' : 'true');
        setInert(controlled, !open);
      }
    });
  }

  function syncCommandPalette() {
    var root = byId('pal');
    var input = byId('palIn');
    var list = byId('palList');
    if (!root || !input || !list) return;

    var open = isSpecOpen(overlaySpecs[0], root);
    input.setAttribute('role', 'combobox');
    input.setAttribute('aria-autocomplete', 'list');
    input.setAttribute('aria-haspopup', 'listbox');
    input.setAttribute('aria-controls', 'palList');
    input.setAttribute('aria-expanded', open ? 'true' : 'false');
    list.setAttribute('role', 'listbox');
    setIfMissing(list, 'aria-label', 'Available commands');

    var options = Array.prototype.slice.call(list.querySelectorAll('.pali'));
    var active = null;
    options.forEach(function (option, index) {
      option.id = 'r4-pal-option-' + index;
      option.setAttribute('role', 'option');
      option.setAttribute('tabindex', '-1');
      option.setAttribute('aria-posinset', String(index + 1));
      option.setAttribute('aria-setsize', String(options.length));
      var selected = option.classList.contains('sel');
      option.setAttribute('aria-selected', selected ? 'true' : 'false');
      if (selected) active = option;
    });
    list.querySelectorAll('.pal-sep').forEach(function (separator) {
      separator.setAttribute('role', 'presentation');
    });

    if (open && active) input.setAttribute('aria-activedescendant', active.id);
    else input.removeAttribute('aria-activedescendant');
  }

  function frameTitle(frame) {
    var source = String(frame.dataset.src || frame.src || '').toLowerCase();
    var next = frame.nextElementSibling;
    var classes = next ? next.className : '';
    if (/frame-fallback-lab/.test(classes) || /(^|[./_-])lab([./_:-]|$)|8888/.test(source)) {
      return 'AI brain embedded dashboard';
    }
    if (/frame-fallback-car/.test(classes) || /(^|[./_-])car([./_:-]|$)|8890/.test(source)) {
      return 'Embodied brain embedded dashboard';
    }
    if (/frame-fallback-arm/.test(classes) || /(^|[./_-])arm([./_:-]|$)|8896/.test(source)) {
      return 'Robot arm workstation embedded dashboard';
    }
    return 'Embedded system view';
  }

  function enhanceFrame(frame) {
    if (frame && !frame.hasAttribute('title')) frame.setAttribute('title', frameTitle(frame));
  }

  function enhanceClickable(element) {
    if (!element || element.nodeType !== 1) return;
    if (!element.matches('div, th, a:not([href])')) return;
    if (!element.hasAttribute('onclick') && typeof element.onclick !== 'function') return;
    if (element.id && OVERLAY_ROOT_IDS.has(element.id)) return;
    if (element.closest('#palList')) return;

    if (element.tagName === 'TH') {
      element.setAttribute('scope', element.getAttribute('scope') || 'col');
      setIfMissing(element, 'aria-sort', 'none');
      if (!element.querySelector(':scope > .r4-th-button')) {
        var button = document.createElement('button');
        button.type = 'button';
        button.className = 'r4-th-button';
        while (element.firstChild) button.appendChild(element.firstChild);
        element.appendChild(button);
      }
      element.removeAttribute('role');
      element.removeAttribute('tabindex');
      element.setAttribute('data-r4-sort-header', 'true');
      return;
    }

    if (!element.hasAttribute('role')) element.setAttribute('role', 'button');
    if (!element.hasAttribute('tabindex')) element.setAttribute('tabindex', '0');
    element.setAttribute('data-r4-keyboard-click', 'true');
  }

  function enhanceTree(root) {
    if (!root || root.nodeType !== 1) return;
    if (root.matches('iframe')) enhanceFrame(root);
    if (root.matches('div, th, a:not([href])')) enhanceClickable(root);
    root.querySelectorAll('iframe').forEach(enhanceFrame);
    root.querySelectorAll('div, th, a:not([href])').forEach(enhanceClickable);
  }

  function setupLiveRegions() {
    var live = [
      { id: 'routeAnnouncer', role: 'status', politeness: 'polite', atomic: 'true' },
      { id: 'toast', role: 'status', politeness: 'polite', atomic: 'true' },
      { id: 'offbar', role: 'status', politeness: 'assertive', atomic: 'true' },
      { id: 'scanMsg', role: 'status', politeness: 'polite', atomic: 'true' },
      { id: 'palCnt', role: 'status', politeness: 'polite', atomic: 'true' },
      { id: 'ncFoot', role: 'status', politeness: 'polite', atomic: 'true' },
      { id: 'thLiveTx', role: 'status', politeness: 'polite', atomic: 'true' },
      { id: 'bdResult', role: 'status', politeness: 'polite', atomic: 'true' },
      { id: 'impResult', role: 'status', politeness: 'polite', atomic: 'true' },
      { id: 'cpBody', role: 'log', politeness: 'polite', atomic: 'false' }
    ];

    live.forEach(function (item) {
      var element = byId(item.id);
      if (!element) return;
      setIfMissing(element, 'role', item.role);
      setIfMissing(element, 'aria-live', item.politeness);
      setIfMissing(element, 'aria-atomic', item.atomic);
      if (item.role === 'log') setIfMissing(element, 'aria-relevant', 'additions text');
    });
  }

  function recordActivator(element) {
    if (!element || element === document.body || element === document.documentElement) return;
    pendingActivator = element;
    pendingActivatorAt = Date.now();
  }

  function isNativeInteractive(element) {
    return !!element.closest('button, input, select, textarea, a[href], [contenteditable="true"]');
  }

  function semanticControlFor(target) {
    if (!target || !target.closest) return null;
    var control = target.closest('[role="button"],[role="tab"],[data-r4-keyboard-click="true"]');
    if (!control || control.tabIndex < 0) return null;
    if (isNativeInteractive(target) && target !== control) return null;
    if (control.matches('button, input, select, textarea, a[href]')) return null;
    return control;
  }

  function singleKeyShortcutsEnabled() {
    try { return window.localStorage.getItem(SINGLE_KEY_STORAGE) !== '0'; }
    catch (error) { return true; }
  }

  function shortcutContext(target) {
    if (!target || !target.closest) return false;
    return !!target.closest([
      'input', 'textarea', 'select', 'button', '[contenteditable="true"]',
      '[role="combobox"]', '[role="listbox"]', '[role="option"]'
    ].join(','));
  }

  function isSingleShortcutKey(event) {
    if (event.ctrlKey || event.metaKey || event.altKey) return false;
    var key = event.key.length === 1 ? event.key.toLowerCase() : event.key;
    return /^[0-9]$/.test(key) || key === '?' || key === '/' || key === 'g' || key === ' ' ||
      ['a', 'c', 'h', 'l', 'm', 'o', 'p', 'q', 'r', 's', 't', 'w'].indexOf(key) !== -1;
  }

  function installSkipLink() {
    var skip = document.querySelector('.skip-link');
    var main = byId('main');
    if (!skip || !main) return;
    skip.addEventListener('click', function (event) {
      event.preventDefault();
      try {
        window.history.replaceState(window.history.state, '',
          window.location.pathname + window.location.search + '#main');
      } catch (error) {}
      try { main.focus({ preventScroll: true }); }
      catch (error) { main.focus(); }
      main.scrollIntoView({ block: 'start', behavior: 'auto' });
    });
  }

  function installEvents() {
    document.addEventListener('pointerdown', function (event) {
      var target = event.target.closest && event.target.closest('button, a, input, select, textarea, [role="button"], [role="tab"], [tabindex]');
      if (target) recordActivator(target);
    }, true);

    document.addEventListener('click', function (event) {
      var target = event.target.closest && event.target.closest('button, a, [role="button"], [role="tab"], [tabindex]');
      if (target) recordActivator(target);
    }, true);

    document.addEventListener('focusin', function (event) {
      var top = topOverlay();
      if (!top) {
        lastExternalFocus = event.target;
        return;
      }
      var root = byId(top.id);
      if (root && !root.contains(event.target)) {
        window.requestAnimationFrame(function () { focusInto(top, root); });
      }
    }, true);

    document.addEventListener('keydown', function (event) {
      if ((event.ctrlKey || event.metaKey) && String(event.key).toLowerCase() === 'k') {
        recordActivator(document.activeElement);
      }

      var semantic = semanticControlFor(event.target);
      if (semantic && (event.key === 'Enter' || event.key === ' ')) {
        event.preventDefault();
        event.stopImmediatePropagation();
        semantic.click();
        return;
      }

      var top = topOverlay();
      if (top && event.key === 'Escape') {
        event.preventDefault();
        event.stopImmediatePropagation();
        closeOverlay(top);
        return;
      }
      var copilot = byId('cpPanel');
      if (!top && event.key === 'Escape' && copilot &&
          copilot.classList.contains('show') && copilot.contains(event.target)) {
        event.preventDefault();
        event.stopImmediatePropagation();
        if (typeof window.cpToggle === 'function') window.cpToggle();
        return;
      }
      if (top && event.key === 'Tab') trapTab(event, top);
    }, true);

    document.addEventListener('keydown', function (event) {
      if (!isSingleShortcutKey(event)) return;
      if (!singleKeyShortcutsEnabled() || shortcutContext(event.target) || topOverlay()) {
        event.stopPropagation();
      }
    });
  }

  function installObserver() {
    function scheduleSync() {
      if (syncFrame) return;
      syncFrame = window.requestAnimationFrame(function () {
        syncFrame = 0;
        syncAllOverlays();
      });
    }

    observer = new MutationObserver(function (mutations) {
      var changed = false;
      mutations.forEach(function (mutation) {
        mutation.addedNodes.forEach(function (node) {
          if (node.nodeType === 1) {
            enhanceTree(node);
            changed = true;
          }
        });
        if (mutation.type === 'attributes') changed = true;
      });
      if (changed) scheduleSync();
    });
    observer.observe(document.body, {
      subtree: true,
      childList: true
    });
    overlaySpecs.concat([
      { id: 'cpPanel' }, { id: 'moreMenu' }
    ]).forEach(function (spec) {
      var root = byId(spec.id);
      if (root) observer.observe(root, { attributes: true, attributeFilter: ['class', 'style'] });
    });
  }

  function init() {
    enhanceTree(document.body);
    setupLiveRegions();
    syncAllOverlays();
    installSkipLink();
    installEvents();
    installObserver();

    if (!window.xrdR4Accessibility) {
      window.xrdR4Accessibility = Object.freeze({
        singleKeyShortcutStorageKey: SINGLE_KEY_STORAGE,
        singleKeyShortcutsEnabled: singleKeyShortcutsEnabled,
        setSingleKeyShortcutsEnabled: function (enabled) {
          try { window.localStorage.setItem(SINGLE_KEY_STORAGE, enabled ? '1' : '0'); }
          catch (error) {}
          return singleKeyShortcutsEnabled();
        },
        refresh: syncAllOverlays
      });
    }
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init, { once: true });
  else init();
}());
