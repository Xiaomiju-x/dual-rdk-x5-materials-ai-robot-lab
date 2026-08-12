#!/usr/bin/env node
/* Static browser contract for Site31 R4 and Site32. Uses only Node built-ins. */
'use strict';

const fs = require('fs');
const path = require('path');

const root = path.resolve(__dirname, '..');
const read = (name) => fs.readFileSync(path.join(root, name), 'utf8');
const appPy = read('app.py');
const configPy = fs.existsSync(path.join(root, 'cmdcenter', 'config.py')) ? read('cmdcenter/config.py') : '';
const html = read('static/index.html');
const appJs = read('static/app.js');
const i18nJs = read('static/i18n.js');
const twinJs = read('static/twin.js');
const r4Js = read('static/r4.js');
const r4Performance = read('static/r4-performance.js');
const r4Accessibility = read('static/r4-accessibility.js');
const site32Js = fs.existsSync(path.join(root, 'static', 'site32.js')) ? read('static/site32.js') : '';
const style = read('static/style.css');
const r4Style = read('static/r4.css');
const site32Style = fs.existsSync(path.join(root, 'static', 'site32.css')) ? read('static/site32.css') : '';
const sw = read('static/sw.js');
const frontendJs = [appJs, i18nJs, twinJs, r4Js, r4Performance, r4Accessibility, site32Js].join('\n');
const frontendCss = `${style}\n${r4Style}\n${site32Style}`;
const failures = [];
let passed = 0;

function check(name, condition, detail) {
  if (condition) {
    passed += 1;
    process.stdout.write(`PASS ${name}\n`);
  } else {
    failures.push(`${name}${detail ? `: ${detail}` : ''}`);
    process.stdout.write(`FAIL ${name}${detail ? `: ${detail}` : ''}\n`);
  }
}

const releaseMatch = (configPy || appPy).match(/^ASSET_VER\s*=\s*["']([^"']+)["']/m);
const release = releaseMatch ? releaseMatch[1] : '';
const releasePattern = /^(?:site31-global-commercial-r4(?:\.\d+)?|site32-global-commercial-v\d+(?:\.\d+)?)-\d{8}$/;
check('release.backend_constant', releasePattern.test(release), release || 'missing');
check('release.html_css_query', html.includes(`/style.css?v=${release}`));
check('release.html_js_query', html.includes(`/app.js?v=${release}`));
['r4.css', 'r4.js', 'r4-performance.js', 'r4-accessibility.js'].forEach((asset) => {
  check(`release.html_${asset.replace(/[^a-z0-9]+/gi, '_')}`, html.includes(`/${asset}?v=${release}`));
});
if (release.startsWith('site32-')) {
  check('release.html_site32_css', html.includes(`/site32.css?v=${release}`));
  check('release.html_site32_js', html.includes(`/site32.js?v=${release}`));
  check('release.site32_module_entry', site32Js.includes("window.Site32 = Object.freeze"));
  check('release.site32_contract_endpoint', appPy.includes('register_site32(app'));
}
check('release.i18n_default', new RegExp(`I18N_DEFAULT_VER\\s*=\\s*['"]${release.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}['"]`).test(appJs));
check('release.service_worker_cache', release && sw.includes(release), 'cache names must include full release');

check('a11y.default_chinese', /<html\b[^>]*\blang=["']zh-CN["']/i.test(html));
check('a11y.skip_link', /<a\b[^>]*href=["']#main["'][^>]*class=["'][^"']*skip-link/i.test(html));
check('a11y.main_focus_target', /id=["']main["'][^>]*role=["']main["'][^>]*tabindex=["']-1["']/i.test(html));
check('a11y.skip_link_focus_handoff', r4Accessibility.includes('installSkipLink') &&
  r4Accessibility.includes("window.location.search + '#main'") &&
  r4Accessibility.includes("main.focus({ preventScroll: true })") &&
  r4Accessibility.includes("main.scrollIntoView({ block: 'start', behavior: 'auto' })"));
check('a11y.route_announcer', /id=["']routeAnnouncer["'][^>]*aria-live=["']polite["']/i.test(html));
check('a11y.search_combobox', /id=["']homeSearchInput["'][^>]*role=["']combobox["']/i.test(html));
check('a11y.search_listbox', /role=["']listbox["']/i.test(html));
check('a11y.search_filters', ['homeSearchKind', 'homeSearchStatusFilter', 'homeSearchSource']
  .every((id) => html.includes(`id="${id}"`)));
check('a11y.search_active_descendant', /aria-activedescendant/i.test(html + frontendJs));
check('a11y.search_keyboard_model', ['ArrowDown', 'ArrowUp', 'Escape', 'Enter'].every((key) => appJs.includes(key)));
check('a11y.dynamic_busy_state', /aria-busy/i.test(html + frontendJs));
check('search.share_url_roundtrip', r4Js.includes("params.set('q', query)") &&
  r4Js.includes("window.history.replaceState({ site32Search: true }") &&
  r4Js.includes('restoreFromUrl()'));
check('search.recoverable_errors', r4Js.includes('retryItem(query, reason)') &&
  r4Js.includes("error.status === 429") && r4Js.includes('navigator.onLine === false'));
check('search.evidence_detail_route', r4Js.includes("detailOpen('evidence'") &&
  appJs.includes("DETAIL_KIND==='evidence'") && appJs.includes('detailRenderEvidence()'));
check('a11y.focus_visible', /:focus-visible/.test(frontendCss));
check('a11y.reduced_motion', /prefers-reduced-motion\s*:\s*reduce/.test(frontendCss));
check('a11y.closed_overlay_inert_hook', /\binert\b/.test(html + frontendJs));
check('a11y.semantic_sort_headers', ['tagName === \'TH\'', "setIfMissing(element, 'aria-sort'", 'r4-th-button']
  .every((token) => r4Accessibility.includes(token)));
check('a11y.copilot_non_modal_region', !/\{\s*id:\s*['"]cpPanel['"][^}]*close:/.test(r4Accessibility) &&
  r4Accessibility.includes("controlled.setAttribute('role', 'region')"));
check('a11y.copilot_escape_from_input', r4Accessibility.includes("copilot.classList.contains('show')") &&
  r4Accessibility.includes("typeof window.cpToggle === 'function'"));
check('a11y.single_key_switch', appJs.includes('kbdSingleKeys') && appJs.includes('_singleKeyShortcutsEnabled'));

check('performance.render_tier_event', appJs.includes("addEventListener('r4renderchange'") &&
  r4Performance.includes("CustomEvent('r4renderchange'"));
check('performance.static_stops_raf', appJs.includes("renderTier!=='balanced'") &&
  appJs.includes('cancelAnimationFrame(threeRaf)'));
check('performance.three_balanced_only', appJs.includes("r4RenderTier()!=='balanced'"));
check('performance.twin_lazy_three', appJs.includes('window.ensureThreeLibrary=loadThreeSceneOnce') &&
  twinJs.includes('window.ensureThreeLibrary()'));
check('performance.twin_hidden_poll_stop', twinJs.includes('if(!visible || document.hidden) return;'));
check('performance.sse_route_gated', appJs.includes("function streamWanted(){ return !document.hidden&&cur==='ops'; }") &&
  appJs.includes('syncStreamForRoute'));
check('performance.visibility_poll_gate', /if\(document\.hidden\) return;/.test(appJs) &&
  appJs.includes("if(!document.hidden&&(cur==='home'||cur==='status'))"));
const kpiFallback = appJs.match(/const KPI_FALLBACK=([\s\S]*?)\n\}\};/);
check('truth.no_numeric_kpi_fallback', !!kpiFallback && !/\b(?:91\.7|26\.4|649|1286)\b/.test(kpiFallback[0]) &&
  /source:['"]unknown['"]/.test(kpiFallback[0]));
check('truth.fleet_unknown_not_live_or_mirror', !appJs.includes('const _miss=') &&
  appJs.includes("unknown:'未知 (当前无快照)'") && appJs.includes("const on = s.online===true"));

check('sw.version_mismatch_bypass', sw.includes('requestedVersion !== RELEASE') &&
  sw.includes("fetch(request, { cache: 'no-store' })"));
check('sw.fixed_offline_shell', !sw.includes('cache.put(offlineRequest') && sw.includes('caches.match(offlineRequest)'));

const releaseTokens = new Set((appPy + '\n' + configPy + '\n' + html + '\n' + frontendJs + '\n' + sw)
  .match(/(?:site31-global-commercial-r(?:\d+(?:\.\d+)?)|site32-global-commercial-v(?:\d+(?:\.\d+)?))-\d{8}/g) || []);
check('release.no_split_tokens', releaseTokens.size === 1 && releaseTokens.has(release), [...releaseTokens].join(', '));

process.stdout.write(`\nSite32 browser/static contract: ${passed} PASS / ${failures.length} FAIL\n`);
if (failures.length) {
  failures.forEach((failure) => process.stdout.write(` - ${failure}\n`));
  process.exitCode = 1;
}
