export function installSite32Runtime(release, context = {}) {
  if (!document.body) return;
  document.body.dataset.site32Shell = 'true';
  document.body.dataset.site32Release = release;
  if (context.modules) {
    document.body.dataset.site32Modules = context.modules.join(',');
  }
  const detail = Object.freeze({
    release,
    modules: Object.freeze((context.modules || []).slice())
  });
  window.dispatchEvent(new CustomEvent('site32ready', {
    detail
  }));
  return detail;
}
