import { RELEASE } from './src/site32/release.js?v=site32-global-commercial-v1.13-20260720';
import { installSite32Runtime } from './src/site32/runtime.js?v=site32-global-commercial-v1.13-20260720';
import { createSite32State } from './src/site32/state.js?v=site32-global-commercial-v1.13-20260720';
import { installSite32Appearance } from './src/site32/appearance.js?v=site32-global-commercial-v1.13-20260720';
import { installSite32Telemetry } from './src/site32/telemetry.js?v=site32-global-commercial-v1.13-20260720';
import { installSite32Motion } from './src/site32/motion.js?v=site32-global-commercial-v1.13-20260720';
import { installSite32A11y } from './src/site32/a11y.js?v=site32-global-commercial-v1.13-20260720';
import { installSite32Router } from './src/site32/router.js?v=site32-global-commercial-v1.13-20260720';
import { installSite32Search } from './src/site32/search.js?v=site32-global-commercial-v1.13-20260720';
import { installSite32Theater } from './src/site32/theater.js?v=site32-global-commercial-v1.13-20260720';

let bootStep = 'state';
const bootModules = ['router', 'state', 'appearance', 'search', 'theater', 'motion', 'a11y', 'telemetry'];
window.Site32Boot = Object.freeze({ ok: false, release: RELEASE, step: bootStep, state: 'starting' });

try {
  const state = createSite32State({ release: RELEASE });
  bootStep = 'appearance';
  const appearance = installSite32Appearance({ state });
  bootStep = 'telemetry';
  const telemetry = installSite32Telemetry({ state });
  bootStep = 'motion';
  const motion = installSite32Motion({ state, telemetry });
  bootStep = 'a11y';
  const a11y = installSite32A11y({ state, telemetry });
  bootStep = 'router';
  const router = installSite32Router({ state, telemetry });
  bootStep = 'search';
  const search = installSite32Search({ state, telemetry, router });
  bootStep = 'theater';
  const theater = installSite32Theater({ state, telemetry, router });

  window.Site32 = Object.freeze({
    release: RELEASE,
    contract: '/api/site32/contract',
    accessMatrix: '/api/site32/access-matrix',
    state,
    appearance,
    router,
    search,
    theater,
    motion,
    a11y,
    telemetry
  });

  bootStep = 'runtime';
  installSite32Runtime(RELEASE, { modules: bootModules });
  window.Site32Boot = Object.freeze({ ok: true, release: RELEASE, step: 'complete', state: 'ready' });
} catch (error) {
  document.body.dataset.site32Shell = 'degraded';
  document.body.dataset.site32BootError = bootStep;
  window.Site32Boot = Object.freeze({
    ok: false,
    release: RELEASE,
    step: bootStep,
    state: 'degraded',
    error: error && error.name ? String(error.name) : 'Error'
  });
  console.error('[Site32 boot]', bootStep, error);
}
