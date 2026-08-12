## What changed

<!-- Describe the smallest reviewable change and why it is needed. -->

## Evidence

- [ ] `python tools/publication/audit_release.py --root . --strict`
- [ ] `python -m unittest discover -s tests_public -v`
- [ ] Frontend build, when `workstation_frontend_public/` changed
- [ ] Documentation links and claim/evidence mapping updated

## Boundary and safety

- [ ] No credential, private host identity/address, personal data, restricted dataset, model weight, or build cache was added.
- [ ] Any live/shadow/replay/sim-only claim is labeled accurately.
- [ ] No physical-motion test is part of automated CI.
- [ ] New media has provenance, metadata removal, and required participant permission.
