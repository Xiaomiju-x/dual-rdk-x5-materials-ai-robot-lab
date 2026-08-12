## What changed

<!-- Describe the smallest reviewable change and why it is needed. -->

## Evidence

- [ ] `python -B tools/publication/audit_release.py --root . --strict`
- [ ] `python -B tools/publication/check_markdown_links.py . --format text`
- [ ] `python -B tools/publication/render_award_status.py --check`
- [ ] `python -B tools/publication/generate_sbom.py --check`
- [ ] `python -B tools/publication/verify_media.py --root .`
- [ ] `python -B -m unittest discover -s tests_public -p "test_*.py" -v`
- [ ] `gitleaks dir . --config .gitleaks.toml --redact=100 --no-banner`
- [ ] Frontend build, when `workstation_frontend_public/` changed
- [ ] Documentation links and claim/evidence mapping updated

## Boundary and safety

- [ ] No credential, private host identity/address, personal data, restricted dataset, model weight, or build cache was added.
- [ ] Any live/shadow/replay/sim-only claim is labeled accurately.
- [ ] No physical-motion test is part of automated CI.
- [ ] New media has provenance, metadata removal, and required participant permission.
