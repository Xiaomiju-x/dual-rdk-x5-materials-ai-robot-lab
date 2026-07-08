# Public Boundary

This repository is intentionally a public engineering boundary, not the full laboratory repository.

The uploaded code is sufficient to review the embedded system shape: public UI, report generation, interface contracts, workstation visualization, collision interlock, mock telemetry, and edge-facing adapters. The private project tree remains local because it contains unpublished materials data, hardware deployment details, model artifacts, and control scripts that should not be published during preliminary competition review.

Security checks before packaging:

- No private IP address such as `192.168.*.*` is allowed in this folder.
- No API key assignment or private key block is allowed.
- No model weight formats are included.
- No script in this folder can directly command chassis velocity, mechanical-arm motion, electromagnet state, or lift motion against real hardware.
