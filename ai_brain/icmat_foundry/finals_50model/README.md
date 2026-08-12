# X5-ICMat Foundry 50-Model Finals Candidate

This directory is the isolated, competition-fast-track candidate for the
national finals AI-brain upgrade. It does not modify or replace the frozen
Dashboard, five production ports, camera ownership, startup scripts, or the
existing CPU/BPU model slots.

Current state: `X5_BOARD_PHASE_COMPLETE_WITH_REJECTIONS_AND_EXPERIMENTALS`

The 2026-08-04 isolated board phase is complete: the 38 new candidates have
`31 X5_VALIDATED / 4 BOARD_EXPERIMENTAL / 3 BOARD_REJECTED`. All 24
BPU-primary candidates executed on the actual Bayes-e BPU; 11 of 14
CPU-primary candidates executed on the actual X5 CPU. See
`evidence/x5_board_20260804/final_acceptance_v1/X5_BOARD_PHASE_FINAL_RECEIPT_20260804.md`.

The accounting contract is strict:

- 11 minimum unique logical models in the frozen X5 baseline;
- 38 new unique logical X5 models (14 CPU-primary and 24 BPU-primary);
- 1 PC-offline MACE-MPA-0 model;
- 50 minimum unique logical models in total, with at least 49 X5-local.

The generated registry is built from the approved master plan. CPU/BPU export
variants, prompts, random seeds, checkpoints, and quantization variants do not
create additional logical models.

The 2026-08-01 PC acceptance is complete:

- 50/50 registry contracts are release-ready with no acceptance gap;
- all 38 finals models have model-specific evidence and unique weight lineage;
- all 24 finals BPU-primary models have real PC OpenExplorer compiler receipts;
- canonical weight hash collisions: 0;
- frozen production baseline: 14/14 file hashes pass;
- X5 board-verified models: 0, by design, until the AI-brain X5 is powered on.

The three new domain BPU LLMs are independent Qwen2.5-0.5B weights. Each uses
the hand-written 24-layer Qwen2 implementation split into layers 0-11 and
12-23. HF/manual/ONNX FP32 differential checks pass, and both segments compile
with 648 BPU operator rows each. Compiler output remains PC toolchain evidence,
not actual X5 latency, memory, or INT8 semantic-differential evidence.

## Acceptance commands

```powershell
.venv-icmat\Scripts\python.exe icmat_foundry\finals_50model\tools\build_registry.py
.venv-icmat\Scripts\python.exe icmat_foundry\finals_50model\tools\verify_registry.py
.venv-icmat\Scripts\python.exe icmat_foundry\finals_50model\tools\verify_frozen_baseline.py
.venv-icmat\Scripts\python.exe icmat_foundry\finals_50model\tools\build_final_acceptance.py --strict
```

Authoritative acceptance:

- `evidence/final_acceptance/final_acceptance.v1.json`
- acceptance current workspace / frozen-release embedded SHA-256:
  `128a9d14050af63882054cccd9c3b30e41f8acda190719e0c25af280cf47a9ce`

Content-addressed releases:

- PC: `releases/x5-icmat-foundry-50model-pc-c7aff501602bde2f.zip`
  SHA-256 `c7aff501602bde2f31881bdbc9d872c9570162359e6b6e8596b463c73add2a81`
- X5 staging: `releases/x5-icmat-foundry-50model-x5-staging-c5fa215a58168c0c.zip`
  SHA-256 `c5fa215a58168c0cb7274c2b1cf6d66bcd0f3c1e70d3f4cf13749e9b57dafb52`

Both releases are `MANUAL_STAGING_ONLY`, contain no startup service, cannot
overwrite production, and leave RB-VoE at `DEPLOYED_OFF`. X5 power-on work must
start with read-only identity/resource/frozen-service verification, then deploy
to an isolated model bank and test one model at a time. Dashboard, five frozen
ports, camera ownership, CPU/BPU slots, and the competition demo remain
unchanged until those board gates pass.

The PC-side RB-VoE FleetAudit is implemented at `rb_voe_fleet_audit/`. One
explicit `PASSIVE_ONESHOT` audited both release ZIPs and the complete registry:
44/44 checks passed, receipt SHA-256
`12454475da916f7b59ae31b9a91f82608337ed43129f5f78586bd5b94369ec2e`.
It returned to `DEPLOYED_OFF` and performed no X5/network access, model loading,
production write, service registration, port/camera access, or enforcement.

After board closeout, a separate read-only FleetAudit verified the copied
board receipt and all referenced evidence: `59/59 PASS`, receipt SHA-256
`1a757d570141bb75bc949fd87b6637cc9d5ea3d2df5ef2155d31ec080185d2a7`.
It returned to `DEPLOYED_OFF` and did not contact either X5.
