# X5-Embodied Cortex

Status: **PC foundation accepted; shared v5r1 BPU board primitive accepted,
but Cortex remains `MONITOR_OFFLINE` pending a candidate ROS graph, complete
real sensor session and readable allocation evidence.**

The 2026-08-04 zero-motion board work validated the underlying v5r1
TinyOccFlow/CamSemLite BPU primitives, not the integrated Cortex runtime. No
Cortex deploy package, service or `current` activation was created.

This is an isolated, passive successor candidate for the finals embodied-brain
stack. It adds data, verification, memory, world-model, trajectory-teacher and
trust infrastructure without changing the validated finals demonstration.

The frozen demonstration remains:

```text
bash ~/tools/finals_lift_nav_demo.sh
F407 build 2026071907
0.50 m odometry-closed-loop transport
```

No code in this directory may publish motion commands, TF, services or actions,
open the F407 serial link, modify firmware, register an automatic service, or
train online. Candidate failure is always `MONITOR_OFFLINE`.

## Architecture

```text
LiDAR / depth / odometry / Vision-BEV
                 |
        read-only session recorder
                 |
     +-----------+------------+
     |                        |
skill graph + evidence   sparse episodic memory
     |                        |
     +-----------+------------+
                 |
      CrossBEV knowledge student
 seven distinct maps: obstacle / traversability / semantic /
 dynamic / visibility / unknown / confidence
                 |
          NavTeacher-15
 15 fixed trajectory proposals and decomposed regret metrics
                 |
 TrustLab: calibration / conformal / OOD / drift / disagreement
                 |
 PASSIVE_OK / REVIEW / MONITOR_OFFLINE
                 |
          evidence only; no control
```

## Modules

| Directory | Implemented role | Authority |
|---|---|---|
| `recorder/` | Timestamp/sequence synchronization, provenance, payload and manifest SHA-256, tamper detection | Read-only |
| `skill_graph/` | Verifies pickup, lift, transport, lower, release and reset ordering | Evidence-only |
| `memory/` | SQLite sparse scene graph, TTL/provenance, hard-case mining and hash chain | Read-only |
| `crossbev/` | Temporal monocular student contract, seven-map output and distillation loss | Offline/shadow |
| `navteacher/` | Fifteen candidates, seven cost terms, rank/regret/top-k metrics | Proposal-only |
| `trust/` | Calibration, risk-coverage, dual-track conformal, OOD, drift and cross-modal disagreement | Diagnostic-only |
| `board/` | Fail-closed evaluation of already collected X5 runtime facts | Read-only |
| `runtime/` | Integrated synthetic fixture proving module composition and boundaries | PC fixture only |

## Reproduce The PC Gate

From the repository root:

```powershell
.\.venv-icmat\Scripts\python.exe -m pytest `
  embodied_brain/finals_cortex/tests -q -p no:cacheprovider

.\.venv-icmat\Scripts\ruff.exe check embodied_brain/finals_cortex

.\.venv-icmat\Scripts\python.exe -m `
  embodied_brain.finals_cortex.tools.verify_non_interference --json

.\.venv-icmat\Scripts\python.exe -m `
  embodied_brain.finals_cortex.runtime.passive_replay `
  --output-root embodied_brain/finals_cortex/evidence/pc_fixture

.\.venv-icmat\Scripts\python.exe -m `
  embodied_brain.finals_cortex.tools.build_pc_acceptance

.\.venv-icmat\Scripts\python.exe -m `
  embodied_brain.finals_cortex.tools.package_candidate
```

The package is deliberately labelled `PC_TOOLING_NOT_X5_DEPLOY`. It must not
be copied to either board as a runtime deployment until the board runbook has
produced a valid compatibility receipt.

## Evidence Boundary

The PC fixture proves implementation, integration, integrity checks and
non-interference. It does **not** prove:

- actual X5 BPU execution, latency, TOPS utilization, memory or temperature;
- real LiDAR/depth/4K synchronization or navigation accuracy;
- physical pickup or transport success;
- autonomous-driving control or functional-safety certification.

The accurate FSD-related description is:

> A public BEV, occupancy-flow and planning-oriented autonomous-driving
> inspired shadow world model, adapted to a LiDAR/depth/4K indoor laboratory
> robot. It does not reproduce Tesla software, weights, data or architecture.

See [ALGORITHM_PROVENANCE.md](docs/ALGORITHM_PROVENANCE.md) and
[X5_UNIFIED_DEPLOYMENT_RUNBOOK.md](docs/X5_UNIFIED_DEPLOYMENT_RUNBOOK.md).
