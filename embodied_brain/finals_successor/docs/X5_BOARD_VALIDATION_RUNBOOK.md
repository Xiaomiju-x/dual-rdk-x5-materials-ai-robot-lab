# RDK X5 Board Validation Runbook

Status: **pure BPU/fixed-input board gate complete; live ROS shadow gate pending.**

Measured results and content hashes are frozen in
`X5_BOARD_ACCEPTANCE_20260804.md`. Do not rerun or activate the candidate for
the competition demonstration merely because the board gate passed.

This runbook validates the independent X5-TriBEV-Flow + ShadowGuard candidate.
It does not modify the frozen entry, F407 build, navigation configuration, or
validated `0.50 m` demonstration.

## Preconditions

- The user confirms the PC and embodied X5 are on the same current LAN defined
  by the repository root `AGENTS.md`; this runbook never changes connectivity.
- Use only `rdk@192.0.2.85` after ED25519 host-key and hostname checks.
- Do not change PC Wi-Fi, TUN, VPN, proxy, route, or ARP state.
- Vehicle, lift, and F407 must be idle for model profiling.
- Do not run the frozen physical demonstration merely to validate this
  candidate. A physical run still requires the user's explicit safety
  confirmation at that time.

## 1. Reverify Laptop State

From the repository root:

```powershell
.venv-icmat\Scripts\python.exe embodied_brain\finals_successor\tools\verify_finals_baseline.py --json
.venv-icmat\Scripts\python.exe embodied_brain\finals_successor\tools\build_pc_acceptance_report.py
```

Required result: baseline `ok=true`, PC acceptance
`PC_ACCEPTED_BOARD_PENDING`, and no frozen hash mismatch.

## 2. Read-Only Board Identity and Baseline Profile

Copy only the candidate tooling to a new board directory. Do not overwrite
`~/tools`, `~/ros2_ws`, services, or frozen artifacts. Then run:

```bash
bash ~/x5_tribev_flow_successor/tools/profile_candidate_x5.sh \
  --snapshot-only
```

Record hostname, RDK OS, runtime packages, `hrt_model_exec`, memory, CMA/ION,
PSS, temperatures, BPU status, and frozen service state. This step must not
start or stop any service.

## 3. Compatibility Gate

Compare the board runtime inventory with the pinned conversion evidence:

- target: RDK X5 / `bayes-e`
- `hb_mapper`: 1.24.3
- OpenExplorer image:
  `openexplorer/ai_toolchain_ubuntu_20_x5_cpu:v1.2.8-py310`
- DDK fingerprint:
  `00c8a25ece06fa219283774e6e33645d7e21bea6c17645e93b8d20be2a677251`

Create `bpu/compatibility/compatibility_record.json` only from observed board
facts. Do not copy the example unchanged. Decision must be `compatible`, no
placeholder may remain, and no system upgrade is permitted.

## 4. Immutable Deploy Package

After compatibility is established:

```powershell
.venv-icmat\Scripts\python.exe embodied_brain\finals_successor\tools\package_successor.py --kind deploy --json
```

The package must contain exactly these reviewed model binaries:

```text
bpu/artifacts/tiny_occ_flow/90e01859991c2eab/tiny_occ_flow.bin
bpu/artifacts/cam_sem_lite/cb582808a90ae93c/cam_sem_lite.bin
```

Verify the archive with `package_successor.py --verify-archive` before
transfer. Extract into a new content-addressed directory and point
`X5_TRIFLOW_ROOT` at it; do not replace an existing candidate in place.

## 5. Pure BPU Profiling

While all physical motion is idle:

```bash
bash "$X5_TRIFLOW_ROOT/tools/profile_candidate_x5.sh" \
  --model "$X5_TRIFLOW_ROOT/bpu/artifacts/tiny_occ_flow/90e01859991c2eab/tiny_occ_flow.bin" \
  --model "$X5_TRIFLOW_ROOT/bpu/artifacts/cam_sem_lite/cb582808a90ae93c/cam_sem_lite.bin" \
  --frame-count 200 \
  --ack-idle
```

Compiler estimates are not substituted for this result. Record actual model
load, p50/p95/p99 where available, BPU state, memory/CMA before and after, and
temperature. `10 TOPS` remains a peak specification, not measured utilization.

## 6. Read-Only Live Collection

Start the collector manually:

```bash
X5_TRIFLOW_ROOT="$HOME/<candidate-dir>" \
  bash "$HOME/<candidate-dir>/runtime/start_x5_tribev_collector.sh"
```

Verify:

- subscriptions exist for LiDAR, depth scan, odometry, Vision-BEV, and
  Lab-FSD reference evidence;
- publisher, service, action, TF, serial, F407, and control counts are zero;
- incomplete `+0.4/+0.8/+1.2 s` targets remain raw-only;
- promoted episode manifests say `source=real`, `labels=pseudo`;
- no data are described as human ground truth.

Stop with the paired `stop_x5_tribev_collector.sh`. If the stored PID identity
does not match, the stop script must refuse to kill it.

## 7. Shadow Runtime

Start only after the BPU and topic gates pass:

```bash
X5_TRIFLOW_ROOT="$HOME/<candidate-dir>" \
  bash "$HOME/<candidate-dir>/runtime/start_x5_tribev_shadow.sh"
```

Expected outputs are restricted to `/x5_triflow_shadow/*`. Verify:

- six diagnostic publishers only;
- no `/cmd_vel`, `/cmd_vel_safe`, TF, service, action, serial, or F407 output;
- BPU future occupancy/dynamic/flow outputs update on new LiDAR frames;
- CPU footprint scoring uses the measured `0.50 x 0.40 m` body, `0.08 m`
  margin, and odometry speed;
- evidence records raw model, occupancy-conditioned, reference, and fused
  token distributions separately;
- stale or absent 4K semantics are labeled, not silently treated as live;
- a model/topic error produces `MONITOR_OFFLINE` and does not block the
  existing demo.

Run a bounded idle/rolling observation, then stop with
`stop_x5_tribev_shadow.sh`. Confirm memory/CMA recovery and no orphan process.

## 8. Promotion Decision

Promotion remains manual and shadow-only. Accept only if all are true:

- actual TinyOccFlow p95 is below the 10 ms design gate at 5 Hz;
- added RSS is below 300 MiB;
- incremental CMA stays below the 96 MiB target and 160 MiB hard limit;
- observed CMA reserve remains at least 150 MiB;
- no thermal throttling or unrecovered allocation appears;
- all candidate interfaces remain diagnostic-only;
- stopping or crashing the candidate leaves frozen services unchanged;
- frozen baseline hashes still pass.

CamSemLite remains disabled unless a separate real-camera probe is performed.
No board result promotes it to a real semantic-accuracy claim without labeled
real-camera evaluation.

## Rollback

Stop the collector and shadow monitor with their exact paired stop scripts.
Remove no frozen file and restart no frozen service. Point
`X5_TRIFLOW_ROOT` back to the prior content-addressed candidate, or leave the
candidate stopped. The validated finals entry remains:

```text
bash ~/tools/finals_lift_nav_demo.sh
```
