# X5 Board Receipt

This directory contains PC-only fixtures and the normalized receipt contract
for the finals Cortex shadow candidate. Nothing here opens SSH, changes a
network, starts motion, publishes to ROS, or restarts a frozen service.

## Commands

Generate a read-only collection plan:

```powershell
python embodied_brain/finals_cortex/tools/board_receipt.py commands `
  --model /home/rdk/xrd_candidates/<sha>/model.bin `
  --candidate-node /x5_finals_cortex/shadow
```

Verify the frozen manifest locally:

```powershell
python embodied_brain/finals_cortex/tools/board_receipt.py verify-manifest
```

Evaluate normalized facts:

```powershell
python embodied_brain/finals_cortex/tools/board_receipt.py evaluate `
  --input <receipt.json> `
  --verify-local-manifest
```

An incomplete or failed receipt returns exit code `2`, decision `NO_GO`, and
monitor state `MONITOR_OFFLINE`. The required response is always to leave the
candidate stopped. The tool never restarts a frozen service.

## Normalized facts

The evaluator requires:

- observed `hobot_dnn` or `hbm_runtime` compatibility and the actual backend;
- exact model and output SHA-256 values;
- actual-board p50/p95/p99 with at least 200 samples;
- PSS, MemAvailable, BPU/ION, CMA, temperature and throttle observations;
- 30 complete load/unload/recovery cycles;
- a candidate-only ROS graph with diagnostic publishers only and no service,
  action, TF or serial authority;
- all 12 files in the frozen finals manifest to match.

Compiler estimates, mapper FPS, host timing and placeholder compatibility
records are explicitly rejected.

## Fixtures

`fixtures/go.json` is the only passing fixture. The remaining fixtures cover
an explicit NO-GO hash mismatch, missing fields, compiler-estimate
impersonation, resource limits and a forbidden motion interface.
