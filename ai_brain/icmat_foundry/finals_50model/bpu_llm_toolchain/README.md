# F-LLM-03/04/05 Bayes-e BPU LLM sidecar

This directory is an isolated finals candidate. It reuses the frozen 24-layer
Qwen2/Qwen2.5 manual segmentation method and the official OpenExplorer
`hb_mapper` compiler. It does not replace the vendor compiler.

## Safety and runtime boundary

- Default state: `DEPLOYED_OFF`.
- On-demand only; no autostart, service registration, or listening port.
- One domain model at a time. Both segments must share the same merged-HF
  content hash; cross-model segment mixing is rejected by the manifest.
- The sidecar has no decision authority and cannot block Dashboard, the five
  frozen ports, cameras, prediction, or robot actions.
- Process exit is the required unload operation because historical
  `pyeasy_dnn` evidence shows this is the reliable CMA release boundary.
- PC OpenExplorer compilation is not X5 runtime evidence. Actual Bayes-e
  execution, memory, latency, INT8 differential, recovery, and non-interference
  remain board gates.

## Pipeline

Run from this directory:

```powershell
python audit_legacy.py
python validate_sidecar.py
python sidecar.py status
python sidecar.py inspect --model-id F-LLM-03
python sidecar.py all --model-id F-LLM-03 --calibration-count 8
```

Repeat `all` for `F-LLM-04` and `F-LLM-05` only after each real `merged_hf`
directory is complete. Missing weights produce `WAITING_FOR_MERGED_HF`; the
tool never creates placeholder weights.

The build is content addressed under `work/<inventory-id>/<content-id>/`.
Each model receives independent ONNX segments, representative calibration,
CPU tensors, compiler logs, Bayes-e bins, and evidence receipts.

## What is already evidenced

`evidence/legacy_24layer_chain_audit.v1.json` hashes the frozen manual Qwen2
implementation, mapper configurations, runtime sidecar, and four historical
compiler logs. The logs contain Bayes-e BPU operator rows and successful bin
conversion markers. The old runtime bins are not present in this PC backup, so
the audit does not claim their current availability or repeat X5 performance.

## Remaining board gates

1. Confirm actual X5 identity, runtime version, CMA, memory, and temperature.
2. Load exactly one content-hashed domain model and both matching segments.
3. Run HF/CPU versus actual INT8 BPU differential on frozen prompts.
4. Measure load, inference, unload, CMA restoration, and service
   non-interference.
5. Exit without enabling autostart or changing the frozen production slots.
