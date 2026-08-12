# SEM-Metrology-X5-Lite

This module is an isolated finals candidate for binary segmentation of the
simulated SEM images published by NIST under DOI `10.18434/mds2-3838`.

It is deliberately not connected to the AI Brain Dashboard, services, model
slots, or an RDK X5. The network uses a fixed `1x1x128x128` input and only
Conv, ReLU, MaxPool, nearest-neighbor Resize, and a final 1x1 Conv. ONNX export
has no dynamic axes and no transposed convolution.

## Current Evidence Level

- Candidate: `OFFICIAL_SUBSET_BASELINE`
- Training inputs: the valid maximum-contrast/minimum-noise reference pairs
  from official sets 1, 2, 3, and 5
- Locked test: official set 6
- Full `intensity_sets.zip`: unavailable on the current network path
- Official set 4 mask: rejected because it is pixel-identical to the set 4
  intensity image and is not binary
- Release eligible: no
- Mapper complete: no
- RDK X5 actual backend tested: no

Run the reproducible local build from the repository root:

```powershell
C:\Users\YOUR_USER\miniconda3\envs\xrd\python.exe `
  tools\build_sem_metrology_x5.py --epochs 50 --batch-size 32 --device cuda
```

Use `--require-full-corpus` to exercise the fail-closed gate. It must reject
the current data state rather than silently treating this subset as the full
six-set corpus.

This data consists of simulated SEM images of quasi-circular structures. It is
not real wafer-defect data and does not establish production-fab performance.
