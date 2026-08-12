# SEM-Metrology-X5 v2 candidate

This isolated candidate asks one question: can the frozen Dice ~0.5393 subset
baseline be replaced by a useful, BPU-oriented SEM dimensional-metrology
segmenter without leaking NIST set 6?

The answer is currently **HOLD_DATA**. The official `intensity_sets.zip` is not
present, and the published set 4 mask in the locally verified NIST archive is
not valid binary ground truth. The code therefore refuses to train, calibrate,
open set 6, run the BPU mapper, contact an X5, or integrate with production.

The proposed model is one small static U-Net with skip connections and a
quality head. The quality head supports calibrated refusal on low-quality
simulated SEM images. It is an architecture candidate, not a deployed model.

Run the fail-closed audit:

```powershell
C:\Users\YOUR_USER\miniconda3\envs\xrd\python.exe tools\build_sem_metrology_v2_candidate.py
```

Outputs are written to
`evaluation/icmat_foundry/sem_metrology_v2/`. No set 6 payload is read.
