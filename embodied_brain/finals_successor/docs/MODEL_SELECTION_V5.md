# TinyOccFlow v5 Model Selection

Status: **v5r1 selected for RDK X5 board validation; board and real-session
validation remain pending.**

## Decision

The promoted laptop candidate is `tiny_occ_flow_v5r1`. It keeps the strongest
combined occupancy, dynamic-region, and tail-flow result on the same frozen
72-episode synthetic test split:

| Candidate | Occupancy mIoU | Dynamic IoU | Flow mean EPE | Flow p95 EPE | Decision |
|---|---:|---:|---:|---:|---|
| v5r1 | 0.97297 | 0.73162 | 0.08077 m | 0.27213 m | Promote to board validation |
| v5r2 | 0.96545 | 0.63625 | 0.07990 m | 0.28434 m | Do not promote |
| v5p | 0.96005 | 0.67451 | 0.09035 m | 0.29246 m | Diagnostic only |

v5r1 improves occupancy mIoU by `0.07874` over the persistence baseline and
reduces mean flow EPE by `77.42%` versus zero flow. Its conformal occupancy
interval reached `91.67%` empirical coverage at `90%` nominal coverage on the
independent synthetic test split.

## Trajectory Boundary

The learned nine-token classifier is not promoted as a planner. Its exact
top-1 result is class-imbalanced and does not beat an always-straight shortcut.
The candidate runtime therefore uses:

1. Bayes-e BPU TinyOccFlow for future occupancy, dynamic probability, flow,
   uncertainty, and an auxiliary token distribution.
2. X5 CPU rectangular-footprint arc sampling over predicted occupancy using
   the measured `0.50 x 0.40 m` chassis and `0.08 m` margin.
3. Optional Lab-FSD/Nav2 reference tokens as separate shadow evidence.
4. A weighted log-opinion pool for diagnostics only.

All three sources remain visible in the evidence JSON. No successor component
publishes `cmd_vel`, authoritative TF, or F407 commands.

## Rejected Experiment

`v5p` balanced the synthetic path labels but did not add reference-path intent
to the model input. Its degraded result confirms that route intent must enter
as an explicit CPU-side reference, not as an unobservable training target.

The machine-readable decision is
`evidence/model_selection_v5.json`. These results establish only synthetic
methodology and offline Bayes-e convertibility. Real perception, navigation,
board performance, and finals-demo non-interference remain unverified until
the embodied X5 is powered on.
