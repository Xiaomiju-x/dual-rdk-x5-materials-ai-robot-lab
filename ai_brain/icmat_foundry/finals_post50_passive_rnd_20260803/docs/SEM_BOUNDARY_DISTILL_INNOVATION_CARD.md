# SEM-BoundaryDistill innovation card

Status: `PC_RND_ONLY / RND_INNOVATION_ANCHOR / NOT_RELEASED / NOT_DEPLOYED / NOT_X5_VERIFIED`

## What is new

`SEM-BoundaryDistill` is a perception model family, not an audit tool. A
boundary-aware teacher jointly learns the defect mask, exact boundary behavior,
and a signed-distance auxiliary target. A smaller student receives the ordinary
mask supervision plus teacher-logit distillation, but its exported runtime
contract emits only mask logits. This preserves a minimal single-input,
single-output inference surface for a future manual X5 experiment.

The innovation is the combined teacher objective and the teacher-to-student
compression under a locked, real public SEM segmentation benchmark. It is not
the number of training seeds, a prompt change, or a renamed existing model.

## Locked evidence

- Data: 4,591 Carinthia/Carinthia-S SEM image-mask pairs, CC BY 4.0.
- Split: class-stratified and grouped by exact source-image SHA-256.
- Completed fit receipts: 32.
- Selection: validation only; test opened after the 32-fit selection lock.
- Locked primary variant: `robust`.
- Locked-test Dice: `0.8161325553560742`.
- Locked-test exact boundary F1: `0.12063885024482687`.
- Declared old boundary reference: `0.086495`.
- Relative boundary-F1 improvement over that reference: approximately `39.5%`.
- Dice guardrail: `>= 0.801`, passed.
- ONNX checker, fixed CPU ORT fixture, and FP32 parity: passed.

The family receipt is
`../evidence/RND-SEM-01/family_receipt.v1.json`. The training-intensity and
control-flow corrections are retained in
`../evidence/RND-SEM-01/epoch_extension.v1.json`; negative seeds and superseded
receipts were not deleted.

## Difference from RB-VoE

`SEM-BoundaryDistill` performs scientific perception inference: image in, mask
logits out. RB-VoE/FleetAudit performs passive evidence consistency checks: it
does not infer SEM masks, load this model, select models, or control production.
Neither component has enforcement or motion authority, and neither is a
startup dependency of the other. RB-VoE remains `DEPLOYED_OFF` except for an
explicit future `PASSIVE_ONESHOT` audit.

## Deployment boundary

No worker, service, daemon, listening port, dashboard route, camera owner,
production model slot, or automatic selector was created. A future board test
may only use a manually started one-model worker with one explicit immutable
input, one content-addressed receipt, and immediate resource release. PC ONNX
results are not X5 latency, BPU, memory, temperature, or INT8 evidence.

No X5 access is authorized until the user separately confirms that the AI-brain
X5 is powered and the PC has been manually connected to the same LAN.
