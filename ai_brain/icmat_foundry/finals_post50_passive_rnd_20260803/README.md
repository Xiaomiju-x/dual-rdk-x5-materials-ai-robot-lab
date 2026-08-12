# Post-50 passive/minimal R&D bank

Status: `PC_RND_ONLY / NOT_IN_OFFICIAL_ACCOUNTING / NOT_RELEASED / NOT_DEPLOYED / NOT_X5_VERIFIED`

This directory is an isolated PC-only research bank. It does not modify or
extend the authoritative 50-model registry, acceptance receipt, PC release, or
X5 staging release.

The training target is 144 core fits and at most 16 conditional JARVIS fits.
Eight R&D model families are evaluated, while no more than 70 canonical
candidate weights may be registered in the R&D bank. Trial checkpoints and
seeds are not additional logical models.

Runtime policy is `PASSIVE_MINIMAL_MANUAL`: an eventual candidate worker may
be started explicitly, load exactly one candidate, consume one explicitly
provided immutable input, write a new content-addressed receipt, and exit. No
worker is implemented as a service, daemon, port, watcher, model router, camera
owner, production dependency, or robot controller.

RB-VoE remains external to these models. Its current authoritative state is
`DEPLOYED_OFF`; its only executable mode is `PASSIVE_ONESHOT`. This R&D bank
does not modify FleetAudit and does not claim that the current FleetAudit can
audit post-50 board receipts.

## Completed outcome

The confirmed overnight plan is complete:

- `160/160` completed receipted fits.
- `15` isolated candidate weights with static ONNX and fixed ORT fixtures.
- `3` family gates passed: `RND-SEM-01`, `RND-MAT-02`, and `RND-MAT-03`.
- `RND-SEM-01 SEM-BoundaryDistill` is the innovation anchor; its validation-selected
  robust student achieved locked-test Dice `0.8161325554` and exact boundary F1
  `0.1206388502`.
- `8/8` frozen official files passed the post-run hash comparison.
- Canonical candidate checkpoint hash collisions: `0`.
- New releases, OpenExplorer runs, deployments, X5 contacts, network actions,
  production writes, services, and listening ports: `0`.

Authoritative isolated outputs:

- Registry: `contracts/rnd_registry.v1.json`
- Acceptance: `evidence/final_acceptance/post50_execution_acceptance.v1.json`
- Human summary: `evidence/final_acceptance/EXECUTION_SUMMARY.md`
- Innovation card: `docs/SEM_BOUNDARY_DISTILL_INNOVATION_CARD.md`

The official accounting remains exactly 50. These post-50 candidates are not
released, not deployed, and not X5 verified.
