# X5-RB-VoE FleetAudit

This is the finals PC-side upgrade of the existing passive RB-VoE concept. It
does not replace OpenExplorer and does not classify or route research tasks.
It audits the completed 50-model registry, evidence, content-addressed release
and claim boundaries after vendor compilation.

Default state is `DEPLOYED_OFF`. The only executable mode is an explicit,
read-only `PASSIVE_ONESHOT`:

```powershell
.venv-icmat\Scripts\python.exe icmat_foundry\finals_50model\rb_voe_fleet_audit\fleet_audit.py --mode PASSIVE_ONESHOT
```

The process hashes both release ZIPs, checks the 50/38/14/24 accounting,
unique weight lineage, 24 BPU compile states, BPU LLM split assets,
`SIM_ONLY`/`QUALITY_LIMITED` boundaries, no-autostart/no-overwrite policy, and
then emits a content-addressed receipt under `evidence/` before returning to
`DEPLOYED_OFF`.

It performs no network or X5 access, opens no model/camera/port, registers no
service and has no decision authority. `ENFORCE` is not implemented. After X5
power-on, a separate board receipt may be audited only after actual backend,
INT8 differential, unload recovery and frozen-service non-interference are
measured.

The final board-phase invocation remains a single read-only process and uses
the copied, content-addressed board receipt:

```powershell
.venv-icmat\Scripts\python.exe icmat_foundry\finals_50model\rb_voe_fleet_audit\fleet_audit.py --mode PASSIVE_ONESHOT --board-receipt icmat_foundry\finals_50model\evidence\x5_board_20260804\final_acceptance_v1\x5_board_phase_acceptance.v1.json
```

This optional input only adds hash, accounting, status-completeness,
fixed-contract differential and non-interference checks. It does not contact
the board. After exit the state remains `DEPLOYED_OFF`.
