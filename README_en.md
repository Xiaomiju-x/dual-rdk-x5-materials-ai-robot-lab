# Material-Synthesis AI Prediction and Multi-Robot Embodied Laboratory Assistant Based on Dual-RDK X5 Heterogeneous Collaboration | Dual-RDK X5 Materials Intelligence & Multi-Robot Lab Assistant | 9th (2026) National College Student Embedded Chip & System Design Competition · D-Robotics Topic · Southwest Region 1st Place

**Material-synthesis AI prediction and a multi-agent embodied laboratory assistant built on two cooperating RDK X5 edge computers.**

> The Southwest Region 1st-place result is team-confirmed; its official ranking source is still pending. The national-final award is pending official announcement.

[中文](README.md) · [Documentation](docs/README.md) · [Safe offline start](docs/getting-started/QUICKSTART_OFFLINE.md) · [Evidence index](docs/evidence/EVIDENCE_INDEX.md) · [Known limitations](docs/evaluation/KNOWN_LIMITATIONS.md)

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Platform: RDK X5](https://img.shields.io/badge/Edge-RDK%20X5-orange.svg)](docs/architecture/SYSTEM_ARCHITECTURE.md)
[![Safety: tiered](https://img.shields.io/badge/Safety-Tier%200--4-red.svg)](docs/safety/PHYSICAL_SAFETY.md)
[![Evidence: traceable](https://img.shields.io/badge/Claims-Evidence--linked-green.svg)](docs/evidence/CLAIM_MATRIX.csv)

Material-Synthesis AI Prediction and Multi-Robot Embodied Laboratory Assistant Based on Dual-RDK X5 Heterogeneous Collaboration is an edge materials-intelligence platform for optoelectronic and advanced-packaging functional materials. Near-infrared phosphors are the real validation carrier; public benchmarks and fixed-task evidence extend the engineering work toward electronic materials, XRD, process metrology, SEM, and packaging without presenting those extensions as fab-line validation.

## Competition status

The project was entered by team **荧光具身智研** in the ninth (2026) National College Student Embedded Chip and System Design Competition, Chip Application division, D-Robotics topic. [`docs/competition/award_status.yaml`](docs/competition/award_status.yaml) is the sole source of truth.

<!-- AWARD_STATUS:START -->
| Stage | Current status | Evidence boundary |
| --- | --- | --- |
| Southwest region | 1st place | `team_confirmed`; official ranking source pending |
| National final | Pending official announcement | No award may be predicted or prefilled |
<!-- AWARD_STATUS:END -->

[Award policy](docs/competition/AWARDS.md) · [Official and public sources](docs/competition/OFFICIAL_SOURCES.md)

## System overview

![Material-Synthesis AI Prediction and Multi-Robot Embodied Laboratory Assistant Based on Dual-RDK X5 Heterogeneous Collaboration physical system](assets/images/system/fig_actual_system_global.jpg)

The project is organized into five authority-separated modules:

- **AI brain:** materials candidates, evidence-grounded assistance, XRD/PL vision and numerical analysis, and an on-demand model library.
- **Embodied brain:** sensor integration, a frozen mobile-assistant workflow, and motionless BEV/occupancy-flow research candidates.
- **Dual-arm workstation:** fixed-fixture bag placement, state confirmation, and grinding; learned successors remain isolated from the physical baseline.
- **STM32F407 execution layer:** low-level communication and state for the chassis, lift, pushrod, servo, and electromagnet.
- **Command center:** a sanitized, read-only evidence presentation layer with no reverse hardware-control authority.

![Logical system architecture](assets/images/system/fig_xrd_architecture_html.png)

[Architecture, authority and status vocabulary →](docs/architecture/SYSTEM_ARCHITECTURE.md)

## Verified status

These statements are linked to the PC acceptance, independent board overlay, and dual-board closeout dated 2026-08-04.

| Scope | Result | It does **not** mean |
| --- | --- | --- |
| Model registry | 50/50 unique logical models are PC release-ready | All 50 passed on X5 |
| X5-local library | Complete status for 49 logical models: 11 frozen + 38 new | 49 models are simultaneously resident |
| 38 new candidates | 31 `X5_VALIDATED`, 4 `BOARD_EXPERIMENTAL`, 3 `BOARD_REJECTED` | Every candidate is production-ready |
| BPU-primary | 24/24 executed on an actual X5 Bayes-e BPU | Every semantic or quality gate passed |
| CPU-primary | 11 of 14 executed on an actual X5 CPU | The three rejected models executed |
| Three segmented BPU LLMs | All six segments executed; all three remain experimental | General free generation or fixed-token agreement |
| Embodied v5r1 | Actual-BPU fixed input, 200-run latency, tensor differential and 30 recovery cycles passed | Real-camera accuracy, navigation success or control authority |
| Integrated Cortex | `MONITOR_OFFLINE` | A validated live multisensor loop |
| Dual-arm successor | Passive X5 fixture replay | A real learned policy or motion authority |

The 49 X5-local entries form an **on-demand** model library. Export formats, quantization variants, prompts and random seeds do not create extra logical-model counts.

[Evidence index →](docs/evidence/EVIDENCE_INDEX.md) · [Claim matrix →](docs/evidence/CLAIM_MATRIX.csv)

## Why the three BPU LLMs remain experimental

`F-LLM-03/04/05` use separate domain weights and two BPU segments per model. The six segments executed on actual X5 hardware and their segment-content binding was verified, but every fixed next-token result diverged from its contract. The project therefore preserves both the execution evidence and the `BOARD_EXPERIMENTAL` outcome. It makes no general free-generation claim.

## Reproduction tiers

| Tier | Scope | Default authority |
| ---: | --- | --- |
| 0 | Documentation, figures, reports, static site, evidence | No installation or device |
| 1 | Mocks, fixtures, contracts, offline replay | Local PC; no hardware access |
| 2 | Licensed dataset/model evaluation | Separate versioned downloads and hashes |
| 3 | Fixed-input RDK X5 inference | Board environment; no motion authority |
| 4 | Real sensors, actuators and robots | On-site authorization, emergency stop and isolation |

Start with the [Tier 0/1 offline guide](docs/getting-started/QUICKSTART_OFFLINE.md). The following repository-root commands have been verified without network, camera, serial, GPIO, robot-SDK, or actuator access:

```bash
python -B tools/publication/audit_release.py --root . --strict
python -B tools/publication/check_markdown_links.py . --format text
python -B tools/publication/render_award_status.py --check
python -B tools/publication/generate_sbom.py --check
python -B -m unittest discover -s tests_public -p "test_*.py" -v
python -B examples/offline_demo/run_demo.py
```

The release passed the publication audit, repository-local link check, award single-source check, deterministic SPDX SBOM check, hardware-free tests, offline demo, and workstation-frontend build. See the [v1.0.0 verification record](docs/releases/v1.0.0/VERIFICATION.md) for the exact scope and commands.

> [!CAUTION]
> This repository contains source code capable of interacting with real mobile, lift, arm, pushrod, servo, and electromagnetic hardware. Source availability is not permission to run it. Tier 4 has no generic one-click command; read the [physical safety policy](docs/safety/PHYSICAL_SAFETY.md) first.

## Repository map

| Path | Purpose | Guide |
| --- | --- | --- |
| [`ai_brain/`](ai_brain/) | Materials, XRD/PL, ICMat 50-model candidate | [AI brain](docs/modules/AI_BRAIN.md) |
| [`embodied_brain/`](embodied_brain/) | ROS 2, mobile perception, successor candidates | [Embodied brain](docs/modules/EMBODIED_BRAIN.md) |
| [`workstation/`](workstation/) | Frozen dual-arm chain and replay-only successor | [Workstation](docs/modules/DUAL_ARM_WORKSTATION.md) |
| [`firmware/stm32f407/`](firmware/stm32f407/) | Embedded execution firmware | [STM32F407](docs/modules/STM32F407.md) |
| [`web/command_center/`](web/command_center/) | Read-only command center | [Command center](docs/modules/COMMAND_CENTER.md) |
| [`evidence/`](evidence/) | Acceptance and board receipts | [Evidence index](docs/evidence/EVIDENCE_INDEX.md) |
| [`schemas/`](schemas/) | Read-only status contracts and examples | [Offline start](docs/getting-started/QUICKSTART_OFFLINE.md) |
| [`assets/`](assets/) | Reviewed system images and media | Gallery below |

## Gallery

[Open the complete physical-system and demo gallery →](docs/gallery.md). Its captions preserve source hashes and truth boundaries. Any public-site image shown there is a versioned **archived screenshot/static snapshot**, not a statement about current online state and not a claimed capture of an access-protected live site.

| Mobile assistant | Dual-arm workstation |
| --- | --- |
| ![Mobile laboratory assistant](assets/images/system/fig_actual_mech_car.jpg) | ![Dual-arm workstation](assets/images/system/fig_actual_mech_workstation.jpg) |

Media must identify whether a segment is `live`, `shadow`, `replay`, or `sim-only`. The current gallery contains only reviewed media with provenance and metadata checks; identifiable team photographs remain unpublished until explicit portrait consent is available.

## Data, licensing and publication boundary

Team-owned source files without a more specific notice are provided under [Apache-2.0](LICENSE). Datasets, base models, papers, fonts, web libraries, and media retain their own terms. The machine-readable [source catalog](ai_brain/icmat_foundry/contracts/source_catalog.v1.json) records representative versions, licenses, risks, and claim boundaries.

Credentials, personal or device identities, unauthorized experimental data, per-document restricted corpora, and artifacts that cannot legally be redistributed are excluded from public release artifacts. See [third-party notices](THIRD_PARTY_NOTICES.md), [NOTICE](NOTICE), and the [publication boundary](docs/safety/PUBLICATION_BOUNDARY.md).

## Limitations, roadmap and community

Failures are part of the published engineering record. The release engineering gates pass, but the technical limits remain: three experimental BPU LLMs, no promoted live Cortex session, replay-only learned dual-arm candidates, hardware- and site-specific Tier 4 reproduction, and redistribution limits for some data and models.

[Known limitations](docs/evaluation/KNOWN_LIMITATIONS.md) · [Roadmap](ROADMAP.md) · [Changelog](CHANGELOG.md)

Contributions are welcome under [CONTRIBUTING.md](CONTRIBUTING.md). Use [SUPPORT.md](SUPPORT.md) for ordinary questions and [SECURITY.md](SECURITY.md) for private vulnerability reports. Cite a versioned archive using [`CITATION.cff`](CITATION.cff).

The project avoids unverifiable “world first” or “fully autonomous” language, does not equate phosphors with all integrated-circuit materials, does not turn replay/simulation into real-loop evidence, and does not predict a national award before an official announcement.
