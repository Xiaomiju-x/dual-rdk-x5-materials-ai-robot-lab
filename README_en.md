# Material-Synthesis AI Prediction and Multi-Robot Embodied Laboratory Assistant Based on Dual-RDK X5 Heterogeneous Collaboration | 2026 National College Student Embedded Chip and System Design Competition · Chip Application Division · D-Robotics Topic | Southwest Regional First Prize · National Final Second Prize

**Official Chinese project title:** 基于双 RDK X5 异构协同的材料合成 AI 预测与多机具身实验助理机器人

> **2026 National College Student Embedded Chip and System Design Competition · Chip Application Division · D-Robotics Topic · Southwest Regional First Prize · National Final Second Prize**
>
> Team: **荧光具身智研**. The Southwest Regional First Prize and National Final Second Prize are team-confirmed; official organizing-committee award sources are still pending.

[![Latest project hardware: mobile laboratory assistant and dual-arm workstation](assets/media/hero/project-hardware-hero.webp)](docs/gallery.md)

Two RDK X5 computers divide materials-AI and embodied-perception workloads, connecting material candidates, XRD/PL analysis, a mobile laboratory assistant, dual arms, an STM32F407 execution layer, and a read-only evidence portal into a traceable, authority-separated experimental system.

[![CI](https://github.com/Xiaomiju-x/dual-rdk-x5-materials-ai-robot-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/Xiaomiju-x/dual-rdk-x5-materials-ai-robot-lab/actions/workflows/ci.yml)
[![CodeQL](https://github.com/Xiaomiju-x/dual-rdk-x5-materials-ai-robot-lab/actions/workflows/codeql.yml/badge.svg)](https://github.com/Xiaomiju-x/dual-rdk-x5-materials-ai-robot-lab/actions/workflows/codeql.yml)
[![Gitleaks](https://github.com/Xiaomiju-x/dual-rdk-x5-materials-ai-robot-lab/actions/workflows/gitleaks.yml/badge.svg)](https://github.com/Xiaomiju-x/dual-rdk-x5-materials-ai-robot-lab/actions/workflows/gitleaks.yml)
[![Latest release](https://img.shields.io/github/v/release/Xiaomiju-x/dual-rdk-x5-materials-ai-robot-lab?display_name=tag&sort=semver)](https://github.com/Xiaomiju-x/dual-rdk-x5-materials-ai-robot-lab/releases/latest)
[![License](https://img.shields.io/github/license/Xiaomiju-x/dual-rdk-x5-materials-ai-robot-lab?label=license)](LICENSE)
[![Platform: RDK X5](https://img.shields.io/badge/Edge-RDK%20X5-orange.svg)](docs/architecture/SYSTEM_ARCHITECTURE.md)
[![Safety: tiered](https://img.shields.io/badge/Safety-Tier%200--4-red.svg)](docs/safety/PHYSICAL_SAFETY.md)

[中文](README.md) · [Documentation](docs/README.md) · [Safe offline start](docs/getting-started/QUICKSTART_OFFLINE.md) · [Evidence index](docs/evidence/EVIDENCE_INDEX.md) · [Known limitations](docs/evaluation/KNOWN_LIMITATIONS.md)

## Three physical-system demos

Click a poster to play the repository-hosted MP4. The [media gallery](docs/gallery.md) and [`MEDIA_PROVENANCE.yml`](assets/media/MEDIA_PROVENANCE.yml) record timestamps, SHA-256 hashes, processing, and claim boundaries. These silent clips do not turn fixed fixtures, manual preparation, or one-session UI readings into general autonomous capability.

| Materials AI: XRD visual analysis | Embodied assistant: assisted laboratory workflow | Dual arms: complete fixed-workcell sequence |
| --- | --- | --- |
| [![Play the materials-AI and XRD-analysis preview](assets/media/previews/dashboard-xrd-pipeline.gif)](assets/media/videos/dashboard-xrd-pipeline.mp4) | [![Play the embodied-assistant workflow preview](assets/media/previews/embodied-assisted-workflow.gif)](assets/media/videos/embodied-assisted-workflow.mp4) | [![Play the complete dual-arm workflow preview](assets/media/previews/dual-arm-complete-hardware-demo.gif)](assets/media/videos/dual-arm-complete-hardware-demo.mp4) |
| [Play the materials-AI MP4](assets/media/videos/dashboard-xrd-pipeline.mp4): one recorded tablet and hardware session; displayed values apply only to this clip. | [Play the embodied-workflow MP4](assets/media/videos/embodied-assisted-workflow.mp4): bottle-fixture, lift, and assisted actions in a fixed workflow; see the gallery for its boundary. | [Play the complete dual-arm MP4](assets/media/videos/dual-arm-complete-hardware-demo.mp4): physical motion on a fixed workcell; not a learned policy or arbitrary-task generalization. |

## Competition status

Team **荧光具身智研** entered the project in the **2026 National College Student Embedded Chip and System Design Competition, Chip Application division, D-Robotics topic**. [`docs/competition/award_status.yaml`](docs/competition/award_status.yaml) is the sole award source of truth. When the national result is announced, only that file is updated before the display blocks are regenerated.

<!-- AWARD_STATUS:START -->
| Stage | Current status | Evidence boundary |
| --- | --- | --- |
| Southwest Regional Contest | First Prize | `team_confirmed`; official award source pending |
| National final | Second Prize | `team_confirmed`; official organizing-committee source pending |
<!-- AWARD_STATUS:END -->

[Award policy](docs/competition/AWARDS.md) · [Official and public sources](docs/competition/OFFICIAL_SOURCES.md)

## System overview

This is an edge materials-intelligence and robotic laboratory platform for optoelectronic and advanced-packaging functional materials. Near-infrared phosphors are the real validation carrier. The repository publishes source, configuration examples, fixed inputs, acceptance receipts, demo media, and explicit publication boundaries—not only competition slides.

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
python -B tools/publication/verify_media.py --root .
python -B -m unittest discover -s tests_public -p "test_*.py" -v
python -B examples/offline_demo/run_demo.py
```

The release passed the publication audit, repository-local link check, award single-source check, deterministic SPDX SBOM check, hardware-free tests, offline demo, workstation-frontend build, and embodied-dashboard frontend build. See the [`v1.0.1` verification record](docs/releases/v1.0.1/VERIFICATION.md) for the final exact scope and commands.

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

## Latest physical-system gallery

[Open the complete physical-system and demo gallery →](docs/gallery.md). Its captions preserve timestamps, source hashes, and truth boundaries; the primary gallery uses the latest six photos and three videos selected on 2026-08-13.

| Mobile assistant, three-quarter view | Dual-arm workstation |
| --- | --- |
| ![Latest mobile laboratory assistant](assets/media/photos/embodied-platform-three-quarter-full.webp) | ![Latest dual-arm workstation](assets/media/photos/dual-arm-workcell-full.webp) |

| Mobile assistant, front | Sensor deck and local display |
| --- | --- |
| ![Mobile laboratory assistant front](assets/media/photos/embodied-platform-front-full.webp) | ![Mobile laboratory assistant sensor deck](assets/media/photos/embodied-platform-sensor-deck-full.webp) |

| Original project poster (QR retained) | On-site dual-arm integration photo |
| --- | --- |
| ![Complete project poster with the original QR code](assets/media/photos/project-overview-poster.webp) | ![Dual-arm integration photo with a team member](assets/media/photos/team-dual-arm-integration-full.webp) |

Media must identify whether a segment is `live`, `shadow`, `replay`, or `sim-only`. A photo or video supports only what happened at that time, with that fixture, and within its stated evidence boundary.

## Data, licensing and publication boundary

Team-owned source files without a more specific notice are provided under [Apache-2.0](LICENSE). Datasets, base models, papers, fonts, web libraries, and media retain their own terms. The machine-readable [source catalog](ai_brain/icmat_foundry/contracts/source_catalog.v1.json) records representative versions, licenses, risks, and claim boundaries.

Credentials, personal or device identities, unauthorized experimental data, per-document restricted corpora, and artifacts that cannot legally be redistributed are excluded from public release artifacts. See [third-party notices](THIRD_PARTY_NOTICES.md), [NOTICE](NOTICE), and the [publication boundary](docs/safety/PUBLICATION_BOUNDARY.md).

## Limitations, roadmap and community

Failures are part of the published engineering record. The release engineering gates pass, but the technical limits remain: three experimental BPU LLMs, no promoted live Cortex session, replay-only learned dual-arm candidates, hardware- and site-specific Tier 4 reproduction, and redistribution limits for some data and models.

[Known limitations](docs/evaluation/KNOWN_LIMITATIONS.md) · [Roadmap](ROADMAP.md) · [Changelog](CHANGELOG.md)

Contributions are welcome under [CONTRIBUTING.md](CONTRIBUTING.md). Use [SUPPORT.md](SUPPORT.md) for ordinary questions and [SECURITY.md](SECURITY.md) for private vulnerability reports. Cite a versioned archive using [`CITATION.cff`](CITATION.cff).

The formal project name is **Material-Synthesis AI Prediction and Multi-Robot Embodied Laboratory Assistant Based on Dual-RDK X5 Heterogeneous Collaboration** (Chinese: **基于双 RDK X5 异构协同的材料合成 AI 预测与多机具身实验助理机器人**). `XRD` remains only the technical abbreviation for X-ray diffraction and a compatibility-oriented internal identifier; it is not the repository or project name. The project avoids unverifiable “world first” or “fully autonomous” language, does not equate phosphors with all integrated-circuit materials, and does not turn replay/simulation into real-loop evidence. Award claims retain a `team_confirmed` or `official_verified` evidence boundary.
