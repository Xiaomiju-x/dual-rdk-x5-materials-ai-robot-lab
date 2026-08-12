#!/usr/bin/env python3
"""Train three independent Qwen2.5-0.5B LoRA experts for later BPU splitting.

The script is intentionally self-contained and writes only to the finals_50model
LLM SFT data, BPU LLM artifact, and BPU LLM evidence directories.  It never
uses the legacy 25,228-chunk corpus, contacts an X5, or claims BPU execution.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
BASE_MODEL = ROOT / "research/model_assets/icmat_foundry/qwen25_05b_instruct/snapshot"
CANDIDATE = ROOT / "icmat_foundry/finals_50model"
DATA_ROOT = CANDIDATE / "data/llm_sft"
ARTIFACT_ROOT = CANDIDATE / "artifacts/bpu_llm_bank"
EVIDENCE_ROOT = CANDIDATE / "evidence/bpu_llm_bank"
XRD_SOURCE = CANDIDATE / "artifacts/xrd_bank/theoretical_xrd_dataset.v1.npz"
SEM_SOURCE = (
    ROOT
    / "research/data_assets/icmat_foundry/carinthia_sem/extracted/data/carinthia.csv"
)
SECOM_SOURCE = ROOT / "research/data_assets/icmat_foundry/uci_secom/raw/secom.zip"
PVD_SOURCE = (
    ROOT
    / "research/icmat_foundry/fabyield_replacement_20260728/candidates/zenodo_16881338"
)
PACKAGE_SCRIPT = CANDIDATE / "training/train_package_bank.py"
PACKAGE_EVIDENCE = CANDIDATE / "evidence/package_bank"
PROCESS_EVIDENCE = CANDIDATE / "evidence/process_bank/process_bank_run.v1.json"
LEGACY_RAG_MARKER = "25228"
SEED = 20260801
STATUS = "MERGED_HF_BPU_SPLIT_PENDING"
SYSTEM_COMMON = (
    "You are an edge materials specialist. Return exactly one compact JSON object, "
    "copy supplied numbers exactly, never invent missing measurements, and use the "
    "literal string UNKNOWN when evidence is absent."
)


@dataclass(frozen=True)
class DomainSpec:
    inventory_id: str
    slug: str
    model_name: str
    system: str


DOMAINS = (
    DomainSpec(
        "F-LLM-03",
        "characterization",
        "ICMat-Qwen2.5-0.5B-Characterization",
        SYSTEM_COMMON
        + " Use schema ICMAT_CHARACTERIZATION_V1 for XRD, PL, or SEM records. "
        "A theoretical pattern is not a measured phase identification.",
    ),
    DomainSpec(
        "F-LLM-04",
        "process",
        "ICMat-Qwen2.5-0.5B-Process",
        SYSTEM_COMMON
        + " Use schema ICMAT_PROCESS_V1. SECOM is an anonymous 2008 public "
        "benchmark; PVD values are public-data process records, not our fab data.",
    ),
    DomainSpec(
        "F-LLM-05",
        "reliability",
        "ICMat-Qwen2.5-0.5B-Reliability",
        SYSTEM_COMMON
        + " Use schema ICMAT_RELIABILITY_V1. Every physics-proxy or packaging "
        "surrogate answer must carry evidence_state=SIM_ONLY and must not be "
        "described as measured reliability or production qualification.",
    ),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
    temporary.replace(path)


def write_hash(path: Path) -> str:
    digest = sha256_file(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )
    return digest


def tree_inventory(path: Path) -> dict[str, Any]:
    records = []
    for item in sorted(path.rglob("*")):
        if item.is_file():
            records.append(
                {
                    "path": item.relative_to(path).as_posix(),
                    "bytes": item.stat().st_size,
                    "sha256": sha256_file(item),
                }
            )
    return {
        "file_count": len(records),
        "bytes": sum(record["bytes"] for record in records),
        "tree_sha256": canonical_sha(records),
        "files": records,
    }


def json_text(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def example(
    spec: DomainSpec,
    split: str,
    index: int,
    source_refs: Sequence[Mapping[str, Any]],
    user_payload: Mapping[str, Any],
    answer_payload: Mapping[str, Any],
    *,
    task_type: str,
    protected_numbers: Sequence[str] = (),
    expected_tokens: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "schema": "x5_icmat_foundry.bpu_llm_sft_example.v1",
        "example_id": f"{spec.inventory_id}-{split}-{index:04d}",
        "inventory_id": spec.inventory_id,
        "domain": spec.slug,
        "split": split,
        "task_type": task_type,
        "source_refs": list(source_refs),
        "protected_numbers": list(protected_numbers),
        "expected_tokens": list(expected_tokens),
        "messages": [
            {"role": "system", "content": spec.system},
            {"role": "user", "content": json_text(user_payload)},
            {"role": "assistant", "content": json_text(answer_payload)},
        ],
    }


def split_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result = {"train": [], "validation": [], "blind_test": []}
    for row in rows:
        result[row["split"]].append(row)
    if min(len(value) for value in result.values()) == 0:
        raise ValueError("all dataset splits must be non-empty")
    return result


def _top_peaks(grid: np.ndarray, profile: np.ndarray, count: int = 3) -> tuple[list[str], list[str]]:
    candidates = np.argsort(profile)[::-1]
    selected: list[int] = []
    for raw_index in candidates:
        index = int(raw_index)
        if all(abs(index - existing) >= 5 for existing in selected):
            selected.append(index)
        if len(selected) == count:
            break
    selected.sort(key=lambda item: float(grid[item]))
    angles = [f"{float(grid[item]):.2f}" for item in selected]
    intensities = [f"{float(profile[item]):.3f}" for item in selected]
    return angles, intensities


def build_characterization(spec: DomainSpec) -> list[dict[str, Any]]:
    source_hash = sha256_file(XRD_SOURCE)
    xrd = np.load(XRD_SOURCE, allow_pickle=False)
    split_code = {"train": 0, "validation": 1, "blind_test": 2}
    requested = {"train": 54, "validation": 9, "blind_test": 6}
    rows: list[dict[str, Any]] = []
    counter = 0
    for split, code in split_code.items():
        indices = np.flatnonzero(xrd["split_codes"] == code)[: requested[split]]
        for row_index in indices:
            counter += 1
            angles, intensities = _top_peaks(
                xrd["grid_2theta_deg"], xrd["spectra"][row_index].astype(np.float32)
            )
            formula = str(xrd["formulas"][row_index])
            jid = str(xrd["jids"][row_index])
            user = {
                "schema": "ICMAT_CHARACTERIZATION_REQUEST_V1",
                "modality": "XRD",
                "source_state": "THEORETICAL_DB",
                "formula": formula,
                "jid": jid,
                "peak_2theta_deg": angles,
                "relative_intensity": intensities,
                "request": "structure the evidence without claiming measured phase identity",
            }
            answer = {
                "schema": "ICMAT_CHARACTERIZATION_V1",
                "modality": "XRD",
                "evidence_state": "THEORETICAL_DB",
                "formula": formula,
                "peak_2theta_deg": angles,
                "relative_intensity": intensities,
                "phase_identity": "UNKNOWN",
                "next_check": "match a measured pattern with calibrated reference data",
            }
            rows.append(
                example(
                    spec,
                    split,
                    counter,
                    [
                        {
                            "source_id": "JARVIS_THEORETICAL_XRD_LOCAL",
                            "path": relative(XRD_SOURCE),
                            "sha256": source_hash,
                            "locator": f"jid:{jid}",
                            "evidence_class": "THEORETICAL_DB",
                        }
                    ],
                    user,
                    answer,
                    task_type="structured_json_numeric_copy",
                    protected_numbers=[*angles, *intensities],
                    expected_tokens=["ICMAT_CHARACTERIZATION_V1", "UNKNOWN"],
                )
            )

    # PL is an explicit spectrum proxy, never mislabeled as a measurement.
    rng = np.random.default_rng(SEED + 3)
    counts = {"train": 36, "validation": 6, "blind_test": 3}
    for split, count in counts.items():
        for _ in range(count):
            counter += 1
            peak = f"{rng.uniform(720.0, 1080.0):.1f}"
            fwhm = f"{rng.uniform(28.0, 145.0):.1f}"
            ratio = f"{rng.uniform(0.15, 1.85):.3f}"
            lifetime = f"{rng.uniform(8.0, 680.0):.1f}"
            missing = counter % 7 == 0
            user = {
                "schema": "ICMAT_CHARACTERIZATION_REQUEST_V1",
                "modality": "PL",
                "source_state": "SIM_ONLY_SPECTRUM",
                "peak_nm": peak,
                "fwhm_nm": None if missing else fwhm,
                "integrated_intensity_ratio": ratio,
                "lifetime_us": lifetime,
            }
            answer = {
                "schema": "ICMAT_CHARACTERIZATION_V1",
                "modality": "PL",
                "evidence_state": "SIM_ONLY_SPECTRUM",
                "peak_nm": peak,
                "fwhm_nm": "UNKNOWN" if missing else fwhm,
                "integrated_intensity_ratio": ratio,
                "lifetime_us": lifetime,
                "status": "UNKNOWN" if missing else "STRUCTURED",
            }
            protected = [peak, ratio, lifetime] + ([] if missing else [fwhm])
            rows.append(
                example(
                    spec,
                    split,
                    counter,
                    [
                        {
                            "source_id": "PL_GAUSSIAN_SCHEMA_GENERATOR_V1",
                            "path": relative(Path(__file__)),
                            "sha256": sha256_file(Path(__file__)),
                            "locator": "build_characterization:PL",
                            "evidence_class": "SIM_ONLY_SPECTRUM",
                        }
                    ],
                    user,
                    answer,
                    task_type="refusal" if missing else "structured_json_numeric_copy",
                    protected_numbers=protected,
                    expected_tokens=["UNKNOWN"] if missing else ["ICMAT_CHARACTERIZATION_V1"],
                )
            )

    # Carinthia provides a real image-class label, but no physical scale for CD.
    sem_hash = sha256_file(SEM_SOURCE)
    lines = SEM_SOURCE.read_text(encoding="utf-8").splitlines()[1:]
    counts = {"train": 30, "validation": 6, "blind_test": 3}
    offset = 0
    for split, count in counts.items():
        for raw in lines[offset : offset + count]:
            offset += 1
            counter += 1
            image_path, filename, label = raw.split(";")
            user = {
                "schema": "ICMAT_CHARACTERIZATION_REQUEST_V1",
                "modality": "SEM",
                "dataset_label": label,
                "image_id": Path(filename).stem,
                "scale_nm_per_px": None,
                "request": "report known class and physical quantities only if supported",
            }
            answer = {
                "schema": "ICMAT_CHARACTERIZATION_V1",
                "modality": "SEM",
                "evidence_state": "PUBLIC_IMAGE_LABEL",
                "dataset_label": label,
                "critical_dimension_nm": "UNKNOWN",
                "roughness_nm": "UNKNOWN",
                "status": "PARTIAL",
            }
            rows.append(
                example(
                    spec,
                    split,
                    counter,
                    [
                        {
                            "source_id": "CARINTHIA_SEM",
                            "path": relative(SEM_SOURCE),
                            "sha256": sem_hash,
                            "locator": image_path,
                            "evidence_class": "PUBLIC_IMAGE_LABEL",
                        }
                    ],
                    user,
                    answer,
                    task_type="refusal",
                    protected_numbers=[label],
                    expected_tokens=["UNKNOWN", "PUBLIC_IMAGE_LABEL"],
                )
            )
    return rows


def build_process(spec: DomainSpec) -> list[dict[str, Any]]:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from icmat_foundry.fabyield.data import load_secom_zip

    process_receipt_hash = sha256_file(PROCESS_EVIDENCE)
    secom = load_secom_zip(SECOM_SOURCE)
    rng = np.random.default_rng(SEED + 4)
    rows: list[dict[str, Any]] = []
    counter = 0
    split_indices = {
        "train": np.arange(0, 104),
        "validation": np.arange(1100, 1118),
        "blind_test": np.arange(1500, 1509),
    }
    for split, indices in split_indices.items():
        for row_index in indices:
            counter += 1
            values = secom.features[int(row_index)]
            finite = values[np.isfinite(values)]
            observed = int(finite.size)
            missing = int(values.size - finite.size)
            mean_value = f"{float(np.mean(finite)):.3f}"
            std_value = f"{float(np.std(finite)):.3f}"
            failure = int(secom.labels[int(row_index)])
            status = "PUBLIC_FAILURE_LABEL" if failure else "PUBLIC_PASS_LABEL"
            user = {
                "schema": "ICMAT_PROCESS_REQUEST_V1",
                "dataset": "UCI_SECOM_2008",
                "source_row": int(secom.source_row_ids[int(row_index)]),
                "timestamp": secom.timestamps[int(row_index)].isoformat(),
                "observed_sensor_count": observed,
                "missing_sensor_count": missing,
                "observed_mean": mean_value,
                "observed_std": std_value,
                "public_label": failure,
            }
            answer = {
                "schema": "ICMAT_PROCESS_V1",
                "evidence_state": "PUBLIC_BENCHMARK",
                "benchmark": "UCI_SECOM_2008",
                "source_row": int(row_index),
                "observed_sensor_count": observed,
                "missing_sensor_count": missing,
                "observed_mean": mean_value,
                "observed_std": std_value,
                "quality_label_state": status,
                "production_root_cause": "UNKNOWN",
                "next_check": "inspect model attribution and original sensor context",
            }
            rows.append(
                example(
                    spec,
                    split,
                    counter,
                    [
                        {
                            "source_id": "UCI_SECOM",
                            "path": relative(SECOM_SOURCE),
                            "sha256": secom.source_sha256,
                            "locator": f"source_row:{int(row_index)}",
                            "evidence_class": "PUBLIC_BENCHMARK",
                        },
                        {
                            "source_id": "PROCESS_BANK_RUN_V1",
                            "path": relative(PROCESS_EVIDENCE),
                            "sha256": process_receipt_hash,
                            "locator": "F-PROC-01/F-PROC-02/F-PROC-07/F-PROC-09",
                            "evidence_class": "MODEL_OUTPUT_EXPLANATION_CONTRACT",
                        },
                    ],
                    user,
                    answer,
                    task_type="structured_json_numeric_copy",
                    protected_numbers=[str(observed), str(missing), mean_value, std_value],
                    expected_tokens=["ICMAT_PROCESS_V1", "UNKNOWN"],
                )
            )

    # Use actual PVD target rows. Their values are public-data records, not local-fab results.
    import pandas as pd

    pvd_manifest = PVD_SOURCE / "zenodo_record_16881338.json"
    pvd_manifest_hash = sha256_file(pvd_manifest)
    split_rows_by_domain = {
        "train": [("AlCu", item) for item in range(0, 36)]
        + [("WTi", item) for item in range(0, 36)],
        "validation": [("AlCu", item) for item in range(4000, 4006)]
        + [("WTi", item) for item in range(1400, 1406)],
        "blind_test": [("AlCu", item) for item in range(4800, 4803)]
        + [("WTi", item) for item in range(1700, 1703)],
    }
    cache: dict[str, tuple[np.ndarray, np.ndarray, str, str]] = {}
    for domain in ("AlCu", "WTi"):
        x_path = PVD_SOURCE / f"X_pvd_{domain}.csv"
        y_path = PVD_SOURCE / f"Y_pvd_{domain}.csv"
        cache[domain] = (
            pd.read_csv(x_path).to_numpy(dtype=np.float32),
            pd.read_csv(y_path).to_numpy(dtype=np.float32),
            sha256_file(x_path),
            sha256_file(y_path),
        )
    for split, selections in split_rows_by_domain.items():
        for domain, row_index in selections:
            counter += 1
            x_values, y_values, x_hash, y_hash = cache[domain]
            sensors = x_values[row_index]
            thickness = y_values[row_index]
            finite = sensors[np.isfinite(sensors)]
            sensor_mean = f"{float(np.mean(finite)):.4f}"
            points = [f"{float(value):.4f}" for value in thickness[[0, 8, 16]]]
            spread = f"{float(np.max(thickness) - np.min(thickness)):.4f}"
            user = {
                "schema": "ICMAT_PROCESS_REQUEST_V1",
                "dataset": "ZENODO_PVD_16881338",
                "material_domain": domain,
                "source_row": row_index,
                "sensor_mean": sensor_mean,
                "thickness_points_1_9_17": points,
                "thickness_spread": spread,
            }
            answer = {
                "schema": "ICMAT_PROCESS_V1",
                "evidence_state": "PUBLIC_PROCESS_DATA",
                "material_domain": domain,
                "source_row": row_index,
                "sensor_mean": sensor_mean,
                "thickness_points_1_9_17": points,
                "thickness_spread": spread,
                "local_fab_transfer": "UNKNOWN",
                "next_check": "verify tool-specific calibration before production use",
            }
            rows.append(
                example(
                    spec,
                    split,
                    counter,
                    [
                        {
                            "source_id": f"ZENODO_PVD_X_{domain}",
                            "path": relative(PVD_SOURCE / f"X_pvd_{domain}.csv"),
                            "sha256": x_hash,
                            "locator": f"row:{row_index}",
                            "evidence_class": "PUBLIC_PROCESS_DATA",
                        },
                        {
                            "source_id": f"ZENODO_PVD_Y_{domain}",
                            "path": relative(PVD_SOURCE / f"Y_pvd_{domain}.csv"),
                            "sha256": y_hash,
                            "locator": f"row:{row_index}",
                            "evidence_class": "PUBLIC_PROCESS_DATA",
                        },
                        {
                            "source_id": "ZENODO_RECORD_16881338",
                            "path": relative(pvd_manifest),
                            "sha256": pvd_manifest_hash,
                            "locator": "record",
                            "evidence_class": "SOURCE_METADATA",
                        },
                    ],
                    user,
                    answer,
                    task_type="structured_json_numeric_copy",
                    protected_numbers=[sensor_mean, *points, spread],
                    expected_tokens=["ICMAT_PROCESS_V1", "UNKNOWN"],
                )
            )

    # Explicit missing-input refusals are domain-specific and never hidden routing.
    for split, count in (("train", 16), ("validation", 3), ("blind_test", 3)):
        for _ in range(count):
            counter += 1
            missing_field = random.Random(SEED + counter).choice(
                ["sensor_vector", "model_output", "source_identity"]
            )
            user = {
                "schema": "ICMAT_PROCESS_REQUEST_V1",
                "dataset": "UNSPECIFIED_PROCESS",
                "missing_field": missing_field,
                "request": "declare a production root cause",
            }
            answer = {
                "schema": "ICMAT_PROCESS_V1",
                "evidence_state": "INSUFFICIENT",
                "production_root_cause": "UNKNOWN",
                "missing_field": missing_field,
                "next_check": f"provide {missing_field}",
            }
            rows.append(
                example(
                    spec,
                    split,
                    counter,
                    [],
                    user,
                    answer,
                    task_type="refusal",
                    expected_tokens=["UNKNOWN", missing_field],
                )
            )
    rng.shuffle(rows)
    return rows


def _load_package_tasks() -> Sequence[Any]:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from icmat_foundry.finals_50model.training.train_package_bank import TASKS

    return TASKS[:3]


def build_reliability(spec: DomainSpec) -> list[dict[str, Any]]:
    tasks = _load_package_tasks()
    rng = np.random.default_rng(SEED + 5)
    rows: list[dict[str, Any]] = []
    counter = 0
    counts = {"train": 42, "validation": 7, "blind_test": 4}
    for split, count in counts.items():
        for task_index, task in enumerate(tasks):
            receipt_path = PACKAGE_EVIDENCE / f"f_pkg_0{task_index + 1}_receipt.v1.json"
            receipt_hash = sha256_file(receipt_path)
            for _ in range(count):
                counter += 1
                input_values = np.array(
                    [[rng.uniform(parameter.low, parameter.high) for parameter in task.parameters]],
                    dtype=np.float64,
                )
                output_values = np.asarray(task.simulator(input_values), dtype=np.float64)[0]
                inputs = {
                    parameter.name: f"{float(value):.4g}"
                    for parameter, value in zip(task.parameters, input_values[0], strict=True)
                }
                outputs = {
                    output.name: f"{float(value):.5g}"
                    for output, value in zip(task.outputs, output_values, strict=True)
                }
                user = {
                    "schema": "ICMAT_RELIABILITY_REQUEST_V1",
                    "model": task.model_id,
                    "physics_proxy": task.equation_scope,
                    "inputs": inputs,
                    "simulated_outputs": outputs,
                    "request": "explain the surrogate output and its evidence boundary",
                }
                answer = {
                    "schema": "ICMAT_RELIABILITY_V1",
                    "evidence_state": "SIM_ONLY",
                    "model": task.model_id,
                    "outputs": outputs,
                    "measured_reliability": "UNKNOWN",
                    "production_qualification": "UNKNOWN",
                    "next_check": "validate against independent physical measurements",
                }
                rows.append(
                    example(
                        spec,
                        split,
                        counter,
                        [
                            {
                                "source_id": f"PACKAGE_PROXY_{task.inventory_id}",
                                "path": relative(receipt_path),
                                "sha256": receipt_hash,
                                "locator": task.equation_scope,
                                "evidence_class": "SIM_ONLY",
                                "canonical_reference": task.nist_url,
                            },
                            {
                                "source_id": "PACKAGE_PHYSICS_GENERATOR",
                                "path": relative(PACKAGE_SCRIPT),
                                "sha256": sha256_file(PACKAGE_SCRIPT),
                                "locator": task.model_id,
                                "evidence_class": "SIM_ONLY",
                            },
                        ],
                        user,
                        answer,
                        task_type=f"output_copy_{task.inventory_id}",
                        protected_numbers=list(outputs.values()),
                        expected_tokens=["ICMAT_RELIABILITY_V1", "SIM_ONLY", "UNKNOWN"],
                    )
                )

    for split, count in (("train", 18), ("validation", 3), ("blind_test", 3)):
        for _ in range(count):
            counter += 1
            user = {
                "schema": "ICMAT_RELIABILITY_REQUEST_V1",
                "model": "UNSPECIFIED",
                "physics_proxy": None,
                "inputs": {"temperature_delta": f"{rng.uniform(40, 200):.1f}"},
                "request": "certify measured package lifetime",
            }
            answer = {
                "schema": "ICMAT_RELIABILITY_V1",
                "evidence_state": "SIM_ONLY",
                "model": "UNKNOWN",
                "measured_reliability": "UNKNOWN",
                "production_qualification": "UNKNOWN",
                "missing": ["physics_proxy", "validated_measurement"],
            }
            rows.append(
                example(
                    spec,
                    split,
                    counter,
                    [],
                    user,
                    answer,
                    task_type="refusal",
                    expected_tokens=["SIM_ONLY", "UNKNOWN"],
                )
            )
    return rows


def materialize_datasets() -> dict[str, dict[str, Any]]:
    builders = {
        "characterization": build_characterization,
        "process": build_process,
        "reliability": build_reliability,
    }
    manifests: dict[str, dict[str, Any]] = {}
    for spec in DOMAINS:
        rows = builders[spec.slug](spec)
        groups = split_rows(rows)
        domain_dir = DATA_ROOT / spec.inventory_id
        files: dict[str, Any] = {}
        for split, records in groups.items():
            path = domain_dir / f"{split}.jsonl"
            write_jsonl(path, records)
            files[split] = {
                "path": relative(path),
                "examples": len(records),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "task_counts": {
                    task: sum(row["task_type"] == task for row in records)
                    for task in sorted({row["task_type"] for row in records})
                },
            }
        source_refs = {
            (ref.get("source_id"), ref.get("path"), ref.get("sha256"))
            for row in rows
            for ref in row["source_refs"]
        }
        manifest = {
            "schema": "x5_icmat_foundry.bpu_llm_dataset_manifest.v1",
            "created_at_utc": utc_now(),
            "inventory_id": spec.inventory_id,
            "model_name": spec.model_name,
            "domain": spec.slug,
            "seed": SEED,
            "files": files,
            "dataset_sha256": canonical_sha(files),
            "source_ref_count": len(source_refs),
            "assistant_only_training_required": True,
            "legacy_25228_training_used": False,
            "blind_test_fixed_before_training": True,
            "claim_boundary": (
                "Training records are structured transformations of the listed local public "
                "data or explicit physics proxies. They are not hidden production data."
            ),
        }
        manifest_path = domain_dir / "manifest.json"
        write_json(manifest_path, manifest)
        write_hash(manifest_path)
        manifests[spec.inventory_id] = manifest
    return manifests


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def encode_assistant_only(
    tokenizer: Any, messages: Sequence[Mapping[str, str]], max_length: int
) -> dict[str, list[int]]:
    encoded_messages = [dict(message) for message in messages]
    prefix = tokenizer.apply_chat_template(
        encoded_messages[:-1], tokenize=True, add_generation_prompt=True
    )
    full = tokenizer.apply_chat_template(
        encoded_messages, tokenize=True, add_generation_prompt=False
    )
    if len(full) > max_length:
        encoded_messages[0]["content"] = (
            "Return one compact JSON. Copy numbers exactly; missing evidence is UNKNOWN. "
            "Reliability proxies must be SIM_ONLY."
        )
        prefix = tokenizer.apply_chat_template(
            encoded_messages[:-1], tokenize=True, add_generation_prompt=True
        )
        full = tokenizer.apply_chat_template(
            encoded_messages, tokenize=True, add_generation_prompt=False
        )
    if full[: len(prefix)] != prefix:
        raise ValueError("assistant chat prefix mismatch")
    if len(full) > max_length:
        raise ValueError(f"sequence length {len(full)} exceeds {max_length}")
    labels = [-100] * len(prefix) + list(full[len(prefix) :])
    if not any(value != -100 for value in labels):
        raise ValueError("assistant-only labels are empty")
    return {
        "input_ids": list(full),
        "attention_mask": [1] * len(full),
        "labels": labels,
    }


class EncodedDataset:
    def __init__(self, rows: Sequence[Mapping[str, Any]], tokenizer: Any, max_length: int):
        self.rows = [
            encode_assistant_only(tokenizer, row["messages"], max_length) for row in rows
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Mapping[str, Sequence[int]]:
        return self.rows[index]


class AssistantOnlyCollator:
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def __call__(self, features: Sequence[Mapping[str, Sequence[int]]]) -> dict[str, Any]:
        import torch

        maximum = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        attention = []
        labels = []
        for feature in features:
            padding = maximum - len(feature["input_ids"])
            input_ids.append(
                list(feature["input_ids"]) + [self.tokenizer.pad_token_id] * padding
            )
            attention.append(list(feature["attention_mask"]) + [0] * padding)
            labels.append(list(feature["labels"]) + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def train_domain(
    spec: DomainSpec,
    *,
    max_steps: int,
    max_length: int,
    gradient_accumulation: int,
) -> dict[str, Any]:
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "WANDB_DISABLED": "true",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        }
    )
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("RTX4050 CUDA is required")
    set_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, local_files_only=True, use_fast=True, fix_mistral_regex=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    domain_data = DATA_ROOT / spec.inventory_id
    train_rows = load_jsonl(domain_data / "train.jsonl")
    validation_rows = load_jsonl(domain_data / "validation.jsonl")
    train_set = EncodedDataset(train_rows, tokenizer, max_length)
    validation_set = EncodedDataset(validation_rows, tokenizer, max_length)
    supervised_tokens = sum(
        sum(token != -100 for token in row["labels"]) for row in train_set.rows
    )
    total_tokens = sum(len(row["labels"]) for row in train_set.rows)

    domain_root = ARTIFACT_ROOT / spec.inventory_id
    adapter_dir = domain_root / "adapter"
    merged_dir = domain_root / "merged_hf"
    trainer_dir = domain_root / "trainer"
    for path in (adapter_dir, merged_dir, trainer_dir):
        if path.exists():
            shutil.rmtree(path)

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    started = time.perf_counter()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        local_files_only=True,
        quantization_config=quantization,
        dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    arguments = TrainingArguments(
        output_dir=str(trainer_dir),
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=max(10, max_steps // 4),
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=5,
        max_steps=max_steps,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=2.0e-4,
        warmup_ratio=0.08,
        weight_decay=0.0,
        lr_scheduler_type="cosine",
        report_to=[],
        seed=SEED,
        data_seed=SEED,
        bf16=True,
        fp16=False,
        tf32=False,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        skip_memory_metrics=True,
        disable_tqdm=False,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_set,
        eval_dataset=validation_set,
        data_collator=AssistantOnlyCollator(tokenizer),
        processing_class=tokenizer,
    )
    result = trainer.train()
    evaluation = trainer.evaluate()
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    peak_vram = int(torch.cuda.max_memory_allocated(0))
    train_elapsed = time.perf_counter() - started
    log_history = list(trainer.state.log_history)
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()

    # Merge into a real, standalone HF tree. This is not a BPU binary yet.
    from peft import PeftModel

    merge_started = time.perf_counter()
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        local_files_only=True,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to("cuda")
    adapted = PeftModel.from_pretrained(base, adapter_dir, local_files_only=True)
    merged = adapted.merge_and_unload(safe_merge=True).eval()
    merged.config.use_cache = True
    merged.save_pretrained(
        merged_dir, safe_serialization=True, max_shard_size="4GB"
    )
    tokenizer.save_pretrained(merged_dir)
    merge_elapsed = time.perf_counter() - merge_started
    del adapted, base, merged
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "inventory_id": spec.inventory_id,
        "domain": spec.slug,
        "seed": SEED,
        "train_examples": len(train_rows),
        "validation_examples": len(validation_rows),
        "max_steps": max_steps,
        "global_step": int(result.global_step),
        "train_loss": float(result.training_loss),
        "eval_loss": float(evaluation["eval_loss"]),
        "assistant_only_loss": True,
        "supervised_assistant_tokens": supervised_tokens,
        "supervised_fraction": supervised_tokens / total_tokens,
        "trainable_parameters": trainable,
        "peak_vram_bytes": peak_vram,
        "training_seconds": train_elapsed,
        "merge_seconds": merge_elapsed,
        "log_history": log_history,
        "adapter": tree_inventory(adapter_dir),
        "merged_hf": tree_inventory(merged_dir),
    }


def extract_json(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def evaluate_blind(spec: DomainSpec, max_new_tokens: int = 256) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = ARTIFACT_ROOT / spec.inventory_id / "merged_hf"
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, use_fast=True, fix_mistral_regex=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to("cuda").eval()
    rows = load_jsonl(DATA_ROOT / spec.inventory_id / "blind_test.jsonl")
    results = []
    started = time.perf_counter()
    for row in rows:
        prompt_messages = row["messages"][:-1]
        rendered = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(rendered, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(
            output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()
        parsed = extract_json(generated)
        number_hits = [number in generated for number in row["protected_numbers"]]
        token_hits = [token in generated for token in row["expected_tokens"]]
        results.append(
            {
                "example_id": row["example_id"],
                "task_type": row["task_type"],
                "generated": generated,
                "json_valid": parsed is not None,
                "protected_numbers": row["protected_numbers"],
                "numeric_copy_pass": all(number_hits),
                "expected_tokens": row["expected_tokens"],
                "expected_tokens_pass": all(token_hits),
            }
        )
    elapsed = time.perf_counter() - started
    json_rate = sum(item["json_valid"] for item in results) / len(results)
    numeric_items = [item for item in results if item["protected_numbers"]]
    numeric_rate = (
        sum(item["numeric_copy_pass"] for item in numeric_items) / len(numeric_items)
        if numeric_items
        else 1.0
    )
    refusal_items = [item for item in results if item["task_type"] == "refusal"]
    refusal_rate = (
        sum(item["expected_tokens_pass"] for item in refusal_items) / len(refusal_items)
        if refusal_items
        else 1.0
    )
    token_rate = sum(item["expected_tokens_pass"] for item in results) / len(results)
    output_copy_by_task = {}
    for task_type in sorted(
        {item["task_type"] for item in results if item["task_type"].startswith("output_copy_")}
    ):
        task_items = [item for item in results if item["task_type"] == task_type]
        successes = sum(item["numeric_copy_pass"] for item in task_items)
        output_copy_by_task[task_type.removeprefix("output_copy_")] = {
            "examples": len(task_items),
            "successes": successes,
            "rate": successes / len(task_items),
        }
    each_output_task_has_success = all(
        item["successes"] >= 1 for item in output_copy_by_task.values()
    )
    accepted = (
        json_rate >= 0.80
        and numeric_rate >= 0.70
        and refusal_rate >= 0.80
        and each_output_task_has_success
    )
    receipt = {
        "schema": "x5_icmat_foundry.bpu_llm_blind_smoke.v1",
        "created_at_utc": utc_now(),
        "inventory_id": spec.inventory_id,
        "domain": spec.slug,
        "status": STATUS if accepted else "MERGED_HF_QUALITY_HOLD",
        "accepted": accepted,
        "blind_examples": len(results),
        "metrics": {
            "json_valid_rate": json_rate,
            "numeric_copy_rate": numeric_rate,
            "refusal_contract_rate": refusal_rate,
            "expected_token_rate": token_rate,
            "output_copy_by_task": output_copy_by_task,
            "each_output_task_has_success": each_output_task_has_success,
            "elapsed_seconds": elapsed,
        },
        "thresholds": {
            "json_valid_rate": 0.80,
            "numeric_copy_rate": 0.70,
            "refusal_contract_rate": 0.80,
            "minimum_successes_per_output_task": 1,
        },
        "results": results,
        "claim_boundary": (
            "Fixed local smoke evaluation only. It is not BPU conversion, X5 execution, "
            "production accuracy, or multi-seed evidence."
        ),
    }
    path = EVIDENCE_ROOT / spec.inventory_id / "blind_smoke.json"
    write_json(path, receipt)
    write_hash(path)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return receipt


def check_distinct_weights(records: Sequence[Mapping[str, Any]]) -> bool:
    hashes = []
    for record in records:
        weights = [
            item["sha256"]
            for item in record["training"]["merged_hf"]["files"]
            if item["path"].endswith(".safetensors")
        ]
        if len(weights) != 1:
            raise ValueError(f"expected one merged safetensors for {record['inventory_id']}")
        hashes.append(weights[0])
    return len(set(hashes)) == len(hashes)


def runtime_environment() -> dict[str, Any]:
    import importlib.metadata
    import torch

    versions = {}
    for name in ("torch", "transformers", "peft", "accelerate", "bitsandbytes"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "NOT_INSTALLED"
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        gpu = result.stdout.strip()
    except Exception:
        gpu = "UNAVAILABLE"
    return {
        "python": sys.version,
        "packages": versions,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "gpu": gpu,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument(
        "--only",
        choices=[spec.inventory_id for spec in DOMAINS],
        action="append",
        help="Train selected inventory IDs; repeat for multiple models.",
    )
    args = parser.parse_args()
    if args.max_steps < 10:
        raise ValueError("max-steps must be at least 10")
    if args.max_length < 256:
        raise ValueError("max-length must be at least 256")
    if args.gradient_accumulation < 1:
        raise ValueError("gradient-accumulation must be positive")
    required = [
        BASE_MODEL / "model.safetensors",
        XRD_SOURCE,
        SEM_SOURCE,
        SECOM_SOURCE,
        PVD_SOURCE / "zenodo_record_16881338.json",
        PACKAGE_SCRIPT,
        PROCESS_EVIDENCE,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"required local inputs are missing: {missing}")
    if LEGACY_RAG_MARKER in str(DATA_ROOT):
        raise AssertionError("legacy RAG marker unexpectedly entered training path")

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    manifests = materialize_datasets()
    if args.data_only:
        print(json.dumps(manifests, ensure_ascii=False, indent=2))
        return 0

    selected = [spec for spec in DOMAINS if not args.only or spec.inventory_id in args.only]
    records = []
    for spec in selected:
        print(f"[{spec.inventory_id}] training {spec.slug}", flush=True)
        training = train_domain(
            spec,
            max_steps=args.max_steps,
            max_length=args.max_length,
            gradient_accumulation=args.gradient_accumulation,
        )
        print(f"[{spec.inventory_id}] fixed blind smoke", flush=True)
        blind = evaluate_blind(spec)
        record = {
            "inventory_id": spec.inventory_id,
            "model_name": spec.model_name,
            "domain": spec.slug,
            "dataset_manifest": manifests[spec.inventory_id],
            "training": training,
            "blind_smoke": {
                "status": blind["status"],
                "accepted": blind["accepted"],
                "metrics": blind["metrics"],
            },
        }
        receipt_path = EVIDENCE_ROOT / spec.inventory_id / "training_receipt.json"
        write_json(receipt_path, record)
        write_hash(receipt_path)
        records.append(record)

    distinct = check_distinct_weights(records) if len(records) > 1 else True
    accepted = all(record["blind_smoke"]["accepted"] for record in records) and distinct
    bank_receipt = {
        "schema": "x5_icmat_foundry.bpu_llm_bank_run.v1",
        "created_at_utc": utc_now(),
        "status": STATUS if accepted else "BPU_LLM_BANK_QUALITY_HOLD",
        "accepted": accepted,
        "requested_models": [spec.inventory_id for spec in selected],
        "models_completed": len(records),
        "all_merged_weight_hashes_distinct": distinct,
        "single_seed": SEED,
        "assistant_only_loss": True,
        "legacy_25228_training_used": False,
        "x5_contacted": False,
        "bpu_compiled": False,
        "elapsed_seconds": time.perf_counter() - started,
        "base_model": {
            "path": relative(BASE_MODEL),
            "model_safetensors_sha256": sha256_file(BASE_MODEL / "model.safetensors"),
        },
        "training_script": {
            "path": relative(Path(__file__)),
            "sha256": sha256_file(Path(__file__)),
        },
        "environment": runtime_environment(),
        "models": [
            {
                "inventory_id": record["inventory_id"],
                "domain": record["domain"],
                "dataset_sha256": record["dataset_manifest"]["dataset_sha256"],
                "adapter_tree_sha256": record["training"]["adapter"]["tree_sha256"],
                "merged_hf_tree_sha256": record["training"]["merged_hf"]["tree_sha256"],
                "blind_status": record["blind_smoke"]["status"],
                "blind_metrics": record["blind_smoke"]["metrics"],
            }
            for record in records
        ],
        "written_roots": [
            relative(Path(__file__)),
            relative(DATA_ROOT),
            relative(ARTIFACT_ROOT),
            relative(EVIDENCE_ROOT),
        ],
        "claim_boundary": (
            "Three distinct merged HF candidates are prepared for later segmented BPU "
            "conversion. No GGUF, Bayes-e binary, X5 run, production integration, or "
            "multi-seed quality claim is made."
        ),
    }
    bank_path = EVIDENCE_ROOT / "bank_receipt.json"
    write_json(bank_path, bank_receipt)
    write_hash(bank_path)
    print(json.dumps(bank_receipt, ensure_ascii=False, indent=2), flush=True)
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
