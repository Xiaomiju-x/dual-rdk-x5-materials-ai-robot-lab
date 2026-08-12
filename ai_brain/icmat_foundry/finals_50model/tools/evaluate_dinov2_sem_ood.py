#!/usr/bin/env python3
"""Build CPU-only DINOv2 evidence for finals registry model F-SEM-05.

Default invocation refuses to run. ``--dry-run`` validates the immutable source and
selection contracts without copying a model or running inference. ``--execute`` is
the only mode that loads ONNX Runtime, extracts embeddings, archives the rejected
CNN candidate, and publishes the DINOv2 artifacts/evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import onnx
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
SOURCE_ONNX = ROOT / "tools" / "dinov2_small.onnx"
SPLIT_MANIFEST = (
    ROOT
    / "icmat_foundry"
    / "finals_50model"
    / "evidence"
    / "sem_bank"
    / "real_data_split.v1.json"
)
IMAGE_ROOT = (
    ROOT
    / "research"
    / "data_assets"
    / "icmat_foundry"
    / "carinthia_sem"
    / "extracted"
    / "data"
    / "images"
)
ARTIFACT_DIR = (
    ROOT
    / "icmat_foundry"
    / "finals_50model"
    / "artifacts"
    / "sem_bank"
    / "F-SEM-05"
)
EVIDENCE_DIR = (
    ROOT
    / "icmat_foundry"
    / "finals_50model"
    / "evidence"
    / "sem_bank"
    / "F-SEM-05"
)

MODEL_ID = "DINOv2-SEM-OOD-CPU"
INVENTORY_ID = "F-SEM-05"
SEED = 20260801
DEFAULT_LIMITS = {"train": 64, "val": 48, "test": 48}
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
RESIZE_SHORTEST = 256
CROP_SIZE = 224


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def atomic_write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)
    digest = sha256_file(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    temporary_sidecar = sidecar.with_name(sidecar.name + ".tmp")
    temporary_sidecar.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    os.replace(temporary_sidecar, sidecar)
    return digest


def write_sha256_sidecar(path: Path) -> str:
    digest = sha256_file(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    return digest


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def load_split_contract() -> dict[str, Any]:
    if not SPLIT_MANIFEST.is_file():
        raise FileNotFoundError(f"fixed split manifest is missing: {SPLIT_MANIFEST}")
    payload = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    required_top = {"schema", "seed", "policy", "records"}
    if not required_top.issubset(payload):
        raise ValueError(f"split manifest misses fields: {sorted(required_top - set(payload))}")
    if payload["schema"] != "x5_icmat_foundry.sem_real_split.v1":
        raise ValueError(f"unexpected split schema: {payload['schema']}")
    if len(payload["records"]) != 4591:
        raise ValueError(f"expected 4,591 fixed records, got {len(payload['records'])}")
    required_record = {
        "filename",
        "class_index",
        "class_label_original",
        "content_sha256",
        "split",
    }
    seen_filenames: set[str] = set()
    for index, record in enumerate(payload["records"]):
        if not required_record.issubset(record):
            raise ValueError(
                f"split record {index} misses fields: {sorted(required_record - set(record))}"
            )
        if record["split"] not in DEFAULT_LIMITS:
            raise ValueError(f"invalid split for {record['filename']}: {record['split']}")
        label = int(record["class_index"])
        if label not in range(6) or int(record["class_label_original"]) != label + 1:
            raise ValueError(f"invalid class contract for {record['filename']}")
        digest = str(record["content_sha256"])
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest.lower()):
            raise ValueError(f"invalid SHA-256 for {record['filename']}")
        filename = str(record["filename"])
        if filename in seen_filenames:
            raise ValueError(f"duplicate filename in split manifest: {filename}")
        seen_filenames.add(filename)
        if not (IMAGE_ROOT / filename).is_file():
            raise FileNotFoundError(f"referenced SEM image is missing: {filename}")
    return payload


def inspect_onnx_contract() -> dict[str, Any]:
    if not SOURCE_ONNX.is_file():
        raise FileNotFoundError(f"DINOv2 source ONNX is missing: {SOURCE_ONNX}")
    graph = onnx.load(str(SOURCE_ONNX), load_external_data=False)
    onnx.checker.check_model(graph)
    if len(graph.graph.input) != 1 or len(graph.graph.output) != 1:
        raise ValueError("DINOv2 contract requires exactly one input and one embedding output")
    model_input = graph.graph.input[0]
    model_output = graph.graph.output[0]

    def dimensions(value: Any) -> list[int | str | None]:
        result: list[int | str | None] = []
        for item in value.type.tensor_type.shape.dim:
            if item.HasField("dim_value"):
                result.append(int(item.dim_value))
            elif item.HasField("dim_param"):
                result.append(str(item.dim_param))
            else:
                result.append(None)
        return result

    input_shape = dimensions(model_input)
    output_shape = dimensions(model_output)
    if len(input_shape) != 4 or input_shape[1:] != [3, 224, 224]:
        raise ValueError(f"expected dynamic batch x 3 x 224 x 224, got {input_shape}")
    if isinstance(input_shape[0], int) and input_shape[0] > 0:
        raise ValueError(f"expected dynamic batch dimension, got {input_shape[0]}")
    if len(output_shape) != 2:
        raise ValueError(f"expected rank-2 embedding output, got {output_shape}")
    return {
        "source_path": relative(SOURCE_ONNX),
        "source_sha256": sha256_file(SOURCE_ONNX),
        "source_bytes": SOURCE_ONNX.stat().st_size,
        "onnx_ir_version": int(graph.ir_version),
        "onnx_opsets": [
            {"domain": item.domain or "ai.onnx", "version": int(item.version)}
            for item in graph.opset_import
        ],
        "input_name": model_input.name,
        "input_shape": input_shape,
        "output_name": model_output.name,
        "output_shape": output_shape,
        "onnx_checker": "PASS",
    }


def deterministic_selection(
    records: Sequence[dict[str, Any]], limits: dict[str, int]
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["split"]), int(record["class_index"]))].append(record)
    selection: dict[str, list[dict[str, Any]]] = {split: [] for split in limits}
    for split, limit in limits.items():
        for label in range(6):
            candidates = grouped[(split, label)]
            if not candidates:
                raise ValueError(f"fixed split has no samples for split={split}, class={label}")
            ordered = sorted(
                candidates,
                key=lambda record: hashlib.sha256(
                    (
                        f"{SEED}|{split}|{label}|{record['content_sha256']}|"
                        f"{record['filename']}"
                    ).encode("utf-8")
                ).hexdigest(),
            )
            selection[split].extend(ordered[: min(limit, len(ordered))])
        selection[split].sort(
            key=lambda record: hashlib.sha256(
                f"{SEED}|ordered|{split}|{record['filename']}".encode("utf-8")
            ).hexdigest()
        )
    chosen = [record for records_for_split in selection.values() for record in records_for_split]
    filenames = [str(record["filename"]) for record in chosen]
    if len(filenames) != len(set(filenames)):
        raise ValueError("deterministic subsets overlap")
    return selection


def contract_report(
    split_payload: dict[str, Any],
    model_contract: dict[str, Any],
    selection: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    available = {
        split: dict(
            sorted(
                Counter(
                    int(record["class_index"])
                    for record in split_payload["records"]
                    if record["split"] == split
                ).items()
            )
        )
        for split in DEFAULT_LIMITS
    }
    selected = {
        split: dict(sorted(Counter(int(record["class_index"]) for record in records).items()))
        for split, records in selection.items()
    }
    return {
        "ok": True,
        "mode": "DRY_RUN_CONTRACT_ONLY",
        "inventory_id": INVENTORY_ID,
        "model_id": MODEL_ID,
        "backend": "CPU",
        "x5_contacted": False,
        "inference_executed": False,
        "model_copied": False,
        "source_model": model_contract,
        "fixed_split": {
            "path": relative(SPLIT_MANIFEST),
            "sha256": sha256_file(SPLIT_MANIFEST),
            "record_count": len(split_payload["records"]),
            "policy": split_payload["policy"],
            "available_by_split_class": available,
        },
        "selection": {
            "seed": SEED,
            "per_class_caps": DEFAULT_LIMITS,
            "selected_by_split_class": selected,
            "selected_counts": {split: len(records) for split, records in selection.items()},
        },
        "preprocessing": {
            "lineage": "facebook/dinov2-small",
            "grayscale_to_rgb": "replicate L channel three times",
            "resize": "shortest edge to 256 with bicubic interpolation",
            "crop": "deterministic center crop 224 x 224",
            "scale": "uint8 / 255",
            "mean": IMAGENET_MEAN.tolist(),
            "std": IMAGENET_STD.tolist(),
        },
    }


def resize_center_crop(image: Image.Image) -> np.ndarray:
    image = image.convert("L")
    width, height = image.size
    scale = RESIZE_SHORTEST / min(width, height)
    resized_width = max(CROP_SIZE, int(round(width * scale)))
    resized_height = max(CROP_SIZE, int(round(height * scale)))
    image = image.resize((resized_width, resized_height), Image.Resampling.BICUBIC)
    left = (resized_width - CROP_SIZE) // 2
    top = (resized_height - CROP_SIZE) // 2
    array = np.asarray(image.crop((left, top, left + CROP_SIZE, top + CROP_SIZE)), dtype=np.uint8)
    return np.repeat(array[:, :, None], 3, axis=2)


def normalized_chw(rgb: np.ndarray) -> np.ndarray:
    scaled = rgb.astype(np.float32) / 255.0
    normalized = (scaled - IMAGENET_MEAN[None, None, :]) / IMAGENET_STD[None, None, :]
    return np.transpose(normalized, (2, 0, 1)).astype(np.float32)


def pixel_mean_std(rgb: np.ndarray) -> np.ndarray:
    grayscale = rgb[:, :, 0].astype(np.float32) / 255.0
    return np.asarray([grayscale.mean(), grayscale.std()], dtype=np.float32)


def deterministic_corruption(rgb: np.ndarray, key: str) -> tuple[np.ndarray, str]:
    digest = hashlib.sha256(f"{SEED}|controlled-ood|{key}".encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False)
    rng = np.random.default_rng(seed)
    mode = seed % 3
    grayscale = rgb[:, :, 0].copy()
    if mode == 0:
        noise = rng.normal(0.0, 58.0, grayscale.shape)
        corrupted = np.clip(grayscale.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        name = "gaussian_noise_sigma58"
    elif mode == 1:
        block = 32
        cropped = grayscale[: block * 7, : block * 7]
        blocks = (
            cropped.reshape(7, block, 7, block)
            .transpose(0, 2, 1, 3)
            .reshape(49, block, block)
        )
        shuffled = blocks[rng.permutation(49)]
        corrupted = (
            shuffled.reshape(7, 7, block, block)
            .transpose(0, 2, 1, 3)
            .reshape(block * 7, block * 7)
        )
        name = "block_shuffle_7x7"
    else:
        yy, xx = np.mgrid[:CROP_SIZE, :CROP_SIZE]
        pattern = 127.5 + 112.0 * np.sin(xx / rng.uniform(1.9, 4.7)) * np.cos(
            yy / rng.uniform(2.1, 6.8)
        )
        corrupted = np.clip(pattern, 0, 255).astype(np.uint8)
        name = "periodic_texture"
    return np.repeat(corrupted[:, :, None], 3, axis=2), name


def load_selected_images(
    records: Sequence[dict[str, Any]], include_corruption: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    tensors: list[np.ndarray] = []
    pixel_features: list[np.ndarray] = []
    labels: list[int] = []
    corruption_names: list[str] = []
    for record in records:
        path = IMAGE_ROOT / str(record["filename"])
        raw = path.read_bytes()
        actual_sha = sha256_bytes(raw)
        if actual_sha != record["content_sha256"]:
            raise ValueError(f"image SHA mismatch: {record['filename']}")
        with Image.open(io.BytesIO(raw)) as image:
            rgb = resize_center_crop(image)
        if include_corruption:
            rgb, corruption_name = deterministic_corruption(rgb, str(record["filename"]))
            corruption_names.append(corruption_name)
        tensors.append(normalized_chw(rgb))
        pixel_features.append(pixel_mean_std(rgb))
        labels.append(int(record["class_index"]))
    return (
        np.stack(tensors).astype(np.float32),
        np.stack(pixel_features).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
        corruption_names,
    )


def l2_normalize(array: np.ndarray) -> np.ndarray:
    return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-12)


def infer_embeddings(
    session: Any,
    input_name: str,
    output_name: str,
    array: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    outputs: list[np.ndarray] = []
    started = time.perf_counter()
    for offset in range(0, len(array), batch_size):
        batch = np.ascontiguousarray(array[offset : offset + batch_size], dtype=np.float32)
        result = session.run([output_name], {input_name: batch})[0]
        if result.ndim != 2 or result.shape[0] != len(batch):
            raise ValueError(f"unexpected embedding shape: {result.shape}")
        if not np.isfinite(result).all():
            raise ValueError("DINOv2 returned non-finite embeddings")
        outputs.append(np.asarray(result, dtype=np.float32))
    elapsed = time.perf_counter() - started
    embeddings = np.concatenate(outputs)
    return l2_normalize(embeddings), {
        "sample_count": int(len(array)),
        "batch_size": int(batch_size),
        "batch_count": int((len(array) + batch_size - 1) // batch_size),
        "elapsed_seconds": elapsed,
        "samples_per_second": float(len(array) / max(elapsed, 1e-9)),
        "raw_embedding_shape": list(embeddings.shape),
        "all_finite": True,
        "l2_normalized": True,
    }


def cosine_centroids(embeddings: np.ndarray, labels: np.ndarray) -> np.ndarray:
    centroids = []
    for label in range(6):
        members = embeddings[labels == label]
        if not len(members):
            raise ValueError(f"train subset has no embedding for class {label}")
        centroid = members.mean(axis=0, keepdims=True)
        centroids.append(l2_normalize(centroid)[0])
    return np.stack(centroids).astype(np.float32)


def centroid_ood_score(embeddings: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    return (1.0 - np.max(embeddings @ centroids.T, axis=1)).astype(np.float32)


def nearest_neighbor_predictions(
    train_embeddings: np.ndarray, train_labels: np.ndarray, query_embeddings: np.ndarray
) -> np.ndarray:
    return train_labels[np.argmax(query_embeddings @ train_embeddings.T, axis=1)]


def classification_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "macro_f1": float(
            f1_score(truth, prediction, labels=list(range(6)), average="macro", zero_division=0)
        ),
        "support_by_class": dict(sorted(Counter(truth.tolist()).items())),
    }


def ood_metrics(id_scores: np.ndarray, ood_scores: np.ndarray, threshold: float) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    truth = np.concatenate(
        (np.zeros(len(id_scores), dtype=np.int64), np.ones(len(ood_scores), dtype=np.int64))
    )
    scores = np.concatenate((id_scores, ood_scores))
    return {
        "auroc": float(roc_auc_score(truth, scores)),
        "auprc_ood_positive": float(average_precision_score(truth, scores)),
        "threshold_val_id_p95": float(threshold),
        "test_id_false_positive_rate": float(np.mean(id_scores > threshold)),
        "controlled_ood_true_positive_rate": float(np.mean(ood_scores > threshold)),
        "id_score_mean": float(id_scores.mean()),
        "controlled_ood_score_mean": float(ood_scores.mean()),
    }


def pixel_baseline_scores(
    train_features: np.ndarray, query_features: np.ndarray
) -> np.ndarray:
    center = train_features.mean(axis=0)
    scale = train_features.std(axis=0) + 1e-8
    return np.linalg.norm((query_features - center) / scale, axis=1).astype(np.float32)


def archive_active_candidate(timestamp: str) -> dict[str, Any]:
    """Move, never delete, the mismatched active CNN files into rejected trees."""
    archive_report: dict[str, Any] = {"artifact_files": [], "evidence_files": []}
    for active_dir, report_key in (
        (ARTIFACT_DIR, "artifact_files"),
        (EVIDENCE_DIR, "evidence_files"),
    ):
        active_dir.mkdir(parents=True, exist_ok=True)
        active_files = [item for item in active_dir.iterdir() if item.name != "rejected"]
        if not active_files:
            continue
        rejected_dir = active_dir / "rejected" / f"cnn_mismatch_{timestamp}"
        rejected_dir.mkdir(parents=True, exist_ok=False)
        for item in sorted(active_files):
            if not item.is_file():
                raise ValueError(f"unexpected active directory entry, refusing archive: {item}")
            record = {
                "original_path": relative(item),
                "sha256": sha256_file(item),
                "bytes": item.stat().st_size,
            }
            destination = rejected_dir / item.name
            shutil.move(str(item), str(destination))
            record["rejected_path"] = relative(destination)
            archive_report[report_key].append(record)
    return archive_report


def execute(
    split_payload: dict[str, Any],
    model_contract: dict[str, Any],
    selection: dict[str, list[dict[str, Any]]],
    batch_size: int,
) -> dict[str, Any]:
    # Lazy imports keep --dry-run free of ORT load/inference.
    import onnxruntime as ort

    if batch_size <= 0:
        raise ValueError("batch-size must be positive")
    available_providers = ort.get_available_providers()
    if "CPUExecutionProvider" not in available_providers:
        raise RuntimeError(f"CPUExecutionProvider unavailable: {available_providers}")
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = max(1, min(8, os.cpu_count() or 1))
    session_options.inter_op_num_threads = 1
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    load_started = time.perf_counter()
    session = ort.InferenceSession(
        str(SOURCE_ONNX),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    load_seconds = time.perf_counter() - load_started
    active_providers = session.get_providers()
    if active_providers != ["CPUExecutionProvider"]:
        raise RuntimeError(f"refusing non-CPU execution providers: {active_providers}")
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    if input_meta.name != model_contract["input_name"] or output_meta.name != model_contract["output_name"]:
        raise ValueError("ORT metadata does not match inspected ONNX contract")

    loaded: dict[str, dict[str, Any]] = {}
    inference: dict[str, dict[str, Any]] = {}
    for split in ("train", "val", "test"):
        tensors, pixel_features, labels, _ = load_selected_images(
            selection[split], include_corruption=False
        )
        embeddings, timing = infer_embeddings(
            session, input_meta.name, output_meta.name, tensors, batch_size
        )
        loaded[split] = {
            "tensors": tensors,
            "pixel_features": pixel_features,
            "labels": labels,
            "embeddings": embeddings,
        }
        inference[split] = timing
    ood_tensors, ood_pixel, _, corruption_names = load_selected_images(
        selection["test"], include_corruption=True
    )
    ood_embeddings, ood_timing = infer_embeddings(
        session, input_meta.name, output_meta.name, ood_tensors, batch_size
    )
    inference["controlled_ood"] = ood_timing

    train_embeddings = loaded["train"]["embeddings"]
    train_labels = loaded["train"]["labels"]
    val_embeddings = loaded["val"]["embeddings"]
    test_embeddings = loaded["test"]["embeddings"]
    test_labels = loaded["test"]["labels"]
    centroids = cosine_centroids(train_embeddings, train_labels)
    val_id_scores = centroid_ood_score(val_embeddings, centroids)
    threshold = float(np.quantile(val_id_scores, 0.95))
    test_id_scores = centroid_ood_score(test_embeddings, centroids)
    test_ood_scores = centroid_ood_score(ood_embeddings, centroids)
    centroid_prediction = np.argmax(test_embeddings @ centroids.T, axis=1)
    knn_prediction = nearest_neighbor_predictions(
        train_embeddings, train_labels, test_embeddings
    )
    learned_ood = ood_metrics(test_id_scores, test_ood_scores, threshold)

    train_pixel = loaded["train"]["pixel_features"]
    val_pixel_scores = pixel_baseline_scores(train_pixel, loaded["val"]["pixel_features"])
    pixel_threshold = float(np.quantile(val_pixel_scores, 0.95))
    baseline_id = pixel_baseline_scores(train_pixel, loaded["test"]["pixel_features"])
    baseline_ood = pixel_baseline_scores(train_pixel, ood_pixel)
    baseline_metrics = ood_metrics(baseline_id, baseline_ood, pixel_threshold)

    # All inference and metrics complete before the active candidate is changed.
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_report = archive_active_candidate(timestamp)
    destination_onnx = ARTIFACT_DIR / "dinov2_small_cpu.onnx"
    temporary_onnx = destination_onnx.with_suffix(".onnx.tmp")
    shutil.copy2(SOURCE_ONNX, temporary_onnx)
    if sha256_file(temporary_onnx) != model_contract["source_sha256"]:
        raise RuntimeError("copied DINOv2 ONNX hash mismatch")
    os.replace(temporary_onnx, destination_onnx)
    destination_onnx_sha = write_sha256_sidecar(destination_onnx)

    calibration_path = ARTIFACT_DIR / "cosine_centroid_calibration.npz"
    np.savez_compressed(
        calibration_path,
        centroids_fp32=centroids,
        val_id_p95_threshold=np.asarray([threshold], dtype=np.float32),
        class_indices=np.arange(6, dtype=np.int64),
    )
    calibration_sha = write_sha256_sidecar(calibration_path)
    fixture_path = ARTIFACT_DIR / "fixed_ort_fixture.npz"
    np.savez_compressed(
        fixture_path,
        filename=np.asarray([selection["test"][0]["filename"]]),
        input_fp32=loaded["test"]["tensors"][:1],
        embedding_fp32=test_embeddings[:1],
    )
    fixture_sha = write_sha256_sidecar(fixture_path)

    selection_manifest = {
        "schema": "x5_icmat_foundry.dinov2_sem_selection.v1",
        "seed": SEED,
        "source_split_path": relative(SPLIT_MANIFEST),
        "source_split_sha256": sha256_file(SPLIT_MANIFEST),
        "per_class_caps": DEFAULT_LIMITS,
        "records": {
            split: [
                {
                    "filename": record["filename"],
                    "class_index": int(record["class_index"]),
                    "content_sha256": record["content_sha256"],
                }
                for record in records
            ]
            for split, records in selection.items()
        },
    }
    selection_path = EVIDENCE_DIR / "selection_manifest.v1.json"
    selection_sha = atomic_write_json(selection_path, selection_manifest)

    corruption_counts = dict(sorted(Counter(corruption_names).items()))
    ort_evidence = {
        "schema": "x5_icmat_foundry.dinov2_ort_cpu_evidence.v1",
        "generated_at": utc_now(),
        "backend": "CPU",
        "requested_providers": ["CPUExecutionProvider"],
        "active_providers": active_providers,
        "available_providers_observed": available_providers,
        "session_load_seconds": load_seconds,
        "model_input": {
            "name": input_meta.name,
            "shape": input_meta.shape,
            "type": input_meta.type,
        },
        "model_output": {
            "name": output_meta.name,
            "shape": output_meta.shape,
            "type": output_meta.type,
        },
        "inference": inference,
        "fixed_fixture": {
            "path": relative(fixture_path),
            "sha256": fixture_sha,
            "load_and_infer": "PASS",
            "all_finite": True,
        },
        "x5_contacted": False,
        "gpu_provider_used": False,
    }
    ort_evidence_path = EVIDENCE_DIR / "ort_cpu_load_infer.v1.json"
    ort_evidence_sha = atomic_write_json(ort_evidence_path, ort_evidence)

    report = contract_report(split_payload, model_contract, selection)
    report.update(
        {
            "schema": "x5_icmat_foundry.dinov2_sem_ood_receipt.v1",
            "mode": "EXECUTED",
            "generated_at": utc_now(),
            "candidate_status": "CPU_ORT_EVALUATED_REAL_ID_CONTROLLED_OOD",
            "authority": 0,
            "backend": "CPU",
            "runtime_scope": "X5_ON_DEMAND_BOARD_PENDING",
            "x5_contacted": False,
            "production_integration_allowed": False,
            "claim_boundary": (
                "ID images are real Carinthia SEM. OOD positives are deterministic "
                "controlled corruptions, not unseen fab defect families. Classification "
                "and retrieval metrics are auxiliary and do not establish production "
                "generalization, wafer metrology accuracy, or X5 runtime performance."
            ),
            "source_data": {
                "name": "Carinthia SEM",
                "zenodo_record": "10715190",
                "license": "CC BY 4.0",
                "real_sem_id": True,
                "fixed_split_path": relative(SPLIT_MANIFEST),
                "fixed_split_sha256": sha256_file(SPLIT_MANIFEST),
            },
            "metrics": {
                "cosine_centroid_ood": learned_ood,
                "pixel_mean_std_baseline": baseline_metrics,
                "auxiliary_centroid_classification": classification_metrics(
                    test_labels, centroid_prediction
                ),
                "auxiliary_cosine_1nn_retrieval": classification_metrics(
                    test_labels, knn_prediction
                ),
                "controlled_corruption_counts": corruption_counts,
            },
            "ort_evidence": {
                "path": relative(ort_evidence_path),
                "sha256": ort_evidence_sha,
                "load_and_infer": "PASS",
            },
            "artifacts": {
                "onnx": {
                    "path": relative(destination_onnx),
                    "sha256": destination_onnx_sha,
                    "copied_byte_identical_to_source": destination_onnx_sha
                    == model_contract["source_sha256"],
                },
                "calibration": {
                    "path": relative(calibration_path),
                    "sha256": calibration_sha,
                },
                "fixed_fixture": {
                    "path": relative(fixture_path),
                    "sha256": fixture_sha,
                },
                "selection_manifest": {
                    "path": relative(selection_path),
                    "sha256": selection_sha,
                },
            },
            "rejected_cnn_archive": archive_report,
            "script_sha256": sha256_file(Path(__file__)),
        }
    )
    receipt_path = EVIDENCE_DIR / "evaluation_receipt.v1.json"
    receipt_sha = atomic_write_json(receipt_path, report)
    return {
        "ok": True,
        "inventory_id": INVENTORY_ID,
        "model_id": MODEL_ID,
        "receipt_path": relative(receipt_path),
        "receipt_sha256": receipt_sha,
        "metrics": report["metrics"],
        "onnx_sha256": destination_onnx_sha,
        "x5_contacted": False,
        "backend": "CPU",
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate contracts and deterministic selection; do not copy or infer.",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Run CPU ORT evaluation and publish F-SEM-05 evidence.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.dry_run and not args.execute:
        print(
            "REFUSED: choose --dry-run for a no-write contract check or --execute for "
            "the explicit CPU evaluation.",
            file=sys.stderr,
        )
        return 2
    model_contract = inspect_onnx_contract()
    split_payload = load_split_contract()
    selection = deterministic_selection(split_payload["records"], DEFAULT_LIMITS)
    if args.dry_run:
        print(
            json.dumps(
                contract_report(split_payload, model_contract, selection),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    result = execute(split_payload, model_contract, selection, args.batch_size)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
