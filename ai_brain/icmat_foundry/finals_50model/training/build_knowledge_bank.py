#!/usr/bin/env python3
"""Build the four isolated ICMat knowledge-bank models and their receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import joblib
import numpy as np
import onnx
import onnxruntime as ort
import torch
from huggingface_hub import HfApi, snapshot_download
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(__file__).resolve().parents[3]
CANDIDATE = ROOT / "CIMC_candidates" / "ICMat_PhosFab_Foundry_R1_20260731"
ARTIFACT_ROOT = ROOT / "icmat_foundry" / "finals_50model" / "artifacts" / "knowledge_bank"
EVIDENCE_ROOT = ROOT / "icmat_foundry" / "finals_50model" / "evidence" / "knowledge_bank"
RAG_ROOT = CANDIDATE / "rag" / "build" / "icmat_rag_v2_14p519"
RAG_CORPUS = RAG_ROOT / "corpus.14p519.v1.jsonl"
RAG_MANIFEST = RAG_ROOT / "index_manifest.v1.json"
SEMANTIC_V7 = (
    ROOT
    / "evaluation/icmat_foundry/llm/icmat_semantic_queries_v7_20260730_r7_clean_r1/records.v7.jsonl"
)
POINTER_V8 = (
    ROOT
    / "evaluation/icmat_foundry/llm/icmat_qwen05b_evidence_pointer_sft_v8_pretrain_20260731_r4"
)
NLI_SOURCE = ROOT / "research/model_assets/icmat_foundry/nli_deberta_v3_small/snapshot"
NLI_RECEIPT = ROOT / "research/model_assets/icmat_foundry/nli_deberta_v3_small/model_receipt.v1.json"
N14_ROOT = CANDIDATE / "artifacts" / "public_models" / "N14"
N14_EVIDENCE = CANDIDATE / "evidence" / "public_models" / "N14"
N14_TEACHER = N14_ROOT / "seed-20260731" / "model.joblib"
JARVIS_ARCHIVE = (
    ROOT / "research/data_assets/icmat_foundry/nist_jarvis_dft/raw/jdft_3d-9-24-2025.json.zip"
)

BGE_ID = "BAAI/bge-small-en-v1.5"
CROSS_ID = "cross-encoder/ms-marco-MiniLM-L6-v2"
SEED = 20260801
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def hash_tree(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if ".cache" in path.parts:
            continue
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def tree_sha256(root: Path) -> str:
    return sha256_json(hash_tree(root))


def model_files_present(path: Path) -> bool:
    return (path / "config.json").is_file() and any(
        (path / name).is_file()
        for name in ("model.safetensors", "pytorch_model.bin")
    )


def ensure_snapshot(model_id: str, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(ARTIFACT_ROOT / ".hf_home")
    if not model_files_present(output):
        snapshot_download(
            repo_id=model_id,
            local_dir=output,
            cache_dir=ARTIFACT_ROOT / ".hf_home" / "hub",
            allow_patterns=(
                "*.json",
                "*.txt",
                "*.model",
                "*.safetensors",
                "*.bin",
                "*.md",
            ),
            max_workers=4,
        )
    if not model_files_present(output):
        raise RuntimeError(f"incomplete Hugging Face snapshot: {model_id}")
    revision = "UNKNOWN_OFFLINE_COPY"
    try:
        revision = HfApi().model_info(model_id).sha or revision
    except Exception:
        pass
    return {
        "repo_id": model_id,
        "revision": revision,
        "local_files_only_after_download": True,
        "tree_sha256": tree_sha256(output),
        "file_count": len(hash_tree(output)),
        "total_bytes": sum(row["bytes"] for row in hash_tree(output)),
    }


def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batches(values: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def load_retrieval_data() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    corpus = read_jsonl(RAG_CORPUS)
    records = [
        row
        for row in read_jsonl(SEMANTIC_V7)
        if row.get("acceptance", {}).get("training_eligible") is True
    ]
    upstream_ids = {row["upstream_chunk_id"] for row in corpus}
    if len(corpus) != 519:
        raise RuntimeError(f"RAG v2 corpus count changed: {len(corpus)}")
    if len(records) != 1482:
        raise RuntimeError(f"Semantic Queries v7 accepted count changed: {len(records)}")
    if not all(set(row["chunk_ids"]) <= upstream_ids for row in records):
        raise RuntimeError("Semantic Queries v7 contains an unbound RAG chunk")
    return corpus, records


@torch.inference_mode()
def encode_bge(
    tokenizer: Any,
    model: nn.Module,
    texts: list[str],
    device: torch.device,
    *,
    query: bool,
    batch_size: int = 32,
) -> np.ndarray:
    vectors: list[np.ndarray] = []
    for chunk in batches(texts, batch_size):
        if query:
            chunk = [QUERY_PREFIX + text for text in chunk]
        encoded = tokenizer(
            chunk,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        hidden = model(**encoded).last_hidden_state[:, 0]
        hidden = torch.nn.functional.normalize(hidden, p=2, dim=1)
        vectors.append(hidden.float().cpu().numpy())
    return np.concatenate(vectors, axis=0).astype(np.float32)


def ranking_metrics(
    ranked_indices: np.ndarray,
    corpus: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    reciprocal: list[float] = []
    recalls = {1: 0, 5: 0, 10: 0, 20: 0}
    namespace_leaks = 0
    for order, record in zip(ranked_indices, records, strict=True):
        relevant = set(record["chunk_ids"])
        ranked_ids = [corpus[int(index)]["upstream_chunk_id"] for index in order]
        rank = next(
            (position for position, item in enumerate(ranked_ids, 1) if item in relevant),
            None,
        )
        reciprocal.append(0.0 if rank is None else 1.0 / rank)
        for cutoff in recalls:
            recalls[cutoff] += int(any(item in relevant for item in ranked_ids[:cutoff]))
        namespace_leaks += int(corpus[int(order[0])]["namespace"] != record["namespace"])
    count = len(records)
    return {
        "queries": count,
        "recall_at_1": recalls[1] / count,
        "recall_at_5": recalls[5] / count,
        "recall_at_10": recalls[10] / count,
        "recall_at_20": recalls[20] / count,
        "mrr": float(np.mean(reciprocal)),
        "top1_cross_namespace_rate": namespace_leaks / count,
    }


def build_dense(
    corpus: list[dict[str, Any]], records: list[dict[str, Any]], device: torch.device
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, Any, Any]:
    model_id = "F-KNW-01"
    artifact_dir = ARTIFACT_ROOT / model_id
    evidence_dir = EVIDENCE_ROOT / model_id
    model_dir = artifact_dir / "model"
    model_binding = ensure_snapshot(BGE_ID, model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(model_dir, local_files_only=True).to(device).eval()
    corpus_texts = [row["text"] for row in corpus]
    query_texts = [row["paraphrase"] for row in records]
    started = time.perf_counter()
    corpus_embeddings = encode_bge(tokenizer, model, corpus_texts, device, query=False)
    query_embeddings = encode_bge(tokenizer, model, query_texts, device, query=True)
    similarity = query_embeddings @ corpus_embeddings.T
    ranked = np.argsort(-similarity, axis=1)[:, :20]
    elapsed = time.perf_counter() - started
    metrics = ranking_metrics(ranked, corpus, records)
    np.save(artifact_dir / "corpus_embeddings.fp32.npy", corpus_embeddings)
    index = [
        {
            "row": index,
            "chunk_id": row["chunk_id"],
            "upstream_chunk_id": row["upstream_chunk_id"],
            "namespace": row["namespace"],
            "source_id": row["source_id"],
            "text_sha256": row["text_sha256"],
        }
        for index, row in enumerate(corpus)
    ]
    write_json(artifact_dir / "corpus_index.v1.json", index)
    fixed_input = {
        "schema": "icmat_dense_retriever_fixed_input.v1",
        "query": records[0]["paraphrase"],
        "query_record_id": records[0]["record_id"],
    }
    fixed_order = ranked[0, :5]
    fixed_output = {
        "schema": "icmat_dense_retriever_fixed_output.v1",
        "top5": [
            {
                "rank": rank,
                "upstream_chunk_id": corpus[int(index)]["upstream_chunk_id"],
                "score": float(similarity[0, int(index)]),
            }
            for rank, index in enumerate(fixed_order, 1)
        ],
    }
    write_json(artifact_dir / "fixed_input.v1.json", fixed_input)
    write_json(artifact_dir / "fixed_output.v1.json", fixed_output)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    receipt = {
        "schema": "icmat_knowledge_model_receipt.v1",
        "model_id": model_id,
        "model_name": "ICMat-DenseRetriever-CPU",
        "status": "PC_RUNNABLE_REAL_WEIGHTS",
        "backend": "TRANSFORMERS_CPU_TARGET_PC_VALIDATED",
        "validation_device": str(device),
        "model_binding": model_binding,
        "sources": source_bindings(),
        "metrics": metrics,
        "elapsed_seconds": elapsed,
        "corpus_shape": list(corpus_embeddings.shape),
        "fixed_input_sha256": sha256_file(artifact_dir / "fixed_input.v1.json"),
        "fixed_output_sha256": sha256_file(artifact_dir / "fixed_output.v1.json"),
        "legacy_25228_training_used": False,
        "x5_contacted": False,
    }
    finalize_receipt(artifact_dir, evidence_dir, receipt)
    return receipt, corpus_embeddings, query_embeddings, tokenizer, model_dir


@torch.inference_mode()
def cross_scores(
    tokenizer: Any,
    model: nn.Module,
    pairs: list[tuple[str, str]],
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    for chunk in batches(pairs, batch_size):
        encoded = tokenizer(
            [item[0] for item in chunk],
            [item[1] for item in chunk],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        logits = model(**encoded).logits
        if logits.shape[-1] == 1:
            score = logits[:, 0]
        else:
            score = torch.softmax(logits, dim=-1)[:, -1]
        outputs.append(score.float().cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


def build_cross_encoder(
    corpus: list[dict[str, Any]],
    records: list[dict[str, Any]],
    corpus_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    model_id = "F-KNW-02"
    artifact_dir = ARTIFACT_ROOT / model_id
    evidence_dir = EVIDENCE_ROOT / model_id
    model_dir = artifact_dir / "model"
    model_binding = ensure_snapshot(CROSS_ID, model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, local_files_only=True
    ).to(device).eval()
    dense_similarity = query_embeddings @ corpus_embeddings.T
    dense_top20 = np.argsort(-dense_similarity, axis=1)[:, :20]
    reranked = np.empty_like(dense_top20)
    started = time.perf_counter()
    for start in range(0, len(records), 64):
        stop = min(start + 64, len(records))
        pairs: list[tuple[str, str]] = []
        for record, candidates in zip(
            records[start:stop], dense_top20[start:stop], strict=True
        ):
            pairs.extend((record["paraphrase"], corpus[int(index)]["text"]) for index in candidates)
        scores = cross_scores(tokenizer, model, pairs, device).reshape(stop - start, 20)
        for offset, candidates in enumerate(dense_top20[start:stop]):
            reranked[start + offset] = candidates[np.argsort(-scores[offset])]
    elapsed = time.perf_counter() - started
    metrics = ranking_metrics(reranked, corpus, records)
    dense_metrics = ranking_metrics(dense_top20, corpus, records)
    first_candidates = dense_top20[0]
    first_pairs = [(records[0]["paraphrase"], corpus[int(i)]["text"]) for i in first_candidates]
    first_scores = cross_scores(tokenizer, model, first_pairs, device)
    fixed_input = {
        "schema": "icmat_cross_encoder_fixed_input.v1",
        "query": records[0]["paraphrase"],
        "candidates": [corpus[int(index)]["upstream_chunk_id"] for index in first_candidates],
    }
    fixed_output = {
        "schema": "icmat_cross_encoder_fixed_output.v1",
        "ranking": [
            {
                "rank": rank,
                "upstream_chunk_id": corpus[int(first_candidates[index])]["upstream_chunk_id"],
                "score": float(first_scores[index]),
            }
            for rank, index in enumerate(np.argsort(-first_scores), 1)
        ],
    }
    write_json(artifact_dir / "fixed_input.v1.json", fixed_input)
    write_json(artifact_dir / "fixed_output.v1.json", fixed_output)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    receipt = {
        "schema": "icmat_knowledge_model_receipt.v1",
        "model_id": model_id,
        "model_name": "ICMat-CrossEncoder-Reranker-CPU",
        "status": "PC_RUNNABLE_REAL_WEIGHTS",
        "backend": "TRANSFORMERS_CPU_TARGET_PC_VALIDATED",
        "validation_device": str(device),
        "model_binding": model_binding,
        "sources": source_bindings(),
        "candidate_depth": 20,
        "dense_metrics": dense_metrics,
        "reranked_metrics": metrics,
        "elapsed_seconds": elapsed,
        "fixed_input_sha256": sha256_file(artifact_dir / "fixed_input.v1.json"),
        "fixed_output_sha256": sha256_file(artifact_dir / "fixed_output.v1.json"),
        "legacy_25228_training_used": False,
        "x5_contacted": False,
    }
    finalize_receipt(artifact_dir, evidence_dir, receipt)
    return receipt


def load_n14_module() -> Any:
    if str(CANDIDATE) not in sys.path:
        sys.path.insert(0, str(CANDIDATE))
    from phosfab.models.public_models.common import (  # type: ignore[import-not-found]
        deterministic_family_split,
        load_jarvis_records,
        stable_sha256,
    )
    from phosfab.models.public_models.features import (  # type: ignore[import-not-found]
        canonical_formula,
        linker_pair_features,
        linker_signature,
    )

    def valid_records(raw: list[dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], int]:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        rejected = 0
        for record in raw:
            family = canonical_formula(str(record.get("formula", "")))
            signature = linker_signature(record)
            jid = str(record.get("jid", ""))
            if not family or signature is None or not jid.startswith("JVASP-"):
                rejected += 1
                continue
            groups[family].append(
                {"jid": jid, "formula": str(record["formula"]), "signature": signature}
            )
        return {
            family: sorted(records, key=lambda item: item["jid"])
            for family, records in groups.items()
            if len(records) >= 2
        }, rejected

    def training_pairs(
        groups: dict[str, list[dict[str, Any]]],
        family_split: dict[str, str],
        split: str,
        negative_limit: int = 5,
    ) -> tuple[np.ndarray, np.ndarray]:
        features: list[np.ndarray] = []
        labels: list[int] = []
        for family in sorted(groups):
            if family_split[family] != split:
                continue
            records = groups[family]
            for query in records:
                features.append(linker_pair_features(query["signature"], query["signature"]))
                labels.append(1)
                candidates = [item for item in records if item["jid"] != query["jid"]]
                candidates.sort(key=lambda item: stable_sha256([query["jid"], item["jid"]]))
                for candidate in candidates[:negative_limit]:
                    features.append(
                        linker_pair_features(query["signature"], candidate["signature"])
                    )
                    labels.append(0)
        if not features:
            raise RuntimeError(f"no N14 pairs for {split}")
        return np.stack(features).astype(np.float32), np.asarray(labels, dtype=np.int64)

    return SimpleNamespace(
        load_jarvis_records=load_jarvis_records,
        deterministic_family_split=deterministic_family_split,
        linker_pair_features=linker_pair_features,
        _valid_records=valid_records,
        _training_pairs=training_pairs,
    )


class ChemEntityMLP(nn.Module):
    def __init__(self, mean: torch.Tensor, scale: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mean", mean.reshape(1, 73))
        self.register_buffer("scale", scale.reshape(1, 73))
        self.layers = nn.Sequential(
            nn.Linear(73, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers((features - self.mean) / self.scale)


@torch.inference_mode()
def mlp_probabilities(model: nn.Module, values: np.ndarray, device: torch.device) -> np.ndarray:
    scores: list[np.ndarray] = []
    for start in range(0, len(values), 4096):
        tensor = torch.from_numpy(values[start : start + 4096]).to(device)
        scores.append(torch.sigmoid(model(tensor)[:, 0]).cpu().numpy())
    return np.concatenate(scores).astype(np.float32)


def n14_retrieval_metrics(
    module: Any,
    model: nn.Module,
    groups: dict[str, list[dict[str, Any]]],
    family_split: dict[str, str],
    device: torch.device,
) -> dict[str, Any]:
    reciprocal: list[float] = []
    top1 = 0
    queries = 0
    for family in sorted(groups):
        if family_split[family] != "test":
            continue
        records = groups[family]
        matrix: list[np.ndarray] = []
        spans: list[tuple[int, int, int]] = []
        for query_index, query in enumerate(records):
            start = len(matrix)
            matrix.extend(
                module.linker_pair_features(query["signature"], candidate["signature"])
                for candidate in records
            )
            spans.append((query_index, start, len(matrix)))
        scores = mlp_probabilities(model, np.stack(matrix).astype(np.float32), device)
        for query_index, start, stop in spans:
            ranked = np.argsort(-scores[start:stop], kind="stable")
            rank = int(np.flatnonzero(ranked == query_index)[0]) + 1
            queries += 1
            top1 += int(rank == 1)
            reciprocal.append(1.0 / rank)
    return {
        "queries": queries,
        "link_accuracy": top1 / queries,
        "mrr": float(np.mean(reciprocal)),
    }


def build_chem_entity(device: torch.device) -> dict[str, Any]:
    model_id = "F-KNW-03"
    artifact_dir = ARTIFACT_ROOT / model_id
    evidence_dir = EVIDENCE_ROOT / model_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    module = load_n14_module()
    teacher = joblib.load(N14_TEACHER)
    raw = module.load_jarvis_records(JARVIS_ARCHIVE)
    groups, rejected = module._valid_records(raw)
    family_split = module.deterministic_family_split(list(groups), 20260731)
    x_train, _ = module._training_pairs(groups, family_split, "train")
    x_validation, _ = module._training_pairs(groups, family_split, "validation")
    x_test, y_test = module._training_pairs(
        groups, family_split, "test", negative_limit=2
    )
    positive_index = list(teacher.classes_).index(1)
    teacher_train = teacher.predict_proba(x_train)[:, positive_index].astype(np.float32)
    teacher_validation = teacher.predict_proba(x_validation)[:, positive_index].astype(np.float32)
    teacher_test = teacher.predict_proba(x_test)[:, positive_index].astype(np.float32)
    mean = x_train.mean(axis=0).astype(np.float32)
    scale = x_train.std(axis=0).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    torch.manual_seed(SEED)
    model = ChemEntityMLP(torch.from_numpy(mean), torch.from_numpy(scale)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()
    dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(teacher_train))
    loader = DataLoader(dataset, batch_size=2048, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    started = time.perf_counter()
    for epoch in range(40):
        model.train()
        for features, target in loader:
            features = features.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)[:, 0]
            loss = loss_fn(logits, target)
            loss.backward()
            optimizer.step()
        model.eval()
        validation_prob = mlp_probabilities(model, x_validation, device)
        validation_loss = float(np.mean((validation_prob - teacher_validation) ** 2))
        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 6:
            break
    if best_state is None:
        raise RuntimeError("N14 distillation produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    student_test = mlp_probabilities(model, x_test, device)
    pair_metrics = {
        "pairs": len(x_test),
        "teacher_student_probability_mae": float(np.mean(np.abs(teacher_test - student_test))),
        "teacher_student_probability_max_abs": float(np.max(np.abs(teacher_test - student_test))),
        "teacher_student_decision_agreement": float(
            np.mean((teacher_test >= 0.5) == (student_test >= 0.5))
        ),
        "student_roc_auc": float(roc_auc_score(y_test, student_test)),
        "student_average_precision": float(average_precision_score(y_test, student_test)),
    }
    retrieval = n14_retrieval_metrics(module, model, groups, family_split, device)
    model_cpu = model.cpu().eval()
    checkpoint = {
        "schema": "icmat_chem_entity_mlp_checkpoint.v1",
        "state_dict": model_cpu.state_dict(),
        "architecture": [73, 128, 64, 1],
        "teacher_sha256": sha256_file(N14_TEACHER),
        "seed": SEED,
    }
    torch.save(checkpoint, artifact_dir / "chem_entity_mlp.fp32.pt")
    fixed = x_test[:1].astype(np.float32)
    onnx_path = artifact_dir / "chem_entity_mlp.static_1x73.onnx"
    torch.onnx.export(
        model_cpu,
        torch.from_numpy(fixed),
        onnx_path,
        input_names=["features_fp32"],
        output_names=["match_logit"],
        opset_version=11,
        dynamic_axes=None,
        do_constant_folding=True,
        dynamo=False,
    )
    graph = onnx.load(onnx_path)
    graph.ir_version = min(graph.ir_version, 7)
    onnx.save(graph, onnx_path)
    onnx.checker.check_model(onnx.load(onnx_path))
    np.savez_compressed(artifact_dir / "fixed_input.v1.npz", features_fp32=fixed)
    torch_logit = model_cpu(torch.from_numpy(fixed)).detach().numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_logit = session.run(None, {"features_fp32": fixed})[0]
    fixed_output = {
        "schema": "icmat_chem_entity_fixed_output.v1",
        "logit": float(onnx_logit[0, 0]),
        "match_probability": float(1.0 / (1.0 + np.exp(-onnx_logit[0, 0]))),
        "torch_onnx_max_abs": float(np.max(np.abs(torch_logit - onnx_logit))),
    }
    write_json(artifact_dir / "fixed_output.v1.json", fixed_output)
    receipt = {
        "schema": "icmat_knowledge_model_receipt.v1",
        "model_id": model_id,
        "model_name": "ICMat-ChemEntity-Linker-X5",
        "status": "PC_RUNNABLE_STATIC_ONNX_BPU_COMPILE_PENDING",
        "backend": "PURE_TENSOR_MLP_STATIC_ONNX",
        "validation_device": str(device),
        "teacher": {
            "model": "N14 RandomForest seed-20260731",
            "model_sha256": sha256_file(N14_TEACHER),
            "promotion_receipt_sha256": sha256_file(N14_EVIDENCE / "promotion_receipt.json"),
        },
        "dataset": {
            "source": "NIST JARVIS-DFT v11",
            "source_sha256": sha256_file(JARVIS_ARCHIVE),
            "train_pairs": len(x_train),
            "validation_pairs": len(x_validation),
            "test_pairs": len(x_test),
            "rejected_records": rejected,
            "family_disjoint": True,
        },
        "distillation": {
            "target": "N14 teacher positive-class probability",
            "student_parameters": sum(parameter.numel() for parameter in model_cpu.parameters()),
            "epochs_completed": epoch + 1,
            "best_validation_probability_mse": best_loss,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "pair_metrics": pair_metrics,
        "retrieval_metrics": retrieval,
        "onnx": {
            "input_shape": [1, 73],
            "output_shape": [1, 1],
            "opset": 11,
            "ir_version": onnx.load(onnx_path).ir_version,
            "checker_passed": True,
            "runtime_parity_max_abs": fixed_output["torch_onnx_max_abs"],
        },
        "fixed_input_sha256": sha256_file(artifact_dir / "fixed_input.v1.npz"),
        "fixed_output_sha256": sha256_file(artifact_dir / "fixed_output.v1.json"),
        "legacy_25228_training_used": False,
        "x5_contacted": False,
    }
    finalize_receipt(artifact_dir, evidence_dir, receipt)
    return receipt


def pointer_evidence(row: dict[str, Any]) -> tuple[list[str], str, int]:
    sentences = [
        sentence["text"]
        for evidence in row["compiler_evidence"]
        for sentence in evidence["sentences"]
    ]
    if not sentences:
        raise RuntimeError(f"missing evidence in {row.get('example_id', 'UNKNOWN')}")
    claim = row.get("requested_claim") or row.get("metadata", {}).get("requested_claim")
    if not isinstance(claim, str) or not claim.strip():
        raise RuntimeError(f"missing requested claim in {row.get('example_id', 'UNKNOWN')}")
    return sentences, claim, int(row["decision"] == "ANSWER")


@torch.inference_mode()
def nli_scores(
    tokenizer: Any,
    model: nn.Module,
    rows: list[dict[str, Any]],
    device: torch.device,
) -> np.ndarray:
    flattened: list[tuple[str, str]] = []
    spans: list[tuple[int, int]] = []
    for row in rows:
        sentences, claim, _label = pointer_evidence(row)
        start = len(flattened)
        flattened.extend((sentence, claim) for sentence in sentences)
        spans.append((start, len(flattened)))
    pair_scores: list[np.ndarray] = []
    for chunk in batches(flattened, 16):
        encoded = tokenizer(
            [premise for premise, _hypothesis in chunk],
            [hypothesis for _premise, hypothesis in chunk],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        probabilities = torch.softmax(model(**encoded).logits, dim=-1)
        pair_scores.append(probabilities[:, 1].float().cpu().numpy())
    all_scores = np.concatenate(pair_scores).astype(np.float32)
    return np.asarray([all_scores[start:stop].max() for start, stop in spans], dtype=np.float32)


def choose_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    candidates = np.unique(np.concatenate(([0.0], scores, [1.0])))
    best = (-1.0, 0.5)
    for threshold in candidates:
        metric = balanced_accuracy_score(labels, scores >= threshold)
        if metric > best[0] or (metric == best[0] and abs(threshold - 0.5) < abs(best[1] - 0.5)):
            best = (float(metric), float(threshold))
    return best[1], best[0]


def build_citation_nli(device: torch.device) -> dict[str, Any]:
    model_id = "F-KNW-04"
    artifact_dir = ARTIFACT_ROOT / model_id
    evidence_dir = EVIDENCE_ROOT / model_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(NLI_SOURCE, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        NLI_SOURCE, local_files_only=True
    ).to(device).eval()
    calibration_rows = read_jsonl(POINTER_V8 / "calibration.jsonl")
    validation_rows = read_jsonl(POINTER_V8 / "validation.jsonl")
    started = time.perf_counter()
    calibration_scores = nli_scores(tokenizer, model, calibration_rows, device)
    validation_scores = nli_scores(tokenizer, model, validation_rows, device)
    calibration_labels = np.asarray([pointer_evidence(row)[2] for row in calibration_rows])
    validation_labels = np.asarray([pointer_evidence(row)[2] for row in validation_rows])
    threshold, calibration_balanced = choose_threshold(calibration_labels, calibration_scores)
    validation_prediction = validation_scores >= threshold
    metrics = {
        "calibration_examples": len(calibration_rows),
        "validation_examples": len(validation_rows),
        "threshold": threshold,
        "calibration_balanced_accuracy": calibration_balanced,
        "validation_balanced_accuracy": float(
            balanced_accuracy_score(validation_labels, validation_prediction)
        ),
        "validation_roc_auc": float(roc_auc_score(validation_labels, validation_scores)),
        "validation_average_precision": float(
            average_precision_score(validation_labels, validation_scores)
        ),
        "unsupported_claim_recall": float(
            np.mean(~validation_prediction[validation_labels == 0])
        ),
        "supported_claim_recall": float(
            np.mean(validation_prediction[validation_labels == 1])
        ),
    }
    calibration = {
        "schema": "icmat_citation_nli_calibration.v1",
        "decision": "SUPPORTED" if threshold <= 0.5 else "CALIBRATED_THRESHOLD",
        "entailment_threshold": threshold,
        "metrics": metrics,
    }
    write_json(artifact_dir / "calibration.v1.json", calibration)
    fixed_premises, fixed_hypothesis, fixed_label = pointer_evidence(validation_rows[0])
    fixed_input = {
        "schema": "icmat_citation_nli_fixed_input.v1",
        "premises": fixed_premises,
        "hypothesis": fixed_hypothesis,
        "aggregation": "maximum_entailment_probability",
    }
    fixed_score = float(validation_scores[0])
    fixed_output = {
        "schema": "icmat_citation_nli_fixed_output.v1",
        "entailment_probability": fixed_score,
        "verdict": "SUPPORTED" if fixed_score >= threshold else "UNSUPPORTED",
        "expected_label": "SUPPORTED" if fixed_label else "UNSUPPORTED",
    }
    write_json(artifact_dir / "fixed_input.v1.json", fixed_input)
    write_json(artifact_dir / "fixed_output.v1.json", fixed_output)
    model_binding = {
        "repo_id": "cross-encoder/nli-deberta-v3-small",
        "revision": json.loads(NLI_RECEIPT.read_text(encoding="utf-8"))["revision"],
        "source_tree_sha256": tree_sha256(NLI_SOURCE),
        "source_model_sha256": sha256_file(NLI_SOURCE / "model.safetensors"),
        "source_receipt_sha256": sha256_file(NLI_RECEIPT),
        "real_transformers_load_completed": True,
        "weights_copied": False,
        "source_path": str(NLI_SOURCE.relative_to(ROOT)).replace("\\", "/"),
    }
    write_json(artifact_dir / "model_binding.v1.json", model_binding)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    receipt = {
        "schema": "icmat_knowledge_model_receipt.v1",
        "model_id": model_id,
        "model_name": "ICMat-CitationNLI-CPU",
        "status": "PC_RUNNABLE_REAL_WEIGHTS_CALIBRATED",
        "backend": "LOCAL_TRANSFORMERS_NLI_CPU_TARGET",
        "validation_device": str(device),
        "model_binding": model_binding,
        "sources": source_bindings(),
        "metrics": metrics,
        "elapsed_seconds": time.perf_counter() - started,
        "fixed_input_sha256": sha256_file(artifact_dir / "fixed_input.v1.json"),
        "fixed_output_sha256": sha256_file(artifact_dir / "fixed_output.v1.json"),
        "legacy_25228_training_used": False,
        "x5_contacted": False,
    }
    finalize_receipt(artifact_dir, evidence_dir, receipt)
    return receipt


def source_bindings() -> dict[str, Any]:
    return {
        "rag_v2_14p519": {
            "corpus_sha256": sha256_file(RAG_CORPUS),
            "manifest_sha256": sha256_file(RAG_MANIFEST),
            "chunks": 519,
            "usage": "retrieval_corpus_not_weight_training",
        },
        "semantic_queries_v7": {
            "path_sha256": sha256_file(SEMANTIC_V7),
            "accepted_training_eligible_records": 1482,
            "usage": "fixed_retrieval_evaluation",
        },
        "evidence_pointer_v8": {
            "train_sha256": sha256_file(POINTER_V8 / "train.jsonl"),
            "validation_sha256": sha256_file(POINTER_V8 / "validation.jsonl"),
            "calibration_sha256": sha256_file(POINTER_V8 / "calibration.jsonl"),
            "examples": 550,
            "usage": "citation_nli_calibration_and_validation",
        },
        "legacy_rag_25228": {
            "used_for_training": False,
            "used_for_evaluation": False,
            "decision": "EXCLUDED_BY_USER_CONTRACT",
        },
    }


def finalize_receipt(artifact_dir: Path, evidence_dir: Path, receipt: dict[str, Any]) -> None:
    manifest = {
        "schema": "icmat_knowledge_artifact_manifest.v1",
        "model_id": receipt["model_id"],
        "files": hash_tree(artifact_dir),
    }
    manifest["tree_sha256"] = sha256_json(manifest["files"])
    write_json(evidence_dir / "artifact_manifest.v1.json", manifest)
    receipt["artifact_manifest_sha256"] = sha256_file(
        evidence_dir / "artifact_manifest.v1.json"
    )
    receipt["artifact_tree_sha256"] = manifest["tree_sha256"]
    receipt["receipt_sha256"] = sha256_json(receipt)
    write_json(evidence_dir / "receipt.v1.json", receipt)


def verify_receipts() -> dict[str, Any]:
    models: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    fixed_files = {
        "F-KNW-01": ("fixed_input.v1.json", "fixed_output.v1.json"),
        "F-KNW-02": ("fixed_input.v1.json", "fixed_output.v1.json"),
        "F-KNW-03": ("fixed_input.v1.npz", "fixed_output.v1.json"),
        "F-KNW-04": ("fixed_input.v1.json", "fixed_output.v1.json"),
    }
    for model_id in ("F-KNW-01", "F-KNW-02", "F-KNW-03", "F-KNW-04"):
        artifact_dir = ARTIFACT_ROOT / model_id
        evidence_dir = EVIDENCE_ROOT / model_id
        receipt_path = evidence_dir / "receipt.v1.json"
        manifest_path = evidence_dir / "artifact_manifest.v1.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        actual_files = hash_tree(artifact_dir)
        input_name, output_name = fixed_files[model_id]
        receipt_copy = dict(receipt)
        stored_receipt_sha = receipt_copy.pop("receipt_sha256")
        passed = (
            manifest["files"] == actual_files
            and manifest["tree_sha256"] == sha256_json(actual_files)
            and receipt["artifact_manifest_sha256"] == sha256_file(manifest_path)
            and receipt["fixed_input_sha256"] == sha256_file(artifact_dir / input_name)
            and receipt["fixed_output_sha256"] == sha256_file(artifact_dir / output_name)
            and stored_receipt_sha == sha256_json(receipt_copy)
            and receipt["legacy_25228_training_used"] is False
            and receipt["x5_contacted"] is False
        )
        models.append(
            {
                "model_id": model_id,
                "status": receipt["status"],
                "receipt_sha256": sha256_file(receipt_path),
                "artifact_tree_sha256": manifest["tree_sha256"],
                "verified": passed,
            }
        )
        if model_id == "F-KNW-02":
            summaries[model_id] = receipt["reranked_metrics"]
        elif model_id == "F-KNW-03":
            summaries[model_id] = receipt["retrieval_metrics"]
        else:
            summaries[model_id] = receipt["metrics"]
    result = {
        "schema": "icmat_knowledge_bank_build_receipt.v1",
        "status": "PASS" if all(row["verified"] for row in models) else "FAIL",
        "models": models,
        "summaries": summaries,
        "model_count": len(models),
        "legacy_25228_training_used": False,
        "production_modified": False,
        "registry_modified": False,
        "overlay_modified": False,
        "agents_modified": False,
        "x5_contacted": False,
    }
    result["receipt_sha256"] = sha256_json(result)
    write_json(EVIDENCE_ROOT / "build_receipt.v1.json", result)
    return result


def clean_download_cache() -> None:
    cache = ARTIFACT_ROOT / ".hf_home"
    if cache.is_dir():
        shutil.rmtree(cache)
    for path in ARTIFACT_ROOT.rglob(".cache"):
        if path.is_dir():
            shutil.rmtree(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--start-at",
        choices=("dense", "chem", "nli"),
        default="dense",
        help="Resume after already receipted earlier stages.",
    )
    args = parser.parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if args.verify_only:
        result = verify_receipts()
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result["status"] == "PASS" else 1
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    device = select_device()
    if args.start_at == "dense":
        corpus, records = load_retrieval_data()
        dense, corpus_embeddings, query_embeddings, _tokenizer, _model_dir = build_dense(
            corpus, records, device
        )
        cross = build_cross_encoder(
            corpus, records, corpus_embeddings, query_embeddings, device
        )
    else:
        dense = json.loads(
            (EVIDENCE_ROOT / "F-KNW-01" / "receipt.v1.json").read_text(encoding="utf-8")
        )
        cross = json.loads(
            (EVIDENCE_ROOT / "F-KNW-02" / "receipt.v1.json").read_text(encoding="utf-8")
        )
    if args.start_at in ("dense", "chem"):
        chem = build_chem_entity(device)
    else:
        chem = json.loads(
            (EVIDENCE_ROOT / "F-KNW-03" / "receipt.v1.json").read_text(encoding="utf-8")
        )
    nli = build_citation_nli(device)
    clean_download_cache()
    result = verify_receipts()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
