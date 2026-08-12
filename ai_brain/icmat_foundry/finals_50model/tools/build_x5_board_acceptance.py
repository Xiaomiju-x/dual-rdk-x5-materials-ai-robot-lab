#!/usr/bin/env python3
"""Build the immutable X5 board-phase acceptance overlay and readable receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FINAL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = FINAL_ROOT.parents[1]
BOARD_ROOT = FINAL_ROOT / "evidence" / "x5_board_20260804"
REGISTRY = FINAL_ROOT / "contracts" / "model_registry.v3.json"
PC_ACCEPTANCE = FINAL_ROOT / "evidence" / "final_acceptance" / "final_acceptance.v1.json"
EXECUTION = BOARD_ROOT / "execution_v1" / "board_session_receipt.v1.json"
HRT_SESSION = BOARD_ROOT / "hrt_recovery_v1" / "hrt_recovery_session.v1.json"
LLM_SESSION = BOARD_ROOT / "bpu_llm_part2_recovery_v1" / "bpu_llm_part2_recovery_session.v1.json"
OVERLAY = BOARD_ROOT / "bpu_llm_part2_recovery_v1" / "final_board_state_overlay.v1.json"
NONINTERFERENCE = BOARD_ROOT / "final_noninterference_receipt.v1.json"
VALIDATION_MANIFEST = BOARD_ROOT / "validation_bundle_v2" / "bundle_manifest.json"
STAGING_ZIP = FINAL_ROOT / "releases" / "x5-icmat-foundry-50model-x5-staging-c5fa215a58168c0c.zip"
EXPECTED_REGISTRY_SHA256 = "a2293bce08d6de380dbbbcf8876381e946d329692bc07dc98dec88199d2f7ef2"
DOCUMENTED_PC_ACCEPTANCE_SHA256 = "aa03341e9bc44c5e47e63935035cebaafa4900e18f0afb3a6af6583eb6330668"
EXPECTED_STAGING_SHA256 = "c5fa215a58168c0cb7274c2b1cf6d66bcd0f3c1e70d3f4cf13749e9b57dafb52"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_path(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT.resolve())).replace("\\", "/")


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def evidence_record(path: Path) -> dict[str, Any]:
    return {"path": repo_path(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def parse_meminfo(block: str) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in block.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields = value.strip().split()
        if fields and fields[0].isdigit():
            values[key] = int(fields[0])
    return values


def receipt_model_payload(path: Path) -> dict[str, Any]:
    envelope = load_json(path)
    return envelope.get("model_receipt", envelope)


def build(output_dir: Path) -> tuple[dict[str, Any], str]:
    registry = load_json(REGISTRY)
    pc_acceptance = load_json(PC_ACCEPTANCE)
    execution = load_json(EXECUTION)
    hrt_session = load_json(HRT_SESSION)
    llm_session = load_json(LLM_SESSION)
    overlay = load_json(OVERLAY)
    noninterference = load_json(NONINTERFERENCE)

    assert sha256(REGISTRY) == EXPECTED_REGISTRY_SHA256
    assert sha256(STAGING_ZIP) == EXPECTED_STAGING_SHA256
    actual_pc_acceptance_sha256 = sha256(PC_ACCEPTANCE)
    embedded_acceptance_sha256: dict[str, str] = {}
    for release_path in (
        FINAL_ROOT / "releases" / "x5-icmat-foundry-50model-pc-c7aff501602bde2f.zip",
        STAGING_ZIP,
    ):
        with zipfile.ZipFile(release_path) as archive:
            embedded = archive.read("acceptance.json")
        embedded_acceptance_sha256[release_path.name] = hashlib.sha256(embedded).hexdigest()
    assert set(embedded_acceptance_sha256.values()) == {actual_pc_acceptance_sha256}
    assert noninterference["result"] == "PASS"
    assert noninterference["summary"]["failed"] == 0
    assert overlay["counts"] == {"X5_VALIDATED": 31, "BOARD_REJECTED": 3, "BOARD_EXPERIMENTAL": 4}

    registry_by_id = {item["inventory_id"]: item for item in registry["models"]}
    finals_registry = {key: value for key, value in registry_by_id.items() if key.startswith("F-")}
    frozen = [item for item in registry["models"] if item["runtime_scope"] == "X5_FROZEN_PRODUCTION"]
    assert len(finals_registry) == 38
    assert len(frozen) == 11
    assert len(overlay["models"]) == 38

    execution_by_id = {item["inventory_id"]: item for item in execution["models"]}
    hrt_by_id = {item["inventory_id"]: item for item in hrt_session["candidates"]}
    llm_by_id = {item["inventory_id"]: item for item in llm_session["results"]}
    models: list[dict[str, Any]] = []
    actual_bpu = 0
    actual_cpu = 0
    for state in sorted(overlay["models"], key=lambda item: item["inventory_id"]):
        inventory_id = state["inventory_id"]
        registry_row = finals_registry[inventory_id]
        initial_meta = execution_by_id[inventory_id]
        initial_path = EXECUTION.parent / initial_meta["receipt"]
        evidence = [evidence_record(initial_path)]
        status = state["final_board_status"]
        details: dict[str, Any] = {}
        if inventory_id in llm_by_id:
            recovery_meta = llm_by_id[inventory_id]
            recovery_path = LLM_SESSION.parent / recovery_meta["receipt"]
            evidence.append(evidence_record(recovery_path))
            initial_envelope = load_json(initial_path)
            recovery = load_json(recovery_path)
            part1 = initial_envelope["part1"]["model_receipt"]
            part2 = recovery["part2"]["model_receipt"]
            details = {
                "actual_backend": "X5_BPU_PART1_AND_PART2_WITH_CPU_EMBED_NORM_LM_HEAD",
                "part1_model_sha256": part1["model_sha256"],
                "part2_model_sha256": part2["model_sha256"],
                "part1_load_ms": part1["load_ms"],
                "part1_inference_ms": part1["inference_ms"],
                "part2_load_ms": part2["load_ms"],
                "part2_inference_ms": part2["inference_ms"],
                "part1_output_tensor_sha256": recovery["part1_output_tensor_sha256"],
                "part2_input_tensor_sha256": recovery["part2_input_tensor_sha256"],
                "content_bound": recovery["part1_part2_content_bound"],
                "expected_next_token_id": part2["expected_next_token_id"],
                "actual_next_token_id": part2["actual_next_token_id"],
                "next_token_exact": part2["next_token_exact"],
                "claim_boundary": part2["claim_boundary"],
            }
            assert details["content_bound"] is True
            actual_bpu += 1
        elif inventory_id in hrt_by_id:
            recovery_meta = hrt_by_id[inventory_id]
            recovery_path = HRT_SESSION.parent / recovery_meta["receipt"]
            evidence.append(evidence_record(recovery_path))
            payload = receipt_model_payload(recovery_path)
            details = {
                key: payload.get(key)
                for key in (
                    "actual_backend",
                    "model_sha256",
                    "input_tensor_sha256",
                    "output_tensor_sha256",
                    "load_ms",
                    "inference_ms",
                    "differential",
                )
            }
            actual_bpu += 1
        else:
            payload = receipt_model_payload(initial_path)
            details = {
                key: payload.get(key)
                for key in (
                    "actual_backend",
                    "backend",
                    "model_sha256",
                    "input_tensor_sha256",
                    "output_tensor_sha256",
                    "load_ms",
                    "inference_ms",
                    "reason",
                    "missing_companion_assets",
                )
            }
            if payload.get("actual_backend"):
                if registry_row["primary_backend"] == "BPU":
                    actual_bpu += 1
                else:
                    actual_cpu += 1
        models.append(
            {
                "inventory_id": inventory_id,
                "model_id": registry_row["model_id"],
                "family": registry_row["family"],
                "primary_backend": registry_row["primary_backend"],
                "final_board_status": status,
                "details": details,
                "evidence": evidence,
            }
        )

    assert actual_bpu == 24
    assert actual_cpu == 11
    status_counts = Counter(item["final_board_status"] for item in models)
    backend_status_counts = {
        backend: dict(Counter(item["final_board_status"] for item in models if item["primary_backend"] == backend))
        for backend in ("CPU", "BPU")
    }
    llm_results = [item for item in models if item["inventory_id"] in llm_by_id]
    rejected = [item for item in models if item["final_board_status"] == "BOARD_REJECTED"]
    experimental = [item for item in models if item["final_board_status"] == "BOARD_EXPERIMENTAL"]
    observations = noninterference["observations"]
    final_resources = parse_meminfo(observations["resources"])

    source_paths = [
        REGISTRY,
        PC_ACCEPTANCE,
        STAGING_ZIP,
        VALIDATION_MANIFEST,
        EXECUTION,
        HRT_SESSION,
        LLM_SESSION,
        OVERLAY,
        NONINTERFERENCE,
    ]
    receipt = {
        "schema": "x5_icmat_foundry.board_phase_acceptance.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "board_phase_status": "COMPLETE_WITH_REJECTIONS_AND_EXPERIMENTALS",
        "target_identity": {
            "hostname": "xrd-ai",
            "user": "sunrise",
            "address": "192.0.2.103",
            "host_key_sha256": noninterference["known_hosts_sha256"],
        },
        "accounting": {
            "registry_total": 50,
            "x5_local_logical_models": 49,
            "frozen_production_baseline_preserved": 11,
            "new_finals_candidates_status_complete": 38,
            "pc_offline_only": 1,
            "new_candidates": dict(status_counts),
            "new_candidates_by_primary_backend": backend_status_counts,
            "actual_x5_backend_executed": 35,
            "actual_x5_cpu_executed": actual_cpu,
            "actual_x5_bpu_executed": actual_bpu,
            "actual_x5_bpu_segment_bins_executed": 6,
        },
        "frozen_production_baseline": [
            {
                "inventory_id": item["inventory_id"],
                "model_id": item["model_id"],
                "status": "FROZEN_PRODUCTION_BASELINE_PRESERVED_NOT_REBENCHMARKED",
            }
            for item in frozen
        ],
        "new_candidate_models": models,
        "exceptions": {
            "board_rejected": [
                {
                    "inventory_id": item["inventory_id"],
                    "reason": item["details"].get("reason"),
                    "missing_companion_assets": item["details"].get("missing_companion_assets"),
                }
                for item in rejected
            ],
            "board_experimental": [
                {
                    "inventory_id": item["inventory_id"],
                    "reason": (
                        "FIXED_NEXT_TOKEN_MISMATCH"
                        if item["inventory_id"].startswith("F-LLM-")
                        else "ACTUAL_INT8_TASK_GATE_DIVERGENCE"
                    ),
                }
                for item in experimental
            ],
        },
        "bpu_llm_fixed_contracts": [
            {"inventory_id": item["inventory_id"], **item["details"]} for item in llm_results
        ],
        "final_noninterference": {
            "result": noninterference["result"],
            "checks": noninterference["summary"],
            "endpoint_status": observations["endpoint_status"],
            "final_resources_kib": final_resources,
            "candidate_processes": observations["candidate_processes"],
            "candidate_systemd_units": observations["candidate_units"],
            "release_service_files": observations["release_service_files"],
            "camera_devices": observations["camera_devices"],
            "production_service_state": observations["service_state"],
            "receipt": evidence_record(NONINTERFERENCE),
        },
        "claim_boundaries": {
            "official_pc_registry_and_acceptance_immutable": True,
            "official_pc_acceptance_x5_board_verified_remains": pc_acceptance["counts"]["x5_board_verified"],
            "pc_acceptance_hash_discrepancy": {
                "documented_sha256": DOCUMENTED_PC_ACCEPTANCE_SHA256,
                "workspace_actual_sha256": actual_pc_acceptance_sha256,
                "embedded_release_acceptance_sha256": embedded_acceptance_sha256,
                "assessment": "Workspace acceptance is byte-identical to acceptance.json embedded in both frozen release ZIPs, but differs from the hash still documented in AGENTS/ledger/README; no baseline file was rewritten during board closeout.",
            },
            "board_results_are_separate_overlay": True,
            "x5_validated_means_fixed_task_contract_on_demand_not_simultaneous_residency": True,
            "F_PROC_03": "QUALITY_LIMITED_NOT_PROMOTED",
            "SIM_ONLY": ["F-PKG-01", "F-PKG-02", "F-PKG-03", "F-PKG-04"],
            "bpu_llm": "All six bins executed on actual X5 BPU, but three logical models remain BOARD_EXPERIMENTAL because fixed next-token IDs diverged; no general free-generation claim.",
            "pc_compilation_not_board_performance": True,
            "production_services_not_modified": True,
            "fleet_audit_state_before_final_oneshot": "DEPLOYED_OFF",
        },
        "source_artifacts": [evidence_record(path) for path in source_paths],
        "fleet_audit_ready": True,
    }
    json_path = output_dir / "x5_board_phase_acceptance.v1.json"
    json_payload = canonical_bytes(receipt)
    atomic_write(json_path, json_payload)
    json_digest = sha256(json_path)
    atomic_write(output_dir / "x5_board_phase_acceptance.v1.json.sha256", f"{json_digest}  {json_path.name}\n".encode())

    status_rows = "\n".join(
        f"| {item['inventory_id']} | {item['family']} | {item['primary_backend']} | {item['final_board_status']} |"
        for item in models
    )
    llm_rows = "\n".join(
        f"| {item['inventory_id']} | {item['details']['expected_next_token_id']} | {item['details']['actual_next_token_id']} | {item['details']['part1_load_ms']:.1f} / {item['details']['part2_load_ms']:.1f} | {item['details']['part1_inference_ms']:.1f} / {item['details']['part2_inference_ms']:.1f} | BOARD_EXPERIMENTAL |"
        for item in llm_results
    )
    markdown = f"""# X5-ICMat Foundry 板端阶段最终回执

状态：`COMPLETE_WITH_REJECTIONS_AND_EXPERIMENTALS`

## 结论

- 官方 50 模型 registry 与两份 release 未改动；板端结果以本独立 overlay 记录。
- 38 个决赛新增候选已全部形成逐模型状态：`31 X5_VALIDATED / 3 BOARD_REJECTED / 4 BOARD_EXPERIMENTAL`。
- 24/24 个 BPU-primary 均在 actual X5 BPU 执行；14 个 CPU-primary 中 11 个完成 actual X5 CPU 推理，3 个因 staging 运行资产不完整而拒绝。
- 三套分段 BPU LLM 的 6 个 bin 均逐段 actual X5 执行且 part1/part2 content hash 绑定正确；固定 next-token 不一致，因此三者均保留 `BOARD_EXPERIMENTAL`。
- 最终生产非干扰检查 `33/33 PASS`：五端口、生产 9000–9003 LLM 槽、Dashboard 健康接口、BPU slot 健康接口、相机状态、生产哈希和 staging 哈希均保持；候选进程、19010/19011 和候选 systemd 单元均不存在。

## 模型会计

| 项目 | 数量 |
|---|---:|
| 注册表唯一逻辑模型 | 50 |
| X5-local 逻辑模型状态完整 | 49（冻结生产 11 + 决赛新增 38） |
| PC-only MACE-MPA-0 | 1 |
| 新增 X5_VALIDATED | 31 |
| 新增 BOARD_REJECTED | 3 |
| 新增 BOARD_EXPERIMENTAL | 4 |
| actual X5 backend 已执行 | 35（CPU 11 + BPU 24） |

## 三套 BPU LLM 固定任务结果

| 模型 | 期望 token | 实际 token | part1/part2 加载 ms | part1/part2 推理 ms | 状态 |
|---|---:|---:|---:|---:|---|
{llm_rows}

该结果只证明固定输入下两段 BPU 与 CPU head 的一次 next-token 合同执行，不证明自由生成、通用问答或 FP32 hidden-state 数值一致。

## 例外边界

- `F-KNW-01/02/04`：`BOARD_REJECTED`，仅有 safetensors，缺少完整 config/tokenizer/可执行 loader；未在 X5 上伪装成成功推理。
- `F-KNW-03`：`BOARD_EXPERIMENTAL`，actual BPU 已执行，但固定标量任务门未通过（NRMSE 0.52269）。
- `F-LLM-03/04/05`：`BOARD_EXPERIMENTAL`，6 个 bin 均 actual BPU 执行，但固定 next-token 不一致。
- `F-PROC-03` 继续为 `QUALITY_LIMITED_NOT_PROMOTED`；`F-PKG-01/02/03/04` 的 `SIM_ONLY` 边界继续保留。
- `X5_VALIDATED` 表示按需单模型固定任务合同通过，不表示 49 个 X5 模型同时常驻。
- PC acceptance 中 `x5_board_verified=0` 是冻结的上电前事实；没有改写，板端结果只记录在本 overlay。
- PC acceptance 工作区实际 SHA-256 为 `{actual_pc_acceptance_sha256}`，且与两份冻结 release 内嵌 `acceptance.json` 字节一致；但 AGENTS/ledger/README 仍记录旧哈希 `{DOCUMENTED_PC_ACCEPTANCE_SHA256}`，本回执保留该不一致告警，未覆盖任何基线文件。

## 38 个新增候选状态

| Inventory ID | 域 | Primary backend | 最终板端状态 |
|---|---|---|---|
{status_rows}

## 哈希

- 本 JSON 回执 SHA-256：`{json_digest}`
- registry SHA-256：`{EXPECTED_REGISTRY_SHA256}`
- PC acceptance 实际/两份 release 内嵌 SHA-256：`{actual_pc_acceptance_sha256}`
- 文档仍记录的 PC acceptance SHA-256：`{DOCUMENTED_PC_ACCEPTANCE_SHA256}`（待后续单独核账，不在本次上板中改写）
- X5 staging SHA-256：`{EXPECTED_STAGING_SHA256}`

下一步只允许对本回执执行一次只读 `FleetAudit PASSIVE_ONESHOT`，随后确认 `DEPLOYED_OFF`。
"""
    markdown_path = output_dir / "X5_BOARD_PHASE_FINAL_RECEIPT_20260804.md"
    atomic_write(markdown_path, markdown.encode("utf-8"))
    markdown_digest = sha256(markdown_path)
    atomic_write(output_dir / "X5_BOARD_PHASE_FINAL_RECEIPT_20260804.md.sha256", f"{markdown_digest}  {markdown_path.name}\n".encode())
    return receipt, json_digest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"refusing non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt, digest = build(output_dir)
    print(
        json.dumps(
            {
                "status": receipt["board_phase_status"],
                "counts": receipt["accounting"]["new_candidates"],
                "actual_x5_backend_executed": receipt["accounting"]["actual_x5_backend_executed"],
                "json_sha256": digest,
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
