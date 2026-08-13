#!/usr/bin/env python3
"""Render award-status blocks from the repository's single YAML source.

This intentionally parses only the small two-level scalar schema used by
docs/competition/award_status.yaml, so publication does not require PyYAML.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


START = "<!-- AWARD_STATUS:START -->"
END = "<!-- AWARD_STATUS:END -->"
PENDING_STATUS = "pending_official" + "_announcement"


def parse_scalar(raw: str):
    value = raw.strip()
    if value in {"null", "~"}:
        return None
    if value in {"true", "false"}:
        return value == "true"
    if value.startswith('"') and value.endswith('"'):
        return json.loads(value)
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    return value


def load_status(path: Path) -> dict:
    data: dict = {}
    section: str | None = None
    for number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if "\t" in raw_line:
            raise ValueError(f"tabs are not allowed in {path}:{number}")
        if not raw_line.startswith(" "):
            key, separator, value = raw_line.partition(":")
            if not separator:
                raise ValueError(f"invalid line in {path}:{number}")
            key = key.strip()
            if value.strip():
                data[key] = parse_scalar(value)
                section = None
            else:
                data[key] = {}
                section = key
            continue
        if not raw_line.startswith("  ") or raw_line.startswith("    ") or section is None:
            raise ValueError(f"only two-level scalar YAML is supported: {path}:{number}")
        key, separator, value = raw_line.strip().partition(":")
        if not separator or not value.strip():
            raise ValueError(f"invalid scalar in {path}:{number}")
        data[section][key.strip()] = parse_scalar(value)
    return data


def validate_evidence(root: Path, section: dict, label: str) -> None:
    source_url = section.get("source_url")
    if not isinstance(source_url, str) or not source_url.startswith("https://"):
        raise ValueError(f"{label} official source must be an HTTPS URL")
    evidence_path = section.get("evidence_path")
    if not isinstance(evidence_path, str) or not evidence_path:
        raise ValueError(f"{label} official evidence path is required")
    candidate = (root / evidence_path).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} evidence must stay inside the repository") from exc
    if not candidate.is_file():
        raise ValueError(f"{label} evidence file does not exist: {evidence_path}")
    expected_sha256 = section.get("evidence_sha256")
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError(f"{label} evidence_sha256 must be 64 lowercase hex characters")
    actual_sha256 = hashlib.sha256(candidate.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(f"{label} evidence SHA-256 mismatch")


def validate(data: dict, root: Path) -> None:
    for section in ("competition", "team", "regional", "national", "publication_rules"):
        if not isinstance(data.get(section), dict):
            raise ValueError(f"missing mapping: {section}")

    if data["publication_rules"].get("forbid_predicted_award") is not True:
        raise ValueError("forbid_predicted_award must remain true")

    competition = data["competition"]
    if competition.get("name_zh") != "2026 全国大学生嵌入式芯片与系统设计竞赛":
        raise ValueError("competition name must use the approved 2026 title")
    if competition.get("division") != "芯片应用赛道":
        raise ValueError("competition division must remain 芯片应用赛道")
    if competition.get("topic") != "地瓜机器人赛题":
        raise ValueError("competition topic must remain 地瓜机器人赛题")

    regional = data["regional"]
    if regional.get("region") != "西南赛区" or regional.get("result") != "一等奖":
        raise ValueError("regional award must remain 西南赛区一等奖")
    if regional.get("status") not in {"team_confirmed", "official_verified"}:
        raise ValueError("regional status must be team_confirmed or official_verified")
    if regional.get("status") == "official_verified":
        validate_evidence(root, regional, "regional")

    national = data["national"]
    pending = national.get("status") == PENDING_STATUS
    if pending:
        if any(
            national.get(key) is not None
            for key in ("result", "source_url", "evidence_path", "evidence_sha256", "announced_at")
        ):
            raise ValueError("pending national status must not prefill a result or evidence")
        return

    if national.get("status") == "team_confirmed":
        if national.get("result") != "二等奖":
            raise ValueError("the team-confirmed national result must remain 二等奖")
        if any(
            national.get(key) is not None
            for key in ("source_url", "evidence_path", "evidence_sha256", "announced_at")
        ):
            raise ValueError("team-confirmed national status cannot claim official evidence")
        return

    if national.get("status") != "official_verified":
        raise ValueError(
            "a published national result must use status=team_confirmed or official_verified"
        )
    if data["publication_rules"].get("require_official_source_for_national_result"):
        required = ("result", "source_url", "evidence_path", "evidence_sha256", "announced_at")
        missing = [key for key in required if not national.get(key)]
        if missing:
            raise ValueError(f"official national result is missing: {', '.join(missing)}")
    validate_evidence(root, national, "national")


def source_link(label: str, url: str | None) -> str:
    return f"[{label}]({url})" if url else label


def zh_block(data: dict, boundary_header: str) -> str:
    regional = data["regional"]
    national = data["national"]
    rules = data["publication_rules"]

    regional_result = regional["result"]
    if regional.get("status") == "official_verified":
        regional_boundary = source_link("`official_verified`：官方来源", regional.get("source_url"))
    else:
        regional_boundary = "`team_confirmed`：队伍确认，官方获奖来源待补"

    if national.get("status") == PENDING_STATUS:
        national_result = rules["national_pending_text_zh"]
        national_boundary = "待官方公布：不预测、不预填奖项"
    elif national.get("status") == "team_confirmed":
        national_result = national["result"]
        national_boundary = "`team_confirmed`：队伍确认，组委会官方获奖来源待补"
    else:
        national_result = national["result"]
        national_boundary = source_link("`official_verified`：组委会官方来源", national.get("source_url"))

    return "\n".join(
        [
            START,
            f"| 阶段 | 当前状态 | {boundary_header} |",
            "| --- | --- | --- |",
            f"| {regional['region']} | {regional_result} | {regional_boundary} |",
            f"| {national['stage']} | {national_result} | {national_boundary} |",
            END,
        ]
    )


def en_block(data: dict) -> str:
    regional = data["regional"]
    national = data["national"]
    regional_result = "First Prize" if regional.get("result") == "一等奖" else regional["result"]
    if regional.get("status") == "official_verified":
        regional_boundary = source_link("`official_verified`; official source", regional.get("source_url"))
    else:
        regional_boundary = "`team_confirmed`; official award source pending"

    if national.get("status") == PENDING_STATUS:
        national_result = "Pending official announcement"
        national_boundary = "No award may be predicted or prefilled"
    elif national.get("status") == "team_confirmed":
        national_result = "Second Prize"
        national_boundary = "`team_confirmed`; official organizing-committee source pending"
    else:
        national_result = "Second Prize" if national.get("result") == "二等奖" else national["result"]
        national_boundary = source_link("`official_verified`; official source", national.get("source_url"))

    return "\n".join(
        [
            START,
            "| Stage | Current status | Evidence boundary |",
            "| --- | --- | --- |",
            f"| Southwest Regional Contest | {regional_result} | {regional_boundary} |",
            f"| National final | {national_result} | {national_boundary} |",
            END,
        ]
    )


def replace_block(text: str, block: str, path: Path) -> str:
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.DOTALL)
    updated, count = pattern.subn(block, text)
    if count != 1:
        raise ValueError(f"expected exactly one award block in {path}, found {count}")
    return updated


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if generated blocks are stale")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    status_path = root / "docs" / "competition" / "award_status.yaml"
    data = load_status(status_path)
    validate(data, root)

    targets = {
        root / "README.md": zh_block(data, "事实边界"),
        root / "README_en.md": en_block(data),
        root / "docs" / "competition" / "AWARDS.md": zh_block(data, "证据边界"),
    }
    stale: list[str] = []
    for path, block in targets.items():
        original = path.read_text(encoding="utf-8")
        updated = replace_block(original, block, path)
        if updated == original:
            continue
        stale.append(path.relative_to(root).as_posix())
        if not args.check:
            path.write_text(updated, encoding="utf-8", newline="\n")

    if args.check and stale:
        raise SystemExit("stale award blocks: " + ", ".join(stale))
    if stale:
        print("updated: " + ", ".join(stale))
    else:
        print("award blocks are current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
