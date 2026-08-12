#!/usr/bin/env python3
"""Fail closed if the successor contains hardware or remote-control code."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any


BANNED_IMPORT_ROOTS = {
    "gpiozero",
    "lgpio",
    "paramiko",
    "periphery",
    "pigpio",
    "pymycobot",
    "rclpy",
    "RPi",
    "serial",
    "smbus",
    "smbus2",
    "socket",
    "spidev",
}
BANNED_CALL_NAMES = {
    "power_on",
    "release_all_servos",
    "send_angle",
    "send_angles",
    "send_coord",
    "send_coords",
    "set_basic_output",
    "set_encoder",
    "set_encoders",
    "set_gripper_value",
    "set_pwm_output",
    "set_servo_data",
}
BANNED_LITERAL_PATTERNS = (
    re.compile(r"/dev/tty", re.IGNORECASE),
    re.compile(r"/dev/video", re.IGNORECASE),
    re.compile(r"\b(?:ssh|scp)\s+[^-]", re.IGNORECASE),
)
EXCLUDED_PARTS = {
    ".venv-cpu",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    "evidence",
    "backups",
    "outputs",
    "packages",
}


def call_name(node: ast.Call) -> str:
    current: ast.AST = node.func
    if isinstance(current, ast.Name):
        return current.id
    if isinstance(current, ast.Attribute):
        return current.attr
    return ""


def scan_python(path: Path) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError) as exc:
        return [{"path": str(path), "kind": "PYTHON_PARSE_ERROR", "detail": str(exc)}]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in BANNED_IMPORT_ROOTS:
                    findings.append(
                        {
                            "path": str(path),
                            "line": node.lineno,
                            "kind": "BANNED_IMPORT",
                            "detail": alias.name,
                        }
                    )
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".", 1)[0]
            if root in BANNED_IMPORT_ROOTS:
                findings.append(
                    {
                        "path": str(path),
                        "line": node.lineno,
                        "kind": "BANNED_IMPORT",
                        "detail": node.module,
                    }
                )
        elif isinstance(node, ast.Call):
            name = call_name(node)
            if name in BANNED_CALL_NAMES:
                findings.append(
                    {
                        "path": str(path),
                        "line": node.lineno,
                        "kind": "BANNED_CONTROL_CALL",
                        "detail": name,
                    }
                )
            if name in {"open", "run", "Popen", "system", "check_call", "check_output"}:
                literals = [
                    arg.value
                    for arg in node.args
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                ]
                for literal in literals:
                    for pattern in BANNED_LITERAL_PATTERNS:
                        if pattern.search(literal):
                            findings.append(
                                {
                                    "path": str(path),
                                    "line": node.lineno,
                                    "kind": "BANNED_EXECUTABLE_LITERAL",
                                    "detail": pattern.pattern,
                                }
                            )
    return findings


def scan_shell(path: Path) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    text = path.read_text(encoding="utf-8")
    for line_no, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        for pattern in BANNED_LITERAL_PATTERNS:
            if pattern.search(stripped):
                findings.append(
                    {
                        "path": str(path),
                        "line": line_no,
                        "kind": "BANNED_SHELL_CONTROL",
                        "detail": pattern.pattern,
                    }
                )
    return findings


def should_scan(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    return not any(part in EXCLUDED_PARTS for part in relative.parts)


def audit(root: Path) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    scanned: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or not should_scan(path, root):
            continue
        if path.suffix == ".py":
            scanned.append(str(path.relative_to(root)))
            findings.extend(scan_python(path))
        elif path.suffix in {".sh", ".ps1", ".cmd"}:
            scanned.append(str(path.relative_to(root)))
            findings.extend(scan_shell(path))
    return {
        "schema_version": "xrd-dual-arm-no-motion-audit-v1",
        "status": "PASS" if not findings else "FAIL",
        "root": str(root),
        "files_scanned": scanned,
        "findings": findings,
        "motion_authority": False,
        "execution_allowed": False,
        "actuator_commands_issued": 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    receipt = audit(root)
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "files_scanned": len(receipt["files_scanned"]),
                "findings": len(receipt["findings"]),
            },
            sort_keys=True,
        )
    )
    return 0 if receipt["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
