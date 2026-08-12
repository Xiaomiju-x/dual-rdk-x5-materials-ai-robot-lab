#!/usr/bin/env python3
"""Offline contract tests for the cmdcenter production dependency lock."""

from __future__ import annotations

import ast
import re
import shlex
import sys
import unittest
from pathlib import Path


CMD_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_PATH = CMD_ROOT / "requirements-production.txt"
SERVICE_PATH = CMD_ROOT / "systemd" / "xrd-cmdcenter.service"
APP_PATH = CMD_ROOT / "app.py"
PACKAGE_ROOT = CMD_ROOT / "cmdcenter"

EXPECTED_LOCK = {
    "flask": "3.1.3",
    "werkzeug": "3.1.8",
    "gunicorn": "23.0.0",
}
EXACT_PIN_RE = re.compile(
    r"(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)"
    r"=="
    r"(?P<version>[A-Za-z0-9][A-Za-z0-9.!+_-]*)\Z"
)


def _canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirement_lines() -> list[tuple[int, str]]:
    lines = []
    for line_number, raw_line in enumerate(
        REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if line and not line.startswith("#"):
            lines.append((line_number, line))
    return lines


def _unit_values(unit_text: str, key: str) -> list[str]:
    prefix = f"{key}="
    return [
        line[len(prefix) :]
        for line in unit_text.splitlines()
        if line.startswith(prefix)
    ]


def _read_python(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


class Site32RequirementsContractTests(unittest.TestCase):
    def test_production_lock_is_exact_unique_and_minimal(self):
        locked = {}
        locations = {}

        for line_number, line in _requirement_lines():
            match = EXACT_PIN_RE.fullmatch(line)
            self.assertIsNotNone(
                match,
                f"line {line_number} must be one exact NAME==VERSION pin: {line!r}",
            )
            name = _canonical_name(match.group("name"))
            self.assertNotIn(
                name,
                locked,
                f"duplicate requirement {name!r} on lines {locations.get(name)} and {line_number}",
            )
            locked[name] = match.group("version")
            locations[name] = line_number

        self.assertEqual(locked, EXPECTED_LOCK)

    def test_locked_framework_matches_production_import_surface(self):
        import_roots = set()
        source_paths = [APP_PATH, *sorted(PACKAGE_ROOT.glob("*.py"))]

        for source_path in source_paths:
            tree = ast.parse(_read_python(source_path), filename=str(source_path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    import_roots.update(alias.name.partition(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    import_roots.add(node.module.partition(".")[0])

        stdlib = set(sys.stdlib_module_names) | set(sys.builtin_module_names)
        third_party = import_roots - stdlib - {"cmdcenter"}

        self.assertEqual(third_party, {"flask"})
        self.assertTrue(third_party.issubset(EXPECTED_LOCK))
        self.assertEqual(set(EXPECTED_LOCK) - third_party, {"gunicorn", "werkzeug"})

    def test_systemd_entry_uses_locked_gunicorn_and_real_app_factory(self):
        unit_text = SERVICE_PATH.read_text(encoding="utf-8")
        working_directories = _unit_values(unit_text, "WorkingDirectory")
        exec_starts = _unit_values(unit_text, "ExecStart")

        self.assertEqual(working_directories, ["/home/rdk/cmdcenter"])
        self.assertEqual(len(exec_starts), 1, "systemd unit must have one ExecStart")

        argv = shlex.split(exec_starts[0], posix=True)
        expected_executable = f"{working_directories[0]}/.venv/bin/gunicorn"
        self.assertEqual(argv[0], expected_executable)
        self.assertIn("gunicorn", EXPECTED_LOCK)

        wsgi_targets = [
            argument
            for argument in argv[1:]
            if re.fullmatch(r"[A-Za-z_]\w*:[A-Za-z_]\w*\(\)", argument)
        ]
        self.assertEqual(wsgi_targets, ["app:create_app()"])

        module_name, factory_call = wsgi_targets[0].split(":", maxsplit=1)
        factory_name = factory_call.removesuffix("()")
        module_path = CMD_ROOT / f"{module_name}.py"
        self.assertTrue(module_path.is_file())

        app_tree = ast.parse(_read_python(module_path), filename=str(module_path))
        top_level_functions = {
            node.name
            for node in app_tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.assertIn(factory_name, top_level_functions)
        self.assertIn(
            f"/usr/bin/test -r {working_directories[0]}/{module_name}.py",
            _unit_values(unit_text, "ExecStartPre"),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
