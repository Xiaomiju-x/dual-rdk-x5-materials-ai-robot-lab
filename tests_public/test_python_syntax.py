from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
import tokenize
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRECTORIES = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "venv",
    }
)
RECOVERED_SYNTAX_TARGETS = frozenset(
    {
        "ai_brain/dashboard/dashboard.py",
        "ai_brain/predict_engine/active_learning.py",
        "ai_brain/predict_engine/bpu_slot_manager.py",
        "ai_brain/predict_engine/campaign.py",
        "ai_brain/predict_engine/flybrain.py",
        "ai_brain/predict_engine/frontier_bpu.py",
        "safety/rb_voe/adapters/read_only.py",
        "safety/rb_voe/live_shadow.py",
        "web/command_center/cmdcenter/rb_voe_public.py",
        "web/command_center/tools/test_rb_voe_public_contract.py",
        "web/command_center/tools/test_site32_public_research_contract.py",
        "workstation/dual_arm/arm01_verify_clear.py",
    }
)


def repository_python_files() -> list[Path]:
    return [
        path
        for path in sorted(ROOT.rglob("*.py"))
        if EXCLUDED_DIRECTORIES.isdisjoint(path.relative_to(ROOT).parts)
    ]


class RepositoryPythonSyntaxTests(unittest.TestCase):
    def test_every_repository_python_source_compiles_without_execution(self) -> None:
        paths = repository_python_files()
        self.assertGreater(len(paths), 100, "Python source discovery unexpectedly found too few files")
        relative_paths = {path.relative_to(ROOT).as_posix() for path in paths}
        self.assertTrue(
            RECOVERED_SYNTAX_TARGETS.issubset(relative_paths),
            "a recovered sanitizer-damaged Python target left the compile inventory",
        )

        failures: list[str] = []
        for path in paths:
            relative = path.relative_to(ROOT).as_posix()
            try:
                with tokenize.open(path) as source_file:
                    source = source_file.read()
                ast.parse(source, filename=relative)
                compile(source, relative, "exec", dont_inherit=True)
            except (OSError, SyntaxError, UnicodeError) as exc:
                location = f":{exc.lineno}" if isinstance(exc, SyntaxError) and exc.lineno else ""
                failures.append(f"{relative}{location}: {exc.msg if isinstance(exc, SyntaxError) else exc}")

        self.assertFalse(failures, "Python syntax failures:\n" + "\n".join(failures))

    def test_read_only_security_modules_import_without_side_effect_backends(self) -> None:
        code = "\n".join(
            (
                "import sys",
                f"sys.path.insert(0, {str(ROOT)!r})",
                f"sys.path.insert(0, {str(ROOT / 'safety')!r})",
                f"sys.path.insert(0, {str(ROOT / 'web' / 'command_center')!r})",
                "from rb_voe.semantic_profiles import load_all_profiles",
                "profiles = load_all_profiles()",
                "assert tuple(profiles) == ('ai_x5.v1', 'embodied_x5.v1', 'dual_arm.v1', 'assay_station.v1')",
                "from rb_voe.adapters.read_only import _REMOTE_HOME_PREFIX",
                "from rb_voe.live_shadow import REMOTE_HOME_PREFIX, _REMOTE_SHA256_LINE_RE",
                "assert REMOTE_HOME_PREFIX == _REMOTE_HOME_PREFIX == '/' + 'home/'",
                "assert _REMOTE_SHA256_LINE_RE.fullmatch(('a' * 64) + '  /home/rdk/tools/probe.py\\n')",
                "assert not _REMOTE_SHA256_LINE_RE.fullmatch(('a' * 64) + '  /opt/probe.py\\n')",
                "import importlib.util",
                f"spec = importlib.util.spec_from_file_location('rb_voe_public_smoke', {str(ROOT / 'web' / 'command_center' / 'cmdcenter' / 'rb_voe_public.py')!r})",
                "assert spec is not None and spec.loader is not None",
                "module = importlib.util.module_from_spec(spec)",
                "spec.loader.exec_module(module)",
                "assert module._PRIVATE_HOME_PREFIX == '/' + 'home/'",
                "assert module._PRIVATE_DEPLOYMENT_RE.search('192' + '.168.1.2')",
            )
        )
        completed = subprocess.run(
            [sys.executable, "-I", "-B", "-c", code],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        self.assertEqual(
            0,
            completed.returncode,
            "read-only security module import failed: " + completed.stderr[-1000:],
        )

    def test_embodied_profile_and_collector_pins_are_current(self) -> None:
        profile_path = ROOT / "safety" / "rb_voe" / "semantic_profiles" / "embodied_x5.v1.json"
        collector_path = ROOT / "embodied_brain" / "tools" / "rb_voe_embodied_snapshot.py"
        config_path = ROOT / "safety" / "rb_voe" / "live_shadow_config.template.json"

        profile_digest = json.loads(profile_path.read_text(encoding="utf-8"))["profile_sha256"]
        collector_tree = ast.parse(collector_path.read_text(encoding="utf-8"))
        frozen_profile = next(
            node.value.value
            for node in collector_tree.body
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "FROZEN_PROFILE_SHA256"
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )
        self.assertEqual(profile_digest, frozen_profile)

        collector_digest = hashlib.sha256(collector_path.read_bytes()).hexdigest()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(
            collector_digest,
            config["expected"]["embodied_required_artifact_sha256"]["collector_script"],
        )


if __name__ == "__main__":
    unittest.main()
