"""PC fixture replay for the passive Embodied Cortex candidate."""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["run_passive_fixture"]


def run_passive_fixture(output_root: Path) -> dict[str, Any]:
    from .passive_replay import run_passive_fixture as _run

    return _run(output_root)
