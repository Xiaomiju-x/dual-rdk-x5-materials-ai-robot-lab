from __future__ import annotations

from embodied_brain.finals_cortex.tools.build_pc_acceptance import (
    CORTEX_ROOT,
    REPO_ROOT,
    REQUIRED_MODULES,
    _source_inventory,
    build_report,
)
from embodied_brain.finals_cortex.tools.package_candidate import (
    _files,
    _manifest_rows,
)


def test_acceptance_inventory_excludes_its_own_receipt() -> None:
    receipt = (
        CORTEX_ROOT / "evidence" / "pc_acceptance.v1.json"
    ).relative_to(REPO_ROOT).as_posix()
    paths = {row["path"] for row in _source_inventory()}
    assert receipt not in paths
    assert all(not path.endswith(".pyc") for path in paths)
    assert all("/releases/" not in path for path in paths)


def test_pc_foundation_acceptance_without_nested_test_run() -> None:
    report = build_report(run_tests=False)
    assert report["valid"] is True
    assert report["status"] == "PC_FOUNDATION_ACCEPTED_REAL_DATA_AND_BOARD_PENDING"
    assert set(report["modules"]) == set(REQUIRED_MODULES)
    assert all(report["modules"].values())
    assert report["quality_gates"]["board_not_contacted"] is True
    assert report["quality_gates"]["no_actual_bpu_claim"] is True
    assert report["quality_gates"]["no_real_accuracy_claim"] is True


def test_package_manifest_uses_portable_posix_order() -> None:
    rows = _manifest_rows(_files())
    paths = [row["path"] for row in rows]
    assert paths == sorted(paths)
    assert all("\\" not in path for path in paths)
