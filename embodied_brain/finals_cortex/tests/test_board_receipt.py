from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from embodied_brain.finals_cortex.tools.board_receipt import (
    DEFAULT_MANIFEST,
    DEFAULT_REPO_ROOT,
    build_command_plan,
    evaluate_receipt,
    main,
    verify_frozen_manifest,
)

FIXTURES = Path(__file__).resolve().parents[1] / "board" / "fixtures"


def _load(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _components(path: str) -> list[str | int]:
    return [int(value) if value.isdigit() else value for value in path.split(".")]


def _set_path(payload: Any, path: str, value: Any) -> None:
    components = _components(path)
    current = payload
    for component in components[:-1]:
        current = current[component]
    current[components[-1]] = value


def _delete_path(payload: Any, path: str) -> None:
    components = _components(path)
    current = payload
    for component in components[:-1]:
        current = current[component]
    del current[components[-1]]


def _apply_case(base: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(base)
    for path, value in case.get("set", {}).items():
        _set_path(payload, path, value)
    for path in case.get("delete", []):
        _delete_path(payload, path)
    return payload


def _gate(result: dict[str, Any], gate_id: str) -> dict[str, Any]:
    return next(gate for gate in result["gates"] if gate["id"] == gate_id)


def test_go_fixture_passes_every_hard_gate() -> None:
    result = evaluate_receipt(_load("go.json"))

    assert result["decision"] == "GO"
    assert result["monitor_state"] == "READY_SHADOW"
    assert result["failed_hard_gates"] == []
    assert all(gate["hard"] and gate["passed"] for gate in result["gates"])
    assert result["required_response"] == {
        "restart_frozen_services": False,
        "modify_network": False,
        "modify_frozen_files": False,
        "start_motion": False,
        "candidate_action": "manual_shadow_only",
    }


def test_hbm_runtime_is_supported_when_observed_and_actual() -> None:
    payload = _load("go.json")
    payload["compatibility"]["detected_runtimes"] = [
        {"name": "hobot_dnn", "version": None, "available": False},
        {"name": "hbm_runtime", "version": "1.2.8", "available": True},
    ]
    payload["compatibility"]["selected_runtime"] = "hbm_runtime"
    payload["execution"]["actual_backend"] = "hbm_runtime"

    result = evaluate_receipt(payload)

    assert result["decision"] == "GO"
    assert _gate(result, "compatibility.runtime")["passed"] is True


def test_declared_backend_must_match_observed_runtime() -> None:
    payload = _load("go.json")
    payload["execution"]["actual_backend"] = "hbm_runtime"

    result = evaluate_receipt(payload)

    assert result["decision"] == "NO_GO"
    assert result["monitor_state"] == "MONITOR_OFFLINE"
    assert "compatibility.runtime" in result["failed_hard_gates"]


CASES = _load("cases.json")["no_go"]


@pytest.mark.parametrize(
    "case",
    CASES,
    ids=[case["id"] for case in CASES],
)
def test_no_go_fixtures_fail_closed(case: dict[str, Any]) -> None:
    result = evaluate_receipt(_apply_case(_load("go.json"), case))

    assert result["decision"] == "NO_GO"
    assert result["monitor_state"] == "MONITOR_OFFLINE"
    assert case["expected_failed_gate"] in result["failed_hard_gates"]
    assert result["required_response"]["candidate_action"] == "leave_stopped"
    assert result["required_response"]["restart_frozen_services"] is False


def test_missing_field_does_not_raise_or_default_to_go() -> None:
    payload = _load("go.json")
    del payload["resources"]["during"]["cma_free_mib"]

    result = evaluate_receipt(payload)

    assert result["decision"] == "NO_GO"
    assert "schema.required_fields" in result["failed_hard_gates"]
    assert "resources.cma" in result["failed_hard_gates"]


def test_compiler_estimate_cannot_impersonate_board_measurement() -> None:
    payload = _load("go.json")
    payload["execution"]["compiler_estimate"] = True
    payload["execution"]["latency_ms"].update(
        {"source": "compiler_estimate", "p50": 0.1261, "p95": 0.1261, "p99": 0.1261}
    )

    result = evaluate_receipt(payload)

    assert result["decision"] == "NO_GO"
    assert _gate(result, "execution.actual_measurement")["passed"] is False
    assert _gate(result, "execution.latency")["passed"] is True


def test_all_resource_hard_limits_are_computed_not_trusted() -> None:
    payload = _load("go.json")
    payload["resources"]["during"].update(
        {
            "pss_mib": 621.0,
            "mem_available_mib": 2559.0,
            "bpu_ion_mib": 157.0,
            "cma_free_mib": 149.0,
        }
    )

    result = evaluate_receipt(payload)

    assert result["decision"] == "NO_GO"
    assert {
        "resources.pss",
        "resources.mem_available",
        "resources.bpu_ion",
        "resources.cma",
    }.issubset(set(result["failed_hard_gates"]))


def test_recovery_requires_30_clean_cycles_and_no_orphans() -> None:
    payload = _load("go.json")
    payload["load_recovery"]["cycles"] = payload["load_recovery"]["cycles"][:29]
    payload["load_recovery"]["orphan_processes"] = [1234]

    result = evaluate_receipt(payload)

    assert result["decision"] == "NO_GO"
    assert "resources.load_recovery_30x" in result["failed_hard_gates"]


def test_forbidden_ros_interfaces_fail_even_in_allowed_namespace() -> None:
    payload = _load("go.json")
    payload["ros_graph"]["publishers"].append(
        {
            "node": "/x5_finals_cortex/shadow",
            "topic": "/tf",
            "type": "tf2_msgs/msg/TFMessage",
        }
    )

    result = evaluate_receipt(payload)

    assert result["decision"] == "NO_GO"
    assert "ros.publishers" in result["failed_hard_gates"]


def test_frozen_manifest_is_rehashed_read_only() -> None:
    result = verify_frozen_manifest(DEFAULT_MANIFEST, DEFAULT_REPO_ROOT)

    assert result["available"] is True
    assert result["manifest_hash_match"] is True
    assert result["all_match"] is True
    assert result["declared_file_count"] == 12
    assert len(result["files"]) == 12


def test_command_generator_has_no_connection_or_mutation_commands() -> None:
    plan = build_command_plan(
        "/home/rdk/xrd_candidates/abc/model.bin",
        ["/x5_finals_cortex/shadow"],
        runtime="auto",
    )

    assert plan["execution_policy"]["execute_locally"] is False
    assert plan["execution_policy"]["opens_ssh"] is False
    assert plan["execution_policy"]["changes_network"] is False
    assert plan["execution_policy"]["restarts_services"] is False
    assert plan["execution_policy"]["starts_motion"] is False
    assert all(command["read_only"] for command in plan["commands"])

    shell = "\n".join(command["shell"] for command in plan["commands"]).lower()
    forbidden = (
        "systemctl restart",
        "systemctl stop",
        "systemctl start",
        "ros2 topic pub",
        "ros2 service call",
        "ros2 action send_goal",
        "ssh ",
        "scp ",
        "netsh ",
        "nmcli ",
        "ip route add",
        "ip route del",
    )
    assert not any(token in shell for token in forbidden)


def test_cli_returns_two_for_no_go_and_writes_decision(tmp_path: Path) -> None:
    payload = _load("go.json")
    payload["execution"]["compiler_estimate"] = True
    source = tmp_path / "input.json"
    output = tmp_path / "decision.json"
    source.write_text(json.dumps(payload), encoding="utf-8")

    exit_code = main(
        ["evaluate", "--input", str(source), "--output", str(output)]
    )
    result = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 2
    assert result["decision"] == "NO_GO"
    assert result["monitor_state"] == "MONITOR_OFFLINE"
