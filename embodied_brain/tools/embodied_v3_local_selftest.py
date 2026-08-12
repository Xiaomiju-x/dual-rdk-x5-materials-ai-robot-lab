#!/usr/bin/env python3
"""Local static and fake-data self-test for the embodied-brain v3 stack.

This script is intentionally host-safe:
- no SSH
- no ROS graph writes
- no serial/F407 access
- no vehicle motion

Use it before deploying to the car X5 to catch broken syntax, missing safety
sentinels, and data-loop/audit regressions.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


TOOL = Path(__file__).resolve()
EMBODIED_ROOT = TOOL.parents[1]
REPO_ROOT = TOOL.parents[2]


PYTHON_SYNTAX_FILES = [
    "embodied_brain/tools/embodied_v3_acceptance_audit.py",
    "embodied_brain/tools/data_loop_finalize.py",
    "embodied_brain/tools/data_loop_to_lerobot.py",
    "embodied_brain/tools/verify_cmd_vel_bag.py",
    "embodied_brain/tools/f407_protocol_contract_test.py",
    "embodied_brain/tools/f407_identity_gate_integration.py",
    "embodied_brain/tools/dispatch_stub_integration.py",
    "embodied_brain/tools/dispatch_fixture_integration.py",
    "embodied_brain/tools/physical_evidence_gate_integration.py",
    "embodied_brain/tools/physical_sensor_bridge_integration.py",
    "embodied_brain/tools/video2_overlay_recorder.py",
    "embodied_brain/tools/slam_wasd_mapper.py",
    "embodied_brain/tools/probe_lidar.py",
    "embodied_brain/tools/f407_link_test.py",
    "embodied_brain/tools/f407_postflash_bundle.py",
    "embodied_brain/tools/f407_postflash_report.py",
    "embodied_brain/tools/f407_postflash_recover_readonly.py",
    "embodied_brain/tools/capture_nav_location.py",
    "embodied_brain/tools/nav2_collision_fixture.py",
    "embodied_brain/ros2_ws/src/my_robot_agents/my_robot_agents/cockpit_bridge.py",
    "embodied_brain/ros2_ws/src/my_robot_agents/my_robot_agents/mppi_node.py",
    "embodied_brain/ros2_ws/src/my_robot_agents/my_robot_agents/dispatch_server.py",
    "embodied_brain/ros2_ws/src/my_robot_agents/my_robot_agents/physical_evidence_contracts.py",
    "embodied_brain/ros2_ws/src/my_robot_agents/my_robot_agents/physical_evidence_gate.py",
    "embodied_brain/ros2_ws/src/my_robot_agents/my_robot_agents/physical_sensor_contracts.py",
    "embodied_brain/ros2_ws/src/my_robot_agents/my_robot_agents/physical_sensor_evidence_bridge.py",
    "embodied_brain/ros2_ws/src/my_robot_agents/launch/physical_sensor_bridge.launch.py",
    "embodied_brain/ros2_ws/src/my_robot_agents/setup.py",
    "embodied_brain/ros2_ws/src/my_robot_agents/my_robot_agents/safety_contracts.py",
    "embodied_brain/ros2_ws/src/my_robot_bridge/my_robot_bridge/ai_brain_bridge.py",
    "embodied_brain/ros2_ws/src/my_robot_navigation/scripts/lab_fsd_core.py",
    "embodied_brain/ros2_ws/src/my_robot_navigation/scripts/bev_shadow_planner.py",
    "embodied_brain/ros2_ws/src/my_robot_navigation/scripts/lab_anomaly_bpu.py",
    "embodied_brain/ros2_ws/src/my_robot_navigation/scripts/vision_bev_bridge.py",
    "embodied_brain/ros2_ws/src/my_robot_navigation/launch/nav2.launch.py",
    "embodied_brain/ros2_ws/src/my_robot_navigation/launch/full_nav.launch.py",
    "embodied_brain/ros2_ws/src/my_robot_navigation/launch/localization.launch.py",
    "embodied_brain/ros2_ws/src/my_robot_drivers/launch/serial_f407.launch.py",
    "embodied_brain/ros2_ws/src/my_robot_bridge/launch/bridge.launch.py",
    "embodied_brain/ros2_ws/src/my_robot_bringup/launch/full.launch.py",
    "embodied_brain/ros2_ws/src/my_robot_dashboard/backend/bridge_state.py",
    "embodied_brain/ros2_ws/src/my_robot_dashboard/backend/mission.py",
    "embodied_brain/ros2_ws/src/my_robot_dashboard/backend/api/bridge.py",
    "embodied_brain/ros2_ws/src/my_robot_dashboard/backend/api/ops.py",
    "tests/test_embodied_data_loop_tools.py",
    "tests/test_embodied_physical_evidence.py",
    "tests/test_embodied_physical_sensor_bridge.py",
    "tests/test_embodied_navigation_commissioning.py",
    "tests/test_embodied_postflash_orchestrator.py",
    "tests/test_embodied_safety_contracts.py",
    "embodied_brain/ros2_ws/src/my_robot_agents/test/test_cockpit_bridge_pickup.py",
]


SHELL_SYNTAX_FILES = [
    "embodied_brain/deploy_to_car.sh",
    "embodied_brain/tools/embodied_v3_acceptance_check.sh",
    "embodied_brain/tools/data_loop_start.sh",
    "embodied_brain/tools/data_loop_stop.sh",
    "embodied_brain/tools/cmd_vel_bag_smoke_test.sh",
    "embodied_brain/tools/data_loop_status.sh",
    "embodied_brain/tools/start_embodied_v3_stack.sh",
    "embodied_brain/tools/embodied_v3_runtime_prepare_evidence.sh",
    "embodied_brain/tools/embodied_v3_runtime_restore.sh",
    "embodied_brain/tools/f407_postflash_interlock_acceptance.sh",
    "embodied_brain/tools/f407_postflash_recover_readonly.sh",
    "embodied_brain/tools/start_lab_fsd_shadow.sh",
    "embodied_brain/tools/x5_runtime_preflight.sh",
    "embodied_brain/tools/embodied_hardware_commissioning_preflight.sh",
    "embodied_brain/tools/video2_prepare_demo.sh",
    "embodied_brain/tools/video2_start_capture.sh",
    "embodied_brain/tools/video2_stop_capture.sh",
]


SENTINELS: dict[str, list[str]] = {
    "embodied_brain/ros2_ws/src/my_robot_navigation/scripts/lab_fsd_core.py": [
        "def fuse_policy_with_bpu_prior",
        "tiny_waypoint_policy_prior",
        "cmd_vel_authority",
        "shadow_only",
        "classify_vision_bev_provenance",
        "apply_vision_bev_provenance",
    ],
    "embodied_brain/ros2_ws/src/my_robot_navigation/scripts/bev_shadow_planner.py": [
        "fuse_policy_with_bpu_prior",
        "policy_prior",
        "/lab_fsd/policy_tokens",
        "/lab_fsd/vision_objects",
        "cmd_vel_authority",
        "vision_objects_topic",
        "apply_vision_bev_provenance",
    ],
    "embodied_brain/ros2_ws/src/my_robot_navigation/config/lab_fsd_shadow.yaml": [
        "use_occ_risk_bpu: true",
        "occ_risk_model_bin",
        "tiny_waypoint_policy_prior",
    ],
    "embodied_brain/ros2_ws/src/my_robot_navigation/config/collision_monitor.yaml": [
        "cmd_vel_in_topic: \"/cmd_vel\"",
        "cmd_vel_out_topic: \"/cmd_vel_safe\"",
        "FrontStop",
        "FrontSlow",
        "/scan_depth",
    ],
    "embodied_brain/ros2_ws/src/my_robot_navigation/launch/nav2.launch.py": [
        "use_collision_monitor",
        "nav2_collision_monitor",
        "controller_server -> /cmd_vel_nav -> velocity_smoother -> /cmd_vel",
        "lifecycle_manager_collision_monitor",
    ],
    "embodied_brain/ros2_ws/src/my_robot_navigation/launch/full_nav.launch.py": [
        "localization.launch.py",
        "UnlessCondition",
        "LaunchConfiguration('map')",
        "use_collision_monitor",
        "use_lab_fsd_shadow",
    ],
    "embodied_brain/ros2_ws/src/my_robot_navigation/launch/localization.launch.py": [
        "localization_launch.py",
        "DeclareLaunchArgument",
        '"map"',
        '"params_file"',
        '"autostart"',
        '"use_composition"',
        "map must be an absolute YAML path",
        "map image does not exist",
    ],
    "embodied_brain/ros2_ws/src/my_robot_bringup/launch/full.launch.py": [
        "use_collision_monitor",
        "EB_USE_COLLISION_MONITOR",
        "EB_USE_LAB_FSD_SHADOW', 'true'",
        "pt_camera_source_available",
        "PythonExpression",
        "use_physical_evidence_gate",
        "physical_evidence_mode",
        "localization.launch.py",
        "EB_MAP_YAML",
    ],
    "embodied_brain/ros2_ws/src/my_robot_bridge/launch/bridge.launch.py": [
        "execute_pickup_actuators",
        "EB_EXECUTE_PICKUP_ACTUATORS",
        "ParameterValue",
        "physical_evidence_gate",
        "physical_evidence_mode",
    ],
    "embodied_brain/ros2_ws/src/my_robot_msgs/action/DispatchTask.action": [
        "string completion_class",
        "bool actuator_sequence_completed",
        "bool physical_completed",
        "bool base_motion_requested",
        "string physical_confirmation",
    ],
    "embodied_brain/deploy_to_car.sh": [
        "lab_fsd_tiny_occ_risk.bin",
        "lab_anomaly_autoencoder.bin",
        "lab_fsd_bins",
        "--parallel-workers",
        "sudo -n systemctl restart embodied_brain.service",
        "EXPECTED_CAR_HOSTNAME",
        "cost_mlp.bin",
        "CAR_MPPI_MODEL",
        "F407_POSTFLASH_MANIFEST",
        "EMBODIED_V3_REQUIRE_POSTFLASH_INTERLOCK",
        "unsafe F407_POSTFLASH_MANIFEST path",
        "postflash_interlock(_recovered)?_manifest",
    ],
    "embodied_brain/ros2_ws/src/my_robot_drivers/src/serial_f407_node.cpp": [
        "create_service<my_robot_msgs::srv::SetLiftHeight>",
        "on_set_lift_height",
        "send_lift_target_height",
        "wait_for_lift_arrival",
        "lift_arrival_tolerance_m",
        "requested_lift_target_m",
        "last_lift_height_m",
        "actuator_commands_blocked_by_estop",
        "DownType::CLEAR_ESTOP",
        "hardware_estop_latched",
        "f407_estop_blocked_commands",
        "firmware_identity_valid",
        "TARGET_FIRMWARE_BUILD_ID",
        "cmd_vel_blocked_by_firmware_identity",
        "service_callback_group_",
        "timer_callback_group_",
        "MultiThreadedExecutor",
    ],
    "embodied_brain/stm32_f407/App/proto.h": [
        "UP_SAFETY_STATE",
        "DN_CLEAR_ESTOP",
        "estop_latched",
        "estop_blocked_command_count",
        "PROTO_ACK_LINK_STALE",
        "UP_FIRMWARE_INFO",
        "PROTO_FIRMWARE_BUILD_ID",
        "PROTO_CAPABILITIES",
    ],
    "embodied_brain/stm32_f407/App/proto.c": [
        "PROTO_ACK_ESTOP_LATCHED",
        "case DN_CLEAR_ESTOP",
        "proto_send_safety_state",
        "motion_interlock_status",
        "proto_send_firmware_info",
    ],
    "embodied_brain/stm32_f407/App/main.c": [
        "ps->estop_latched",
        "proto_send_safety_state",
        "proto_send_firmware_info",
    ],
    "embodied_brain/ros2_ws/src/my_robot_drivers/launch/serial_f407.launch.py": [
        "lift_arrival_tolerance_m",
        "lift_arrival_default_timeout_s",
        "SetLiftHeight",
        "require_firmware_identity",
        "firmware_identity_stale_s",
    ],
    "embodied_brain/ros2_ws/src/my_robot_agents/my_robot_agents/mppi_node.py": [
        "def _is_cmd_vel_topic",
        "Unsafe MPPI proposed cmd_vel_topic",
        "cmd_topic = '/mppi/cmd_vel_proposed'",
        "'proposed_only'",
        "'proposed_topic'",
        "self.pub_cmd.publish(cmd)",
    ],
    "embodied_brain/ros2_ws/src/my_robot_agents/my_robot_agents/cockpit_bridge.py": [
        "def _cmd_pickup_flow",
        "pickup_flow_rejected",
        "pickup_flow_stage",
        "send_goal_async(goal, feedback_callback=_feedback_cb)",
        "'pickup_flow'",
        "self._bb_event('pickup_flow', snapshot)",
        "physical_completed",
        "actuator_sequence_completed",
        "base_motion_requested",
        "physical_confirmation",
        "getattr(res, 'completion_class'",
        "F407_REPORTED_COMPLETED:",
        "SIMULATED_ONLY:",
        "self._clear_estop_cli.call_async",
        "Cockpit 急停保持锁存",
        "/f407/firmware_identity_valid",
        "/lab_fsd/fsd_v3_status",
    ],
    "embodied_brain/ros2_ws/src/my_robot_agents/my_robot_agents/dispatch_server.py": [
        "lab_fsd_require_fresh_for_nav",
        "lab_fsd_guard_reasons",
        "execute_pickup_actuators",
        "allow_stationary_pickup_fixture",
        "stationary_pickup_fixture_only",
        "stationary_pickup_fixture_one_shot",
        "pickup_fixture_stationary",
        "/f407/firmware_identity_valid",
        "stationary_fixture=true, dispatch_issued_base_motion=false",
        "result.completion_class",
        "result.actuator_sequence_completed",
        "result.physical_completed",
        "result.base_motion_requested",
        "result.physical_confirmation",
        "physical_evidence_mode",
        "VerifyPhysicalEvidence",
        "PHYSICAL_EVIDENCE_OK",
        "PHYSICAL_COMPLETED:",
        "F407_SERVICE_OK SET_LIFT_HEIGHT",
        "F407_SERVICE_OK SET_ELECTROMAGNET",
        "F407_REPORTED_COMPLETED:",
        "physical_completed=false",
        "SIMULATED_ONLY:",
        "rejecting concurrent dispatch goal",
        "升降台下降到放置高度",
        "升降台回运输安全高度",
    ],
    "embodied_brain/ros2_ws/src/my_robot_agents/my_robot_agents/physical_evidence_contracts.py": [
        "xrd-pickup-physical-confirmation-v1",
        "canonical_evidence_sha256",
        "validate_evidence",
        "payload_sha256 mismatch",
        "evidence_id replayed",
        "independent_object_evidence",
    ],
    "embodied_brain/ros2_ws/src/my_robot_agents/my_robot_agents/physical_evidence_gate.py": [
        "VerifyPhysicalEvidence",
        "/pickup/physical_evidence",
        "/pickup/physical_evidence_request",
        "request_id replayed or already in flight",
        "does not create evidence",
        "MultiThreadedExecutor",
    ],
    "embodied_brain/ros2_ws/src/my_robot_agents/my_robot_agents/physical_sensor_contracts.py": [
        "xrd-physical-sensor-calibration-v1",
        "canonical_sample_sha256",
        "production-authorized",
        "sample sequence is not strictly increasing",
        "sample_sha256 mismatch",
        "evaluate_sample",
    ],
    "embodied_brain/ros2_ws/src/my_robot_agents/my_robot_agents/physical_sensor_evidence_bridge.py": [
        "self.declare_parameter(\"enabled\", False)",
        "expected_calibration_sha256",
        "expected_driver_instance_id",
        "expected_publisher_node",
        "require_unique_publisher",
        "sample was observed before request receipt",
        "canonical_evidence_sha256",
        "commands_published\": False",
    ],
    "embodied_brain/ros2_ws/src/my_robot_agents/launch/physical_sensor_bridge.launch.py": [
        "default_value=\"false\"",
        "condition=IfCondition(enabled)",
        "allow_unapproved_calibration",
        "expected_calibration_sha256",
        "expected_driver_instance_id",
        "require_unique_publisher",
    ],
    "embodied_brain/ros2_ws/src/my_robot_agents/setup.py": [
        "physical_evidence_gate = my_robot_agents.physical_evidence_gate:main",
        "physical_sensor_evidence_bridge = my_robot_agents.physical_sensor_evidence_bridge:main",
    ],
    "embodied_brain/ros2_ws/src/my_robot_msgs/CMakeLists.txt": [
        "msg/HardwareSensorSample.msg",
        "msg/PhysicalEvidenceRequest.msg",
        "msg/PhysicalEvidence.msg",
        "srv/VerifyPhysicalEvidence.srv",
    ],
    "embodied_brain/ros2_ws/src/my_robot_msgs/msg/PhysicalEvidence.msg": [
        "string evidence_id",
        "string request_id",
        "bool hardware_observed",
        "string payload_sha256",
    ],
    "embodied_brain/ros2_ws/src/my_robot_msgs/msg/HardwareSensorSample.msg": [
        "string driver_instance_id",
        "uint64 sequence",
        "bool hardware_observed",
        "string sample_sha256",
    ],
    "embodied_brain/ros2_ws/src/my_robot_msgs/srv/VerifyPhysicalEvidence.srv": [
        "my_robot_msgs/PhysicalEvidenceRequest request",
        "my_robot_msgs/PhysicalEvidence evidence",
    ],
    "embodied_brain/ros2_ws/src/my_robot_agents/my_robot_agents/safety_contracts.py": [
        "(\"safety_gate\", safety_gate_time)",
        "(\"future_risk\", future_risk_time)",
        "(\"input_status\", input_status_time)",
        "reasons.append(f\"{name}_missing\")",
        "input_status={overall}",
        "safety_gate_hard_reasons=",
    ],
    "embodied_brain/ros2_ws/src/my_robot_bridge/my_robot_bridge/ai_brain_bridge.py": [
        "completion_class",
        "actuator_sequence_completed",
        "physical_completed",
        "base_motion_requested",
        "physical_confirmation",
    ],
    "embodied_brain/ros2_ws/src/my_robot_dashboard/backend/api/ops.py": [
        "@router.post('/pickup_flow')",
        "send_command('pickup_flow'",
    ],
    "embodied_brain/ros2_ws/src/my_robot_dashboard/backend/api/bridge.py": [
        "'pickup_flow'",
        "'motion_busy'",
        "'unavailable': True",
    ],
    "embodied_brain/ros2_ws/src/my_robot_dashboard/backend/bridge_state.py": [
        "fixture_only",
        "live_partial",
        "fixture_fields",
        "actuation fail-closed",
        "http://198.51.100.103:8888",
    ],
    "embodied_brain/ros2_ws/src/my_robot_dashboard/backend/mission.py": [
        "'pickup_flow'",
        "140",
        "急停保持锁存",
    ],
    "embodied_brain/tools/embodied_v3_acceptance_check.sh": [
        "/lab_fsd/bev",
        "/lab_fsd/future_bev",
        "/mppi/cmd_vel_proposed",
        "/lab_fsd/policy_tokens",
        "/lab_fsd/vision_objects",
        "/lab_fsd/vision_bev",
        "/lab_fsd/vision_risk",
        "/lift_status",
        "TOPIC_ECHO_ATTEMPTS",
        "capture_topic_once",
        "--full-length",
        "lab_anomaly_autoencoder_bin",
        "cockpit_blackbox_recent",
        "mppi_not_cmd_vel_publisher",
        "f407_interlock_report",
        "/f407/firmware_identity_valid",
        "/f407/firmware_info",
        "set +u",
        "set -u",
        "mppi_cost_bin",
        "runtime_mppi_raw_evidence",
        "dispatch_fixture_integration.py",
        "dispatch_fixture_integration",
        "acceptance_start.json",
        "dispatch_fixture_freshness",
        "MPPI node absent; runtime/raw evidence is audited separately",
        "F407_POSTFLASH_MANIFEST",
        "EMBODIED_V3_REQUIRE_POSTFLASH_INTERLOCK",
        "f407_postflash_bundle.py",
        "f407_postflash_bundle_index.json",
        "f407_postflash_manifest",
        "physical_evidence_config",
        "/verify_physical_evidence",
        "hardware_sensor_sample",
        "physical_sensor_evidence_bridge",
    ],
    "embodied_brain/tools/start_embodied_v3_stack.sh": [
        "EMBODIED_V3_SETTLE_S",
        "--settle-s",
        "use_yolo_world:=false",
        "use_edgesam:=false",
        "use_xfeat:=false",
        "use_bottle_ocr:=false",
        "/lab_fsd/policy_tokens",
        "/lift_status",
        "/f407/firmware_identity_valid",
        "/f407/firmware_info",
        "use_mppi:=\"$USE_MPPI_PROPOSED\"",
        "mppi_publish_direct_cmd_vel:=false",
    ],
    "embodied_brain/tools/start_lab_fsd_shadow.sh": [
        "LAB_FSD_SHADOW_ALREADY_MANAGED",
        "owner=embodied_brain.service",
        "standalone duplication is refused",
        "EB_USE_LAB_FSD_SHADOW=true",
    ],
    "embodied_brain/tools/embodied_v3_runtime_prepare_evidence.sh": [
        "--verify-estop-interlock",
        "--require-ack",
        "ros2 topic pub --rate 5 /cmd_vel geometry_msgs/msg/Twist '{}'",
        "--cmd-vel-expect zero",
        "--no-video2-copy",
        "xrd-embodied-v3-runtime-prepare-v4",
        "data_manifest_sha256",
        "interlock_report_sha256",
        "postflash_manifest_reuse",
        "f407_postflash_bundle.py",
        "runtime_interlock_revalidation.json",
        "RUNTIME_PREP_FAILURE_OWNER_RESTORED",
        "nonzero_cmd_vel_published\": False",
        "f407_estop_left_latched\": True",
        "data_loop_stop.sh",
        "shadow_plus_mppi_proposed_only",
        "MPPI_STATS_SHA256",
    ],
    "embodied_brain/tools/embodied_v3_runtime_restore.sh": [
        "start_embodied_v3_stack.sh\" stop",
        "managed_service_stopped",
        "systemctl start embodied_brain.service",
        "never clears F407 estop",
    ],
    "embodied_brain/tools/x5_runtime_preflight.sh": [
        "nav2_collision_monitor",
        "/dev/F407",
        "lab_fsd_tiny_occ_risk",
        "hobot_dnn import",
        "embodied_brain.service",
        "set +u",
        "set -u",
        "mppi_cost",
        "rosbag2_storage_mcap",
    ],
    "embodied_brain/tools/nav2_collision_fixture.py": [
        "ROS_LOCALHOST_ONLY=1 is required",
        "ROS_DOMAIN_ID must be >= 200",
        "--fixture-only",
        "scan.header.stamp = self.get_clock().now().to_msg()",
        "it never publishes velocity",
    ],
    "embodied_brain/tools/embodied_hardware_commissioning_preflight.sh": [
        "Read-only guarantee",
        "no ROS publish/service call",
        "expected embodied-x5",
        "/f407/firmware_identity_valid",
        "/f407/estop_latched",
        "/cmd_vel_safe",
        "TF odom -> base_footprint",
        "summary: FAIL",
    ],
    "embodied_brain/tools/embodied_v3_acceptance_audit.py": [
        "topic_lab_fsd_bev",
        "topic_lab_fsd_future_bev",
        "fsd_v3_status_tiny_occ_risk_content",
        "fsd_v3_status_shadow_only_content",
        "policy_tokens_prior_content",
        "policy_tokens_prior_json_content",
        "load_json_objects_from_ros_text",
        "data_model_artifact",
        "data_rosbag_info_readable",
        "data_bag_files_nonempty",
        "data_cmd_vel_semantic_evidence",
        "data_cmd_vel_source_binding",
        "data_ledger_chain",
        "training_source_manifest_binding",
        "training_quality_gate",
        "cockpit_pickup_blackbox_content",
        "cockpit_pickup_terminal_truth",
        "cockpit_pickup_physical_evidence",
        "cockpit_pickup_actuator_report_evidence",
        "f407_interlock_report_content",
        "f407_firmware_identity_topic_content",
        "f407_firmware_identity_valid_content",
        "identity_enforcement_enabled",
        "vision_bev_provenance_truthful",
        "lab_fsd_geometry_input_truth",
        "topic_lab_fsd_vision_objects",
        "runtime_prepare_report_content",
        "mppi_stats_proposed_only_content",
        "mppi_proposed_estop_zero_content",
        "dispatch_fixture_integration_content",
        "xrd-dispatch-fixture-integration-v2",
        "xrd-embodied-v3-audit-v4",
        "xrd-embodied-v3-acceptance-policy-v4",
        "validate_physical_confirmation",
        "physical_evidence_default_safe",
        "sensor_bridge_present",
        "raw_sample_topic_present",
        "dispatch_fixture_check_contract",
        "dispatch_fixture_monitor_evidence",
        "f407_postflash_orchestration_content",
        "audit_postflash_orchestration",
        "xrd-embodied-v3-runtime-prepare-v4",
        "postflash_manifest_reuse",
        "xrd-f407-postflash-interlock-orchestration-v2",
        "f407_postflash_recovered_source_chain",
        "f407_postflash_readonly_recovery_contract",
    ],
    "embodied_brain/tools/data_loop_finalize.py": [
        "DEFAULT_MODEL_ARTIFACTS",
        "scan_model_artifacts",
        "model_artifacts",
        "append_ledger_entry",
        "preserve_existing_terminal_manifest",
        "SKIP_HASH_PREFIXES",
        "scan_cmd_vel_evidence",
        "mppi_cost",
        "/pickup/hardware_sensor_sample",
    ],
    "embodied_brain/tools/data_loop_start.sh": [
        "/lab_fsd/bev",
        "/lab_fsd/future_bev",
        "/lab_fsd/policy_tokens",
        "/lab_fsd/vision_objects",
        "/lab_fsd/vision_bev",
        "/lab_fsd/vision_risk",
        "/lift_status",
        "REQUIRED_TOPIC_GATE_FILE",
        "^/mppi(/|$)",
        "hardware_sensor_sample",
        "physical_evidence_bridge_status",
        "/f407/cmd_vel_expired",
        "/f407/firmware_identity_valid",
        "/f407/firmware_info",
        "CMD_VEL_EXPECTATION",
    ],
    "embodied_brain/tools/data_loop_stop.sh": [
        "ros2 bag info",
        "rosbag_info.txt",
        "--status stopped",
        "verify_cmd_vel_bag.py",
        "cmd_vel_evidence.json",
    ],
    "embodied_brain/tools/data_loop_to_lerobot.py": [
        "verify_manifest_sha256",
        "source_manifest_integrity",
        "skeleton_manifest.sha256",
        "build_quality_gate",
        "conversion_status.json",
        "/f407/firmware_identity_valid",
        "/f407/firmware_info",
        "/lab_fsd/vision_objects",
        "/lab_fsd/vision_bev",
        "/lab_fsd/vision_risk",
        "/pickup/hardware_sensor_sample",
        "cmd_vel_semantic_evidence",
    ],
    "embodied_brain/tools/verify_cmd_vel_bag.py": [
        "xrd-cmd-vel-bag-evidence-v1",
        "rosbag2_py.SequentialReader",
        "geometry_msgs/msg/Twist",
        "nonzero_count",
        "--expect",
    ],
    "embodied_brain/tools/cmd_vel_bag_smoke_test.sh": [
        "ROS_LOCALHOST_ONLY=1",
        "ROS2CLI_NO_DAEMON=1",
        "geometry_msgs/msg/Twist '{}'",
        "--expect zero",
        "no bag playback",
    ],
    "embodied_brain/tools/f407_link_test.py": [
        "EXPECTED_BUILD_ID = 2026071907",
        "FIRMWARE_INFO missing",
        "identity_verified_before_commands",
        "xrd-f407-interlock-evidence-v2",
    ],
    "embodied_brain/tools/f407_postflash_report.py": [
        "EXPECTED_FIRMWARE",
        "EXPECTED_ACKS",
        "clear_estop_forbidden",
        "firmware_identity_exact",
        "estop_left_latched",
        "ack_set_exact",
        "xrd-f407-postflash-interlock-validation-v1",
    ],
    "embodied_brain/tools/f407_postflash_bundle.py": [
        "xrd-f407-postflash-bundle-index-v1",
        "xrd-f407-postflash-interlock-orchestration-v1",
        "xrd-f407-postflash-interlock-orchestration-v2",
        "xrd-f407-postflash-readonly-recovery-v1",
        "f407_postflash_acceptance",
        "postflash_interlock_manifest.json",
        "postflash_interlock_recovered_manifest.json",
        "physical tool snapshot",
        "read-only recovery contract failed",
        "must not be a symlink",
        "path is not canonical",
        "physical_hardware_touched",
    ],
    "embodied_brain/tools/f407_postflash_interlock_acceptance.sh": [
        "NO_LOAD_PATH_CLEAR_BASE_FIXED_HANDS_CLEAR_OPERATOR_PRESENT",
        "--acknowledge-magnet-off-can-drop-load",
        "EXPECTED_CAR_HOSTNAME:-embodied-x5",
        "restore_original_services",
        "trap cleanup EXIT",
        "--verify-estop-interlock",
        "--require-ack",
        "SERIAL_OWNERS_AFTER_STOP",
        "refusing an unowned maintenance state",
        "post_f407_estop_latched.txt",
        "xrd-f407-postflash-interlock-orchestration-v1",
        "physical_completion_claimed\": False",
        "F407_POSTFLASH_INTERLOCK_ACCEPTANCE_PASS",
    ],
    "embodied_brain/tools/f407_postflash_recover_readonly.py": [
        "xrd-f407-postflash-interlock-orchestration-v2",
        "xrd-f407-postflash-readonly-recovery-v1",
        "post_restore_readonly_only",
        "serial_device_opened",
        "ros_graph_writes",
        "physical_commands_sent",
        "source_failed_manifest.json",
        "recovery_interlock_revalidation.json",
    ],
    "embodied_brain/tools/f407_postflash_recover_readonly.sh": [
        "set +u",
        "set -u",
        "f407_postflash_recover_readonly.py",
    ],
    "embodied_brain/tools/f407_identity_gate_integration.py": [
        "xrd-f407-identity-gate-pty-integration-v1",
        "real_hardware_touched",
        "missing_identity_blocks_nonzero_cmd_vel",
        "mismatched_build_blocks_nonzero_cmd_vel",
        "stale_identity_blocks_clear_estop",
        "physical_runtime_audit_still_required",
    ],
    "embodied_brain/tools/dispatch_stub_integration.py": [
        "xrd-dispatch-stub-integration-v1",
        "EXPECTED_FETCH_STAGES",
        "SIMULATED_ONLY:",
        "stub_fetch_no_f407_service_calls",
        "stub_fetch_no_cmd_vel_messages",
        "stub_goal_rejected_by_f407_estop",
        "stub_goal_ignores_navigation_only_lab_fsd_guard",
        "dispatch_global_single_task_mutex",
        "real_hardware_touched",
        "physical_runtime_audit_still_required",
    ],
    "embodied_brain/tools/dispatch_fixture_integration.py": [
        "xrd-dispatch-fixture-integration-v2",
        "procfs-process-tree-fd-monitor-v1",
        "ROS_LOCALHOST_ONLY",
        "pickup_fixture_stationary",
        "invalid_firmware_identity_rejected",
        "estop_latched_rejected",
        "stationary_fixture_one_shot_rejected",
        "stationary_no_cmd_vel_messages",
        "dispatch_monitor_coverage_complete",
        "dispatch_process_clean_exit",
        "real_hardware_touched",
        "physical_completed",
    ],
    "embodied_brain/tools/physical_evidence_gate_integration.py": [
        "xrd-physical-evidence-gate-integration-v1",
        "simulation_only",
        "real_hardware_touched",
        "tampered_hash_rejected",
        "wrong_source_rejected",
        "evidence_replay_rejected",
        "no_physical_device_fd",
        "no_cmd_vel_publisher",
    ],
    "embodied_brain/tools/physical_sensor_bridge_integration.py": [
        "xrd-physical-sensor-bridge-integration-v1",
        "simulation_only",
        "production_authorized",
        "pre_request_sample_rejected",
        "tampered_sample_hash_rejected",
        "multiple_sample_publishers_rejected",
        "no_physical_device_fd",
        "no_cmd_vel_publisher",
    ],
    "embodied_brain/tools/capture_nav_location.py": [
        "buffer.lookup_transform",
        "motion_command_published\": False",
        "tf2_lookup_read_only",
    ],
}


@dataclass
class Result:
    name: str
    status: str
    detail: str = ""


def rel(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def add(results: list[Result], name: str, status: str, detail: str = "") -> None:
    results.append(Result(name=name, status=status, detail=detail))
    suffix = f" - {detail}" if detail else ""
    print(f"[{status}] {name}{suffix}")


def path_for(relative: str) -> Path:
    return REPO_ROOT / relative


def check_python_syntax(results: list[Result]) -> None:
    for relative in PYTHON_SYNTAX_FILES:
        path = path_for(relative)
        name = f"python_syntax:{relative}"
        if not path.exists():
            add(results, name, "FAIL", "missing")
            continue
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
            compile(source, str(path), "exec", ast.PyCF_ONLY_AST)
        except Exception as exc:  # pragma: no cover - printed for operators
            add(results, name, "FAIL", f"{type(exc).__name__}: {exc}")
        else:
            add(results, name, "PASS")


def find_bash() -> str | None:
    if os.name == "nt":
        # Windows may expose a WSL app-execution alias as system32/bash.exe even
        # when no WSL distribution exists. Prefer a real Git Bash installation.
        candidates = [
            Path(os.environ.get("GIT_BASH", "")) if os.environ.get("GIT_BASH") else None,
            Path(r"C:\Program Files\Git\bin\bash.exe"),
            Path(r"C:\Program Files\Git\usr\bin\bash.exe"),
        ]
        for candidate in candidates:
            if candidate is not None and candidate.is_file():
                return str(candidate)
    found = shutil.which("bash")
    if found:
        return found
    return None


def run_cmd(cmd: list[str], timeout_s: int = 60) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_s,
    )
    return proc.returncode, proc.stdout


def check_shell_syntax(results: list[Result]) -> None:
    bash = find_bash()
    if not bash:
        add(results, "bash_available", "FAIL", "bash not found")
        return
    add(results, "bash_available", "PASS", bash)
    for relative in SHELL_SYNTAX_FILES:
        path = path_for(relative)
        name = f"shell_syntax:{relative}"
        if not path.exists():
            add(results, name, "FAIL", "missing")
            continue
        try:
            rc, out = run_cmd([bash, "-n", str(path)], timeout_s=30)
        except Exception as exc:
            add(results, name, "FAIL", f"{type(exc).__name__}: {exc}")
            continue
        if rc == 0:
            add(results, name, "PASS")
        else:
            add(results, name, "FAIL", out.strip()[-500:])


def check_sentinels(results: list[Result]) -> None:
    for relative, needles in SENTINELS.items():
        path = path_for(relative)
        if not path.exists():
            add(results, f"sentinel:{relative}", "FAIL", "missing")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        missing = [needle for needle in needles if needle not in text]
        if missing:
            add(results, f"sentinel:{relative}", "FAIL", "missing: " + ", ".join(missing))
        else:
            add(results, f"sentinel:{relative}", "PASS", f"{len(needles)} tokens")

    mission_path = path_for("embodied_brain/ros2_ws/src/my_robot_dashboard/backend/mission.py")
    mission_text = mission_path.read_text(encoding="utf-8", errors="replace") if mission_path.exists() else ""
    abort_start = mission_text.find("    async def abort(self)")
    abort_end = mission_text.find("    def _log(", abort_start + 1) if abort_start >= 0 else -1
    abort_body = mission_text[abort_start:abort_end] if abort_start >= 0 and abort_end > abort_start else ""
    if abort_body and "send_command('estop'" in abort_body and "clear_estop" not in abort_body:
        add(results, "mission_abort_estop_latches", "PASS", "abort asserts estop without automatic clear")
    else:
        add(results, "mission_abort_estop_latches", "FAIL", "mission abort must not auto-clear estop")


def check_model_bin(results: list[Result]) -> None:
    import hashlib

    specs = [
        (
            "lab_fsd_tiny_occ_risk",
            "models/bpu_compiled/lab_fsd_tiny_occ_risk/model_output/lab_fsd_tiny_occ_risk.bin",
            395641,
            "3b1a96483351f72746fdcacfb179b69f4527076046e5dd73d5bcae7688d99c90",
        ),
        (
            "lab_anomaly_autoencoder",
            "models/bpu_compiled/lab_anomaly_autoencoder/model_output/lab_anomaly_autoencoder.bin",
            2492577,
            "1045be38ff947ad3c97c365416170970f59735504a1f38663bd8cce8d112ad7f",
        ),
        (
            "mppi_cost",
            "research/round_4/c2_mppi/bpu_compile/output_cost_mlp/cost_mlp.bin",
            264036,
            "fe54f08d12285cf66c37ee7168b51a6762bb086b30a681a12f18374d8eea853d",
        ),
    ]
    for model_name, relative, expected_size, expected_sha in specs:
        path = path_for(relative)
        if not path.exists():
            add(results, f"model_bin:{model_name}", "FAIL", "missing")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        size = path.stat().st_size
        if size == expected_size and digest == expected_sha:
            add(results, f"model_bin:{model_name}", "PASS", f"{size} bytes {digest}")
        else:
            add(results, f"model_bin:{model_name}", "FAIL", f"{size} bytes {digest}")


def check_data_loop_fake(results: list[Result]) -> None:
    sys.path.insert(0, str(REPO_ROOT))
    name = "fake_data_loop_and_acceptance_audit"
    try:
        mod = importlib.import_module("tests.test_embodied_data_loop_tools")
        with tempfile.TemporaryDirectory(prefix="embodied_v3_selftest_") as tmp:
            mod.test_data_loop_finalize_skeleton_and_audit(Path(tmp))
            postflash_tmp = Path(tmp) / "postflash_tamper"
            postflash_tmp.mkdir()
            mod.test_postflash_orchestration_v3_positive_and_tampering(postflash_tmp)
    except Exception as exc:
        add(results, name, "FAIL", f"{type(exc).__name__}: {exc}")
    else:
        add(results, name, "PASS")


def check_f407_protocol_contract(results: list[Result]) -> None:
    name = "f407_protocol_contract"
    tools_path = REPO_ROOT / "embodied_brain" / "tools"
    sys.path.insert(0, str(tools_path))
    try:
        module = importlib.import_module("f407_protocol_contract_test")
        report = module.run_contract()
        if report.get("overall") != "PASS":
            raise AssertionError(f"failed={report.get('failed')}")
    except Exception as exc:
        add(results, name, "FAIL", f"{type(exc).__name__}: {exc}")
    else:
        add(results, name, "PASS", f"checks={len(report.get('checks') or [])}")


def check_postflash_orchestrator_contract(results: list[Result]) -> None:
    name = "f407_postflash_orchestrator_contract"
    sys.path.insert(0, str(REPO_ROOT))
    try:
        module = importlib.import_module("tests.test_embodied_postflash_orchestrator")
        module.test_postflash_shell_contract_is_fail_closed()
        module.test_postflash_validator_rejects_safety_and_ack_tampering()
        module.test_runtime_reuses_exact_postflash_manifest_without_second_probe()
        with tempfile.TemporaryDirectory(prefix="f407_postflash_bundle_") as tmp:
            root = Path(tmp)
            module.test_postflash_bundler_copies_only_canonical_hash_matched_files(root / "positive")
            module.test_postflash_bundler_rejects_path_and_hash_substitution(root / "negative")
    except Exception as exc:
        add(results, name, "FAIL", f"{type(exc).__name__}: {exc}")
    else:
        add(results, name, "PASS", "shell/validator/bundler contracts + path/hash substitution rejection; hardware untouched")


def check_f407_keil_build(results: list[Result]) -> None:
    name = "f407_keil_test_mode0_build"
    root = REPO_ROOT / "embodied_brain" / "stm32_f407"
    log_path = root / "keil_build.log"
    hex_path = root / "Objects" / "a.hex"
    sources = [
        root / "App" / "main.c",
        root / "App" / "proto.c",
        root / "App" / "proto.h",
        root / "App" / "bsp_lift.c",
        root / "App" / "bsp_lift.h",
        root / "App" / "bsp_uart.c",
        root / "App" / "bsp_uart.h",
        root / "App" / "bsp_imu.c",
        root / "App" / "bsp_imu.h",
    ]
    if not log_path.exists() or not hex_path.exists() or any(not path.exists() for path in sources):
        add(results, name, "FAIL", "Keil log/hex/source missing")
        return
    log = log_path.read_text(encoding="utf-8", errors="replace")
    newest_source = max(path.stat().st_mtime for path in sources)
    artifact_time = min(log_path.stat().st_mtime, hex_path.stat().st_mtime)
    main_text = sources[0].read_text(encoding="utf-8", errors="replace")
    ok = (
        '0 Error(s), 0 Warning(s).' in log
        and '#define TEST_MODE       0' in main_text
        and hex_path.stat().st_size > 0
        and artifact_time >= newest_source
    )
    if ok:
        add(results, name, "PASS", f"hex={hex_path.stat().st_size} bytes; build newer than safety sources")
    else:
        add(results, name, "FAIL", "Keil build missing/failed/stale or TEST_MODE is not 0")


def check_safety_contracts(results: list[Result]) -> None:
    name = "dispatch_lab_fsd_safety_contracts"
    sys.path.insert(0, str(REPO_ROOT))
    try:
        module = importlib.import_module("tests.test_embodied_safety_contracts")
        module.test_lab_fsd_guard_contracts()
    except Exception as exc:
        add(results, name, "FAIL", f"{type(exc).__name__}: {exc}")
    else:
        add(results, name, "PASS")


def check_lab_fsd_core(results: list[Result]) -> None:
    name = "lab_fsd_core_bev_and_tensor"
    nav_scripts = REPO_ROOT / "embodied_brain" / "ros2_ws" / "src" / "my_robot_navigation" / "scripts"
    sys.path.insert(0, str(nav_scripts))
    try:
        import numpy as np
        from lab_fsd_core import BevConfig, bev_tensor_for_bpu, points_to_bev, scan_to_points, score_candidate_trajectories

        ranges = [2.0 for _ in range(181)]
        ranges[90] = 0.8
        pts = scan_to_points(ranges, -math.pi / 2, math.pi / 180, 0.05, 5.0, 4.5)
        if pts.shape[1] != 2:
            raise AssertionError(f"scan_to_points shape={pts.shape}")
        bev = points_to_bev(pts, BevConfig(grid_size=48, resolution_m=0.10, inflation_cells=1))
        if bev.shape != (48, 48) or int(bev.max()) != 100:
            raise AssertionError(f"bev shape/max={bev.shape}/{int(bev.max())}")
        scores = score_candidate_trajectories(bev, goal_xy=(1.5, 0.0), resolution_m=0.10)
        if scores.get("mode") != "shadow_only" or len(scores.get("candidates", [])) < 5:
            raise AssertionError("trajectory scores missing shadow candidates")
        conf = float(scores.get("shadow_confidence", -1.0))
        if not 0.0 <= conf <= 1.0:
            raise AssertionError(f"shadow_confidence={conf}")
        tensor = bev_tensor_for_bpu(np.zeros((48, 48), dtype=np.int16), goal_xy=(1.0, 0.2))
        if tensor.shape != (1, 3, 48, 48) or tensor.dtype != np.float32:
            raise AssertionError(f"tensor shape/dtype={tensor.shape}/{tensor.dtype}")
    except ModuleNotFoundError as exc:
        if exc.name == "numpy":
            add(results, name, "WARN", "numpy not installed in this PC Python; runtime check skipped")
        else:
            add(results, name, "FAIL", f"{type(exc).__name__}: {exc}")
    except Exception as exc:
        add(results, name, "FAIL", f"{type(exc).__name__}: {exc}")
    else:
        add(results, name, "PASS")


def check_preflight(results: list[Result], require_pass: bool) -> None:
    bash = find_bash()
    if not bash:
        add(results, "network_preflight", "FAIL" if require_pass else "WARN", "bash not found")
        return
    deploy = EMBODIED_ROOT / "deploy_to_car.sh"
    try:
        rc, out = run_cmd([bash, str(deploy), "preflight"], timeout_s=90)
    except Exception as exc:
        add(results, "network_preflight", "FAIL" if require_pass else "WARN", f"{type(exc).__name__}: {exc}")
        return
    tail = "\n".join(out.splitlines()[-20:])
    network_path_ok = (
        "OK: WLAN IPv4 is 192.168.31." in out
        or "OK: device overlay host routes are bound to WLAN" in out
    )
    ok = "OK: SSID is xrd-lab_5G" in out and network_path_ok
    if rc != 0:
        add(results, "network_preflight", "FAIL" if require_pass else "WARN", f"rc={rc}\n{tail}")
    elif ok:
        add(results, "network_preflight", "PASS", "PC has a valid xrd-lab_5G native or /32 host-route overlay path")
    else:
        add(results, "network_preflight", "FAIL" if require_pass else "WARN", tail)


def summarize_results(results: list[Result]) -> dict:
    counts = {"PASS": 0, "WARN": 0, "FAIL": 0}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    overall = "FAIL" if counts["FAIL"] else ("PASS" if counts["WARN"] == 0 else "PASS_WITH_WARN")
    return {"counts": counts, "overall": overall}


def write_report(results: list[Result], out_path: str) -> str:
    target = Path(out_path).expanduser()
    if not target.is_absolute():
        target = REPO_ROOT / target
    target.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize_results(results)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo": str(REPO_ROOT),
        "embodied_root": str(EMBODIED_ROOT),
        "overall": summary["overall"],
        "counts": summary["counts"],
        "results": [result.__dict__ for result in results],
    }
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return rel(target)


def print_summary(results: list[Result], out_path: str = "") -> int:
    summary = summarize_results(results)
    counts = summary["counts"]
    report_path = write_report(results, out_path) if out_path else ""
    print("")
    print("EMBODIED_V3_LOCAL_SELFTEST")
    print(f"repo: {REPO_ROOT}")
    print(f"counts: {counts}")
    if report_path:
        print(f"report: {report_path}")
    print(f"overall: {summary['overall']}")
    if summary["overall"] == "FAIL":
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true", help="Also run read-only PC overlay network preflight.")
    parser.add_argument(
        "--require-preflight-pass",
        action="store_true",
        help="Make a non-xrd-lab_5G / non-192.168.31.x network preflight a hard failure.",
    )
    parser.add_argument("--out", default="", help="Write a JSON self-test report to this path.")
    args = parser.parse_args()

    os.environ.setdefault("PYTHONUTF8", "1")
    results: list[Result] = []
    check_python_syntax(results)
    check_shell_syntax(results)
    check_sentinels(results)
    check_model_bin(results)
    check_f407_protocol_contract(results)
    check_postflash_orchestrator_contract(results)
    check_f407_keil_build(results)
    check_safety_contracts(results)
    check_data_loop_fake(results)
    check_lab_fsd_core(results)
    if args.preflight or args.require_preflight_pass:
        check_preflight(results, require_pass=args.require_preflight_pass)
    return print_summary(results, out_path=args.out)


if __name__ == "__main__":
    raise SystemExit(main())
