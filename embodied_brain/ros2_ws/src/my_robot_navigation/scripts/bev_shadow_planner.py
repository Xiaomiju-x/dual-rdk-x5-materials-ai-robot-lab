#!/usr/bin/env python3
"""Lab-FSD v3 BEV Occupancy Shadow Planner.

This node is intentionally shadow-only:
  * subscribes to scan/depth scan/odom/goal
  * publishes BEV, a proposed local path, and JSON risk diagnostics
  * never publishes cmd_vel
Nav2 remains the authority and safety fallback.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab_fsd_core import (
    BevConfig,
    apply_vision_bev_provenance,
    bev_tensor_for_bpu,
    classify_vision_bev_provenance,
    forecast_future_occupancy,
    fuse_policy_with_bpu_prior,
    merge_bev,
    points_to_bev,
    safety_gate_decision,
    scan_to_points,
    score_candidate_trajectories,
    score_candidate_trajectories_v2,
)


def main() -> None:
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import OccupancyGrid, Odometry, Path as NavPath
    from rclpy.node import Node
    from sensor_msgs.msg import LaserScan
    from std_msgs.msg import Float32, String

    class BevShadowPlanner(Node):
        def __init__(self) -> None:
            super().__init__("lab_fsd_bev_shadow_planner")
            self.declare_parameter("grid_size", 48)
            self.declare_parameter("resolution_m", 0.10)
            self.declare_parameter("max_range_m", 4.5)
            self.declare_parameter("inflation_cells", 2)
            self.declare_parameter("publish_rate_hz", 5.0)
            self.declare_parameter("scan_topic", "/scan")
            self.declare_parameter("depth_scan_topic", "/scan_depth")
            self.declare_parameter("odom_topic", "/odom")
            self.declare_parameter("goal_topic", "/goal_pose")
            self.declare_parameter("vx_shadow_mps", 0.18)
            self.declare_parameter("horizon_s", 2.4)
            self.declare_parameter("model_bin", "")
            self.declare_parameter("anomaly_model_bin", "/home/rdk/models/lab_fsd/lab_anomaly_autoencoder.bin")
            self.declare_parameter("use_vision_bev", True)
            self.declare_parameter("vision_bev_topic", "/lab_fsd/vision_bev")
            self.declare_parameter("vision_objects_topic", "/lab_fsd/vision_objects")
            self.declare_parameter("vision_bev_ttl_s", 4.0)
            self.declare_parameter("fsd_v2_enabled", True)
            self.declare_parameter("temporal_history_len", 5)
            self.declare_parameter("future_horizons", 3)
            self.declare_parameter("safety_min_confidence", 0.12)
            self.declare_parameter("safety_max_risk", 0.82)
            self.declare_parameter("fsd_v3_enabled", True)
            self.declare_parameter("sensor_stale_after_s", 1.5)
            self.declare_parameter("sensor_offline_after_s", 5.0)
            self.declare_parameter("odom_stale_after_s", 2.5)
            self.declare_parameter("odom_offline_after_s", 8.0)
            self.declare_parameter("future_risk_alert", 0.82)
            self.declare_parameter("use_occ_risk_bpu", False)
            self.declare_parameter("occ_risk_model_bin", "/home/rdk/models/lab_fsd/lab_fsd_tiny_occ_risk.bin")

            self.config = BevConfig(
                grid_size=int(self.get_parameter("grid_size").value),
                resolution_m=float(self.get_parameter("resolution_m").value),
                max_range_m=float(self.get_parameter("max_range_m").value),
                inflation_cells=int(self.get_parameter("inflation_cells").value),
            )
            self.scan_msg = None
            self.depth_scan_msg = None
            self.vision_bev_msg = None
            self.vision_bev_seen_monotonic = 0.0
            self.vision_provenance = classify_vision_bev_provenance({})
            self.odom_msg = None
            self.goal_xy = (1.5, 0.0)
            self.goal_frame = "fallback"
            self.goal_frame_valid = False
            self.goal_frame_note = "no goal received; using forward fallback"
            self.bev_history = []
            self.bev_history_meta = []
            self.sensor_seen_monotonic = {
                "scan": 0.0,
                "scan_depth": 0.0,
                "odom": 0.0,
                "goal": 0.0,
                "vision_bev": 0.0,
            }

            self.create_subscription(LaserScan, str(self.get_parameter("scan_topic").value), self._on_scan, 10)
            self.create_subscription(LaserScan, str(self.get_parameter("depth_scan_topic").value), self._on_depth_scan, 10)
            self.create_subscription(Odometry, str(self.get_parameter("odom_topic").value), self._on_odom, 10)
            self.create_subscription(PoseStamped, str(self.get_parameter("goal_topic").value), self._on_goal, 10)
            self.create_subscription(OccupancyGrid, str(self.get_parameter("vision_bev_topic").value), self._on_vision_bev, 10)
            self.create_subscription(String, str(self.get_parameter("vision_objects_topic").value), self._on_vision_objects, 10)

            self.pub_bev = self.create_publisher(OccupancyGrid, "/lab_fsd/bev", 10)
            self.pub_path = self.create_publisher(NavPath, "/lab_fsd/shadow_path", 10)
            self.pub_diag = self.create_publisher(String, "/lab_fsd/trajectory_scores", 10)
            self.pub_risk = self.create_publisher(Float32, "/lab_fsd/risk", 10)
            self.pub_anomaly = self.create_publisher(Float32, "/lab_fsd/anomaly_score", 10)
            self.pub_future_bev = self.create_publisher(OccupancyGrid, "/lab_fsd/future_bev", 10)
            self.pub_future_risk = self.create_publisher(Float32, "/lab_fsd/future_risk", 10)
            self.pub_policy = self.create_publisher(String, "/lab_fsd/policy_tokens", 10)
            self.pub_safety = self.create_publisher(String, "/lab_fsd/safety_gate", 10)
            self.pub_input_status = self.create_publisher(String, "/lab_fsd/input_status", 10)
            self.pub_status_v2 = self.create_publisher(String, "/lab_fsd/fsd_v2_status", 10)
            self.pub_status_v3 = self.create_publisher(String, "/lab_fsd/fsd_v3_status", 10)

            self.anomaly = None
            anomaly_bin = str(self.get_parameter("anomaly_model_bin").value or "")
            if anomaly_bin:
                try:
                    from lab_anomaly_bpu import LabAnomalyBpu
                    self.anomaly = LabAnomalyBpu(anomaly_bin)
                    if not self.anomaly.available():
                        self.get_logger().warn(f"anomaly BPU bin missing: {anomaly_bin}")
                        self.anomaly = None
                except Exception as exc:
                    self.get_logger().warn(f"anomaly BPU unavailable: {exc}")

            self.occ_risk_bpu = None
            self.occ_risk_bpu_error = ""
            occ_bin = str(self.get_parameter("occ_risk_model_bin").value or "")
            if bool(self.get_parameter("use_occ_risk_bpu").value) and occ_bin:
                try:
                    from lab_anomaly_bpu import LabOccRiskBpu
                    self.occ_risk_bpu = LabOccRiskBpu(occ_bin)
                    if not self.occ_risk_bpu.available():
                        self.occ_risk_bpu_error = f"tiny occ-risk BPU bin missing: {occ_bin}"
                        self.get_logger().warn(self.occ_risk_bpu_error)
                        self.occ_risk_bpu = None
                except Exception as exc:
                    self.occ_risk_bpu_error = str(exc)[:160]
                    self.get_logger().warn(f"tiny occ-risk BPU unavailable: {exc}")
            self.occ_risk_bpu_diag = self._occ_risk_bpu_diag()
            hz = max(0.5, float(self.get_parameter("publish_rate_hz").value))
            self.create_timer(1.0 / hz, self._tick)
            model_bin = str(self.get_parameter("model_bin").value or "")
            self.get_logger().info(
                f"Lab-FSD shadow planner online: grid={self.config.grid_size}, "
                f"res={self.config.resolution_m}, model_bin={model_bin or 'heuristic'}, "
                f"anomaly_bin={anomaly_bin or 'disabled'}, "
                f"fsd_v2={bool(self.get_parameter('fsd_v2_enabled').value)}, "
                f"fsd_v3={bool(self.get_parameter('fsd_v3_enabled').value)}, "
                f"occ_risk_bpu={self.occ_risk_bpu_diag.get('state')}"
            )

        def _on_scan(self, msg) -> None:
            self.scan_msg = msg
            self.sensor_seen_monotonic["scan"] = time.monotonic()

        def _on_depth_scan(self, msg) -> None:
            self.depth_scan_msg = msg
            self.sensor_seen_monotonic["scan_depth"] = time.monotonic()

        def _on_odom(self, msg) -> None:
            self.odom_msg = msg
            self.sensor_seen_monotonic["odom"] = time.monotonic()

        def _on_vision_bev(self, msg) -> None:
            self.vision_bev_msg = msg
            self.vision_bev_seen_monotonic = time.monotonic()
            self.sensor_seen_monotonic["vision_bev"] = self.vision_bev_seen_monotonic

        def _on_vision_objects(self, msg) -> None:
            try:
                payload = json.loads(msg.data)
            except (TypeError, json.JSONDecodeError):
                self.vision_provenance = classify_vision_bev_provenance({})
                return
            self.vision_provenance = classify_vision_bev_provenance(payload)

        def _on_goal(self, msg) -> None:
            frame_id = (msg.header.frame_id or "").strip()
            if frame_id not in ("", "base_link", "base_footprint"):
                self.goal_xy = (1.5, 0.0)
                self.goal_frame = frame_id
                self.goal_frame_valid = False
                self.goal_frame_note = f"unsupported goal frame {frame_id}; using forward fallback"
                self.sensor_seen_monotonic["goal"] = time.monotonic()
                self.get_logger().warn(self.goal_frame_note)
                return
            try:
                self.goal_xy = (float(msg.pose.position.x), float(msg.pose.position.y))
                self.goal_frame = frame_id or "base_link_assumed"
                self.goal_frame_valid = True
                self.goal_frame_note = "goal accepted in robot base frame"
                self.sensor_seen_monotonic["goal"] = time.monotonic()
            except Exception:
                self.goal_xy = (1.5, 0.0)
                self.goal_frame = frame_id or "unknown"
                self.goal_frame_valid = False
                self.goal_frame_note = "invalid goal payload; using forward fallback"

        def _occ_risk_bpu_diag(self) -> dict:
            bin_path = str(self.get_parameter("occ_risk_model_bin").value or "")
            enabled = bool(self.get_parameter("use_occ_risk_bpu").value)
            bin_exists = bool(bin_path and Path(bin_path).expanduser().exists())
            runtime_ready = self.occ_risk_bpu is not None
            if enabled and runtime_ready:
                state = "runtime_ready"
                reason = "tiny_occ_risk_bpu_loaded_on_first_forward"
            elif enabled:
                state = "missing_bin"
                reason = self.occ_risk_bpu_error or "tiny_occ_risk_bin_missing_cpu_shadow_fallback_active"
            elif bin_exists:
                state = "available_not_enabled"
                reason = "tiny_occ_risk_bin_present_but_disabled"
            else:
                state = "disabled"
                reason = "optional_tiny_occ_risk_bpu_not_required"
            return {
                "enabled": bool(enabled),
                "bin_path": bin_path,
                "bin_exists": bool(bin_exists),
                "state": state,
                "used": False,
                "runtime": "hobot_dnn" if runtime_ready else "none",
                "reason": reason,
            }

        def _score_occ_risk_bpu(self, bev: np.ndarray, input_status: dict) -> dict:
            diag = dict(self.occ_risk_bpu_diag)
            if self.occ_risk_bpu is None:
                return diag
            if input_status.get("overall") == "offline":
                diag.update({
                    "used": False,
                    "state": "input_offline",
                    "reason": "BEV inputs offline; BPU prior skipped",
                })
                return diag
            try:
                tensor = bev_tensor_for_bpu(bev, self.goal_xy)
                result = self.occ_risk_bpu.score(tensor)
                result.update({
                    "enabled": True,
                    "bin_path": str(self.get_parameter("occ_risk_model_bin").value or ""),
                    "state": "forward_ok",
                    "runtime": "hobot_dnn",
                    "authority": "shadow_diagnostic_only",
                })
                return result
            except Exception as exc:
                diag.update({
                    "used": False,
                    "state": "forward_error",
                    "runtime": "hobot_dnn",
                    "reason": str(exc)[:160],
                })
                return diag

        def _source_state(self, name: str, seen_monotonic: float, stale_s: float, offline_s: float,
                          enabled: bool = True) -> dict:
            if not enabled:
                return {
                    "name": name,
                    "state": "disabled",
                    "age_s": None,
                    "fresh": False,
                    "usable": False,
                }
            if seen_monotonic <= 0.0:
                return {
                    "name": name,
                    "state": "offline",
                    "age_s": None,
                    "fresh": False,
                    "usable": False,
                }
            age_s = max(0.0, time.monotonic() - float(seen_monotonic))
            offline_s = max(float(stale_s), float(offline_s))
            if age_s > offline_s:
                state = "offline"
                fresh = False
                usable = False
            elif age_s > float(stale_s):
                state = "stale"
                fresh = False
                usable = True
            else:
                state = "live"
                fresh = True
                usable = True
            return {
                "name": name,
                "state": state,
                "age_s": round(age_s, 3),
                "fresh": bool(fresh),
                "usable": bool(usable),
            }

        def _collect_input_status(self) -> dict:
            sensor_stale_s = float(self.get_parameter("sensor_stale_after_s").value)
            sensor_offline_s = float(self.get_parameter("sensor_offline_after_s").value)
            odom_stale_s = float(self.get_parameter("odom_stale_after_s").value)
            odom_offline_s = float(self.get_parameter("odom_offline_after_s").value)
            vision_ttl_s = float(self.get_parameter("vision_bev_ttl_s").value)
            vision_enabled = bool(self.get_parameter("use_vision_bev").value)

            sources = {
                "scan": self._source_state(
                    "scan",
                    self.sensor_seen_monotonic["scan"],
                    sensor_stale_s,
                    sensor_offline_s,
                    enabled=True,
                ),
                "scan_depth": self._source_state(
                    "scan_depth",
                    self.sensor_seen_monotonic["scan_depth"],
                    sensor_stale_s,
                    sensor_offline_s,
                    enabled=True,
                ),
                "odom": self._source_state(
                    "odom",
                    self.sensor_seen_monotonic["odom"],
                    odom_stale_s,
                    odom_offline_s,
                    enabled=True,
                ),
                "vision_bev": apply_vision_bev_provenance(
                    self._source_state(
                        "vision_bev",
                        self.sensor_seen_monotonic["vision_bev"],
                        vision_ttl_s,
                        max(sensor_offline_s, vision_ttl_s * 2.0),
                        enabled=vision_enabled,
                    ),
                    self.vision_provenance,
                ),
            }
            if self.sensor_seen_monotonic["goal"] <= 0.0 or not self.goal_frame_valid:
                goal_state = {
                    "name": "goal",
                    "state": "fallback",
                    "age_s": None,
                    "fresh": False,
                    "usable": True,
                    "target_xy": [round(float(self.goal_xy[0]), 3), round(float(self.goal_xy[1]), 3)],
                    "frame_id": self.goal_frame,
                    "reason": self.goal_frame_note,
                }
            else:
                goal_state = self._source_state(
                    "goal",
                    self.sensor_seen_monotonic["goal"],
                    odom_offline_s,
                    odom_offline_s * 3.0,
                    enabled=True,
                )
                goal_state["target_xy"] = [round(float(self.goal_xy[0]), 3), round(float(self.goal_xy[1]), 3)]
                goal_state["frame_id"] = self.goal_frame
                goal_state["reason"] = self.goal_frame_note
            sources["goal"] = goal_state

            proximity = [sources["scan"], sources["scan_depth"]]
            if any(src["state"] == "live" for src in proximity):
                overall = "live"
            elif any(src["usable"] for src in proximity):
                overall = "stale"
            else:
                overall = "offline"
            bev_sources = [name for name in ("scan", "scan_depth", "vision_bev") if sources[name]["usable"]]
            return {
                "stack": "Lab-FSD v3",
                "overall": overall,
                "bev_sources": bev_sources,
                "sources": sources,
                "shadow_only": True,
                "cmd_vel_authority": False,
            }

        def _future_risk_summary(self, future_bevs, score: dict, input_status: dict) -> dict:
            horizons = []
            for idx, grid in enumerate(future_bevs, start=1):
                arr = np.clip(np.asarray(grid, dtype=np.float32), 0.0, 100.0)
                horizons.append({
                    "horizon": idx,
                    "max_occ": round(float(arr.max()) / 100.0, 4) if arr.size else 0.0,
                    "mean_occ": round(float(arr.mean()) / 100.0, 4) if arr.size else 0.0,
                    "occupied_ratio": round(float(np.mean(arr >= 65.0)), 4) if arr.size else 0.0,
                })
            best = score.get("best", {}) if isinstance(score, dict) else {}
            path_future_risk = float(best.get("future_risk", best.get("risk", 1.0)))
            peak_grid_risk = max([float(row["max_occ"]) for row in horizons], default=0.0)
            health_floor = 0.0
            if input_status.get("overall") == "stale":
                health_floor = 0.65
            elif input_status.get("overall") == "offline":
                health_floor = 1.0
            return {
                "horizons": horizons,
                "path_future_risk": round(path_future_risk, 4),
                "peak_grid_risk": round(peak_grid_risk, 4),
                "published_future_risk": round(max(path_future_risk, health_floor), 4),
                "input_health_floor": round(health_floor, 4),
            }

        def _arc_token_diag(self, score: dict) -> dict:
            candidates = list(score.get("candidates", [])) if isinstance(score, dict) else []
            policy = score.get("policy", {}) if isinstance(score, dict) else {}
            tokens = []
            for rank, cand in enumerate(candidates):
                probability = float(cand.get("probability", 0.0) or 0.0)
                token_score = float(cand.get("score", 0.0) or 0.0)
                token_risk = float(cand.get("risk", 1.0) or 1.0)
                token_conf = max(0.0, min(1.0, 0.55 * probability + 0.30 * token_score + 0.15 * (1.0 - token_risk)))
                tokens.append({
                    "rank": rank,
                    "token_id": int(cand.get("token_id", rank)),
                    "omega": round(float(cand.get("omega", 0.0) or 0.0), 4),
                    "score": round(token_score, 4),
                    "probability": round(probability, 4),
                    "confidence": round(token_conf, 4),
                    "risk": round(token_risk, 4),
                    "now_risk": round(float(cand.get("now_risk", token_risk) or token_risk), 4),
                    "future_risk": round(float(cand.get("future_risk", token_risk) or token_risk), 4),
                    "clearance": round(float(cand.get("clearance", 0.0) or 0.0), 4),
                    "goal_alignment": round(float(cand.get("goal_alignment", 0.0) or 0.0), 4),
                })
            top_prob = tokens[0]["probability"] if tokens else 0.0
            second_prob = tokens[1]["probability"] if len(tokens) > 1 else 0.0
            best = score.get("best", {}) if isinstance(score, dict) else {}
            return {
                "vocabulary": policy.get("vocabulary", "arc_omega_tokens"),
                "winner": {
                    "token_id": int(best.get("token_id", tokens[0]["token_id"] if tokens else 0)),
                    "omega": round(float(best.get("omega", tokens[0]["omega"] if tokens else 0.0)), 4),
                },
                "confidence": round(float(score.get("shadow_confidence", policy.get("confidence", 0.0)) or 0.0), 4),
                "entropy": round(float(policy.get("entropy", 1.0) or 1.0), 4),
                "probability_margin": round(max(0.0, top_prob - second_prob), 4),
                "token_count": len(tokens),
                "tokens": tokens,
            }

        def _apply_v3_safety_markers(self, safety_gate: dict, input_status: dict,
                                     future_risk: dict, arc_tokens: dict) -> dict:
            out = dict(safety_gate or {})
            reasons = list(out.get("reasons") or [])
            overall = str(input_status.get("overall") or "offline")
            if overall == "offline":
                reasons.append("bev_inputs_offline")
            elif overall == "stale":
                reasons.append("bev_inputs_stale")
            odom_state = input_status.get("sources", {}).get("odom", {}).get("state")
            if odom_state in ("stale", "offline"):
                reasons.append(f"odom_{odom_state}")
            future_alert = float(self.get_parameter("future_risk_alert").value)
            if float(future_risk.get("published_future_risk", 1.0)) > future_alert:
                reasons.append("future_occupancy_risk_high")
            min_conf = float(self.get_parameter("safety_min_confidence").value)
            if float(arc_tokens.get("confidence", 0.0)) < min_conf and "low_policy_confidence" not in reasons:
                reasons.append("low_policy_confidence")

            deduped = []
            for reason in reasons:
                if reason not in deduped:
                    deduped.append(reason)
            out["reasons"] = deduped
            out["assist_allowed"] = bool(out.get("assist_allowed", False) and not deduped)
            out["shadow_policy"] = "assist_candidate" if out["assist_allowed"] else "observe_only"
            out["cmd_vel_authority"] = False
            out["shadow_only"] = True
            out["input_state"] = overall
            out["future_risk"] = future_risk
            out["arc_token_confidence"] = arc_tokens.get("confidence", 0.0)
            return out

        def _scan_to_bev(self, msg) -> np.ndarray:
            pts = scan_to_points(
                msg.ranges,
                float(msg.angle_min),
                float(msg.angle_increment),
                float(msg.range_min),
                float(msg.range_max),
                self.config.max_range_m,
            )
            return points_to_bev(pts, self.config)

        def _vision_to_bev(self, source_state: dict | None = None) -> tuple[np.ndarray | None, dict]:
            msg = self.vision_bev_msg
            if msg is None or not bool(self.get_parameter("use_vision_bev").value):
                return None, {"used": False, "reason": "disabled_or_missing"}
            if source_state is not None and not bool(source_state.get("usable")):
                return None, {
                    "used": False,
                    "reason": "unusable_provenance",
                    "source_state": source_state,
                }
            age_s = time.monotonic() - float(self.vision_bev_seen_monotonic or 0.0)
            ttl_s = float(self.get_parameter("vision_bev_ttl_s").value)
            if age_s > ttl_s:
                return None, {"used": False, "reason": "stale", "age_s": round(age_s, 3)}
            w = int(msg.info.width)
            h = int(msg.info.height)
            if w <= 0 or h <= 0 or len(msg.data) != w * h:
                return None, {"used": False, "reason": "bad_shape", "width": w, "height": h}
            arr = np.array(msg.data, dtype=np.int16).reshape((h, w))
            if arr.shape != (self.config.grid_size, self.config.grid_size):
                return None, {
                    "used": False,
                    "reason": "shape_mismatch",
                    "shape": list(arr.shape),
                    "expected": [self.config.grid_size, self.config.grid_size],
                }
            arr = np.clip(arr, 0, 100).astype(np.int16)
            return arr, {
                "used": True,
                "age_s": round(age_s, 3),
                "max": int(arr.max()),
                "source_state": source_state or {},
            }

        def _tick(self) -> None:
            input_status = self._collect_input_status()
            grids = []
            if self.scan_msg is not None and input_status["sources"]["scan"]["usable"]:
                grids.append(self._scan_to_bev(self.scan_msg))
            if self.depth_scan_msg is not None and input_status["sources"]["scan_depth"]["usable"]:
                grids.append(self._scan_to_bev(self.depth_scan_msg))
            vision_bev, vision_diag = self._vision_to_bev(
                input_status["sources"]["vision_bev"]
            )
            if vision_bev is not None:
                grids.append(vision_bev)
            elif "vision_bev" in input_status["bev_sources"]:
                input_status["bev_sources"].remove("vision_bev")

            if grids:
                bev = merge_bev(*grids)
            else:
                bev = np.zeros((self.config.grid_size, self.config.grid_size), dtype=np.int16)

            fsd_v3_enabled = bool(self.get_parameter("fsd_v3_enabled").value)
            fsd_v2_enabled = bool(self.get_parameter("fsd_v2_enabled").value) or fsd_v3_enabled
            future_bevs = []
            if fsd_v2_enabled:
                if grids:
                    self.bev_history.append(bev.copy())
                    self.bev_history_meta.append({
                        "max_occ": int(np.max(bev)) if bev.size else 0,
                        "mean_occ": round(float(np.mean(np.clip(bev, 0, 100))) / 100.0, 4) if bev.size else 0.0,
                        "sources": list(input_status["bev_sources"]),
                        "input_state": input_status["overall"],
                    })
                else:
                    self.bev_history = []
                    self.bev_history_meta = []
                max_hist = max(1, int(self.get_parameter("temporal_history_len").value))
                self.bev_history = self.bev_history[-max_hist:]
                self.bev_history_meta = self.bev_history_meta[-max_hist:]
                future_bevs = forecast_future_occupancy(
                    self.bev_history,
                    horizons=int(self.get_parameter("future_horizons").value),
                ) if self.bev_history else []
                score = score_candidate_trajectories_v2(
                    bev,
                    future_bevs=future_bevs,
                    goal_xy=self.goal_xy,
                    resolution_m=self.config.resolution_m,
                    vx=float(self.get_parameter("vx_shadow_mps").value),
                    horizon_s=float(self.get_parameter("horizon_s").value),
                )
            else:
                score = score_candidate_trajectories(
                    bev,
                    goal_xy=self.goal_xy,
                    resolution_m=self.config.resolution_m,
                    vx=float(self.get_parameter("vx_shadow_mps").value),
                    horizon_s=float(self.get_parameter("horizon_s").value),
                )
            self._publish_bev(bev)
            if future_bevs:
                self._publish_future_bev(merge_bev(*future_bevs))
            elif fsd_v2_enabled:
                self._publish_future_bev(np.zeros_like(bev))
            self._publish_path(score["best"]["points_xy"])
            future_risk = self._future_risk_summary(future_bevs, score, input_status)
            risk = max(float(score["best"]["risk"]), float(future_risk["published_future_risk"]))
            self.pub_risk.publish(Float32(data=risk))
            self.pub_future_risk.publish(Float32(data=float(future_risk["published_future_risk"])))
            anomaly_diag = None
            if self.anomaly is not None:
                try:
                    if input_status["overall"] == "offline":
                        anomaly_diag = {"ok": False, "reason": "bev_inputs_offline"}
                    else:
                        tensor = bev_tensor_for_bpu(bev, self.goal_xy)
                        anomaly_diag = self.anomaly.score(tensor)
                        self.pub_anomaly.publish(Float32(data=float(anomaly_diag["mse"])))
                except Exception as exc:
                    anomaly_diag = {"ok": False, "error": str(exc)[:160]}
            occ_risk_diag = self._score_occ_risk_bpu(bev, input_status)
            arc_tokens = self._arc_token_diag(score)
            policy_prior = fuse_policy_with_bpu_prior(score, occ_risk_diag)
            safety_gate = safety_gate_decision(
                score,
                vision_diag=vision_diag,
                anomaly_diag=anomaly_diag,
                min_confidence=float(self.get_parameter("safety_min_confidence").value),
                max_risk=float(self.get_parameter("safety_max_risk").value),
            )
            safety_gate = self._apply_v3_safety_markers(safety_gate, input_status, future_risk, arc_tokens)
            mode = "lab_fsd_v3_arc_token_shadow" if fsd_v3_enabled else score.get("mode", "shadow_only")
            policy_payload = dict(score.get("policy") or {})
            policy_payload.update({
                "stack": "Lab-FSD v3" if fsd_v3_enabled else "Lab-FSD v2",
                "mode": mode,
                "arc_tokens": arc_tokens,
                "future_risk": future_risk,
                "bpu_occ_risk_prior": occ_risk_diag,
                "policy_prior": policy_prior,
                "cmd_vel_authority": False,
            })
            fsd_status = {
                "stack": "Lab-FSD v3" if fsd_v3_enabled else "Lab-FSD v2",
                "mode": mode,
                "research_lineage": [
                    "Tesla-style BEV occupancy",
                    "BEVFormer-style temporal memory",
                    "UniAD-style planning-oriented diagnostics",
                    "VADv2-style probabilistic trajectory tokens",
                ],
                "shadow_only": True,
                "cmd_vel_authority": False,
                "input_status": input_status,
                "temporal_history": len(self.bev_history),
                "temporal_history_meta": list(self.bev_history_meta),
                "future_horizons": len(future_bevs),
                "future_risk": future_risk,
                "arc_tokens": {
                    "winner": arc_tokens.get("winner"),
                    "confidence": arc_tokens.get("confidence"),
                    "probability_margin": arc_tokens.get("probability_margin"),
                    "token_count": arc_tokens.get("token_count"),
                },
                "bpu": {
                    "anomaly_autoencoder": bool(self.anomaly is not None),
                    "tiny_occ_risk": occ_risk_diag,
                    "policy_prior": {
                        "source": policy_prior.get("source"),
                        "used_bpu": policy_prior.get("used_bpu"),
                        "fused_best_index": policy_prior.get("fused_best_index"),
                        "agreement": policy_prior.get("agreement"),
                        "confidence": policy_prior.get("confidence"),
                    },
                    "policy_interface": "BPU Conv/MLP prior is shadow diagnostic; deterministic CPU/Nav2 remain safety fallback",
                },
                "safety_gate": safety_gate,
            }
            if "policy" in score or arc_tokens.get("tokens"):
                self.pub_policy.publish(String(data=json.dumps(policy_payload, ensure_ascii=False)))
            self.pub_safety.publish(String(data=json.dumps(safety_gate, ensure_ascii=False)))
            status_payload = json.dumps(fsd_status, ensure_ascii=False)
            self.pub_status_v2.publish(String(data=status_payload))
            self.pub_status_v3.publish(String(data=status_payload))
            self.pub_input_status.publish(String(data=json.dumps(input_status, ensure_ascii=False)))
            diag = {
                "version": "v3" if fsd_v3_enabled else "v2",
                "mode": mode,
                "risk": round(risk, 4),
                "raw_trajectory_risk": round(float(score["best"]["risk"]), 4),
                "future_risk": future_risk,
                "shadow_confidence": score["shadow_confidence"],
                "arc_tokens": arc_tokens,
                "anomaly": anomaly_diag,
                "occ_risk_bpu": occ_risk_diag,
                "policy_prior": policy_prior,
                "safety_gate": safety_gate,
                "policy": score.get("policy"),
                "future_occupancy": {
                    "enabled": fsd_v2_enabled,
                    "history_len": len(self.bev_history),
                    "history": list(self.bev_history_meta),
                    "horizons": len(future_bevs),
                    "max": int(max([int(f.max()) for f in future_bevs], default=0)),
                },
                "best": {k: v for k, v in score["best"].items() if k != "points_xy"},
                "candidates": score["candidates"],
                "bpu": {
                    "tiny_occ_risk": occ_risk_diag,
                    "anomaly_autoencoder": bool(self.anomaly is not None),
                },
                "shadow_only": True,
                "cmd_vel_authority": False,
                "inputs": {
                    "scan": self.scan_msg is not None,
                    "scan_depth": self.depth_scan_msg is not None,
                    "odom": self.odom_msg is not None,
                    "vision_bev": vision_diag,
                    "state": input_status,
                },
            }
            self.pub_diag.publish(String(data=json.dumps(diag, ensure_ascii=False)))

        def _publish_bev(self, bev: np.ndarray) -> None:
            msg = OccupancyGrid()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "base_footprint"
            msg.info.resolution = float(self.config.resolution_m)
            msg.info.width = int(bev.shape[1])
            msg.info.height = int(bev.shape[0])
            msg.info.origin.position.x = -bev.shape[1] * self.config.resolution_m / 2.0
            msg.info.origin.position.y = -bev.shape[0] * self.config.resolution_m / 2.0
            msg.info.origin.orientation.w = 1.0
            msg.data = [int(max(0, min(100, v))) for v in bev.flatten()]
            self.pub_bev.publish(msg)

        def _publish_future_bev(self, bev: np.ndarray) -> None:
            msg = OccupancyGrid()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "base_footprint"
            msg.info.resolution = float(self.config.resolution_m)
            msg.info.width = int(bev.shape[1])
            msg.info.height = int(bev.shape[0])
            msg.info.origin.position.x = -bev.shape[1] * self.config.resolution_m / 2.0
            msg.info.origin.position.y = -bev.shape[0] * self.config.resolution_m / 2.0
            msg.info.origin.orientation.w = 1.0
            msg.data = [int(max(0, min(100, v))) for v in bev.flatten()]
            self.pub_future_bev.publish(msg)

        def _publish_path(self, points_xy) -> None:
            path = NavPath()
            path.header.stamp = self.get_clock().now().to_msg()
            path.header.frame_id = "base_footprint"
            for x, y in points_xy:
                ps = PoseStamped()
                ps.header = path.header
                ps.pose.position.x = float(x)
                ps.pose.position.y = float(y)
                ps.pose.orientation.w = 1.0
                path.poses.append(ps)
            self.pub_path.publish(path)

    rclpy.init()
    node = BevShadowPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
