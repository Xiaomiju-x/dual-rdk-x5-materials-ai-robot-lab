#!/usr/bin/env python3
"""Keyboard teleop helper for slow SLAM mapping on the car-side RDK X5.

Run this script on the car X5 from a local terminal with a keyboard attached.
It publishes /cmd_vel for the STM32F407 chassis bridge and can save the current
slam_toolbox map through nav2_map_server.
"""

from __future__ import annotations

import argparse
import os
import select
import subprocess
import sys
import termios
import time
import tty
from datetime import datetime
from pathlib import Path

import rclpy
from geometry_msgs.msg import Twist


HELP = """\
WASD SLAM mapping control

  w : forward
  s : backward
  a : rotate left
  d : rotate right
  space/x : stop
  m : save map to ~/maps/lab_YYYYmmdd_HHMMSS
  h/? : help
  q : stop and quit

Hold a movement key for continuous motion. If no movement key arrives for the
deadman timeout, the script publishes zero velocity automatically.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WASD teleop for SLAM mapping")
    parser.add_argument("--linear", type=float, default=0.02,
                        help="linear command in m/s; F407 normalizes nonzero commands to CRUISE_PPS")
    parser.add_argument("--angular", type=float, default=0.45,
                        help="angular command in rad/s")
    parser.add_argument("--rate", type=float, default=15.0,
                        help="publish rate in Hz")
    parser.add_argument("--deadman", type=float, default=0.45,
                        help="seconds after the last movement key before auto-stop")
    parser.add_argument("--map-dir", default=str(Path.home() / "maps"),
                        help="directory for saved maps")
    parser.add_argument("--map-prefix", default="lab",
                        help="saved map filename prefix")
    return parser.parse_args()


class RawTerminal:
    def __enter__(self) -> "RawTerminal":
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)


def read_key(timeout_s: float) -> str | None:
    ready, _, _ = select.select([sys.stdin], [], [], timeout_s)
    if not ready:
        return None
    return os.read(sys.stdin.fileno(), 1).decode(errors="ignore").lower()


def twist(vx: float, wz: float) -> Twist:
    msg = Twist()
    msg.linear.x = float(vx)
    msg.angular.z = float(wz)
    return msg


def save_map(map_dir: Path, prefix: str) -> None:
    map_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = map_dir / f"{prefix}_{stamp}"
    cmd = [
        "ros2", "run", "nav2_map_server", "map_saver_cli",
        "-f", str(base),
        "--ros-args", "-p", "map_subscribe_transient_local:=true",
    ]
    print(f"\nSaving map: {base}.yaml")
    try:
        completed = subprocess.run(cmd, check=False, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   timeout=20)
        output = completed.stdout.strip()
        if output:
            print(output)
        if completed.returncode == 0:
            print("Map saved.")
            audit = Path.home() / "tools" / "map_quality_audit.py"
            if audit.exists():
                audit_out = base.with_name(base.name + "_quality.json")
                audited = subprocess.run(
                    [sys.executable, str(audit), str(base.with_suffix('.yaml')),
                     '--out', str(audit_out)],
                    check=False, text=True, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, timeout=30)
                print(audited.stdout.strip())
                print("Map quality PASS." if audited.returncode == 0 else
                      "Map quality FAIL; do not use this map for Nav2.")
        else:
            print(f"Map save failed with exit code {completed.returncode}.")
    except Exception as exc:  # pragma: no cover - field diagnostic path
        print(f"Map save failed: {exc}")


def main() -> int:
    args = parse_args()
    rclpy.init()
    node = rclpy.create_node("slam_wasd_mapper")
    pub = node.create_publisher(Twist, "/cmd_vel", 10)

    period = 1.0 / max(args.rate, 1.0)
    current = twist(0.0, 0.0)
    last_motion_key = 0.0
    last_pub = 0.0
    stopped = True

    def publish_stop() -> None:
        nonlocal current, stopped
        current = twist(0.0, 0.0)
        for _ in range(3):
            pub.publish(current)
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(0.03)
        stopped = True

    print(HELP)
    print(f"linear={args.linear:.3f} angular={args.angular:.3f} deadman={args.deadman:.2f}s")

    try:
        with RawTerminal():
            while rclpy.ok():
                now = time.monotonic()
                key = read_key(0.02)
                if key:
                    if key == "w":
                        current = twist(args.linear, 0.0)
                        last_motion_key = now
                        stopped = False
                    elif key == "s":
                        current = twist(-args.linear, 0.0)
                        last_motion_key = now
                        stopped = False
                    elif key == "a":
                        current = twist(0.0, args.angular)
                        last_motion_key = now
                        stopped = False
                    elif key == "d":
                        current = twist(0.0, -args.angular)
                        last_motion_key = now
                        stopped = False
                    elif key in (" ", "x"):
                        publish_stop()
                        print("\nSTOP")
                    elif key == "m":
                        publish_stop()
                        save_map(Path(args.map_dir), args.map_prefix)
                    elif key in ("h", "?"):
                        print("\n" + HELP)
                    elif key == "q":
                        publish_stop()
                        print("\nQuit.")
                        break

                if not stopped and (now - last_motion_key) > args.deadman:
                    current = twist(0.0, 0.0)
                    stopped = True

                if (now - last_pub) >= period:
                    pub.publish(current)
                    rclpy.spin_once(node, timeout_sec=0.0)
                    last_pub = now

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        publish_stop()
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
