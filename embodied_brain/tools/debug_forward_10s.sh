#!/usr/bin/env bash

# ROS setup scripts read optional variables that are unset in a desktop launch.
set +u
source /opt/ros/humble/setup.bash
source /home/rdk/ros2_ws/install/setup.bash
set -u

LOCK_FILE="/tmp/xrd_debug_forward_10s.lock"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "Another forward test is already running."
  sleep 3
  exit 1
fi

python3 - <<'PY'
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_srvs.srv import Trigger


class ForwardDebug(Node):
    def __init__(self) -> None:
        super().__init__("debug_forward_10s")
        self.publisher = self.create_publisher(Twist, "/cmd_vel", 10)
        self.clear_estop = self.create_client(Trigger, "/clear_estop")
        self.estop = self.create_client(Trigger, "/estop")

    def trigger(self, client, label: str, timeout_s: float = 8.0):
        if not client.wait_for_service(timeout_sec=timeout_s):
            raise RuntimeError(f"{label} service unavailable")
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_s)
        response = future.result()
        if response is None:
            raise RuntimeError(f"{label} response timeout")
        print(f"{label}: ok={response.success} message={response.message}", flush=True)
        return response

    def publish_for(self, seconds: float) -> None:
        command = Twist()
        command.linear.x = 0.02
        deadline = time.monotonic() + seconds
        next_report = 1
        while time.monotonic() < deadline:
            self.publisher.publish(command)
            rclpy.spin_once(self, timeout_sec=0.0)
            remaining = max(0, int(deadline - time.monotonic() + 0.999))
            elapsed = int(seconds) - remaining
            if elapsed >= next_report:
                print(f"Forward: {elapsed}/{int(seconds)} s", flush=True)
                next_report += 1
            time.sleep(0.05)

    def publish_zero(self) -> None:
        zero = Twist()
        for _ in range(8):
            self.publisher.publish(zero)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(0.03)


def main() -> int:
    rclpy.init()
    node = ForwardDebug()
    exit_code = 1
    try:
        clear = node.trigger(node.clear_estop, "CLEAR_ESTOP")
        if not clear.success:
            raise RuntimeError("F407 refused CLEAR_ESTOP")
        print("Starting 10-second forward test.", flush=True)
        node.publish_for(10.0)
        print("Forward test complete.", flush=True)
        exit_code = 0
    except KeyboardInterrupt:
        print("Interrupted; stopping.", flush=True)
    except Exception as exc:
        print(f"ERROR: {exc}", flush=True)
    finally:
        try:
            node.publish_zero()
            node.trigger(node.estop, "EMERGENCY_STOP", timeout_s=12.0)
        except Exception as exc:
            print(f"STOP ERROR: {exc}", flush=True)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    print("Window closes in 3 seconds.", flush=True)
    time.sleep(3)
    return exit_code


raise SystemExit(main())
PY
