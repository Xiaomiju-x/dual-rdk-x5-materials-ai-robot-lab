# Metric Navigation Shadow Library

Pure Python, offline diagnostics for the independent finals candidate.

## Scope

- LiDAR, IMU, and wheel-odometry timing/frequency/covariance health.
- Wheel/IMU yaw-rate innovation and Mahalanobis gating.
- Stationary drift, scan overlap, and planar geometry degeneracy.
- Pose-graph and loop-closure event evidence contracts.
- AMCL, `slam_toolbox`, and MPPI shadow A/B records and recommendations.

Every `to_dict()` result is strict JSON data and includes:

```json
{
  "safety_boundary": {
    "shadow_only": true,
    "cmd_vel_authority": false,
    "publishes_tf": false,
    "writes_nav_stack": false
  }
}
```

The package has no ROS imports, publishers, services, serial access, or
configuration writers. A `RECOMMEND` result means that a candidate is suitable
for human-reviewed follow-up; it never authorizes activation.
