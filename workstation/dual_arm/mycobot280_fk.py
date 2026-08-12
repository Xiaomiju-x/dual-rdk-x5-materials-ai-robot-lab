#!/usr/bin/env python3
"""Dependency-free forward kinematics for the myCobot 280 Pi.

Geometry source:
  https://github.com/elephantrobotics/mycobot_ros/blob/
  15503ee4f8b859a8db69791eb1c1cccd2508b5ed/
  mycobot_description/urdf/mycobot_280_pi/mycobot_280_pi.urdf

Source URDF SHA-256:
  99F9DB4B54FC4C40FE3D2FFB64CE64F1EA0A60A253B71964DB948511046CDB28
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Iterable, List, Sequence


Matrix = List[List[float]]


# parent -> revolute-joint origins from the official URDF, in metres/radians.
JOINT_ORIGINS = (
    ((0.0, 0.0, 0.13956), (0.0, 0.0, 0.0)),
    ((0.0, 0.0, -0.001), (0.0, 1.5708, -1.5708)),
    ((-0.1104, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ((-0.096, 0.0, 0.06462), (0.0, 0.0, -1.5708)),
    ((0.0, -0.07318, -0.001), (1.5708, -1.5708, 0.0)),
    ((0.0, 0.0456, 0.0), (-1.5708, 0.0, 0.0)),
)


def identity() -> Matrix:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def matmul(left: Matrix, right: Matrix) -> Matrix:
    return [
        [sum(left[row][k] * right[k][col] for k in range(4)) for col in range(4)]
        for row in range(4)
    ]


def translation(x: float, y: float, z: float) -> Matrix:
    result = identity()
    result[0][3] = x
    result[1][3] = y
    result[2][3] = z
    return result


def rotation_x(angle: float) -> Matrix:
    c, s = math.cos(angle), math.sin(angle)
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, c, -s, 0.0],
        [0.0, s, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def rotation_y(angle: float) -> Matrix:
    c, s = math.cos(angle), math.sin(angle)
    return [
        [c, 0.0, s, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-s, 0.0, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def rotation_z(angle: float) -> Matrix:
    c, s = math.cos(angle), math.sin(angle)
    return [
        [c, -s, 0.0, 0.0],
        [s, c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def origin_transform(xyz: Sequence[float], rpy: Sequence[float]) -> Matrix:
    # URDF fixed-axis RPY is Rz(yaw) * Ry(pitch) * Rx(roll).
    rotation = matmul(
        rotation_z(float(rpy[2])),
        matmul(rotation_y(float(rpy[1])), rotation_x(float(rpy[0]))),
    )
    return matmul(translation(*map(float, xyz)), rotation)


def forward_frames(angles_deg: Sequence[float]) -> list[Matrix]:
    if len(angles_deg) != 6:
        raise ValueError("exactly six joint angles are required")
    transform = identity()
    frames = [transform]
    for angle_deg, (xyz, rpy) in zip(angles_deg, JOINT_ORIGINS):
        transform = matmul(transform, origin_transform(xyz, rpy))
        transform = matmul(transform, rotation_z(math.radians(float(angle_deg))))
        frames.append(transform)
    return frames


def xyz_mm(transform: Matrix) -> list[float]:
    return [round(transform[index][3] * 1000.0, 6) for index in range(3)]


def forward_kinematics(angles_deg: Sequence[float]) -> dict[str, object]:
    frames = forward_frames(angles_deg)
    points = [xyz_mm(frame) for frame in frames]
    flange = points[-1]
    return {
        "angles_deg": [float(value) for value in angles_deg],
        "joint_points_mm": points,
        "flange_xyz_mm": flange,
        "flange_radius_mm": round(math.hypot(flange[0], flange[1]), 6),
    }


def parse_angles(values: Iterable[str]) -> list[float]:
    result = [float(value) for value in values]
    if len(result) != 6:
        raise argparse.ArgumentTypeError("provide exactly six angles")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("angles", nargs=6)
    args = parser.parse_args()
    print(json.dumps(forward_kinematics(parse_angles(args.angles)), indent=2))


if __name__ == "__main__":
    main()
