#!/usr/bin/env python3

import math
import unittest

from mycobot280_fk import forward_kinematics


MEASURED = {
    "START": (
        [142.55, -142.03, 31.72, 138.6, 104.41, -50.97],
        [-77.5, 83.5, 71.5],
    ),
    "PICK": (
        [168.13, -140.88, 31.02, 91.14, 18.63, -59.15],
        [-210.0, 94.0, 70.1],
    ),
    "DISH_DROP": (
        [-148.79, -124.01, 47.72, 73.03, 68.81, -55.98],
        [-186.5, -88.5, 166.2],
    ),
    "LEFT_HANDLE": (
        [-170.33, -140.62, 29.53, 22.41, 112.5, -146.68],
        [-234.4, -18.3, 30.5],
    ),
}


class ForwardKinematicsTest(unittest.TestCase):
    def test_matches_recorded_arm01_positions(self):
        for name, (angles, measured) in MEASURED.items():
            with self.subTest(name=name):
                predicted = forward_kinematics(angles)["flange_xyz_mm"]
                error = math.dist(predicted, measured)
                self.assertLess(error, 8.0, (name, predicted, measured, error))

    def test_compact_pose_envelope(self):
        compact = [0.0, -13.63, 28.97, -18.66, 104.74, 0.0]
        result = forward_kinematics(compact)
        points = result["joint_points_mm"][2:]
        max_radius = max(math.hypot(point[0], point[1]) for point in points)
        min_height = min(point[2] for point in points)
        self.assertLess(max_radius, 65.0)
        self.assertGreater(min_height, 138.0)
        self.assertGreater(result["flange_xyz_mm"][2], 410.0)


if __name__ == "__main__":
    unittest.main()
