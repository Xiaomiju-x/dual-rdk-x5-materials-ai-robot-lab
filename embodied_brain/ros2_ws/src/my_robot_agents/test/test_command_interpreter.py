"""test_command_interpreter.py — RuleInterpreter 单元测试.

跑法:
    cd ~/ros2_ws/src/my_robot_agents
    python3 test/test_command_interpreter.py
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))

from my_robot_agents.command_interpreter import RuleInterpreter  # noqa: E402


class TestRuleInterpreter(unittest.TestCase):
    def setUp(self):
        self.r = RuleInterpreter()

    def test_emergency_stop(self):
        for u in ['急停', '停下', '强制停车']:
            res = self.r.parse(u)
            self.assertTrue(res.success, u)
            self.assertEqual(res.task_type, 'home')
            self.assertEqual(res.priority, 3)

    def test_home(self):
        for u in ['回工位', '回去', '回 home']:
            res = self.r.parse(u)
            self.assertTrue(res.success, u)
            self.assertEqual(res.task_type, 'home')

    def test_patrol(self):
        for u in ['巡更', '巡视一圈', '走一圈']:
            res = self.r.parse(u)
            self.assertTrue(res.success, u)
            self.assertEqual(res.task_type, 'patrol')

    def test_monitor_furnace(self):
        for u in ['监控 1 号炉', '监控2号烧结炉', '监控三号炉子']:
            res = self.r.parse(u)
            self.assertTrue(res.success, u)
            self.assertEqual(res.task_type, 'monitor_furnace')
            self.assertTrue(res.to_location.startswith('furnace_'))

    def test_fetch_sample(self):
        cases = [
            ('去 1 号试剂柜取 SYGO1 号瓶',     'SYGO1', 'shelf_1'),
            ('去 2 号架子 取 YCAS-3 瓶',       'YCAS-3', 'shelf_2'),
            ('去取 3 号瓶',                    '3', ''),  # 没指定 shelf
        ]
        for u, expect_bottle, expect_shelf in cases:
            res = self.r.parse(u)
            self.assertTrue(res.success, f'{u} → {res}')
            self.assertEqual(res.task_type, 'fetch_sample', u)
            # bottle_id 非空 (具体格式宽松, 只要含目标关键字)
            self.assertTrue(expect_bottle.lower() in res.bottle_id.lower(),
                           f'{u}: expected {expect_bottle} in {res.bottle_id}')
            if expect_shelf:
                self.assertEqual(res.from_location, expect_shelf, u)

    def test_deliver_to_furnace(self):
        for u in [
            '把 SYGO-1 送到 1 号炉',
            '送 YCAS3 到 2 号烧结炉',
        ]:
            res = self.r.parse(u)
            self.assertTrue(res.success, u)
            self.assertEqual(res.task_type, 'deliver_to_furnace', u)
            self.assertTrue(res.to_location.startswith('furnace_'), u)

    def test_unknown(self):
        for u in ['今天天气怎么样', '帮我点杯奶茶']:
            res = self.r.parse(u)
            self.assertFalse(res.success, u)


if __name__ == '__main__':
    unittest.main(verbosity=2)
