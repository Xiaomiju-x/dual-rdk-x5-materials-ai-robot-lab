"""Focused pickup_flow action lifecycle tests without a ROS2 runtime."""
from __future__ import annotations

import importlib.util
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest import mock


def _stub_module(name: str, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


class _Dummy:
    def __init__(self, *args, **kwargs):
        pass


class _DispatchGoal:
    pass


class _DispatchTask:
    Goal = _DispatchGoal


def _load_cockpit_bridge():
    rclpy = _stub_module('rclpy')
    stubs = {
        'rclpy': rclpy,
        'rclpy.action': _stub_module('rclpy.action', ActionClient=_Dummy),
        'rclpy.callback_groups': _stub_module(
            'rclpy.callback_groups', ReentrantCallbackGroup=_Dummy
        ),
        'rclpy.node': _stub_module('rclpy.node', Node=_Dummy),
        'rclpy.qos': _stub_module(
            'rclpy.qos',
            QoSDurabilityPolicy=_Dummy,
            QoSProfile=_Dummy,
            ReliabilityPolicy=_Dummy,
        ),
        'geometry_msgs': _stub_module('geometry_msgs'),
        'geometry_msgs.msg': _stub_module(
            'geometry_msgs.msg', PoseWithCovarianceStamped=_Dummy, Twist=_Dummy
        ),
        'nav_msgs': _stub_module('nav_msgs'),
        'nav_msgs.msg': _stub_module(
            'nav_msgs.msg', OccupancyGrid=_Dummy, Odometry=_Dummy
        ),
        'sensor_msgs': _stub_module('sensor_msgs'),
        'sensor_msgs.msg': _stub_module(
            'sensor_msgs.msg', Image=_Dummy, LaserScan=_Dummy
        ),
        'std_msgs': _stub_module('std_msgs'),
        'std_msgs.msg': _stub_module('std_msgs.msg', Bool=_Dummy, String=_Dummy),
        'std_srvs': _stub_module('std_srvs'),
        'std_srvs.srv': _stub_module('std_srvs.srv', Trigger=_Dummy),
        'my_robot_msgs': _stub_module('my_robot_msgs'),
        'my_robot_msgs.msg': _stub_module(
            'my_robot_msgs.msg', Alarm=_Dummy, FurnaceReading=_Dummy, SystemTelemetry=_Dummy
        ),
        'my_robot_msgs.action': _stub_module(
            'my_robot_msgs.action', DispatchTask=_DispatchTask
        ),
        'my_robot_msgs.srv': _stub_module('my_robot_msgs.srv', VlmQuery=_Dummy),
        'requests': _stub_module('requests', RequestException=RuntimeError),
        'ai_msgs': None,
    }
    source = Path(__file__).resolve().parents[1] / 'my_robot_agents' / 'cockpit_bridge.py'
    spec = importlib.util.spec_from_file_location('_cockpit_bridge_pickup_target', source)
    module = importlib.util.module_from_spec(spec)
    stubs[spec.name] = module
    with mock.patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


COCKPIT = _load_cockpit_bridge()


class _ImmediateFuture:
    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value

    def add_done_callback(self, callback):
        callback(self)


class _DeferredFuture:
    def __init__(self):
        self._callback = None
        self._value = None

    def result(self):
        return self._value

    def add_done_callback(self, callback):
        self._callback = callback

    def resolve(self, value):
        self._value = value
        self._callback(self)


class _NonBlockingEvent:
    def __init__(self):
        self._set = False

    def set(self):
        self._set = True

    def wait(self, timeout=None):
        return self._set


class _GoalHandle:
    def __init__(self, result_future, accepted=True):
        self.accepted = accepted
        self.result_future = result_future
        self.cancel_calls = 0

    def get_result_async(self):
        return self.result_future

    def cancel_goal_async(self):
        self.cancel_calls += 1
        return _ImmediateFuture(types.SimpleNamespace(goals_canceling=[self]))


class _ActionClient:
    def __init__(self, goal_handles):
        self.goal_handles = list(goal_handles)
        self.feedback_callbacks = []
        self.goals = []
        self.wait_calls = 0

    def wait_for_server(self, timeout_sec):
        self.wait_calls += 1
        return True

    def send_goal_async(self, goal, feedback_callback):
        self.goals.append(goal)
        self.feedback_callbacks.append(feedback_callback)
        return _ImmediateFuture(self.goal_handles.pop(0))


class _DeferredAcceptActionClient:
    def __init__(self, goal_future):
        self.goal_future = goal_future
        self.feedback_callback = None

    def wait_for_server(self, timeout_sec):
        return True

    def send_goal_async(self, goal, feedback_callback):
        self.feedback_callback = feedback_callback
        return self.goal_future


def _result(
    success: bool,
    message: str,
    elapsed_s: float = 1.0,
    *,
    completion_class=None,
    actuator_sequence_completed=None,
    physical_completed=None,
    base_motion_requested=None,
    physical_confirmation=None,
):
    fields = {'success': success, 'message': message, 'elapsed_s': elapsed_s}
    optional = {
        'completion_class': completion_class,
        'actuator_sequence_completed': actuator_sequence_completed,
        'physical_completed': physical_completed,
        'base_motion_requested': base_motion_requested,
        'physical_confirmation': physical_confirmation,
    }
    fields.update({key: value for key, value in optional.items() if value is not None})
    result = types.SimpleNamespace(**fields)
    return types.SimpleNamespace(result=result)


def _feedback(stage: int, progress_pct: float, stage_message: str):
    feedback = types.SimpleNamespace(
        stage=stage, progress_pct=progress_pct, stage_message=stage_message
    )
    return types.SimpleNamespace(feedback=feedback)


def _make_bridge(action_client):
    bridge = object.__new__(COCKPIT.CockpitBridge)
    bridge._lock = threading.Lock()
    bridge._estop = False
    bridge._motion_busy = False
    bridge._pickup_generation = 0
    bridge._pickup_active_generation = None
    bridge._pickup_motion_generation = None
    bridge._pickup_goal_handle = None
    bridge._pickup_state = {
        'active': False,
        'state': 'idle',
        'flow_id': '',
        'task_id': '',
        'task_type': '',
        'bottle_id': '',
        'from_location': '',
        'to_location': '',
        'message': '',
        'error': '',
        'stage': 0,
        'progress_pct': 0.0,
        'stage_message': '',
        'elapsed_s': 0.0,
        'completion_class': '',
        'actuator_sequence_completed': False,
        'physical_completed': False,
        'physical_confirmation': '',
        'base_motion_requested': False,
        'updated_at': 0.0,
    }
    bridge._dispatch_cli = action_client
    bridge.blackbox_events = []
    bridge._bb_event = lambda kind, data: bridge.blackbox_events.append(
        {'k': kind, **dict(data)}
    )
    return bridge


class TestCockpitBridgePickupFlow(unittest.TestCase):
    def test_pickup_motion_reservation_is_atomic_and_generation_owned(self):
        bridge = _make_bridge(_ActionClient([]))

        first = bridge._begin_pickup_flow()
        self.assertIsInstance(first, int)
        self.assertIsNone(bridge._begin_pickup_flow())
        self.assertTrue(bridge._motion_busy)

        bridge._end_pickup_flow_motion(first)
        bridge._pickup_active_generation = None
        bridge._pickup_state['active'] = False
        second = bridge._begin_pickup_flow()
        self.assertIsInstance(second, int)
        bridge._end_pickup_flow_motion(first)
        self.assertTrue(bridge._motion_busy)
        bridge._end_pickup_flow_motion(second)
        self.assertFalse(bridge._motion_busy)

    def test_timeout_cancels_stored_goal_and_persists_terminal_fields(self):
        goal_handle = _GoalHandle(_DeferredFuture())
        bridge = _make_bridge(_ActionClient([goal_handle]))

        with mock.patch.object(COCKPIT.threading, 'Event', _NonBlockingEvent):
            result = bridge._cmd_pickup_flow(
                {'flow_id': 'flow-timeout', 'task_id': 'task-timeout', 'timeout_s': 5}
            )

        self.assertFalse(result['ok'])
        self.assertTrue(result['cancel_requested'])
        self.assertEqual(goal_handle.cancel_calls, 1)
        self.assertEqual(bridge._pickup_state['state'], 'timeout')
        self.assertEqual(bridge._pickup_state['completion_class'], 'timeout')
        self.assertFalse(bridge._pickup_state['actuator_sequence_completed'])
        self.assertFalse(bridge._pickup_state['physical_completed'])
        self.assertIsNone(bridge._pickup_goal_handle)
        self.assertIsNone(bridge._pickup_active_generation)
        terminal = [
            event for event in bridge.blackbox_events
            if event['k'] == 'pickup_flow' and event['state'] == 'timeout'
        ][-1]
        self.assertEqual(terminal['completion_class'], 'timeout')
        self.assertFalse(terminal['actuator_sequence_completed'])
        self.assertFalse(terminal['physical_completed'])

    def test_reported_completion_fields_reach_state_and_blackbox(self):
        result_future = _ImmediateFuture(
            _result(True, 'F407_REPORTED_COMPLETED: actuator sequence done', 2.5)
        )
        bridge = _make_bridge(_ActionClient([_GoalHandle(result_future)]))

        with mock.patch.object(COCKPIT.threading, 'Event', _NonBlockingEvent):
            result = bridge._cmd_pickup_flow(
                {'flow_id': 'flow-complete', 'task_id': 'task-complete'}
            )

        self.assertTrue(result['ok'])
        self.assertEqual(result['completion_class'], 'f407_reported')
        self.assertTrue(result['actuator_sequence_completed'])
        self.assertFalse(result['physical_completed'])
        self.assertEqual(bridge._pickup_state['completion_class'], 'f407_reported')
        self.assertTrue(bridge._pickup_state['actuator_sequence_completed'])
        self.assertFalse(bridge._pickup_state['physical_completed'])
        terminal = [
            event for event in bridge.blackbox_events
            if event['k'] == 'pickup_flow' and event['state'] == 'reported_completed'
        ][-1]
        self.assertEqual(terminal['completion_class'], 'f407_reported')
        self.assertTrue(terminal['actuator_sequence_completed'])
        self.assertFalse(terminal['physical_completed'])

    def test_structured_result_is_authoritative_over_message_prefix(self):
        result_future = _ImmediateFuture(
            _result(
                True,
                'structured result without a legacy prefix',
                1.5,
                completion_class='f407_reported',
                actuator_sequence_completed=True,
                physical_completed=False,
                base_motion_requested=False,
            )
        )
        bridge = _make_bridge(_ActionClient([_GoalHandle(result_future)]))

        with mock.patch.object(COCKPIT.threading, 'Event', _NonBlockingEvent):
            result = bridge._cmd_pickup_flow(
                {'flow_id': 'flow-structured', 'task_id': 'task-structured'}
            )

        self.assertTrue(result['ok'])
        self.assertEqual(result['completion_class'], 'f407_reported')
        self.assertTrue(result['actuator_sequence_completed'])
        self.assertFalse(result['physical_completed'])
        self.assertFalse(result['base_motion_requested'])
        self.assertFalse(bridge._pickup_state['base_motion_requested'])

    def test_structured_actuator_false_cannot_be_overridden_by_message(self):
        result_future = _ImmediateFuture(
            _result(
                True,
                'F407_REPORTED_COMPLETED: misleading legacy text',
                completion_class='f407_reported',
                actuator_sequence_completed=False,
                physical_completed=False,
            )
        )
        bridge = _make_bridge(_ActionClient([_GoalHandle(result_future)]))

        with mock.patch.object(COCKPIT.threading, 'Event', _NonBlockingEvent):
            result = bridge._cmd_pickup_flow(
                {'flow_id': 'flow-inconsistent', 'task_id': 'task-inconsistent'}
            )

        self.assertFalse(result['ok'])
        self.assertEqual(bridge._pickup_state['state'], 'failed')

    def test_physical_completion_requires_actuator_sequence(self):
        result_future = _ImmediateFuture(
            _result(
                True,
                'PHYSICAL_COMPLETED: inconsistent structured result',
                completion_class='physical',
                actuator_sequence_completed=False,
                physical_completed=True,
            )
        )
        bridge = _make_bridge(_ActionClient([_GoalHandle(result_future)]))

        with mock.patch.object(COCKPIT.threading, 'Event', _NonBlockingEvent):
            result = bridge._cmd_pickup_flow(
                {'flow_id': 'flow-physical-invalid', 'task_id': 'task-physical-invalid'}
            )

        self.assertFalse(result['ok'])
        self.assertEqual(bridge._pickup_state['state'], 'failed')

    def test_physical_confirmation_reaches_state_and_blackbox(self):
        confirmation = '{"confirmed":true,"schema_version":"xrd-pickup-physical-confirmation-v1"}'
        result_future = _ImmediateFuture(
            _result(
                True,
                'PHYSICAL_COMPLETED: independent evidence passed',
                completion_class='physical',
                actuator_sequence_completed=True,
                physical_completed=True,
                base_motion_requested=False,
                physical_confirmation=confirmation,
            )
        )
        bridge = _make_bridge(_ActionClient([_GoalHandle(result_future)]))

        with mock.patch.object(COCKPIT.threading, 'Event', _NonBlockingEvent):
            result = bridge._cmd_pickup_flow(
                {'flow_id': 'flow-physical-valid', 'task_id': 'task-physical-valid'}
            )

        self.assertTrue(result['ok'])
        self.assertEqual(result['completion_class'], 'physical')
        self.assertEqual(result['physical_confirmation'], confirmation)
        self.assertEqual(bridge._pickup_state['physical_confirmation'], confirmation)
        terminal = [
            event for event in bridge.blackbox_events
            if event['k'] == 'pickup_flow' and event['state'] == 'completed'
        ][-1]
        self.assertEqual(terminal['physical_confirmation'], confirmation)

    def test_stale_feedback_and_result_cannot_overwrite_newer_flow(self):
        old_result = _DeferredFuture()
        old_handle = _GoalHandle(old_result)
        new_result = _ImmediateFuture(_result(True, 'SIMULATED_ONLY: newer flow', 0.5))
        client = _ActionClient([old_handle, _GoalHandle(new_result)])
        bridge = _make_bridge(client)

        with mock.patch.object(COCKPIT.threading, 'Event', _NonBlockingEvent):
            first = bridge._cmd_pickup_flow(
                {'flow_id': 'reused-flow-id', 'task_id': 'old-task', 'timeout_s': 5}
            )
            second = bridge._cmd_pickup_flow(
                {'flow_id': 'reused-flow-id', 'task_id': 'new-task'}
            )

        self.assertEqual(first['completion_class'], 'timeout')
        self.assertEqual(old_handle.cancel_calls, 1)
        self.assertEqual(second['completion_class'], 'simulated')
        expected_state = dict(bridge._pickup_state)
        event_count = len(bridge.blackbox_events)

        client.feedback_callbacks[0](_feedback(99, 99.0, 'stale feedback'))
        old_result.resolve(_result(True, 'PHYSICAL_COMPLETED: stale result', 99.0))

        self.assertEqual(bridge._pickup_state, expected_state)
        self.assertEqual(bridge._pickup_state['task_id'], 'new-task')
        self.assertEqual(bridge._pickup_state['completion_class'], 'simulated')
        self.assertEqual(len(bridge.blackbox_events), event_count)

    def test_goal_accepted_after_timeout_is_cancelled_as_stale(self):
        delayed_accept = _DeferredFuture()
        bridge = _make_bridge(_DeferredAcceptActionClient(delayed_accept))

        with mock.patch.object(COCKPIT.threading, 'Event', _NonBlockingEvent):
            timed_out = bridge._cmd_pickup_flow(
                {'flow_id': 'late-accept', 'task_id': 'old-task', 'timeout_s': 5}
            )
            bridge._dispatch_cli = _ActionClient([
                _GoalHandle(_ImmediateFuture(_result(True, 'SIMULATED_ONLY: new flow')))
            ])
            newer = bridge._cmd_pickup_flow(
                {'flow_id': 'late-accept', 'task_id': 'new-task'}
            )

        self.assertEqual(timed_out['completion_class'], 'timeout')
        self.assertFalse(timed_out['cancel_requested'])
        self.assertEqual(newer['completion_class'], 'simulated')
        expected_state = dict(bridge._pickup_state)
        pickup_event_count = sum(
            event['k'] == 'pickup_flow' for event in bridge.blackbox_events
        )
        late_handle = _GoalHandle(_DeferredFuture())

        delayed_accept.resolve(late_handle)

        self.assertEqual(late_handle.cancel_calls, 1)
        self.assertEqual(bridge._pickup_state, expected_state)
        self.assertEqual(
            sum(event['k'] == 'pickup_flow' for event in bridge.blackbox_events),
            pickup_event_count,
        )
        self.assertEqual(bridge.blackbox_events[-1]['k'], 'pickup_flow_cancel')
        self.assertEqual(bridge.blackbox_events[-1]['reason'], 'stale_goal_accept')

    def test_estop_rejects_without_dispatch_and_remains_latched(self):
        client = _ActionClient([])
        bridge = _make_bridge(client)
        bridge._estop = True

        result = bridge._cmd_pickup_flow(
            {'flow_id': 'flow-estop', 'task_id': 'task-estop'}
        )

        self.assertFalse(result['ok'])
        self.assertIn('ESTOP', result['error'])
        self.assertTrue(bridge._estop)
        self.assertEqual(client.wait_calls, 0)
        self.assertEqual(bridge._pickup_state['state'], 'rejected')
        self.assertEqual(bridge._pickup_state['completion_class'], 'rejected')
        self.assertFalse(bridge._pickup_state['actuator_sequence_completed'])
        self.assertFalse(bridge._pickup_state['physical_completed'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
