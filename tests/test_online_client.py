"""Failure-path tests with fake ROS transport; does not claim ROS integration."""
from concurrent.futures import Future
import importlib.util
from pathlib import Path
import sys
import time
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1] / "tools/fr3_comparison"


def loaded_client():
    stubs = {}
    for name, entries in {
        "rclpy": {}, "rclpy.action": {"ActionClient": object}, "rclpy.node": {"Node": object},
        "rclpy.qos": {"qos_profile_sensor_data": None, "QoSProfile": object, "DurabilityPolicy": object},
        "control_msgs.action": {"FollowJointTrajectory": types.SimpleNamespace(Goal=types.SimpleNamespace)},
        "control_msgs.msg": {"JointTolerance": types.SimpleNamespace},
        "controller_manager_msgs.srv": {"ListControllers": object},
        "geometry_msgs.msg": {"PoseStamped": object},
        "moveit_msgs.srv": {"GetStateValidity": object}, "sensor_msgs.msg": {"JointState": object},
        "std_msgs.msg": {"String": object}, "std_srvs.srv": {"SetBool": object, "Trigger": object},
        "trajectory_msgs.msg": {"JointTrajectory": object, "JointTrajectoryPoint": object},
    }.items():
        mod = types.ModuleType(name)
        mod.__dict__.update(entries)
        stubs[name] = mod
    spec = importlib.util.spec_from_file_location("protocol", ROOT / "protocol.py")
    protocol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(protocol)
    stubs["protocol"] = protocol
    with patch.dict(sys.modules, stubs):
        spec = importlib.util.spec_from_file_location("online_client_test", ROOT / "online_client.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


CLIENT = loaded_client().Client


class Handle:
    accepted = True
    def __init__(self):
        self.cancelled = 0
    def cancel_goal_async(self):
        self.cancelled += 1
        f = Future()
        f.set_result(None)
        return f


class ClientFailureTests(unittest.TestCase):
    def make(self):
        node = CLIENT.__new__(CLIENT)
        node.enabled = True
        node.generation = 1
        node.goal = Handle()
        node.goal_sequence = None
        node.goal_futures = []
        node.awaiting_acceptance = None
        node.recording_until = None
        node.args = types.SimpleNamespace(
            plan_timeout=.15, tracking_tolerance=.1, post_stop_seconds=1.)
        node.started = time.monotonic()
        node.trial = {"duration_s": 8.8}
        node.record = lambda *a, **kw: None
        node.get_logger = lambda: types.SimpleNamespace(warning=lambda *a: None)
        node.fresh_state = lambda: [0.] * 14
        node.pending = None
        return node

    def test_stale_state_cancels_active_goal(self):
        node = self.make()
        old = node.goal
        def stale():
            raise ValueError("stale state")
        node.fresh_state = stale
        node.tick()
        self.assertFalse(node.enabled)
        self.assertEqual(old.cancelled, 1)

    def test_late_inference_cancels_active_goal(self):
        node = self.make()
        old = node.goal
        node.pending = dict(sent_at=time.monotonic() - .2, future=Future(), stage="inference")
        node.tick()
        self.assertFalse(node.enabled)
        self.assertEqual(old.cancelled, 1)

    def test_stop_before_goal_acceptance_cancels_late_goal(self):
        node = self.make()
        accepted = Future()
        node.action = types.SimpleNamespace(
            send_goal_async=lambda goal, feedback_callback=None: accepted)
        node.send_goal(object(), 10)
        node.stop("operator")
        late = Handle()
        accepted.set_result(late)
        self.assertEqual(late.cancelled, 1)
        self.assertFalse(node.enabled)

    def test_invalid_collision_never_sends_goal(self):
        node = self.make()
        old = node.goal
        checked = Future()
        checked.set_result(types.SimpleNamespace(valid=False))
        node.pending = dict(sent_at=time.monotonic(), future=Future(), stage="collision", checks=[checked])
        node.tick()
        self.assertFalse(node.enabled)
        self.assertEqual(old.cancelled, 1)

    def test_disabled_client_does_not_plan(self):
        node = self.make()
        node.enabled = False
        node.step()
        self.assertIsNone(node.pending)

    def test_external_cancellation_stops_even_with_zero_error_code(self):
        node = self.make()
        accepted = Future()
        result = Future()
        node.action = types.SimpleNamespace(
            send_goal_async=lambda goal, feedback_callback=None: accepted)
        node.send_goal(object(), 10)
        handle = Handle()
        handle.get_result_async = lambda: result
        accepted.set_result(handle)
        result.set_result(types.SimpleNamespace(status=5,
            result=types.SimpleNamespace(error_code=0, error_string="cancelled")))
        self.assertFalse(node.enabled)
        self.assertEqual(handle.cancelled, 1)


if __name__ == "__main__":
    unittest.main()
