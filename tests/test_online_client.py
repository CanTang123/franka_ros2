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
            plan_timeout=.15, tracking_tolerance=.1, post_stop_seconds=1.,
            ee_record_rate=50., time_scale=1., move_group="fr3_arm",
            collision_retry_after=.02)
        node.started = time.monotonic()
        node.trial = {"duration_s": 8.8}
        node.record = lambda *a, **kw: None
        node.get_logger = lambda: types.SimpleNamespace(warning=lambda *a: None)
        node.fresh_state = lambda: [0.] * 14
        node.pending = None
        return node

    def test_ee_recording_is_rate_limited(self):
        node = self.make()
        node.ee = None
        node.last_ee_recorded = None
        events = []
        node.recording = lambda: True
        node.record = lambda event, **values: events.append(event)
        msg = types.SimpleNamespace(
            pose=types.SimpleNamespace(
                position=types.SimpleNamespace(x=.1, y=.2, z=.3),
                orientation=types.SimpleNamespace(x=0., y=0., z=0., w=1.)),
            header=types.SimpleNamespace(
                stamp=types.SimpleNamespace(sec=1, nanosec=1), frame_id="base"))
        client_time = CLIENT.on_ee_pose.__globals__["time"]
        with patch.object(client_time, "monotonic", side_effect=[1., 1.001]):
            node.on_ee_pose(msg)
            msg.header.stamp.nanosec = 2
            node.on_ee_pose(msg)
        self.assertEqual(events, ["measured_ee"])

    def test_verified_controller_is_not_queried_again(self):
        node = self.make()
        node.args.execute = True
        node.controller_verified = True
        node.limits = ([-1.] * 7, [1.] * 7, 1.)
        node.next_request = time.monotonic() + 1.
        valid = Future()
        valid.set_result(types.SimpleNamespace(valid=True))
        node.pending = dict(
            sent_at=time.monotonic(), stage="collision", checks=[valid],
            current=[0.] * 7, prefix=[[.001] * 7],
            request={"request_id": 1}, reply={},
            inference_done_at=time.monotonic())
        node.controllers = types.SimpleNamespace(
            call_async=lambda *_: self.fail("controller manager queried again"))
        node.preview = types.SimpleNamespace(publish=lambda *_: None)
        node.make_trajectory = lambda *_: object()
        sent = []
        node.send_goal = lambda trajectory, sequence: sent.append(sequence)
        node.step()
        self.assertEqual(sent, [1])

    def test_safe_trajectory_carries_velocity_and_acceleration_knots(self):
        class Duration:
            sec = 0
            nanosec = 0
        class Point:
            def __init__(self):
                self.positions = []
                self.velocities = []
                self.accelerations = []
                self.time_from_start = Duration()
        class Trajectory:
            def __init__(self):
                self.joint_names = []
                self.points = []
        node = self.make()
        node.joints = [f"fr3_joint{i}" for i in range(1, 8)]
        positions = [[i*.001] * 7 for i in range(1, 5)]
        velocities = [[i*.01] * 7 for i in range(1, 5)]
        accelerations = [[i*.1] * 7 for i in range(1, 5)]
        with patch.dict(CLIENT.make_trajectory.__globals__, {
                "JointTrajectory": Trajectory, "JointTrajectoryPoint": Point}):
            trajectory = node.make_trajectory(
                [0.] * 7, positions, 1., [0.] * 7, velocities,
                [0.] * 7, accelerations,
                {"duration_s": .2, "terminal_position": [.03] * 7,
                 "terminal_velocity": [0.] * 7,
                 "terminal_acceleration": [0.] * 7})
        self.assertEqual(trajectory.points[0].velocities, [0.] * 7)
        self.assertEqual(trajectory.points[1].accelerations, accelerations[0])
        self.assertEqual(len(trajectory.points), 6)
        self.assertEqual(trajectory.points[-1].time_from_start.nanosec, 400_000_000)
        self.assertEqual(trajectory.points[-1].velocities, [0.] * 7)

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

    def test_scaled_validation_budget_allows_longer_ros_checks(self):
        node = self.make()
        node.args.time_scale = 3.
        node.pending = dict(sent_at=time.monotonic() - .2, future=Future(), stage="inference")
        node.tick()
        self.assertTrue(node.enabled)

    def test_unanswered_collision_sample_is_retried_once(self):
        node = self.make()
        node.args.time_scale = 3.
        node.args.execute = False
        original = Future()
        replacement = Future()
        calls = []
        node.validity = types.SimpleNamespace(
            call_async=lambda request: calls.append(request) or replacement)
        events = []
        node.record = lambda event, **values: events.append((event, values))
        node.pending = dict(
            sent_at=time.monotonic() - .2, stage="collision",
            checks=[original], check_positions=[[0.] * 7],
            request={"request_id": 7})
        request_type = types.SimpleNamespace(Request=lambda: types.SimpleNamespace(
            robot_state=types.SimpleNamespace(
                joint_state=types.SimpleNamespace(name=[], position=[]), is_diff=False),
            group_name=""))
        with patch.dict(CLIENT.step.__globals__, {"GetStateValidity": request_type}):
            node.tick()
        self.assertTrue(node.enabled)
        self.assertIs(node.pending["checks"][0], replacement)
        self.assertEqual(len(calls), 1)
        self.assertEqual(events[0][0], "collision_retry")

    def test_collision_retry_happens_before_unchanged_final_deadline(self):
        node = self.make()
        node.args.execute = False
        original = Future()
        replacement = Future()
        calls = []
        node.validity = types.SimpleNamespace(
            call_async=lambda request: calls.append(request) or replacement)
        node.pending = dict(
            sent_at=time.monotonic() - .08,
            collision_started_at=time.monotonic() - .03,
            stage="collision", checks=[original], check_positions=[[0.] * 7],
            request={"request_id": 8})
        request_type = types.SimpleNamespace(Request=lambda: types.SimpleNamespace(
            robot_state=types.SimpleNamespace(
                joint_state=types.SimpleNamespace(name=[], position=[]), is_diff=False),
            group_name=""))
        with patch.dict(CLIENT.step.__globals__, {"GetStateValidity": request_type}):
            node.tick()
        self.assertTrue(node.enabled)
        self.assertEqual(len(calls), 1)
        self.assertTrue(node.pending["collision_retried"])

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
