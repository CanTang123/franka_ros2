#!/usr/bin/env python3
"""FR3 measured-state replanning client. Starts disabled, with dry-run enabled."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTolerance
from controller_manager_msgs.srv import ListControllers
from moveit_msgs.srv import GetStateValidity
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from protocol import SCHEMA, JOINTS, vector, request, response, guard_prefix, collision_samples


class Client(Node):
    def __init__(self, args):
        super().__init__("fr3_comparison")
        self.args = args
        self.trial = json.loads(args.trial.read_text())
        vector(self.trial["initial_q_rad"], 7, "initial_q_rad")
        self.state = self.received = self.state_stamp = None
        self.limits = None
        self.enabled = False
        self.sequence = 0
        self.pending = None
        self.goal = None
        self.goal_sequence = None
        self.goal_futures = []
        self.awaiting_acceptance = None
        self.generation = 0
        self.run_id = 0
        self.recording_until = None
        self.ee = None
        self.report_processes = []
        self.pool = ThreadPoolExecutor(max_workers=1)
        args.log.parent.mkdir(parents=True, exist_ok=True)
        self.log = args.log.open("x", buffering=1)
        self.create_subscription(JointState, args.joint_states, self.on_state, qos_profile_sensor_data)
        self.create_subscription(PoseStamped, args.ee_pose, self.on_ee_pose, qos_profile_sensor_data)
        self.create_subscription(String, args.robot_description, self.on_description,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.preview = self.create_publisher(JointTrajectory, "~/reference_trajectory", 1)
        self.action = ActionClient(self, FollowJointTrajectory, args.controller + "/follow_joint_trajectory")
        self.validity = self.create_client(GetStateValidity, args.state_validity)
        self.controllers = self.create_client(ListControllers, args.controller_manager + "/list_controllers")
        self.create_service(SetBool, "~/enable", self.enable)
        self.create_service(Trigger, "~/stop", self.stop_service)
        self.create_timer(.01, self.tick)
        self.record("initialized", method=args.method, dry_run=not args.execute, trial=self.trial,
                    deployment=vars(args), protocol="H16; execute first4; 0.05s position-linear knots; request period0.2s")
        self.get_logger().info("Disabled. Inspect ~/reference_trajectory in dry-run; enable via ~/enable.")

    def record(self, event, **values):
        self.log.write(json.dumps(dict(event=event, monotonic_s=time.monotonic(),
            ros_time_ns=self.get_clock().now().nanoseconds, run_id=self.run_id, **values), default=str, allow_nan=False) + "\n")

    def recording(self):
        return self.enabled or (self.recording_until is not None and time.monotonic() <= self.recording_until)

    def on_ee_pose(self, msg):
        p, q = msg.pose.position, msg.pose.orientation
        values = [p.x, p.y, p.z, q.x, q.y, q.z, q.w]
        stamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
        try:
            vector(values, 7, "end effector pose")
            if stamp <= 0 or not msg.header.frame_id or sum(v * v for v in values[3:]) < 1e-12:
                raise ValueError("Missing EE timestamp/frame or invalid quaternion")
            if self.ee is not None and stamp <= self.ee["stamp_ns"]:
                return
            self.ee = dict(position_m=values[:3], quaternion_xyzw=values[3:],
                           stamp_ns=stamp, frame_id=msg.header.frame_id,
                           sample_monotonic_s=time.monotonic())
            if self.recording():
                self.record("measured_ee", **self.ee)
        except ValueError as error:
            if self.recording():
                self.record("telemetry_rejected", topic=self.args.ee_pose, reason=str(error))

    def finish_recording(self):
        if self.recording_until is None:
            return
        self.recording_until = None
        self.record("recording_complete")
        self.log.flush()
        if self.args.no_auto_report:
            return
        root = self.args.log.with_name(self.args.log.stem + "_reports")
        try:
            root.mkdir(parents=True, exist_ok=True)
            out = root / f"run_{self.run_id:03d}"
            report_log = root / f"run_{self.run_id:03d}_report.log"
            with report_log.open("x") as handle:
                process = subprocess.Popen([sys.executable,
                    str(Path(__file__).with_name("report_results.py")), str(self.args.log.resolve()),
                    "--run-id", str(self.run_id), "--output", str(out.resolve())],
                    stdout=handle, stderr=subprocess.STDOUT)
            self.report_processes.append(process)
            self.get_logger().info(f"Recording saved; generating report in {out} (log: {report_log})")
        except (OSError, ValueError) as error:
            self.get_logger().warning(f"Automatic report failed; raw data is saved: {error}")

    def on_description(self, msg):
        try:
            joints = {j.attrib["name"]: j for j in ET.fromstring(msg.data).findall("joint")}
            limits = [joints[name].find("limit") for name in JOINTS]
            lower = [float(v.attrib["lower"]) for v in limits]
            upper = [float(v.attrib["upper"]) for v in limits]
            speed = [float(v.attrib["velocity"]) for v in limits]
            if not all(math.isfinite(v) for v in lower + upper + speed) or any(l >= u for l, u in zip(lower, upper)):
                raise ValueError("Invalid robot limits")
            self.limits = lower, upper, min(speed)
        except Exception as error:
            self.limits = None
            self.stop(f"robot_description: {error}")

    def on_state(self, msg):
        try:
            if len(msg.name) != len(set(msg.name)):
                raise ValueError("Duplicate joint names")
            ids = [msg.name.index(j) for j in JOINTS]
            state = [msg.position[i] for i in ids] + [msg.velocity[i] for i in ids]
            vector(state, 14, "joint state")
            stamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
            if stamp <= 0:
                raise ValueError("Joint state needs a nonzero timestamp")
            if self.state_stamp is not None and stamp <= self.state_stamp:
                return  # Repeated/out-of-order messages cannot refresh freshness.
            self.state, self.received, self.state_stamp = state, time.monotonic(), stamp
            if self.recording():
                self.record("measured_state", state=state, stamp_ns=stamp, sample_monotonic_s=self.received)
        except (ValueError, IndexError) as error:
            self.stop(f"joint_states: {error}")

    def fresh_state(self):
        now = time.monotonic()
        if self.state is None or now - self.received > self.args.state_timeout:
            raise ValueError("Measured state missing/stale")
        age = (self.get_clock().now().nanoseconds - self.state_stamp) / 1e9
        if not -.02 <= age <= self.args.state_timeout:
            raise ValueError("Measured ROS timestamp stale or in the future")
        if max(abs(v) for v in self.state[7:]) > self.args.max_speed:
            raise ValueError("Measured velocity limit exceeded")
        return list(self.state)

    def enable(self, req, res):
        if not req.data:
            self.stop("disabled by operator")
            res.success = True
            return res
        try:
            if self.enabled or self.recording_until is not None or self.pending is not None or any(not f.done() for f in self.goal_futures):
                raise ValueError("Busy; wait for pending operations to finish")
            state = self.fresh_state()
            if self.limits is None:
                raise ValueError("Waiting for robot_description with seven joint limits")
            if max(abs(q - p) for q, p in zip(state[:7], self.trial["initial_q_rad"])) > self.args.start_tolerance:
                raise ValueError("Measured state differs from declared trial start; position robot before enabling")
            if max(abs(v) for v in state[7:]) > .05:
                raise ValueError("Robot must be stationary at trial start")
            if not self.validity.service_is_ready():
                raise ValueError("MoveIt state validity service unavailable")
            if self.args.execute and (not self.action.server_is_ready() or not self.controllers.service_is_ready()):
                raise ValueError("Controller action/list_controllers unavailable")
            self.enabled = True
            self.run_id += 1
            self.generation += 1
            self.started = self.next_request = time.monotonic()
            self.control_step = 0
            self.record("enabled")
            self.record("measured_state", state=state, stamp_ns=self.state_stamp, sample_monotonic_s=self.received)
            if self.ee is not None and time.monotonic() - self.ee["sample_monotonic_s"] <= self.args.state_timeout:
                self.record("measured_ee", **self.ee)
            else:
                self.record("telemetry_missing", topic=self.args.ee_pose, reason="No fresh EE pose at enable")
            res.success, res.message = True, "Enabled" if self.args.execute else "Enabled dry-run"
        except ValueError as error:
            res.success, res.message = False, str(error)
        return res

    def stop_service(self, req, res):
        self.stop("operator stop")
        res.success, res.message = True, "Disabled; action cancellation requested"
        return res

    def stop(self, reason):
        was_enabled = self.enabled
        self.enabled = False
        self.generation += 1
        if self.goal is not None:
            self.goal_futures.append(self.goal.cancel_goal_async())
            self.goal = None
        if was_enabled:
            self.record("stopped", reason=reason)
            self.recording_until = time.monotonic() + self.args.post_stop_seconds
            self.get_logger().warning(reason)

    def http_plan(self, value):
        req = Request(self.args.server.rstrip("/") + "/plan",
                      data=json.dumps(value, allow_nan=False).encode(), headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=self.args.plan_timeout) as handle:
            data = handle.read(65537)
            if len(data) > 65536:
                raise ValueError("Oversized inference reply")
            return json.loads(data)

    def tick(self):
        try:
            self.step()
        except Exception as error:
            self.stop(str(error))

    def step(self):
        now = time.monotonic()
        if not self.enabled:
            if self.recording_until is not None and now >= self.recording_until:
                self.finish_recording()
            if self.pending and self.pending["future"].done():
                self.pending = None
            return
        self.fresh_state()
        if self.awaiting_acceptance and now - self.awaiting_acceptance[1] > .1:
            raise ValueError("Controller acceptance deadline missed")
        if now - self.started >= self.trial["duration_s"]:
            self.stop("trial duration reached")
            return
        if self.pending:
            item = self.pending
            if now - item["sent_at"] > self.args.plan_timeout:
                raise ValueError("Planning/communication/collision-check deadline missed")
            if item["stage"] == "inference" and item["future"].done():
                reply = item["future"].result()
                actions = response(reply, item["request"])
                # Only four targets are ever authorized in one action goal.
                prefix = actions[:4]
                current = self.fresh_state()[:7]
                low, high, speed = self.limits
                guard_prefix(prefix, current, low, high, self.args.max_step, min(speed, self.args.max_speed))
                checks = []
                for q in collision_samples(current, prefix):
                    check = GetStateValidity.Request()
                    check.robot_state.joint_state.name = JOINTS
                    check.robot_state.joint_state.position = q
                    check.robot_state.is_diff = True
                    check.group_name = self.args.move_group
                    checks.append(self.validity.call_async(check))
                item.update(stage="collision", reply=reply, prefix=prefix, checks=checks, current=current)
                if self.args.execute:
                    item["controller_check"] = self.controllers.call_async(ListControllers.Request())
            if item["stage"] == "collision" and all(f.done() for f in item["checks"]):
                if not all(f.result().valid for f in item["checks"]):
                    raise ValueError("MoveIt rejected a sampled state in the four-step prefix")
                if self.args.execute:
                    if not item["controller_check"].done():
                        return
                    controllers = item["controller_check"].result().controller
                    expected = self.args.controller.rsplit("/", 1)[-1]
                    active = [c for c in controllers if c.name == expected and c.state == "active"
                              and c.type == "joint_trajectory_controller/JointTrajectoryController"]
                    if len(active) != 1:
                        raise ValueError("Expected JointTrajectoryController is not active")
                # Collision samples began at item.current; reject appreciable drift.
                current = self.fresh_state()[:7]
                if max(abs(q - p) for q, p in zip(current, item["current"])) > .01:
                    raise ValueError("Robot moved during collision validation")
                trajectory = self.make_trajectory(item["current"], item["prefix"])
                self.preview.publish(trajectory)
                self.record("plan_ready", request=item["request"], response=item["reply"],
                            latency_s=now - item["sent_at"], applied_prefix=item["prefix"], dry_run=not self.args.execute)
                if self.args.execute:
                    self.send_goal(trajectory, item["request"]["request_id"])
                self.pending = None
        if self.pending is None and now >= self.next_request:
            # Do not catch up by sending bursts or alter the physical phase.
            if now - self.next_request > .05:
                raise ValueError("0.2s request schedule deadline missed")
            state = self.fresh_state()
            self.sequence += 1
            value = request(dict(schema=SCHEMA, request_id=self.sequence, joint_names=JOINTS,
                method=self.args.method, state=state, descriptor=self.trial["path_descriptor"],
                condition_id=self.trial["condition_id"], source=self.trial["source"],
                control_step=self.control_step, phase_time_s=now - self.started))
            self.pending = dict(request=value, sent_at=now, future=self.pool.submit(self.http_plan, value), stage="inference")
            self.control_step += 4
            self.next_request += .2

    @staticmethod
    def make_trajectory(current, prefix):
        trajectory = JointTrajectory()
        trajectory.joint_names = JOINTS
        for i, positions in enumerate([current] + prefix):
            point = JointTrajectoryPoint()
            point.positions = positions
            point.time_from_start.nanosec = i * 50_000_000
            trajectory.points.append(point)
        return trajectory

    def send_goal(self, trajectory, sequence):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory
        goal.path_tolerance = [JointTolerance(name=j, position=self.args.tracking_tolerance) for j in JOINTS]
        epoch = self.generation
        self.goal_sequence = sequence
        def feedback(msg):
            if epoch != self.generation or not self.recording():
                return
            data = msg.feedback
            try:
                ids = [data.joint_names.index(name) for name in JOINTS]
                values = {}
                for name in ("desired", "actual", "error"):
                    point = getattr(data, name)
                    positions = [point.positions[i] for i in ids]
                    vector(positions, 7, name)
                    values[name + "_q_rad"] = positions
                self.record("controller_feedback", sequence=sequence, **values)
            except (ValueError, IndexError) as error:
                self.record("telemetry_rejected", topic="controller_feedback", reason=str(error))
        future = self.action.send_goal_async(goal, feedback_callback=feedback)
        self.goal_futures.append(future)
        sent = time.monotonic()
        self.awaiting_acceptance = sequence, sent
        def accepted(f):
            try:
                if self.awaiting_acceptance and self.awaiting_acceptance[0] == sequence:
                    self.awaiting_acceptance = None
                handle = f.result()
                if not handle.accepted:
                    if epoch == self.generation:
                        self.stop("Controller rejected trajectory")
                    return
                if epoch != self.generation or sequence != self.goal_sequence or not self.enabled or time.monotonic() - sent > .1:
                    self.goal_futures.append(handle.cancel_goal_async())
                    if epoch == self.generation:
                        self.stop("Controller acceptance deadline missed")
                    return
                self.goal = handle
                result = handle.get_result_async()
                self.goal_futures.append(result)
                def finished(done):
                    try:
                        result = done.result()
                        self.record("controller_result", sequence=sequence, status=result.status,
                                    error_code=result.result.error_code, error_string=result.result.error_string)
                        # A newer goal intentionally preempts its predecessor.
                        if epoch == self.generation and sequence == self.goal_sequence and (result.status != 4 or result.result.error_code != 0):
                            self.stop("Active controller goal failed")
                    except Exception as error:
                        self.stop(f"Controller result failed: {error}")
                result.add_done_callback(finished)
            except Exception as error:
                self.stop(f"Controller send failed: {error}")
        future.add_done_callback(accepted)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--method", choices=list("ABCD"), required=True)
    parser.add_argument("--log", type=Path, required=True, help="New JSONL log file")
    parser.add_argument("--ee-pose", default="/franka_robot_state_broadcaster/current_pose")
    parser.add_argument("--post-stop-seconds", type=float, default=1., help="Record stopping motion after disabling")
    parser.add_argument("--no-auto-report", action="store_true", help="Save raw data; generate CSV/plots manually later")
    parser.add_argument("--server", default="http://127.0.0.1:8765")
    parser.add_argument("--execute", action="store_true", help="Allow controller commands after explicit enable")
    parser.add_argument("--joint-states", default="/joint_states")
    parser.add_argument("--robot-description", default="/robot_description")
    parser.add_argument("--controller", default="/fr3_arm_controller")
    parser.add_argument("--controller-manager", default="/controller_manager")
    parser.add_argument("--state-validity", default="/check_state_validity")
    parser.add_argument("--move-group", default="fr3_arm")
    parser.add_argument("--plan-timeout", type=float, default=.15)
    parser.add_argument("--state-timeout", type=float, default=.1)
    parser.add_argument("--start-tolerance", type=float, default=.03)
    parser.add_argument("--tracking-tolerance", type=float, default=.1)
    parser.add_argument("--max-step", type=float, default=.03)
    parser.add_argument("--max-speed", type=float, default=.6)
    args = parser.parse_args()
    for key in ("plan_timeout", "state_timeout", "start_tolerance", "tracking_tolerance", "max_step", "max_speed"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error(f"{key} must be finite and positive")
    if args.plan_timeout >= .2:
        parser.error("plan_timeout must be below the 0.2s replanning period")
    if not math.isfinite(args.post_stop_seconds) or not 0 <= args.post_stop_seconds <= 10:
        parser.error("post_stop_seconds must be within [0, 10]")
    # Keep ROS alive long enough to request action cancellation on shutdown.
    from home_checks import ControlLease
    lease = ControlLease(args.controller)
    from rclpy.signals import SignalHandlerOptions
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = Client(args)
    exiting = False
    def interrupt(signum, frame):
        nonlocal exiting
        exiting = True
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    try:
        while rclpy.ok() and not exiting:
            rclpy.spin_once(node, timeout_sec=.02)
    finally:
        node.stop("process shutdown")
        deadline = time.monotonic() + max(2., args.post_stop_seconds + .1)
        while rclpy.ok() and time.monotonic() < deadline and (node.recording_until is not None or any(not f.done() for f in node.goal_futures)):
            rclpy.spin_once(node, timeout_sec=.02)
        node.finish_recording()
        node.pool.shutdown(wait=True, cancel_futures=True)
        node.log.close()
        node.destroy_node()
        rclpy.shutdown()
        lease.close()
        for process in node.report_processes:
            try:
                process.wait(timeout=30)
                if process.returncode:
                    print("Report generation failed; see *_report.log. Raw JSONL is preserved.", file=sys.stderr)
            except subprocess.TimeoutExpired:
                print("Report generation is still running; see *_reports/.", file=sys.stderr)


if __name__ == "__main__":
    main()
