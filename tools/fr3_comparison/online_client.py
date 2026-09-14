#!/usr/bin/env python3
"""FR3 measured-state replanning client. Starts disabled, with dry-run enabled."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
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
try:
    from fixed_replay import (DELAYS_MS, ReplayCursor, braking_tail, guard_acceleration,
                              load_sequence, load_site_config, motion_metrics,
                              simulated_reply)
except ModuleNotFoundError:  # Supports source-tree importlib tests without altering sys.path.
    import importlib.util
    _fixed_spec = importlib.util.spec_from_file_location(
        "fixed_replay", Path(__file__).with_name("fixed_replay.py"))
    _fixed_module = importlib.util.module_from_spec(_fixed_spec)
    sys.modules[_fixed_spec.name] = _fixed_module
    _fixed_spec.loader.exec_module(_fixed_module)
    DELAYS_MS = _fixed_module.DELAYS_MS
    ReplayCursor = _fixed_module.ReplayCursor
    braking_tail = _fixed_module.braking_tail
    load_sequence = _fixed_module.load_sequence
    load_site_config = _fixed_module.load_site_config
    motion_metrics = _fixed_module.motion_metrics
    simulated_reply = _fixed_module.simulated_reply
    guard_acceleration = _fixed_module.guard_acceleration


class Client(Node):
    def __init__(self, args):
        super().__init__("fr3_comparison")
        self.args = args
        self.fixed_sequence = getattr(args, "fixed_sequence_data", None)
        self.joints = list(args.site_data["joint_names"]) if self.fixed_sequence else JOINTS
        self.cursor = ReplayCursor(self.fixed_sequence) if self.fixed_sequence else None
        self.trial = (dict(initial_q_rad=list(self.fixed_sequence.points[0]),
                           duration_s=self.fixed_sequence.duration_s,
                           path_descriptor=[0.] * 18 + [self.fixed_sequence.duration_s],
                           condition_id="fixed_sequence_delay_simulation", source=0)
                      if self.fixed_sequence else json.loads(args.trial.read_text()))
        vector(self.trial["initial_q_rad"], 7, "initial_q_rad")
        self.state = self.received = self.state_stamp = None
        self.limits = None
        self.enabled = False
        self.sequence = 0
        self.pending = None
        self.goal = None
        self.goal_sequence = None
        self.goal_active = False
        self.goal_futures = []
        self.awaiting_acceptance = None
        self.generation = 0
        self.run_id = 0
        self.recording_until = None
        self.ee = None
        self.last_ee_recorded = None
        self.controller_verified = False
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
        experiment = "simulated_delay_fixed_replay" if self.fixed_sequence else "online_model_replanning"
        deployment = {key: value for key, value in vars(args).items()
                      if key not in ("site_data", "fixed_sequence_data")}
        self.record("initialized", method=(f"fixed-{args.method}" if self.fixed_sequence else args.method),
                    trajectory_label=args.method, experiment_type=experiment,
                    model_called=not bool(self.fixed_sequence), simulated_delay_ms=(args.simulated_delay_ms if self.fixed_sequence else None),
                    fixed_sequence_path=(str(self.fixed_sequence.path) if self.fixed_sequence else None),
                    fixed_sequence_debug_slow5x=(self.fixed_sequence.debug_slow5x if self.fixed_sequence else None),
                    fixed_sequence_safe_smooth=(self.fixed_sequence.safe_smooth if self.fixed_sequence else None),
                    fixed_sequence_continuous_smooth=(self.fixed_sequence.continuous_smooth if self.fixed_sequence else None),
                    site_config=(args.site_data if self.fixed_sequence else None),
                    dry_run=not args.execute, trial=self.trial,
                    deployment=deployment, protocol=(
                        "fixed H16 lookahead; execute sequential first4; no model; simulated delay; "
                        + ("50ms global-C2 position/velocity/acceleration knots; controlled rolling action replacement; 200ms request period"
                           if self.fixed_sequence.continuous_smooth else
                           "50ms stationary-block position/velocity/acceleration quintic knots; 200ms request period"
                           if self.fixed_sequence.safe_smooth else
                           "50ms position-linear knots; 200ms request period") if self.fixed_sequence else
                        "H16; execute first4; position-linear knots; "
                        f"time_scale={args.time_scale:g}; point_dt={.05 * args.time_scale:g}s; "
                        f"request_period={.2 * args.time_scale:g}s"))
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
            sample_time = time.monotonic()
            self.ee = dict(position_m=values[:3], quaternion_xyzw=values[3:],
                           stamp_ns=stamp, frame_id=msg.header.frame_id,
                           sample_monotonic_s=sample_time)
            period = 1. / self.args.ee_record_rate
            if self.recording() and (self.last_ee_recorded is None or
                                     sample_time - self.last_ee_recorded >= period):
                self.record("measured_ee", **self.ee)
                self.last_ee_recorded = sample_time
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
                report_env = os.environ.copy()
                # ROS deployment uses declared system Python dependencies; ignore an
                # incompatible user-site NumPy/Matplotlib pair during auto-reporting.
                report_env["PYTHONNOUSERSITE"] = "1"
                process = subprocess.Popen([sys.executable,
                    str(Path(__file__).with_name("report_results.py")), str(self.args.log.resolve()),
                    "--run-id", str(self.run_id), "--output", str(out.resolve())],
                    stdout=handle, stderr=subprocess.STDOUT, env=report_env)
            self.report_processes.append(process)
            self.get_logger().info(f"Recording saved; generating report in {out} (log: {report_log})")
        except (OSError, ValueError) as error:
            self.get_logger().warning(f"Automatic report failed; raw data is saved: {error}")

    def on_description(self, msg):
        try:
            joints = {j.attrib["name"]: j for j in ET.fromstring(msg.data).findall("joint")}
            limits = [joints[name].find("limit") for name in self.joints]
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
            ids = [msg.name.index(j) for j in self.joints]
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
            if self.fixed_sequence:
                # Every run starts at index 1 after independently rechecking the measured start.
                self.cursor = ReplayCursor(self.fixed_sequence)
                metrics = motion_metrics(self.fixed_sequence, self.args.site_data)
                self.record("offline_motion_validation", metrics=metrics,
                            passed=not any(metrics[key] for key in ("position_violations", "step_violations",
                                "speed_violations", "acceleration_violations")))
                if self.args.execute and any(metrics[key] for key in ("position_violations", "step_violations",
                        "speed_violations", "acceleration_violations")):
                    raise ValueError("Fixed sequence exceeds confirmed motion limits; see offline validation")
            self.enabled = True
            self.run_id += 1
            self.generation += 1
            self.controller_verified = False
            self.last_ee_recorded = None
            self.started = self.next_request = time.monotonic()
            self.control_step = 0
            self.record("enabled")
            self.record("measured_state", state=state, stamp_ns=self.state_stamp, sample_monotonic_s=self.received)
            if self.ee is not None and time.monotonic() - self.ee["sample_monotonic_s"] <= self.args.state_timeout:
                self.record("measured_ee", **self.ee)
                self.last_ee_recorded = time.monotonic()
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
        if self.fixed_sequence:
            time.sleep(self.args.simulated_delay_ms / 1000.)
            return simulated_reply(value, self.args.simulated_delay_ms, self.fixed_sequence)
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
        fixed_sequence = getattr(self, "fixed_sequence", None)
        joint_names = getattr(self, "joints", JOINTS)
        if not self.enabled:
            if self.recording_until is not None and now >= self.recording_until:
                self.finish_recording()
            if self.pending and self.pending["future"].done():
                self.pending = None
            return
        self.fresh_state()
        if self.awaiting_acceptance and now - self.awaiting_acceptance[1] > .1:
            raise ValueError("Controller acceptance deadline missed")
        if not fixed_sequence and now - self.started >= self.trial["duration_s"] * self.args.time_scale:
            self.stop("trial duration reached")
            return
        if self.pending:
            item = self.pending
            elapsed = now - item["sent_at"]
            collision_started = item.get("collision_started_at", item["sent_at"])
            if (item["stage"] == "collision" and
                    now - collision_started > self.args.collision_retry_after and
                    not item.get("collision_retried")):
                unresolved = [i for i, future in enumerate(item["checks"]) if not future.done()]
                if unresolved:
                    for i in unresolved:
                        check = GetStateValidity.Request()
                        check.robot_state.joint_state.name = joint_names
                        check.robot_state.joint_state.position = item["check_positions"][i]
                        check.robot_state.is_diff = True
                        check.group_name = self.args.move_group
                        item["checks"][i] = self.validity.call_async(check)
                    item["collision_retried"] = True
                    self.record("collision_retry", sequence=item["request"]["request_id"],
                                unresolved_samples=len(unresolved), elapsed_s=elapsed,
                                collision_elapsed_s=now-collision_started,
                                remaining_budget_s=max(0.,
                                    self.args.plan_timeout*self.args.time_scale-elapsed))
            validation_budget = self.args.plan_timeout * self.args.time_scale
            if elapsed > validation_budget:
                blocked = item["stage"]
                if blocked == "collision" and all(f.done() for f in item["checks"]):
                    check = item.get("controller_check")
                    blocked = "controller-check" if check is not None and not check.done() else "validation"
                raise ValueError(
                    f"Planning deadline missed during {blocked}: "
                    f"elapsed={elapsed:.6f}s, budget={validation_budget:.6f}s")
            if item["stage"] == "inference" and item["future"].done():
                reply = item["future"].result()
                item["inference_done_at"] = time.monotonic()
                if fixed_sequence:
                    expected = item["request"]
                    if (reply.get("experiment_type") != "fr3_fixed_delay_replay_v1" or
                            reply.get("request_id") != expected["request_id"] or
                            reply.get("method") != self.args.method or
                            reply.get("condition_id") != "fixed_sequence_delay_simulation" or
                            reply.get("source") != 0 or
                            reply.get("control_step") != expected["start_index"]-1 or
                            reply.get("trajectory_start_index") != expected["start_index"] or
                            reply.get("trajectory_end_index") != expected["end_index"] or
                            reply.get("joint_names") != list(fixed_sequence.joint_names)):
                        raise ValueError("Stale/mismatched fixed replay response")
                    actions = reply["target_q_rad"]
                    if not isinstance(actions, list) or len(actions) != 16:
                        raise ValueError("Expected the full fixed H16 plan")
                    for row in actions:
                        vector(row, 7, "target_q_rad")
                    if actions[:4] != expected["targets"]:
                        raise ValueError("Fixed replay response changed the reserved targets")
                    if fixed_sequence.safe_smooth or fixed_sequence.continuous_smooth:
                        target_velocities = reply.get("target_dq_rad_s")
                        target_accelerations = reply.get("target_ddq_rad_s2")
                        if (not isinstance(target_velocities, list) or len(target_velocities) != 16 or
                                not isinstance(target_accelerations, list) or len(target_accelerations) != 16):
                            raise ValueError("Safe fixed reply needs full H16 velocity and acceleration")
                        for row in target_velocities:
                            vector(row, 7, "target_dq_rad_s")
                        for row in target_accelerations:
                            vector(row, 7, "target_ddq_rad_s2")
                else:
                    actions = response(reply, item["request"])
                # Only four targets are ever authorized in one action goal.
                prefix = actions[:4]
                measured_state = self.fresh_state()
                current = (list(fixed_sequence.points[item["request"]["start_index"]-1])
                           if fixed_sequence and (not self.args.execute or fixed_sequence.safe_smooth or
                                                  fixed_sequence.continuous_smooth)
                           else measured_state[:7])
                low, high, speed = self.limits
                point_dt = .05 * self.args.time_scale
                guard_prefix(prefix, current, low, high,
                             self.args.max_step * self.args.time_scale,
                             min(speed, self.args.max_speed), control_dt=point_dt)
                if (fixed_sequence and self.args.execute and not fixed_sequence.safe_smooth and
                        not fixed_sequence.continuous_smooth):
                    guard_acceleration(current, measured_state[7:], prefix,
                                       self.args.site_data["acceleration_limits_rad_s2"], point_dt)
                check_positions = collision_samples(current, prefix)
                brake = None
                if (fixed_sequence and fixed_sequence.continuous_smooth and
                        item["request"]["end_index"] < fixed_sequence.command_count):
                    brake = braking_tail(prefix[-1], target_velocities[3],
                                         target_accelerations[3], self.args.site_data)
                    check_positions.extend(brake["collision_samples"])
                checks = []
                for q in check_positions:
                    check = GetStateValidity.Request()
                    check.robot_state.joint_state.name = joint_names
                    check.robot_state.joint_state.position = q
                    check.robot_state.is_diff = True
                    check.group_name = self.args.move_group
                    checks.append(self.validity.call_async(check))
                item.update(stage="collision", reply=reply, prefix=prefix, checks=checks,
                            collision_started_at=time.monotonic(),
                            check_positions=check_positions, current=current,
                            measured_at_validation=measured_state[:7], braking_tail=brake)
                if fixed_sequence.safe_smooth or fixed_sequence.continuous_smooth:
                    start = item["request"]["start_index"]
                    item["start_velocity"] = list(fixed_sequence.velocities[start-1])
                    item["start_acceleration"] = list(fixed_sequence.accelerations[start-1])
                    item["prefix_velocities"] = target_velocities[:4]
                    item["prefix_accelerations"] = target_accelerations[:4]
                if self.args.execute and not self.controller_verified:
                    item["controller_check"] = self.controllers.call_async(ListControllers.Request())
            if item["stage"] == "collision" and all(f.done() for f in item["checks"]):
                if not all(f.result().valid for f in item["checks"]):
                    raise ValueError("MoveIt rejected a sampled state in the four-step prefix")
                item.setdefault("collision_done_at", time.monotonic())
                if self.args.execute and not self.controller_verified:
                    if not item["controller_check"].done():
                        return
                    controllers = item["controller_check"].result().controller
                    expected = self.args.controller.rsplit("/", 1)[-1]
                    active = [c for c in controllers if c.name == expected and c.state == "active"
                              and c.type == "joint_trajectory_controller/JointTrajectoryController"]
                    if len(active) != 1:
                        raise ValueError("Expected JointTrajectoryController is not active")
                    if fixed_sequence:
                        claimed = getattr(active[0], "claimed_interfaces", [])
                        if not all(any(interface.rsplit("/", 1)[0] == joint for interface in claimed)
                                   for joint in joint_names):
                            raise ValueError("Active controller does not claim every configured arm joint")
                    self.controller_verified = True
                    item["controller_done_at"] = time.monotonic()
                if (fixed_sequence and fixed_sequence.safe_smooth and self.args.execute and
                        self.goal_active):
                    # Keep exactly one action active. The completed block is stationary, so
                    # its successor can start from the identical C2 boundary without preemption.
                    return
                # Collision samples began at item.current; reject appreciable drift.
                state_now = self.fresh_state()
                current = state_now[:7]
                measured_at_validation = item.get("measured_at_validation", item["current"])
                if max(abs(q - p) for q, p in zip(current, measured_at_validation)) > .01:
                    raise ValueError("Robot moved during collision validation")
                if fixed_sequence and fixed_sequence.safe_smooth and self.args.execute:
                    boundary_error = max(abs(q-p) for q, p in zip(current, item["current"]))
                    if boundary_error > self.args.start_tolerance:
                        raise ValueError(
                            f"Measured state missed fixed block boundary: error={boundary_error:.6f} rad")
                    if max(abs(v) for v in state_now[7:]) > .05:
                        raise ValueError("Robot is not stationary at fixed block boundary")
                elif fixed_sequence and fixed_sequence.continuous_smooth and self.args.execute:
                    boundary_error = max(abs(q-p) for q, p in zip(current, item["current"]))
                    if boundary_error > self.args.start_tolerance:
                        raise ValueError(
                            f"Measured state missed rolling trajectory boundary: error={boundary_error:.6f} rad")
                trajectory = self.make_trajectory(
                    item["current"], item["prefix"], self.args.time_scale,
                    item.get("start_velocity"), item.get("prefix_velocities"),
                    item.get("start_acceleration"), item.get("prefix_accelerations"),
                    item.get("braking_tail"))
                self.preview.publish(trajectory)
                ready_at = time.monotonic()
                timings = {
                    "inference_reply_s": item["inference_done_at"] - item["sent_at"],
                    "collision_checks_s": item["collision_done_at"] - item["inference_done_at"],
                }
                if "controller_done_at" in item:
                    timings["controller_check_s"] = item["controller_done_at"] - item["collision_done_at"]
                self.record("plan_ready", request=item["request"], response=item["reply"],
                            latency_s=ready_at - item["sent_at"], stage_timings_s=timings,
                            request_to_send_s=ready_at-item["sent_at"],
                            simulated_delay_ms=(self.args.simulated_delay_ms if fixed_sequence else None),
                            trajectory_start_index=(item["request"].get("start_index") if fixed_sequence else None),
                            trajectory_end_index=(item["request"].get("end_index") if fixed_sequence else None),
                            applied_prefix=item["prefix"],
                            applied_velocity_prefix=item.get("prefix_velocities"),
                            applied_acceleration_prefix=item.get("prefix_accelerations"),
                            fallback_braking_tail=item.get("braking_tail"),
                            dry_run=not self.args.execute)
                if fixed_sequence:
                    self.cursor.commit(item["request"]["request_id"])
                if self.args.execute:
                    self.send_goal(trajectory, item["request"]["request_id"])
                self.pending = None
                if fixed_sequence and self.cursor.done and not self.args.execute:
                    self.stop("fixed trajectory dry-run complete")
                    return
        if self.pending is None and now >= self.next_request:
            if fixed_sequence and self.cursor.done:
                return
            # Do not catch up by sending bursts or alter the physical phase.
            if now - self.next_request > .05:
                raise ValueError(
                    f"{.2 * self.args.time_scale:g}s request schedule deadline missed")
            state = self.fresh_state()
            self.sequence += 1
            if fixed_sequence:
                value = self.cursor.reserve()
                value.update(schema=SCHEMA, experiment_type="fr3_fixed_delay_replay_v1",
                             method=self.args.method, joint_names=list(fixed_sequence.joint_names),
                             condition_id="fixed_sequence_delay_simulation", source=0,
                             control_step=value["start_index"]-1, state=state,
                             descriptor=self.trial["path_descriptor"],
                             phase_time_s=(value["start_index"]-1)*fixed_sequence.dt_s)
                self.record("fixed_request", request_id=value["request_id"],
                            trajectory_start_index=value["start_index"], trajectory_end_index=value["end_index"],
                            simulated_delay_ms=self.args.simulated_delay_ms)
            else:
                value = request(dict(schema=SCHEMA, request_id=self.sequence, joint_names=JOINTS,
                    method=self.args.method, state=state, descriptor=self.trial["path_descriptor"],
                    condition_id=self.trial["condition_id"], source=self.trial["source"],
                    control_step=self.control_step,
                    phase_time_s=(now - self.started) / self.args.time_scale))
            self.pending = dict(request=value, sent_at=now, future=self.pool.submit(self.http_plan, value), stage="inference")
            self.control_step += 4
            self.next_request += .2 * self.args.time_scale

    def make_trajectory(self, current, prefix, time_scale=1., start_velocity=None,
                        prefix_velocities=None, start_acceleration=None,
                        prefix_accelerations=None, tail=None):
        trajectory = JointTrajectory()
        trajectory.joint_names = self.joints
        for i, positions in enumerate([current] + prefix):
            point = JointTrajectoryPoint()
            point.positions = positions
            if prefix_velocities is not None:
                point.velocities = (start_velocity if i == 0 else prefix_velocities[i-1])
            if prefix_accelerations is not None:
                point.accelerations = (start_acceleration if i == 0 else
                                       prefix_accelerations[i-1])
            time_ns = round(i * .05 * time_scale * 1_000_000_000)
            point.time_from_start.sec = time_ns // 1_000_000_000
            point.time_from_start.nanosec = time_ns % 1_000_000_000
            trajectory.points.append(point)
        if tail is not None:
            point = JointTrajectoryPoint()
            point.positions = tail["terminal_position"]
            point.velocities = tail["terminal_velocity"]
            point.accelerations = tail["terminal_acceleration"]
            time_ns = round((.2*time_scale + tail["duration_s"])*1_000_000_000)
            point.time_from_start.sec = time_ns // 1_000_000_000
            point.time_from_start.nanosec = time_ns % 1_000_000_000
            trajectory.points.append(point)
        return trajectory

    def send_goal(self, trajectory, sequence):
        fixed = getattr(self, "fixed_sequence", None)
        if (fixed and not fixed.continuous_smooth and
                getattr(self, "goal_active", False)):
            raise ValueError("Previous trajectory goal is still active; refusing concurrent accumulation")
        if fixed and fixed.continuous_smooth and getattr(self, "goal_active", False):
            self.record("rolling_action_replacement", previous_sequence=self.goal_sequence,
                        replacement_sequence=sequence)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory
        goal.path_tolerance = [JointTolerance(name=j, position=self.args.tracking_tolerance)
                               for j in getattr(self, "joints", JOINTS)]
        epoch = self.generation
        self.goal_sequence = sequence
        def feedback(msg):
            if epoch != self.generation or not self.recording():
                return
            data = msg.feedback
            try:
                ids = [data.joint_names.index(name) for name in getattr(self, "joints", JOINTS)]
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
                self.goal_active = True
                result = handle.get_result_async()
                self.goal_futures.append(result)
                def finished(done):
                    try:
                        result = done.result()
                        self.record("controller_result", sequence=sequence, status=result.status,
                                    error_code=result.result.error_code, error_string=result.result.error_string)
                        if epoch == self.generation and sequence == self.goal_sequence:
                            self.goal_active = False
                        # A newer goal intentionally preempts its predecessor.
                        if epoch == self.generation and sequence == self.goal_sequence and (result.status != 4 or result.result.error_code != 0):
                            self.stop("Active controller goal failed")
                        elif (getattr(self, "fixed_sequence", None) and epoch == self.generation and
                              sequence == self.goal_sequence and self.cursor.done):
                            self.stop("fixed trajectory complete")
                        elif epoch == self.generation and sequence == self.goal_sequence:
                            self.goal = None
                    except Exception as error:
                        self.stop(f"Controller result failed: {error}")
                result.add_done_callback(finished)
            except Exception as error:
                self.stop(f"Controller send failed: {error}")
        future.add_done_callback(accepted)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path)
    parser.add_argument("--method", choices=list("ABCD"), required=True)
    parser.add_argument("--log", type=Path, required=True, help="New JSONL log file")
    parser.add_argument("--ee-pose", default="/franka_robot_state_broadcaster/current_pose")
    parser.add_argument("--ee-record-rate", type=float, default=50.,
                        help="Maximum EE samples written per second")
    parser.add_argument("--post-stop-seconds", type=float, default=1., help="Record stopping motion after disabling")
    parser.add_argument("--no-auto-report", action="store_true", help="Save raw data; generate CSV/plots manually later")
    parser.add_argument("--server", default="http://127.0.0.1:8765")
    parser.add_argument("--fixed-sequences", type=Path,
                        help="Use local fixed A/B/C/D data (prefers continuous, safe_smooth, then debug_slow5x); never call a model")
    parser.add_argument("--simulated-delay-ms", type=int, choices=DELAYS_MS, default=0)
    parser.add_argument("--site-config", type=Path,
                        help="Explicit fixed-replay ROS/tool/payload/collision configuration JSON")
    parser.add_argument("--execute", action="store_true", help="Allow controller commands after explicit enable")
    parser.add_argument("--joint-states", default="/joint_states")
    parser.add_argument("--robot-description", default="/robot_description")
    parser.add_argument("--controller", default="/fr3_arm_controller")
    parser.add_argument("--controller-manager", default="/controller_manager")
    parser.add_argument("--state-validity", default="/check_state_validity")
    parser.add_argument("--move-group", default="fr3_arm")
    parser.add_argument("--plan-timeout", type=float, default=.15)
    parser.add_argument("--collision-retry-after", type=float, default=.02,
                        help="Retry only unresolved collision samples after this stage time; total plan timeout is unchanged")
    parser.add_argument("--state-timeout", type=float, default=.1)
    parser.add_argument("--start-tolerance", type=float, default=.03)
    parser.add_argument("--tracking-tolerance", type=float, default=.1)
    parser.add_argument("--max-step", type=float, default=.03)
    parser.add_argument("--max-speed", type=float, default=.6)
    parser.add_argument("--time-scale", type=float, default=1.,
                        help="Slow model phase and trajectory timing uniformly; range [1, 5]")
    args = parser.parse_args()
    if args.fixed_sequences:
        if not args.site_config:
            parser.error("--fixed-sequences requires --site-config")
        try:
            args.site_data = load_site_config(args.site_config, require_confirmed=args.execute)
            args.fixed_sequence_data = load_sequence(args.fixed_sequences, args.method)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            parser.error(str(error))
        for key in ("controller", "controller_manager", "joint_states", "robot_description",
                    "state_validity", "move_group", "ee_pose"):
            setattr(args, key, args.site_data[key])
        args.max_step = args.site_data["max_step_rad"]
        args.max_speed = args.site_data["max_speed_rad_s"]
        if args.time_scale != 1.:
            parser.error("fixed sequence timing is stored in its 50 ms points; --time-scale must remain 1")
    else:
        args.site_data = None
        args.fixed_sequence_data = None
        if args.trial is None:
            parser.error("--trial is required unless --fixed-sequences is used")
    for key in ("plan_timeout", "collision_retry_after", "state_timeout", "start_tolerance",
                "tracking_tolerance", "max_step", "max_speed"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error(f"{key} must be finite and positive")
    if not math.isfinite(args.ee_record_rate) or not 1 <= args.ee_record_rate <= 200:
        parser.error("ee_record_rate must be within [1, 200] Hz")
    if not math.isfinite(args.time_scale) or not 1 <= args.time_scale <= 5:
        parser.error("time_scale must be within [1, 5]")
    if args.plan_timeout >= .2:
        parser.error("plan_timeout must be below the 0.2s replanning period")
    if args.collision_retry_after >= args.plan_timeout:
        parser.error("collision_retry_after must be below plan_timeout")
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
