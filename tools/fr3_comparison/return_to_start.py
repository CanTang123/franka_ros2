#!/usr/bin/env python3
"""MoveIt trial-start positioning: plan by default, explicit --execute to move."""
import argparse
import json
import math
from pathlib import Path
import signal
import time
import xml.etree.ElementTree as ET

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger
from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.msg import Constraints, JointConstraint, DisplayTrajectory
from moveit_msgs.srv import GetStateValidity
from controller_manager_msgs.srv import ListControllers

from protocol import JOINTS, vector, collision_samples
from home_checks import ControlLease, trial_start, arrived, check_plan


class Home(Node):
    def __init__(self, args):
        super().__init__('fr3_return_to_start')
        self.args = args
        self.target = trial_start(json.loads(args.trial.read_text()))
        self.state = self.limits = None
        self.stopping = False
        self.handles, self.acceptances = [], []
        args.log.parent.mkdir(parents=True, exist_ok=True)
        self.log = args.log.open('x', buffering=1)
        self.create_subscription(JointState, args.joint_states, self.on_state, qos_profile_sensor_data)
        self.create_subscription(String, args.robot_description, self.on_description,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.planner = ActionClient(self, MoveGroup, args.move_action)
        # ``Node.executor`` is owned by rclpy.  Overwriting it with an
        # ActionClient makes rclpy.spin_once() treat the action client as an
        # executor and fail while trying to call add_node().
        self.trajectory_executor = ActionClient(self, ExecuteTrajectory, args.execute_action)
        self.validity = self.create_client(GetStateValidity, args.state_validity)
        self.controllers = self.create_client(ListControllers, args.controller_manager + '/list_controllers')
        self.preview = self.create_publisher(DisplayTrajectory, '/display_planned_path',
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_service(Trigger, '~/stop', self.stop_service)
        self.record('initialized', target_q_rad=self.target, execute=args.execute,
                    arrival_tolerance_rad=args.arrival_tolerance, trial=str(args.trial))

    def record(self, event, **values):
        self.log.write(json.dumps(dict(event=event, monotonic_s=time.monotonic(), **values),
                                  allow_nan=False) + '\n')

    def on_state(self, msg):
        try:
            if len(msg.name) != len(set(msg.name)):
                raise ValueError('Duplicate joint names')
            order = [msg.name.index(j) for j in JOINTS]
            q = vector([float(msg.position[i]) for i in order], 7, 'q')
            dq = vector([float(msg.velocity[i]) for i in order], 7, 'dq')
            stamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
            if stamp <= 0 or (self.state is not None and stamp <= self.state[3]):
                return
            self.state = q, dq, time.monotonic(), stamp
            self.record('measured_state', q_rad=q, dq_rad_s=dq, stamp_ns=stamp)
        except (ValueError, IndexError):
            self.state = None

    def on_description(self, msg):
        try:
            joints = {j.attrib['name']: j for j in ET.fromstring(msg.data).findall('joint')}
            limits = [joints[j].find('limit') for j in JOINTS]
            lo, hi, speed = ([float(v.attrib[k]) for v in limits] for k in ('lower', 'upper', 'velocity'))
            if not all(math.isfinite(v) for v in lo+hi+speed) or any(a >= b for a,b in zip(lo,hi)) or min(speed) <= 0:
                raise ValueError('Invalid limits')
            self.limits = lo, hi, speed
        except (ValueError, KeyError, AttributeError, ET.ParseError):
            self.limits = None

    def fresh(self):
        if self.state is None or time.monotonic() - self.state[2] > .1:
            raise ValueError('Missing/stale joint state (arrival >100ms)')
        age = (self.get_clock().now().nanoseconds - self.state[3]) / 1e9
        if not -.02 <= age <= .1:
            raise ValueError('Stale joint-state source timestamp')
        return self.state[:2]

    def spin(self):
        if self.stopping or not rclpy.ok():
            raise RuntimeError('Positioning stopped')
        rclpy.spin_once(self, timeout_sec=.01)
        if self.stopping:
            raise RuntimeError('Positioning stopped')

    def wait(self, future, seconds, monitor=False):
        deadline = time.monotonic() + seconds
        while not future.done():
            self.spin()
            if monitor:
                self.fresh()
            if time.monotonic() > deadline:
                raise TimeoutError('ROS request/action timed out')
        if self.stopping:
            raise RuntimeError('Positioning stopped')
        return future.result()

    def stop(self):
        self.stopping = True
        for handle in self.handles:
            handle.cancel_goal_async()

    def stop_service(self, req, res):
        self.stop()
        res.success, res.message = True, 'Cancellation requested; confirm physical stop'
        return res

    def action(self, client, goal, seconds, monitor=False):
        future = client.send_goal_async(goal)
        self.acceptances.append(future)
        def accepted(done):
            try:
                handle = done.result()
                if handle.accepted:
                    self.handles.append(handle)
                    if self.stopping:
                        handle.cancel_goal_async()
            except Exception:
                self.stop()
        future.add_done_callback(accepted)
        handle = self.wait(future, 5., monitor)
        if not handle.accepted:
            raise ValueError('MoveIt rejected action')
        result = self.wait(handle.get_result_async(), seconds, monitor)
        if result.status != 4 or result.result.error_code.val != 1:
            raise ValueError(f'MoveIt action failed: status={result.status}, code={result.result.error_code.val}')
        self.handles.remove(handle)
        return result.result

    def valid(self, q):
        req = GetStateValidity.Request()
        req.group_name = self.args.move_group
        req.robot_state.is_diff = True
        req.robot_state.joint_state.name = JOINTS
        req.robot_state.joint_state.position = q
        if not self.wait(self.validity.call_async(req), 2.).valid:
            raise ValueError('MoveIt collision/state validity check rejected configuration')

    def settle(self):
        until, stable = time.monotonic() + 5., None
        while time.monotonic() < until:
            self.spin()
            q, dq = self.fresh()
            if arrived(q, dq, self.target, tolerance=self.args.arrival_tolerance):
                stable = stable or time.monotonic()
                if time.monotonic() - stable >= .3:
                    self.record('at_start', q_rad=q, dq_rad_s=dq,
                                max_error_rad=max(abs(a-b) for a,b in zip(q,self.target)),
                                arrival_tolerance_rad=self.args.arrival_tolerance)
                    print(f'起点检查通过：各关节误差 ≤{self.args.arrival_tolerance:.3f} rad，'
                          '速度 ≤0.05 rad/s，持续 0.3 秒。', flush=True)
                    return
            else:
                stable = None
        q, dq = self.fresh()
        errors = [abs(a-b) for a,b in zip(q,self.target)]
        worst = max(range(7), key=errors.__getitem__)
        raise ValueError(
            'Measured trial-start position/stationarity check failed: '
            f'worst=fr3_joint{worst + 1}, error={errors[worst]:.6f} rad, '
            f'tolerance={self.args.arrival_tolerance:.6f} rad, '
            f'max_speed={max(map(abs, dq)):.6f} rad/s')

    def run(self):
        deadline = time.monotonic() + 15.
        while True:
            self.spin()
            try:
                q, dq = self.fresh()
                ready = self.limits is not None and self.validity.service_is_ready()
                if ready:
                    break
            except ValueError:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError('Need fresh /joint_states, /robot_description and MoveIt /check_state_validity')
        lo, hi, speed = self.limits
        if any(not a < x < b for x,a,b in zip(self.target,lo,hi)):
            raise ValueError('Trial start is outside robot URDF limits')
        self.valid(q)
        self.valid(self.target)
        if self.args.check_only:
            self.settle()
            return
        if max(map(abs,dq)) > .05:
            raise ValueError('Robot must be stationary before planning')
        if arrived(q, dq, self.target, tolerance=self.args.arrival_tolerance):
            self.settle()
            return
        if not self.planner.wait_for_server(timeout_sec=5.):
            raise ValueError('MoveGroup action unavailable')
        goal = MoveGroup.Goal()
        goal.request.group_name = self.args.move_group
        goal.request.num_planning_attempts = 1
        goal.request.allowed_planning_time = 5.
        goal.request.max_velocity_scaling_factor = min(.1, .3/max(speed))
        goal.request.max_acceleration_scaling_factor = .1
        goal.request.start_state.is_diff = True
        goal.request.start_state.joint_state.name = JOINTS
        goal.request.start_state.joint_state.position = q
        constraints = Constraints()
        for name, target in zip(JOINTS,self.target):
            c = JointConstraint()
            c.joint_name, c.position, c.weight = name, target, 1.
            c.tolerance_above = c.tolerance_below = .002
            constraints.joint_constraints.append(c)
        goal.request.goal_constraints = [constraints]
        goal.planning_options.plan_only = True
        goal.planning_options.replan = False
        planned = self.action(self.planner, goal, 15.)
        trajectory = planned.planned_trajectory.joint_trajectory
        if planned.planned_trajectory.multi_dof_joint_trajectory.points:
            raise ValueError('Unexpected multi-DOF trajectory')
        points = [dict(positions=list(p.positions), velocities=list(p.velocities),
                       time_s=p.time_from_start.sec+p.time_from_start.nanosec/1e9)
                  for p in trajectory.points]
        rows = check_plan(list(trajectory.joint_names), points, q, self.target, lo, hi,
                          [min(.3,v*.1) for v in speed])
        if points[-1]['time_s'] > 120.:
            raise ValueError('Planned positioning duration exceeds 120 seconds')
        self.record('planned', joint_names=list(trajectory.joint_names), points=points)
        preview = DisplayTrajectory()
        preview.trajectory_start = planned.trajectory_start
        preview.trajectory = [planned.planned_trajectory]
        self.preview.publish(preview)
        if not self.args.execute:
            print('仅规划完成，未执行。规划轨迹已写入日志并发布至 /display_planned_path。', flush=True)
            return
        if not self.trajectory_executor.wait_for_server(timeout_sec=5.) or not self.controllers.wait_for_service(timeout_sec=5.):
            raise ValueError('MoveIt execution/controller manager unavailable')
        controllers = self.wait(self.controllers.call_async(ListControllers.Request()), 3.)
        controller_name = self.args.controller.rstrip('/').rsplit('/', 1)[-1]
        if not any(c.name == controller_name and c.state == 'active'
                   and c.type == 'joint_trajectory_controller/JointTrajectoryController' for c in controllers.controller):
            raise ValueError('Required JointTrajectoryController is not active')
        # Recheck the current scene before sending the planned trajectory.
        samples = collision_samples(rows[0], rows[1:])
        if len(samples) > 10000:
            raise ValueError('Positioning path exceeds 10000 validation samples')
        validation_deadline = time.monotonic() + 30.
        for sample in samples:
            self.valid(sample)
            if time.monotonic() > validation_deadline:
                raise TimeoutError('Positioning path validation exceeded 30 seconds')
        current, velocity = self.fresh()
        if max(abs(a-b) for a,b in zip(current,q)) > .01 or max(map(abs,velocity)) > .05:
            raise ValueError('Robot moved after planning; run positioning again')
        execution = ExecuteTrajectory.Goal()
        execution.trajectory = planned.planned_trajectory
        execution.controller_names = [controller_name]
        self.record('execution_started')
        self.action(self.trajectory_executor, execution, points[-1]['time_s'] + 10., monitor=True)
        self.settle()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trial', type=Path, required=True)
    p.add_argument('--log', type=Path, required=True)
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--execute', action='store_true')
    modes.add_argument('--check-only', action='store_true')
    p.add_argument('--joint-states', default='/joint_states')
    p.add_argument('--robot-description', default='/robot_description')
    p.add_argument('--state-validity', default='/check_state_validity')
    p.add_argument('--move-group', default='fr3_arm')
    p.add_argument('--move-action', default='/move_action')
    p.add_argument('--execute-action', default='/execute_trajectory')
    p.add_argument('--controller', default='/fr3_arm_controller')
    p.add_argument('--controller-manager', default='/controller_manager')
    p.add_argument('--arrival-tolerance', type=float, default=.01,
                   help='Measured joint arrival tolerance in rad; maximum 0.03')
    args = p.parse_args()
    if not math.isfinite(args.arrival_tolerance) or not 0 < args.arrival_tolerance <= .03:
        p.error('--arrival-tolerance must be finite and within (0, 0.03] rad')
    lease = ControlLease(args.controller)
    from rclpy.signals import SignalHandlerOptions
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = None
    success = False
    try:
        node = Home(args)
        signal.signal(signal.SIGINT, lambda *_: node.stop())
        signal.signal(signal.SIGTERM, lambda *_: node.stop())
        node.run()
        success = True
    except Exception as error:
        print(f'回起点失败：{error}', flush=True)
        if node is not None:
            node.record('failed', reason=str(error))
    finally:
        if node is not None:
            node.stop()
            deadline = time.monotonic()+2.
            while rclpy.ok() and time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=.02)
            node.log.close()
            node.destroy_node()
        rclpy.shutdown()
        lease.close()
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
