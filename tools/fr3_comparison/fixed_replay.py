#!/usr/bin/env python3
"""ROS-independent fixed trajectory loading and simulated-delay scheduling."""
from dataclasses import dataclass
import json
import math
from pathlib import Path


SCHEMA = "fr3_fixed_delay_replay_v1"
DELAYS_MS = (0, 10, 50, 100)


def _finite_vector(value, size, name):
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{name} must have {size} entries")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
           for x in value):
        raise ValueError(f"{name} must contain finite numbers")
    return [float(x) for x in value]


@dataclass(frozen=True)
class FixedSequence:
    method: str
    path: Path
    joint_names: tuple
    points: tuple
    velocities: tuple
    accelerations: tuple
    dt_s: float
    duration_s: float
    debug_slow5x: bool
    safe_smooth: bool
    continuous_smooth: bool
    continuous_max_speed: tuple
    continuous_max_acceleration: tuple

    @property
    def command_count(self):
        return len(self.points) - 1


def load_sequence(root, method, prefer_debug_slow5x=True):
    """Load the structured sequence; slow5x is mandatory when present and preferred."""
    if method not in "ABCD" or len(method) != 1:
        raise ValueError("method label must be A, B, C, or D")
    root = Path(root)
    continuous = root / method / "debug_continuous_smooth" / "joint_trajectory_points.json"
    safe = root / method / "debug_safe_smooth" / "joint_trajectory_points.json"
    slow = root / method / "debug_slow5x" / "joint_trajectory_points.json"
    normal = root / method / "joint_trajectory_points.json"
    path = (continuous if continuous.is_file() else safe if safe.is_file() else
            slow if prefer_debug_slow5x and slow.is_file() else normal)
    if not path.is_file():
        raise ValueError(f"trajectory file not found: {path}")
    data = json.loads(path.read_text())
    expected = ("fr3_continuous_smooth_sequence_v1" if path == continuous else
                "fr3_safe_smooth_sequence_v1" if path == safe else
                "fr3_derived_debug_sequence_v1" if path == slow else None)
    if expected and data.get("schema") != expected:
        raise ValueError(f"unexpected fixed-sequence schema in {path}")
    if data.get("method") != method:
        raise ValueError(f"trajectory method does not match directory label {method}")
    names = data.get("joint_names")
    if not isinstance(names, list) or len(names) != 7 or len(set(names)) != 7 or not all(names):
        raise ValueError("trajectory must declare seven unique joint names")
    dt = float(data.get("sample_dt_s", .05))
    duration = float(data.get("duration_s", -1))
    if not math.isfinite(dt) or abs(dt - .05) > 1e-9:
        raise ValueError("fixed replay requires 50 ms trajectory points")
    raw = data.get("points")
    if not isinstance(raw, list) or len(raw) < 5 or (len(raw) - 1) % 4:
        raise ValueError("trajectory needs one start plus a multiple of four targets")
    points, velocities, accelerations = [], [], []
    for index, row in enumerate(raw):
        if not isinstance(row, dict):
            raise ValueError(f"point {index} is not an object")
        stamp = row.get("time_from_start_s")
        if not isinstance(stamp, (int, float)) or not math.isfinite(stamp) or abs(stamp-index*dt) > 1e-8:
            raise ValueError(f"point {index} timestamp is not on the 50 ms grid")
        points.append(tuple(_finite_vector(row.get("positions"), 7, f"point {index}")))
        if path in (safe, continuous):
            velocities.append(tuple(_finite_vector(row.get("velocities"), 7, f"point {index} velocities")))
            accelerations.append(tuple(_finite_vector(row.get("accelerations"), 7, f"point {index} accelerations")))
    if abs(duration - (len(points)-1)*dt) > 1e-8:
        raise ValueError("duration does not match the final point")
    if path in (safe, continuous):
        if data.get("request_period_s") != .2 or data.get("targets_per_request") != 4:
            raise ValueError("smooth trajectory must declare four 50 ms targets per 0.2 s request")
    if path == safe:
        for index in range(0, len(points), 4):
            if max(map(abs, velocities[index] + accelerations[index])) > 1e-9:
                raise ValueError(f"safe trajectory block boundary {index} is not stationary")
        speed_coefficient = 1.875
        acceleration_coefficient = 10.0 / math.sqrt(3.0)
        computed_speed = [0.] * 7
        computed_acceleration = [0.] * 7
        for start in range(0, len(points)-1, 4):
            delta = [b-a for a, b in zip(points[start], points[start+4])]
            for offset in range(5):
                u = offset/4
                s = 10*u**3 - 15*u**4 + 6*u**5
                ds = 30*u**2 - 60*u**3 + 30*u**4
                dds = 60*u - 180*u**2 + 120*u**3
                for joint in range(7):
                    expected_position = points[start][joint] + delta[joint]*s
                    expected_velocity = delta[joint]*ds/.2
                    expected_acceleration = delta[joint]*dds/(.2**2)
                    if (abs(points[start+offset][joint]-expected_position) > 1e-9 or
                            abs(velocities[start+offset][joint]-expected_velocity) > 1e-9 or
                            abs(accelerations[start+offset][joint]-expected_acceleration) > 1e-8):
                        raise ValueError(
                            f"safe trajectory block {start//4} is not the declared minimum-jerk curve")
                computed_speed = [max(value, speed_coefficient*abs(change)/.2)
                                  for value, change in zip(computed_speed, delta)]
                computed_acceleration = [
                    max(value, acceleration_coefficient*abs(change)/(.2**2))
                    for value, change in zip(computed_acceleration, delta)]
    if path == continuous:
        if max(map(abs, velocities[0] + accelerations[0] +
                   velocities[-1] + accelerations[-1])) > 1e-8:
            raise ValueError("continuous trajectory endpoints must be stationary")
        deviation = data.get("maximum_source_deviation_rad")
        allowed = data.get("allowed_source_deviation_rad")
        if (not isinstance(deviation, (int, float)) or not math.isfinite(deviation) or
                not isinstance(allowed, (int, float)) or not math.isfinite(allowed) or
                allowed <= 0 or deviation < 0 or deviation > allowed + 1e-12):
            raise ValueError("continuous trajectory has invalid source-deviation metadata")
        if data.get("rolling_action_replacement") is not True:
            raise ValueError("continuous trajectory must declare rolling action replacement")
    if path in (safe, continuous):
        continuous_speed = tuple(_finite_vector(data.get("continuous_max_speed_rad_s"), 7,
                                                "continuous_max_speed_rad_s"))
        continuous_acceleration = tuple(_finite_vector(data.get("continuous_max_acceleration_rad_s2"), 7,
                                                       "continuous_max_acceleration_rad_s2"))
        if any(abs(row[j]) > continuous_speed[j] + 1e-9
               for row in velocities for j in range(7)):
            raise ValueError("smooth trajectory speed metadata is below a trajectory knot")
        if any(abs(row[j]) > continuous_acceleration[j] + 1e-8
               for row in accelerations for j in range(7)):
            raise ValueError("smooth trajectory acceleration metadata is below a trajectory knot")
        if path == safe and (
                any(abs(a-b) > 1e-9 for a, b in zip(continuous_speed, computed_speed)) or
                any(abs(a-b) > 1e-8 for a, b in zip(continuous_acceleration,
                                                    computed_acceleration))):
            raise ValueError("safe trajectory continuous-limit metadata does not match its points")
    else:
        velocities, accelerations = [], []
        continuous_speed, continuous_acceleration = (), ()
    return FixedSequence(method, path.resolve(), tuple(names), tuple(points), tuple(velocities),
                         tuple(accelerations), dt, duration, path == slow, path == safe,
                         path == continuous,
                         continuous_speed, continuous_acceleration)


def load_site_config(path, require_confirmed=False):
    """Read explicit site facts. Execution never accepts placeholders or implicit defaults."""
    path = Path(path)
    data = json.loads(path.read_text())
    if data.get("schema") != "fr3_fixed_replay_site_v1":
        raise ValueError("invalid site configuration schema")
    names = data.get("joint_names")
    if not isinstance(names, list) or len(names) != 7 or len(set(names)) != 7 or not all(names):
        raise ValueError("site config needs seven unique joint_names")
    data["acceleration_limits_rad_s2"] = _finite_vector(
        data.get("acceleration_limits_rad_s2"), 7, "acceleration_limits_rad_s2")
    data["position_lower_rad"] = _finite_vector(data.get("position_lower_rad"), 7, "position_lower_rad")
    data["position_upper_rad"] = _finite_vector(data.get("position_upper_rad"), 7, "position_upper_rad")
    if any(x <= 0 for x in data["acceleration_limits_rad_s2"]):
        raise ValueError("acceleration limits must be positive")
    if any(a >= b for a, b in zip(data["position_lower_rad"], data["position_upper_rad"])):
        raise ValueError("position lower limits must be below upper limits")
    for key in ("max_step_rad", "max_speed_rad_s"):
        if not isinstance(data.get(key), (int, float)) or not math.isfinite(data[key]) or data[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    required_strings = ("controller", "controller_manager", "joint_states", "robot_description",
                        "state_validity", "move_group", "ee_pose")
    missing = [key for key in required_strings if not isinstance(data.get(key), str) or not data[key]]
    confirmations = ("tool_transform", "payload", "collision_model")
    for key in confirmations:
        item = data.get(key)
        if not isinstance(item, dict) or not isinstance(item.get("description"), str):
            missing.append(key)
        elif require_confirmed and (item.get("verified") is not True or not item["description"].strip()):
            missing.append(key)
    if missing:
        raise ValueError("site configuration missing/unconfirmed: " + ", ".join(missing))
    return data


def motion_metrics(sequence, limits=None):
    velocities = [[(b[j]-a[j])/sequence.dt_s for j in range(7)]
                  for a, b in zip(sequence.points, sequence.points[1:])]
    accelerations = [[(b[j]-a[j])/sequence.dt_s for j in range(7)]
                     for a, b in zip(velocities, velocities[1:])]
    max_step = [max(abs(b[j]-a[j]) for a, b in zip(sequence.points, sequence.points[1:]))
                for j in range(7)]
    max_speed = ([max(abs(row[j]) for row in sequence.velocities) for j in range(7)]
                 if sequence.velocities else [max(abs(row[j]) for row in velocities) for j in range(7)])
    max_acceleration = ([max(abs(row[j]) for row in sequence.accelerations) for j in range(7)]
                        if sequence.accelerations else
                        [max(abs(row[j]) for row in accelerations) for j in range(7)])
    if sequence.continuous_max_speed:
        max_speed = [max(a, b) for a, b in zip(max_speed, sequence.continuous_max_speed)]
    if sequence.continuous_max_acceleration:
        max_acceleration = [max(a, b) for a, b in zip(max_acceleration,
                                                       sequence.continuous_max_acceleration)]
    acceleration_violations = []
    position_violations = []
    step_violations = []
    speed_violations = []
    if limits is not None:
        acceleration_limits = (limits["acceleration_limits_rad_s2"] if isinstance(limits, dict) else limits)
        acceleration_limits = _finite_vector(acceleration_limits, 7, "acceleration limits")
        acceleration_violations = [dict(index=i, joint=sequence.joint_names[i], measured=max_acceleration[i], limit=acceleration_limits[i])
                                   for i in range(7) if max_acceleration[i] > acceleration_limits[i] + 1e-9]
        if isinstance(limits, dict):
            lower, upper = limits["position_lower_rad"], limits["position_upper_rad"]
            for point_index, row in enumerate(sequence.points):
                for joint_index, value in enumerate(row):
                    if not lower[joint_index] < value < upper[joint_index]:
                        position_violations.append(dict(point_index=point_index,
                            joint=sequence.joint_names[joint_index], measured=value,
                            lower=lower[joint_index], upper=upper[joint_index]))
            step_violations = [dict(joint=sequence.joint_names[i], measured=max_step[i], limit=limits["max_step_rad"])
                               for i in range(7) if max_step[i] > limits["max_step_rad"] + 1e-9]
            speed_violations = [dict(joint=sequence.joint_names[i], measured=max_speed[i], limit=limits["max_speed_rad_s"])
                                for i in range(7) if max_speed[i] > limits["max_speed_rad_s"] + 1e-9]
    return dict(max_step_rad=max_step, max_speed_rad_s=max_speed,
                max_acceleration_rad_s2=max_acceleration,
                position_violations=position_violations, step_violations=step_violations,
                speed_violations=speed_violations, acceleration_violations=acceleration_violations)


def guard_acceleration(current, current_velocity, targets, acceleration_limits, dt_s=.05):
    current = _finite_vector(current, 7, "current position")
    previous_velocity = _finite_vector(current_velocity, 7, "current velocity")
    limits = _finite_vector(acceleration_limits, 7, "acceleration limits")
    previous = current
    for point_index, target in enumerate(targets, 1):
        target = _finite_vector(target, 7, f"target {point_index}")
        velocity = [(q-p)/dt_s for q, p in zip(target, previous)]
        acceleration = [abs(v-pv)/dt_s for v, pv in zip(velocity, previous_velocity)]
        if any(a > limit + 1e-9 for a, limit in zip(acceleration, limits)):
            joint_index = max(range(7), key=lambda i: acceleration[i]/limits[i])
            raise ValueError("Target acceleration exceeds confirmed deployment limits: "
                f"point={point_index}, joint={joint_index+1}, measured={acceleration[joint_index]:.6f} "
                f"rad/s^2, limit={limits[joint_index]:.6f}")
        previous, previous_velocity = target, velocity


def braking_tail(position, velocity, acceleration, limits, duration_s=.2):
    """Return a quintic fallback stop after a rolling four-point prefix."""
    position = _finite_vector(list(position), 7, "brake position")
    velocity = _finite_vector(list(velocity), 7, "brake velocity")
    acceleration = _finite_vector(list(acceleration), 7, "brake acceleration")
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("brake duration must be finite and positive")
    terminal = [q + v*duration_s/2 for q, v in zip(position, velocity)]
    if max(abs(a-b) for a, b in zip(position, terminal)) > limits["max_step_rad"] + 1e-9:
        raise ValueError("rolling fallback stop exceeds confirmed step limit")
    coefficients = []
    for q0, v0, a0, q1 in zip(position, velocity, acceleration, terminal):
        b0, b1, b2 = q0, v0*duration_s, .5*a0*duration_s**2
        p = q1-(b0+b1+b2)
        v = -(b1+2*b2)
        a = -2*b2
        coefficients.append((b0, b1, b2, 10*p-4*v+.5*a,
                             -15*p+7*v-a, 6*p-3*v+.5*a))

    def evaluate(u):
        q, dq, ddq = [], [], []
        for b0, b1, b2, b3, b4, b5 in coefficients:
            q.append(b0+b1*u+b2*u**2+b3*u**3+b4*u**4+b5*u**5)
            dq.append((b1+2*b2*u+3*b3*u**2+4*b4*u**3+5*b5*u**4)/duration_s)
            ddq.append((2*b2+6*b3*u+12*b4*u**2+20*b5*u**3)/duration_s**2)
        return q, dq, ddq

    dense = [evaluate(i/200) for i in range(201)]
    lower, upper = limits["position_lower_rad"], limits["position_upper_rad"]
    acceleration_limits = limits["acceleration_limits_rad_s2"]
    if any(not lower[j] < row[0][j] < upper[j]
           for row in dense for j in range(7)):
        raise ValueError("rolling fallback stop leaves confirmed position limits")
    if any(abs(row[1][j]) > limits["max_speed_rad_s"] + 1e-9
           for row in dense for j in range(7)):
        raise ValueError("rolling fallback stop exceeds confirmed speed limit")
    if any(abs(row[2][j]) > .8*acceleration_limits[j] + 1e-9
           for row in dense for j in range(7)):
        raise ValueError("rolling fallback stop exceeds 80% of acceleration limit")
    samples = [evaluate(i/4)[0] for i in range(1, 5)]
    return dict(duration_s=duration_s, terminal_position=terminal,
                terminal_velocity=[0.]*7, terminal_acceleration=[0.]*7,
                collision_samples=samples)


class ReplayCursor:
    """Transactional sequential cursor: at most one reserved four-point block."""
    def __init__(self, sequence):
        self.sequence = sequence
        self.next_index = 1
        self._reserved = None
        self.request_id = 0

    @property
    def done(self):
        return self.next_index >= len(self.sequence.points)

    def reserve(self):
        if self._reserved is not None:
            raise ValueError("a fixed replay request is already pending")
        if self.done:
            raise StopIteration
        end = self.next_index + 4
        if end > len(self.sequence.points):
            raise ValueError("incomplete final four-point block")
        self.request_id += 1
        self._reserved = dict(request_id=self.request_id, start_index=self.next_index,
                              end_index=end-1, targets=[list(q) for q in self.sequence.points[self.next_index:end]])
        return dict(self._reserved)

    def commit(self, request_id):
        if self._reserved is None or self._reserved["request_id"] != request_id:
            raise ValueError("stale/mismatched fixed replay response")
        self.next_index = self._reserved["end_index"] + 1
        self._reserved = None

    def abort(self, request_id):
        if self._reserved is None or self._reserved["request_id"] != request_id:
            raise ValueError("stale/mismatched fixed replay abort")
        self._reserved = None


def simulated_reply(reservation, delay_ms, sequence):
    if delay_ms not in DELAYS_MS:
        raise ValueError("simulated delay must be one of 0, 10, 50, 100 ms")
    # Preserve the existing H16 shape. Only the first four sequential targets are
    # authorized; the tail is look-ahead and final-point padding is never executed.
    start = reservation["start_index"]
    targets = [list(q) for q in sequence.points[start:min(start+16, len(sequence.points))]]
    targets.extend([list(sequence.points[-1])] * (16-len(targets)))
    target_velocities = []
    target_accelerations = []
    if sequence.safe_smooth or sequence.continuous_smooth:
        target_velocities = [list(q) for q in sequence.velocities[start:min(start+16, len(sequence.points))]]
        target_accelerations = [list(q) for q in sequence.accelerations[start:min(start+16, len(sequence.points))]]
        target_velocities.extend([[0.]*7 for _ in range(16-len(target_velocities))])
        target_accelerations.extend([[0.]*7 for _ in range(16-len(target_accelerations))])
    return dict(schema="fr3_online_v1", experiment_type=SCHEMA,
                request_id=reservation["request_id"], method=sequence.method,
                condition_id="fixed_sequence_delay_simulation", source=0,
                control_step=reservation["start_index"]-1,
                joint_names=list(sequence.joint_names), control_dt_s=sequence.dt_s,
                trajectory_start_index=reservation["start_index"],
                trajectory_end_index=reservation["end_index"],
                simulated_delay_ms=delay_ms, target_q_rad=targets,
                target_dq_rad_s=target_velocities, target_ddq_rad_s2=target_accelerations)
