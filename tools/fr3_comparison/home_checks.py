"""ROS-independent checks for planned trial-start positioning."""
import fcntl
import hashlib
import math
import os
from pathlib import Path
import tempfile

from protocol import JOINTS, vector


class ControlLease:
    """Same-host exclusion between our homing and comparison processes.

    This does not lock other ROS applications or hardware interfaces.
    """
    def __init__(self, controller):
        key = hashlib.sha256(controller.strip('/').encode()).hexdigest()[:16]
        path = Path(tempfile.gettempdir()) / f"fr3-control-{os.getuid()}-{key}.lock"
        self.fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(self.fd)
            self.fd = None
            raise RuntimeError("Another comparison/home process owns this controller; exit it first")

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


def trial_start(trial):
    if trial.get('joint_names') != JOINTS:
        raise ValueError('Trial joint names/order do not match FR3')
    q = vector(trial.get('initial_q_rad'), 7, 'trial initial_q_rad')
    dq = vector(trial.get('initial_dq_rad_s'), 7, 'trial initial_dq_rad_s')
    if max(map(abs, dq)) > 1e-6:
        raise ValueError('Homing supports stationary trial starts only')
    return q


def arrived(q, dq, target, tolerance=.01):
    return max(abs(a-b) for a, b in zip(q, target)) <= tolerance and max(map(abs, dq)) <= .05


def check_plan(names, points, current, target, lower, upper, speed):
    """Reject malformed, wrong-start/goal, out-of-limit or overspeed plans.

    MoveIt performs collision checking; this check does not prove continuous
    spline safety. Serialized points are in the planner's joint order.
    """
    if len(names) != 7 or set(names) != set(JOINTS) or len(points) < 2:
        raise ValueError('Expected a nonempty seven-joint MoveIt trajectory')
    order = [names.index(j) for j in JOINTS]
    rows = []
    previous_t = -1.
    for p in points:
        q = [vector(p['positions'], 7, 'planned positions')[i] for i in order]
        v = [vector(p['velocities'], 7, 'planned velocities')[i] for i in order]
        t = p['time_s']
        if not math.isfinite(t) or t < 0 or t <= previous_t:
            raise ValueError('Invalid trajectory timestamps')
        if any(not lo < x < hi for x, lo, hi in zip(q, lower, upper)):
            raise ValueError('Planned joint position outside URDF limits')
        if any(abs(x) > limit + 1e-6 for x, limit in zip(v, speed)):
            raise ValueError('Planned velocity exceeds homing limit')
        if rows and any(abs(a-b)/(t-previous_t) > limit + 1e-6
                        for a, b, limit in zip(q, rows[-1], speed)):
            raise ValueError('Planned segment exceeds homing speed limit')
        rows.append(q)
        previous_t = t
    if max(abs(a-b) for a, b in zip(rows[0], current)) > .01:
        raise ValueError('Planned start differs from measured state')
    if max(abs(a-b) for a, b in zip(rows[-1], target)) > .005:
        raise ValueError('Planned endpoint differs from trial start')
    if max(map(abs, points[-1]['velocities'])) > .01:
        raise ValueError('Planned endpoint is not stationary')
    return rows
