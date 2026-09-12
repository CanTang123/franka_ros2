"""ROS-independent validation shared by the inference server and robot client."""
import math

SCHEMA = "fr3_online_v1"
JOINTS = [f"fr3_joint{i}" for i in range(1, 8)]


def vector(value, length, name):
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must have {length} entries")
    if any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) for x in value):
        raise ValueError(f"{name} must contain finite numbers")
    return value


def request(value):
    if value.get("schema") != SCHEMA or value.get("method") not in ("A", "B", "C", "D"):
        raise ValueError("Invalid schema/method")
    if value.get("joint_names") != JOINTS:
        raise ValueError("Joint order mismatch")
    vector(value.get("state"), 14, "state")
    vector(value.get("descriptor"), 19, "descriptor")
    if value["descriptor"][18] <= 0:
        raise ValueError("Task duration must be positive")
    vector([value.get("phase_time_s")], 1, "phase_time_s")
    if value["phase_time_s"] < 0:
        raise ValueError("Negative phase time")
    for key in ("request_id", "source", "control_step"):
        if type(value.get(key)) is not int or not 0 <= value[key] < 2**32:
            raise ValueError(f"Invalid {key}")
    if not isinstance(value.get("condition_id"), str) or not value["condition_id"]:
        raise ValueError("Missing condition_id")
    return value


def response(value, sent):
    if value.get("schema") != SCHEMA or value.get("request_id") != sent["request_id"]:
        raise ValueError("Stale/mismatched response")
    for key in ("method", "condition_id", "source", "control_step"):
        if value.get(key) != sent[key]:
            raise ValueError(f"Response {key} mismatch")
    if value.get("joint_names") != JOINTS or value.get("control_dt_s") != .05:
        raise ValueError("Response joint order or time step mismatch")
    actions = value.get("target_q_rad")
    if not isinstance(actions, list) or len(actions) != 16:
        raise ValueError("Expected the full H16 plan")
    for row in actions:
        vector(row, 7, "target_q_rad")
    return actions


def guard_prefix(actions, current, lower, upper, max_step, max_speed):
    vector(current, 7, "current joints")
    previous = current
    for row in actions:
        vector(row, 7, "target joints")
        if any(not low < q < high for low, q, high in zip(lower, row, upper)):
            raise ValueError("Joint target outside robot URDF limits")
        step = max(abs(q - p) for q, p in zip(row, previous))
        if step > max_step or step / .05 > max_speed:
            raise ValueError("Target discontinuity/speed exceeds deployment limits")
        previous = row


def collision_samples(current, actions, resolution=.01):
    """Include endpoints and intermediate joint-linear states, without smoothing."""
    result = [current]
    previous = current
    for row in actions:
        n = max(1, math.ceil(max(abs(q - p) for q, p in zip(row, previous)) / resolution))
        for i in range(1, n + 1):
            result.append([p + (q - p) * i / n for q, p in zip(row, previous)])
        previous = row
    return result
