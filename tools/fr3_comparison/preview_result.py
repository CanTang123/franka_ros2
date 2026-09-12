#!/usr/bin/env python3
"""Validate an imported record; optionally publish an RViz display (no execution)."""
import argparse
import json
import math
from pathlib import Path


JOINTS = [f"fr3_joint{i}" for i in range(1, 8)]


def trajectory_payload(record, time_scale=1.0):
    if not math.isfinite(time_scale) or time_scale < 1:
        raise ValueError("time_scale must be finite and >= 1")
    if record.get("schema") != "fr3_comparison_record_v1":
        raise ValueError("Unsupported record schema")
    if record.get("joint_names") != JOINTS or record.get("units") != {
            "position": "rad", "velocity": "rad/s", "time": "s"}:
        raise ValueError("Unexpected joint order or units")
    if record.get("mode") != "recorded_simulation_targets":
        raise ValueError("Expected a recorded simulation target sequence")
    points = [record["initial_q_rad"]] + record["target_q_rad"]
    if len(points) < 2 or any(len(p) != 7 or not all(math.isfinite(v) for v in p) for p in points):
        raise ValueError("Expected finite seven-joint positions")
    starts = record["target_start_times_s"]
    dt = record["control_dt_s"]
    if not math.isfinite(dt) or abs(dt - .05) > 1e-9 or len(starts) != len(points) - 1:
        raise ValueError("Invalid control times")
    if any(not math.isfinite(t) or abs(t - i * dt) > 1e-7 for i, t in enumerate(starts)):
        raise ValueError("Target start times must follow the 0.05s source grid")
    if not math.isfinite(record["duration_s"]) or abs(record["duration_s"] - len(starts) * dt) > 1e-7:
        raise ValueError("Duration does not match target count")
    return {"joint_names": JOINTS, "points": [
        {"positions": q, "time_from_start_s": i * dt * time_scale} for i, q in enumerate(points)]}


def publish_display(payload):
    # ROS imports are deliberately optional for server-side validation.
    import rclpy
    from rclpy.qos import QoSProfile, DurabilityPolicy
    from moveit_msgs.msg import DisplayTrajectory, RobotTrajectory
    from trajectory_msgs.msg import JointTrajectoryPoint

    rclpy.init()
    node = rclpy.create_node("fr3_comparison_preview")
    qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    publisher = node.create_publisher(DisplayTrajectory, "/display_planned_path", qos)
    display = DisplayTrajectory()
    display.trajectory_start.joint_state.name = payload["joint_names"]
    display.trajectory_start.joint_state.position = payload["points"][0]["positions"]
    display.trajectory_start.is_diff = True
    trajectory = RobotTrajectory()
    trajectory.joint_trajectory.joint_names = payload["joint_names"]
    for p in payload["points"]:
        point = JointTrajectoryPoint()
        point.positions = p["positions"]
        ns = round(p["time_from_start_s"] * 1_000_000_000)
        point.time_from_start.sec, point.time_from_start.nanosec = divmod(ns, 1_000_000_000)
        trajectory.joint_trajectory.points.append(point)
    display.trajectory.append(trajectory)
    publisher.publish(display)
    node.get_logger().info("RViz preview published; no controller commands. Ctrl+C to exit.")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--rviz", action="store_true")
    parser.add_argument("--trajectory-json", type=Path)
    args = parser.parse_args()
    record = json.loads(args.record.read_text())
    payload = trajectory_payload(record, args.time_scale)
    print(json.dumps({"method": record["method"], "condition_id": record["condition_id"],
                      "source": record["source"], "points": len(payload["points"]),
                      "preview_duration_s": payload["points"][-1]["time_from_start_s"],
                      "hardware_connected": False,
                      "note": "Position interpolation preview; targets placed at control interval ends. "
                              "Not the original zero-order-hold rollout or a collision check."}))
    if args.trajectory_json:
        with args.trajectory_json.open("x") as out:
            json.dump(payload, out, indent=2, allow_nan=False)
            out.write("\n")
    if args.rviz:
        publish_display(payload)


if __name__ == "__main__":
    main()
