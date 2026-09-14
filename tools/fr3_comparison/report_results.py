#!/usr/bin/env python3
"""Export one FR3 run to CSV, measured-motion summaries, and PNG/PDF figures."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def finite_vector(value, size):
    return isinstance(value, list) and len(value) == size and all(
        isinstance(x, (int, float)) and math.isfinite(x) for x in value)


def read_run(path, run_id):
    records, initialized, warnings = [], {}, []
    active = 0
    raw = Path(path).read_bytes()
    lines = raw.splitlines()
    for i, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if i == len(lines) - 1:
                warnings.append("Ignored truncated final JSONL record")
                break
            raise ValueError(f"Invalid JSONL at line {i + 1}")
        if row.get("event") == "initialized":
            initialized = row
        if row.get("event") == "enabled":
            active += 1
        if row.get("run_id", active) == run_id:
            records.append(row)
            if row.get("event") == "recording_complete":
                break  # Stable snapshot even while the next run appends to the log.
    start = next((r for r in records if r.get("event") == "enabled"), None)
    if start is None:
        raise ValueError(f"No enabled run {run_id} in {path}")
    return initialized, records, float(start["monotonic_s"]), warnings, hashlib.sha256(raw).hexdigest()


def write_csv(path, fields, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sample_base(row, start, stop):
    sample_time = row.get("sample_monotonic_s", row["monotonic_s"])
    return dict(time_s=sample_time - start, receipt_monotonic_s=sample_time,
                stamp_ns=row.get("stamp_ns", ""), receipt_ros_time_ns=row.get("ros_time_ns", ""),
                phase="post_stop" if stop is not None and sample_time > stop else "active")


def export_report(log, output, run_id=1, plots=True):
    init, records, start, warnings, digest = read_run(log, run_id)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    stopped = next((r for r in records if r["event"] == "stopped"), None)
    stop = stopped["monotonic_s"] if stopped else None
    joints, poses, feedback = [], [], []
    keys = [f"q{i}_rad" for i in range(1, 8)] + [f"dq{i}_rad_s" for i in range(1, 8)]
    for row in records:
        event = row.get("event")
        if event == "measured_state":
            if not finite_vector(row.get("state"), 14):
                warnings.append("Skipped invalid measured joint state")
                continue
            joints.append(dict(sample_base(row, start, stop), **dict(zip(keys, row["state"]))))
        elif event == "measured_ee":
            xyz, quat = row.get("position_m"), row.get("quaternion_xyzw")
            if not finite_vector(xyz, 3) or not finite_vector(quat, 4) or not row.get("frame_id"):
                warnings.append("Skipped invalid measured EE pose")
                continue
            if sum(v * v for v in quat) < 1e-12:
                warnings.append("Skipped zero quaternion")
                continue
            poses.append(dict(sample_base(row, start, stop), frame_id=row["frame_id"],
                              **dict(zip(["x_m", "y_m", "z_m", "qx", "qy", "qz", "qw"], xyz + quat))))
        elif event == "controller_feedback":
            names = ("desired", "actual", "error")
            if not all(finite_vector(row.get(name + "_q_rad"), 7) for name in names):
                warnings.append("Skipped invalid controller feedback")
                continue
            feedback.append(dict(time_s=row["monotonic_s"] - start, sequence=row["sequence"], **{
                f"{name}_q{i+1}_rad": row[name + "_q_rad"][i] for name in names for i in range(7)}))
    # Preserve actual observation order; gaps are visible rather than resampled.
    base = ["time_s", "receipt_monotonic_s", "stamp_ns", "receipt_ros_time_ns", "phase"]
    write_csv(out / "joint_states.csv", base + keys, joints)
    write_csv(out / "ee_pose.csv", base + ["frame_id", "x_m", "y_m", "z_m", "qx", "qy", "qz", "qw"], poses)
    write_csv(out / "controller_feedback.csv", ["time_s", "sequence"] + [
        f"{name}_q{i}_rad" for name in ("desired", "actual", "error") for i in range(1, 8)], feedback)
    plans = [r for r in records if r.get("event") == "plan_ready"]
    with (out / "plans.jsonl").open("w") as handle:
        for row in plans:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    latency_rows = [dict(request_id=r.get("request", {}).get("request_id", ""),
                         trajectory_start_index=r.get("trajectory_start_index", ""),
                         trajectory_end_index=r.get("trajectory_end_index", ""),
                         set_delay_ms=r.get("simulated_delay_ms", ""),
                         request_to_send_ms=1000*r.get("request_to_send_s", r.get("latency_s", 0)),
                         simulated_wait_ms=1000*r.get("stage_timings_s", {}).get("inference_reply_s", 0),
                         collision_checks_ms=1000*r.get("stage_timings_s", {}).get("collision_checks_s", 0))
                    for r in plans]
    write_csv(out / "delay_timing.csv", ["request_id", "trajectory_start_index", "trajectory_end_index",
              "set_delay_ms", "request_to_send_ms", "simulated_wait_ms", "collision_checks_ms"], latency_rows)
    if not joints:
        warnings.append("No measured joint samples")
    if not poses:
        warnings.append("No measured end effector poses; check --ee-pose and the broadcaster")
    frames = sorted({p["frame_id"] for p in poses})
    if len(frames) > 1:
        warnings.append("EE frame changed: frame-specific plots; no combined displacement")
    def coverage(rows):
        gaps = [b["time_s"] - a["time_s"] for a, b in zip(rows, rows[1:])]
        return dict(samples=len(rows), first_time_s=rows[0]["time_s"] if rows else None,
                    last_time_s=rows[-1]["time_s"] if rows else None,
                    max_gap_s=max(gaps) if gaps else None,
                    gaps_over_100ms=sum(g > .1 for g in gaps),
                    nonincreasing_receipt_times=sum(g <= 0 for g in gaps))
    joint_summary = []
    active_joints = [p for p in joints if p["phase"] == "active"]
    configured_names = (init.get("site_config") or {}).get(
        "joint_names", [f"fr3_joint{i}" for i in range(1, 8)])
    for i in range(1, 8):
        q, dq = f"q{i}_rad", f"dq{i}_rad_s"
        if joints:
            joint_summary.append(dict(joint=configured_names[i-1], initial_rad=joints[0][q],
                last_active_rad=active_joints[-1][q] if active_joints else None,
                final_recorded_rad=joints[-1][q], min_rad=min(r[q] for r in joints),
                max_rad=max(r[q] for r in joints), peak_abs_velocity_rad_s=max(abs(r[dq]) for r in joints)))
    def pose_summary(rows):
        if not rows:
            return None
        first, last = rows[0], rows[-1]
        xyz_keys = ["x_m", "y_m", "z_m"]
        delta = [last[k] - first[k] for k in xyz_keys]
        quat_keys = ["qx", "qy", "qz", "qw"]
        q0, q1 = [[r[k] for k in quat_keys] for r in (first, last)]
        dot = sum(a*b for a, b in zip(q0, q1)) / math.sqrt(sum(a*a for a in q0) * sum(b*b for b in q1))
        return dict(initial_position_m=[first[k] for k in xyz_keys], final_position_m=[last[k] for k in xyz_keys],
            final_quaternion_xyzw=q1, displacement_xyz_m=delta, displacement_norm_m=math.sqrt(sum(x*x for x in delta)),
            orientation_change_deg=math.degrees(2*math.acos(min(1., abs(dot)))),
            sampled_path_length_m=sum(math.sqrt(sum((b[k]-a[k])**2 for k in xyz_keys))
                for a, b in zip(rows, rows[1:]) if 0 < b["time_s"] - a["time_s"] <= .1),
            last_time_s=last["time_s"], last_stamp_ns=last["stamp_ns"],
            age_at_stop_s=(stop - start - last["time_s"]) if stop is not None and last["time_s"] <= stop-start else None)
    ee_summary = {}
    for frame in frames:
        rows = [r for r in poses if r["frame_id"] == frame]
        ee_summary[frame] = dict(active=pose_summary([r for r in rows if r["phase"] == "active"]),
                                 including_post_stop=pose_summary(rows))
    summary = dict(schema="fr3_motion_report_v1", run_id=run_id, method=init.get("method"),
        experiment_type=init.get("experiment_type"), model_called=init.get("model_called"),
        simulated_delay_ms=init.get("simulated_delay_ms"),
        dry_run=init.get("dry_run"), synthetic=init.get("synthetic", False),
        condition_id=init.get("trial", {}).get("condition_id"), source=init.get("trial", {}).get("source"),
        log=str(Path(log).resolve()), log_snapshot_sha256=digest,
        stop_reason=stopped.get("reason") if stopped else None,
        stop_time_s=(stop-start) if stop is not None else None,
        recording_complete=any(r["event"] == "recording_complete" for r in records),
        joint_coverage=coverage(joints), ee_coverage=coverage(poses), joints=joint_summary,
        end_effector=ee_summary, controller_feedback_samples=len(feedback), plans=len(plans),
        warnings=warnings, figures=[],
        interpretation="Measured motion, not task-success certification. EE is the broadcaster's O_T_EE in its stated frame, "
                       "not necessarily fr3_hand_tcp. Path length skips observation gaps >100ms. "
                       "Joint data is not resampled; controller desired positions are separate from model plans.")
    if plots:
        try:
            summary["figures"] = plot_motion(out, summary, joints, poses, feedback, latency_rows)
        except ImportError as error:
            summary["warnings"].append(f"Plot dependency missing: {error}; CSV/JSON saved. Install python3-matplotlib.")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (out / "README.txt").write_text(
        "This report contains measured joint/EE data. Time is local receipt time relative to enable; "
        "source ROS timestamps are retained in CSV. Dashed vertical lines mark disable; later samples show stopping motion.\n"
        "EE position is metres, quaternion is xyzw. Frames are never combined or silently transformed. "
        "Quaternion plots normalize and align signs only for visualization; CSV retains the received values.\n"
        "Missing data/aborted/dry-run trials remain visible in summary.json. No endpoint error to the task is inferred.\n"
        + ("SYNTHETIC TEST DATA: not a hardware experiment.\n" if summary["synthetic"] else ""))
    return summary


def plot_motion(out, summary, joints, poses, feedback, latency_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .22})
    title = f"FR3 {summary['method']} | run {summary['run_id']} | " + (
        "SYNTHETIC TEST" if summary["synthetic"] else "dry-run" if summary["dry_run"] else "measured hardware telemetry")
    files = []
    def save(fig, name):
        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0, 1, .97))
        for ext in ("png", "pdf"):
            file = f"{name}.{ext}"
            fig.savefig(out / file, dpi=160)
            files.append(file)
        plt.close(fig)
    def mark(ax):
        if summary["stop_time_s"] is not None:
            ax.axvline(summary["stop_time_s"], color="0.4", ls=":", lw=1, label="disable")
    def line(ax, times, values, **kwargs):
        # Break lines across gaps rather than inventing unobserved trajectories.
        x, y = [], []
        for i, (t, value) in enumerate(zip(times, values)):
            if i and not 0 < t - times[i-1] <= .1:
                x.append(np.nan); y.append(np.nan)
            x.append(t); y.append(value)
        ax.plot(x, y, **kwargs)
    if joints:
        t = [r["time_s"] for r in joints]
        for prefix, unit, filename in (("q", "rad", "joint_positions"), ("dq", "rad_s", "joint_velocities")):
            fig, axes = plt.subplots(7, 1, figsize=(11, 12), sharex=True)
            for i, ax in enumerate(axes, 1):
                line(ax, t, [r[f"{prefix}{i}_{unit}"] for r in joints], label="measured", color="#2463ad")
                if prefix == "q" and feedback:
                    line(ax, [r["time_s"] for r in feedback], [r[f"desired_q{i}_rad"] for r in feedback],
                         label="controller desired", color="#dd8935", ls="--")
                ax.set_ylabel(f"J{i} ({'rad/s' if prefix == 'dq' else 'rad'})")
                mark(ax)
            axes[0].legend(loc="upper right")
            axes[-1].set_xlabel("Time since enable (s)")
            save(fig, filename)
    if feedback:
        fig, axes = plt.subplots(7, 1, figsize=(11, 12), sharex=True)
        t = [r["time_s"] for r in feedback]
        for i, ax in enumerate(axes, 1):
            line(ax, t, [r[f"error_q{i}_rad"] for r in feedback], color="#b23a48")
            ax.set_ylabel(f"J{i} (rad)")
            mark(ax)
        axes[0].set_title("Controller tracking error (desired - actual)")
        axes[-1].set_xlabel("Time since enable (s)")
        save(fig, "joint_tracking_error")
    if latency_rows:
        fig, ax = plt.subplots(figsize=(11, 5))
        x = [r["request_id"] for r in latency_rows]
        actual = [r["request_to_send_ms"] for r in latency_rows]
        ax.plot(x, actual, color="#2463ad", marker=".", ms=3, label="request to send/preview")
        configured = latency_rows[0]["set_delay_ms"]
        if configured != "":
            ax.axhline(float(configured), color="#dd8935", ls="--", label="configured simulated delay")
        ax.set(xlabel="Request id", ylabel="Latency (ms)", title="Configured delay vs actual request-to-send latency")
        ax.legend()
        save(fig, "delay_comparison")
    frames = sorted({r["frame_id"] for r in poses})
    for f, frame in enumerate(frames):
        rows = [r for r in poses if r["frame_id"] == frame]
        t = np.array([r["time_s"] for r in rows])
        xyz = np.array([[r[k] for k in ("x_m", "y_m", "z_m")] for r in rows])
        quat = np.array([[r[k] for k in ("qx", "qy", "qz", "qw")] for r in rows])
        quat /= np.linalg.norm(quat, axis=1)[:, None]
        for i in range(1, len(quat)):
            if np.dot(quat[i-1], quat[i]) < 0:
                quat[i] *= -1
        suffix = "" if len(frames) == 1 else f"_frame{f}"
        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot(121, projection="3d")
        broken = []
        for i, point in enumerate(xyz):
            if i and not 0 < t[i] - t[i-1] <= .1:
                broken.append([np.nan]*3)
            broken.append(point)
        broken = np.array(broken)
        ax.plot(*broken.T, color="#2463ad")
        ax.scatter(*xyz[0], color="green", label="first")
        ax.scatter(*xyz[-1], color="red", label="last recorded")
        ax.set(xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)", title=f"EE in {frame}")
        ax.legend()
        span = np.maximum(np.ptp(xyz, axis=0), .001)
        ax.set_box_aspect(span)
        ax2 = fig.add_subplot(122)
        ax2.plot(broken[:, 0], broken[:, 1], color="#2463ad")
        ax2.scatter(xyz[0, 0], xyz[0, 1], color="green")
        ax2.scatter(xyz[-1, 0], xyz[-1, 1], color="red")
        ax2.set(xlabel="X (m)", ylabel="Y (m)", title="XY projection")
        ax2.set_aspect("equal", adjustable="datalim")
        save(fig, "ee_trajectory" + suffix)
        fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
        for i, ax in enumerate(axes):
            line(ax, t, xyz[:, i], color="#2463ad")
            ax.set_ylabel(f"{'XYZ'[i]} (m)")
            mark(ax)
        axes[0].set_title(f"Measured EE position in {frame}")
        axes[-1].set_xlabel("Time since enable (s)")
        save(fig, "ee_position" + suffix)
        fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
        for i, ax in enumerate(axes):
            line(ax, t, quat[:, i], color="#2463ad")
            ax.set_ylabel(f"q{'xyzw'[i]}")
            mark(ax)
        axes[0].set_title(f"Measured EE quaternion in {frame} (sign aligned for display)")
        axes[-1].set_xlabel("Time since enable (s)")
        save(fig, "ee_orientation" + suffix)
    return files


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("log", type=Path)
    p.add_argument("--run-id", type=int, default=1)
    p.add_argument("--output", type=Path, required=True, help="New output directory")
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()
    result = export_report(args.log, args.output, args.run_id, not args.no_plots)
    print(json.dumps({k: result[k] for k in ("run_id", "method", "dry_run", "joint_coverage", "ee_coverage", "warnings", "figures")}))


if __name__ == "__main__":
    main()
