#!/usr/bin/env python3
"""Derive C2, fixed-50ms replay data from debug_slow5x without loosening limits."""
import argparse
import json
import math
from pathlib import Path

from fixed_replay import load_sequence, load_site_config


BLOCK_S = .2
DT_S = .05
ACCELERATION_MARGIN = .8
MAX_FACTOR = 20
MAX_SPEED_COEFFICIENT = 1.875
MAX_ACCELERATION_COEFFICIENT = 10.0 / math.sqrt(3.0)  # max of minimum-jerk s''


def smoothstep(u):
    return 10*u**3 - 15*u**4 + 6*u**5


def smoothstep_d1(u):
    return 30*u**2 - 60*u**3 + 30*u**4


def smoothstep_d2(u):
    return 60*u - 180*u**2 + 120*u**3


def source_at(sequence, time_s):
    position = min(time_s / sequence.dt_s, len(sequence.points)-1)
    left = min(int(math.floor(position)), len(sequence.points)-1)
    right = min(left+1, len(sequence.points)-1)
    fraction = position-left
    return [a+(b-a)*fraction for a, b in zip(sequence.points[left], sequence.points[right])]


def blocks_for(sequence, factor):
    count = sequence.command_count * factor // 4
    result = []
    for block in range(count):
        q0 = source_at(sequence, block*BLOCK_S/factor)
        q1 = source_at(sequence, (block+1)*BLOCK_S/factor)
        result.append((q0, q1, [b-a for a, b in zip(q0, q1)]))
    return result


def continuous_limits(blocks):
    max_delta = [max(abs(row[2][j]) for row in blocks) for j in range(7)]
    speed = [MAX_SPEED_COEFFICIENT*d/BLOCK_S for d in max_delta]
    acceleration = [MAX_ACCELERATION_COEFFICIENT*d/(BLOCK_S**2) for d in max_delta]
    return speed, acceleration


def choose_factor(sequence, site):
    for factor in range(2, MAX_FACTOR+1):
        blocks = blocks_for(sequence, factor)
        speed, acceleration = continuous_limits(blocks)
        max_quarter_step = max(abs(d)*.396484375 for _, _, delta in blocks for d in delta)
        if (max_quarter_step <= site["max_step_rad"] and
                all(v <= site["max_speed_rad_s"] for v in speed) and
                all(a <= ACCELERATION_MARGIN*limit
                    for a, limit in zip(acceleration, site["acceleration_limits_rad_s2"]))):
            return factor, blocks, speed, acceleration
    raise ValueError(f"no safe factor found through {MAX_FACTOR}x")


def derive(sequence, site):
    factor, blocks, continuous_speed, continuous_acceleration = choose_factor(sequence, site)
    points = []
    for block_index, (q0, _, delta) in enumerate(blocks):
        start_k = 0 if block_index == 0 else 1
        for k in range(start_k, 5):
            u = k/4
            points.append(dict(
                time_from_start_s=(block_index*4+k)*DT_S,
                positions=[a+d*smoothstep(u) for a, d in zip(q0, delta)],
                velocities=[d*smoothstep_d1(u)/BLOCK_S for d in delta],
                accelerations=[d*smoothstep_d2(u)/(BLOCK_S**2) for d in delta]))
    expected = sequence.command_count*factor+1
    if len(points) != expected or (len(points)-1) % 4:
        raise AssertionError("derived sequence length invariant failed")
    return dict(schema="fr3_safe_smooth_sequence_v1", method=sequence.method,
        joint_names=list(sequence.joint_names), units="rad, rad/s, rad/s^2 and seconds",
        derived_from=str(sequence.path), source_is_debug_slow5x=sequence.debug_slow5x,
        additional_time_scale=factor, sample_dt_s=DT_S,
        request_period_s=BLOCK_S, targets_per_request=4,
        duration_s=(len(points)-1)*DT_S, points_count=len(points),
        interpolation="Per-200ms minimum-jerk quintic; zero velocity and acceleration at every action boundary",
        acceleration_limit_margin=ACCELERATION_MARGIN,
        continuous_max_speed_rad_s=continuous_speed,
        continuous_max_acceleration_rad_s2=continuous_acceleration,
        hardware_validated=False, original_method_result=False,
        note="Safety-derived fixed replay only; no model inference and no new simulation result.",
        points=points)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--site-config", type=Path, required=True)
    p.add_argument("--methods", default="ABCD")
    args = p.parse_args()
    site = load_site_config(args.site_config, require_confirmed=False)
    if args.output.exists():
        p.error(f"output already exists; refusing to overwrite: {args.output}")
    generated = {}
    for method in args.methods:
        sequence = load_sequence(args.source, method)
        if not sequence.debug_slow5x:
            p.error(f"{method} source is not debug_slow5x")
        data = derive(sequence, site)
        folder = args.output / method / "debug_safe_smooth"
        folder.mkdir(parents=True)
        path = folder / "joint_trajectory_points.json"
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
        generated[method] = dict(path=str(path.resolve()), additional_time_scale=data["additional_time_scale"],
            duration_s=data["duration_s"], points=data["points_count"],
            continuous_max_speed_rad_s=max(data["continuous_max_speed_rad_s"]),
            continuous_max_acceleration_rad_s2=max(data["continuous_max_acceleration_rad_s2"]))
    manifest = dict(schema="fr3_safe_smooth_bundle_v1", source=str(args.source.resolve()),
                    site_config=str(args.site_config.resolve()), methods=generated)
    (args.output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
