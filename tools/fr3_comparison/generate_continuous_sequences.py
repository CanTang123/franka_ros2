#!/usr/bin/env python3
"""Generate a 44 s C2 fixed replay that stays close to debug_slow5x."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
try:
    from scipy.interpolate import BPoly, PPoly, make_smoothing_spline
except ImportError as error:
    raise SystemExit("continuous generation requires python3-scipy") from error

from fixed_replay import load_sequence, load_site_config


DT_S = .05
REQUEST_S = .2
SMOOTHING_LAMBDA = 1e-4
END_BLEND_S = .8
MAX_DEVIATION_RAD = .0035
ACCELERATION_MARGIN = .8


def _piece_extreme(curve, order, start, end):
    derivative = curve.derivative(order)
    critical = curve.derivative(order+1).roots(extrapolate=False)
    candidates = [start, end]
    candidates.extend(float(x) for x in critical
                      if np.isfinite(x) and start <= x <= end)
    return max(abs(float(derivative(x))) for x in candidates)


def fit(sequence):
    positions = np.asarray(sequence.points)
    times = np.arange(len(positions))*DT_S
    duration = float(times[-1])
    if duration <= 2*END_BLEND_S:
        raise ValueError("source trajectory is too short for endpoint blending")
    middle = [make_smoothing_spline(times, positions[:, joint],
                                    lam=SMOOTHING_LAMBDA)
              for joint in range(7)]
    left = [BPoly.from_derivatives([0., END_BLEND_S],
        [[positions[0, joint], 0., 0.],
         [curve(END_BLEND_S), curve(END_BLEND_S, 1), curve(END_BLEND_S, 2)]])
        for joint, curve in enumerate(middle)]
    right = [BPoly.from_derivatives([duration-END_BLEND_S, duration],
        [[curve(duration-END_BLEND_S), curve(duration-END_BLEND_S, 1),
          curve(duration-END_BLEND_S, 2)],
         [positions[-1, joint], 0., 0.]])
        for joint, curve in enumerate(middle)]

    def evaluate(query, order=0):
        query = np.asarray(query)
        result = np.empty((query.size, 7))
        for joint in range(7):
            result[:, joint] = np.where(
                query < END_BLEND_S, left[joint](query, order),
                np.where(query > duration-END_BLEND_S,
                         right[joint](query, order), middle[joint](query, order)))
        return result

    # Compute derivative extrema from polynomial critical points, not sampling.
    max_speed = [0.] * 7
    max_acceleration = [0.] * 7
    polynomials = []
    for joint in range(7):
        pieces = [(PPoly.from_bernstein_basis(left[joint]), 0., END_BLEND_S),
                  (PPoly.from_spline(middle[joint]), END_BLEND_S,
                   duration-END_BLEND_S),
                  (PPoly.from_bernstein_basis(right[joint]),
                   duration-END_BLEND_S, duration)]
        polynomials.append(pieces)
        for curve, start, end in pieces:
            max_speed[joint] = max(max_speed[joint],
                                   _piece_extreme(curve, 1, start, end))
            max_acceleration[joint] = max(
                max_acceleration[joint], _piece_extreme(curve, 2, start, end))

    # Find the exact maximum against every linear source interval. Endpoint
    # blends align to the 50 ms grid, so each interval belongs to one polynomial.
    max_deviation = 0.
    for index in range(len(times)-1):
        start, end = times[index], times[index+1]
        for joint in range(7):
            slope = (positions[index+1, joint]-positions[index, joint])/DT_S
            curve = next(curve for curve, low, high in polynomials[joint]
                         if low-1e-12 <= (start+end)/2 <= high+1e-12)
            difference_derivative = curve.derivative()
            coefficients = difference_derivative.c.copy()
            coefficients[-1] -= slope
            roots = PPoly(coefficients, difference_derivative.x,
                          extrapolate=False).roots(extrapolate=False)
            candidates = [start, end]
            candidates.extend(float(x) for x in roots
                              if np.isfinite(x) and start <= x <= end)
            for value in candidates:
                reference = positions[index, joint] + slope*(value-start)
                max_deviation = max(max_deviation,
                                    abs(float(curve(value))-reference))
    return times, evaluate(times), evaluate(times, 1), evaluate(times, 2), \
        max_deviation, max_speed, max_acceleration


def derive(sequence, site):
    times, q, dq, ddq, deviation, speed, acceleration = fit(sequence)
    if deviation > MAX_DEVIATION_RAD + 1e-12:
        raise ValueError(
            f"{sequence.method} smoothing deviation {deviation:.9f} exceeds "
            f"{MAX_DEVIATION_RAD:.9f} rad")
    if any(value > site["max_speed_rad_s"] for value in speed):
        raise ValueError(f"{sequence.method} continuous speed exceeds site limit")
    if any(value > ACCELERATION_MARGIN*limit
           for value, limit in zip(acceleration,
                                   site["acceleration_limits_rad_s2"])):
        raise ValueError(
            f"{sequence.method} cannot meet the 20% acceleration margin at 44 s")
    points = [dict(time_from_start_s=float(t), positions=p.tolist(),
                   velocities=v.tolist(), accelerations=a.tolist())
              for t, p, v, a in zip(times, q, dq, ddq)]
    return dict(schema="fr3_continuous_smooth_sequence_v1", method=sequence.method,
        joint_names=list(sequence.joint_names), units="rad, rad/s, rad/s^2 and seconds",
        derived_from=str(sequence.path), source_is_debug_slow5x=sequence.debug_slow5x,
        sample_dt_s=DT_S, request_period_s=REQUEST_S, targets_per_request=4,
        duration_s=float(times[-1]), points_count=len(points),
        interpolation="Global C2 smoothing spline with clamped C2 endpoint blends",
        smoothing_lambda=SMOOTHING_LAMBDA, endpoint_blend_s=END_BLEND_S,
        maximum_source_deviation_rad=deviation,
        allowed_source_deviation_rad=MAX_DEVIATION_RAD,
        acceleration_limit_margin=ACCELERATION_MARGIN,
        continuous_max_speed_rad_s=speed,
        continuous_max_acceleration_rad_s2=acceleration,
        rolling_action_replacement=True, hardware_validated=False,
        original_method_result=False,
        note="Continuous safety-derived fixed replay; no model inference or online closed loop.",
        points=points)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--site-config", type=Path, required=True)
    parser.add_argument("--methods", default="ABCD")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists; refusing to overwrite: {args.output}")
    site = load_site_config(args.site_config, require_confirmed=False)
    generated = {}
    for method in args.methods:
        sequence = load_sequence(args.source, method)
        if not sequence.debug_slow5x:
            parser.error(f"{method} source is not debug_slow5x")
        data = derive(sequence, site)
        folder = args.output / method / "debug_continuous_smooth"
        folder.mkdir(parents=True)
        path = folder / "joint_trajectory_points.json"
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False,
                                   allow_nan=False)+"\n")
        generated[method] = dict(path=str(path.resolve()),
            duration_s=data["duration_s"], points=data["points_count"],
            maximum_source_deviation_rad=data["maximum_source_deviation_rad"],
            continuous_max_speed_rad_s=max(data["continuous_max_speed_rad_s"]),
            continuous_max_acceleration_rad_s2=max(
                data["continuous_max_acceleration_rad_s2"]))
    manifest = dict(schema="fr3_continuous_smooth_bundle_v1",
                    source=str(args.source.resolve()),
                    site_config=str(args.site_config.resolve()), methods=generated)
    (args.output / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
