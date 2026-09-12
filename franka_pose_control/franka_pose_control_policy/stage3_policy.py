"""Pure-Python deployment contract and waypoint logic for the Stage 3 policy."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _point_segment_distance(start: np.ndarray, end: np.ndarray, point: np.ndarray) -> float:
    direction = end - start
    denominator = float(np.dot(direction, direction))
    fraction = (
        0.0 if denominator <= 1.0e-12 else float(np.dot(point - start, direction) / denominator)
    )
    closest = start + np.clip(fraction, 0.0, 1.0) * direction
    return float(np.linalg.norm(closest - point))


@dataclass(frozen=True)
class DeploymentContract:
    """Validated values required to reproduce the simulation policy interface."""

    bundle_dir: Path
    model_path: Path
    input_name: str
    output_name: str
    observation_dim: int
    action_dim: int
    joint_names: tuple[str, ...]
    q_default: np.ndarray
    model_max_step: float
    target_height: float
    target_orientation_xyzw: np.ndarray
    action_workspace_x: tuple[float, float]
    action_workspace_y: tuple[float, float]
    planner_workspace_x: tuple[float, float]
    planner_workspace_y: tuple[float, float]
    waypoint_candidates: int
    waypoint_clearance: float
    waypoint_path_clearance: float
    waypoint_switch_distance: float
    waypoint_radial_scales: tuple[float, ...]

    @classmethod
    def load(cls, metadata_path: str | Path, validate_hashes: bool = True) -> DeploymentContract:
        """Load and validate an exported deployment bundle.

        Args:
            metadata_path: Exported YAML contract path.
            validate_hashes: Whether to verify all artifact SHA-256 hashes.

        Returns:
            Parsed deployment contract.
        """
        metadata_path = Path(metadata_path).expanduser().resolve()
        with metadata_path.open(encoding='utf-8') as stream:
            data: dict[str, Any] = yaml.safe_load(stream)
        bundle_dir = metadata_path.parent
        if data.get('bundle_version') != 1:
            raise ValueError(f'Unsupported bundle version: {data.get("bundle_version")}')
        model = data['model']
        if model['input_shape'] != [1, 32] or model['output_shape'] != [1, 2]:
            raise ValueError('Stage 3 deployment requires model shapes [1,32] -> [1,2]')
        observations = data['observations']
        expected_terms = (
            ('joint_pos', 0, 7),
            ('joint_vel', 7, 14),
            ('pose_command', 14, 21),
            ('actions', 21, 23),
            ('tcp_position', 23, 26),
            ('constraint', 26, 30),
            ('detour_waypoint', 30, 32),
        )
        actual_terms = tuple((term['name'], term['start'], term['stop']) for term in observations)
        if actual_terms != expected_terms:
            raise ValueError(f'Unexpected observation layout: {actual_terms}')
        artifacts = data.get('artifacts', {})
        if validate_hashes:
            for name, expected_hash in artifacts.items():
                artifact = bundle_dir / name
                if not artifact.is_file():
                    raise FileNotFoundError(f'Missing deployment artifact: {artifact}')
                actual_hash = _sha256(artifact)
                if actual_hash != expected_hash:
                    raise ValueError(f'SHA-256 mismatch for {name}: {actual_hash}')
        control = data['control']
        planner = data['waypoint_planner']
        return cls(
            bundle_dir=bundle_dir,
            model_path=bundle_dir / 'stage3_policy.onnx',
            input_name=model['input_name'],
            output_name=model['output_name'],
            observation_dim=model['input_shape'][1],
            action_dim=model['output_shape'][1],
            joint_names=tuple(data['robot']['joint_names']),
            q_default=np.asarray(data['robot']['q_default_rad'], dtype=np.float32),
            model_max_step=float(data['action_processing']['max_step_m']),
            target_height=float(control['target_height_m']),
            target_orientation_xyzw=np.asarray(
                control['target_orientation_xyzw'], dtype=np.float32
            ),
            action_workspace_x=tuple(float(value) for value in control['workspace_x_m']),
            action_workspace_y=tuple(float(value) for value in control['workspace_y_m']),
            planner_workspace_x=tuple(
                float(value) for value in planner.get('workspace_x_m', control['workspace_x_m'])
            ),
            planner_workspace_y=tuple(
                float(value) for value in planner.get('workspace_y_m', control['workspace_y_m'])
            ),
            waypoint_candidates=int(planner['candidate_count']),
            waypoint_clearance=float(planner['candidate_clearance_m']),
            waypoint_path_clearance=float(planner['path_clearance_m']),
            waypoint_switch_distance=float(planner['switch_distance_m']),
            waypoint_radial_scales=tuple(
                float(value) for value in planner['candidate_radial_scales']
            ),
        )

    def build_observation(
        self,
        joint_position: np.ndarray,
        joint_velocity: np.ndarray,
        target_pose_xyzw: np.ndarray,
        previous_action: np.ndarray,
        tcp_position: np.ndarray,
        forbidden_center_xy: np.ndarray,
        forbidden_radius: float,
        avoidance_enabled: bool,
        active_waypoint_xy: np.ndarray,
    ) -> np.ndarray:
        """Build one model observation using the exported ordering."""
        observation = np.concatenate(
            (
                np.asarray(joint_position, dtype=np.float32) - self.q_default,
                np.asarray(joint_velocity, dtype=np.float32),
                np.asarray(target_pose_xyzw, dtype=np.float32),
                np.asarray(previous_action, dtype=np.float32),
                np.asarray(tcp_position, dtype=np.float32),
                np.asarray(forbidden_center_xy, dtype=np.float32)
                - np.asarray(tcp_position[:2], dtype=np.float32),
                np.asarray((forbidden_radius, float(avoidance_enabled)), dtype=np.float32),
                np.asarray(active_waypoint_xy, dtype=np.float32)
                - np.asarray(tcp_position[:2], dtype=np.float32),
            )
        )
        if observation.shape != (self.observation_dim,) or not np.all(np.isfinite(observation)):
            raise ValueError(f'Invalid policy observation with shape {observation.shape}')
        return observation

    def process_action(
        self, raw_action: np.ndarray, deployment_max_step: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Clip an actor output and return the previous action and XY increment [m]."""
        if deployment_max_step <= 0.0 or deployment_max_step > self.model_max_step:
            raise ValueError(f'deployment_max_step must be in (0, {self.model_max_step}]')
        previous_action = np.clip(np.asarray(raw_action, dtype=np.float32), -1.0, 1.0)
        if previous_action.shape != (self.action_dim,) or not np.all(np.isfinite(previous_action)):
            raise ValueError(f'Invalid policy action with shape {previous_action.shape}')
        bounded = previous_action / max(float(np.linalg.norm(previous_action)), 1.0)
        return previous_action, bounded * deployment_max_step


@dataclass(frozen=True)
class WaypointPlan:
    """One circular-constraint waypoint plan."""

    direct_path_blocked: bool
    waypoint_valid: bool
    waypoint_xy: np.ndarray


class WaypointPlanner:
    """NumPy reproduction of the Stage 3 sampled waypoint planner."""

    def __init__(self, contract: DeploymentContract):
        self.contract = contract

    def plan(
        self,
        start_xy: np.ndarray,
        target_xy: np.ndarray,
        forbidden_center_xy: np.ndarray,
        forbidden_radius: float,
        avoidance_enabled: bool,
    ) -> WaypointPlan:
        """Select the shortest sampled collision-free detour waypoint."""
        start = np.asarray(start_xy, dtype=np.float64)
        target = np.asarray(target_xy, dtype=np.float64)
        center = np.asarray(forbidden_center_xy, dtype=np.float64)
        blocked = (
            avoidance_enabled
            and _point_segment_distance(start, target, center) <= forbidden_radius
        )
        if not blocked:
            return WaypointPlan(False, True, target.copy())

        angles = np.arange(self.contract.waypoint_candidates) * (
            2.0 * np.pi / self.contract.waypoint_candidates
        )
        directions = np.column_stack((np.cos(angles), np.sin(angles)))
        candidates = []
        required_clearance = forbidden_radius + self.contract.waypoint_path_clearance
        for radial_scale in self.contract.waypoint_radial_scales:
            radius = forbidden_radius + self.contract.waypoint_clearance * radial_scale
            for candidate in center + radius * directions:
                inside = (
                    self.contract.planner_workspace_x[0]
                    <= candidate[0]
                    <= self.contract.planner_workspace_x[1]
                    and self.contract.planner_workspace_y[0]
                    <= candidate[1]
                    <= self.contract.planner_workspace_y[1]
                )
                if not inside:
                    continue
                if _point_segment_distance(start, candidate, center) < required_clearance:
                    continue
                if _point_segment_distance(candidate, target, center) < required_clearance:
                    continue
                length = float(
                    np.linalg.norm(candidate - start) + np.linalg.norm(target - candidate)
                )
                candidates.append((length, candidate.copy()))
        if not candidates:
            return WaypointPlan(True, False, target.copy())
        return WaypointPlan(True, True, min(candidates, key=lambda item: item[0])[1])


class Stage3Policy:
    """ONNX Runtime wrapper for one exported Stage 3 actor."""

    def __init__(self, contract: DeploymentContract):
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError(
                'Install onnxruntime in the ROS 2 container to run the Stage 3 policy'
            ) from error
        self.contract = contract
        session_options = ort.SessionOptions()
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(contract.model_path),
            sess_options=session_options,
            providers=['CPUExecutionProvider'],
        )

    def infer(self, observation: np.ndarray) -> np.ndarray:
        """Run one deterministic actor inference."""
        batch = np.asarray(observation, dtype=np.float32).reshape(1, self.contract.observation_dim)
        output = self.session.run([self.contract.output_name], {self.contract.input_name: batch})[
            0
        ]
        action = np.asarray(output[0], dtype=np.float32)
        if action.shape != (self.contract.action_dim,) or not np.all(np.isfinite(action)):
            raise RuntimeError('Policy returned an invalid action')
        return action
