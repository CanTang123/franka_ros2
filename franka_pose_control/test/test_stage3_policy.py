"""Tests for the Stage 3 deployment contract and waypoint planner."""

from pathlib import Path

from franka_pose_control_policy.stage3_policy import (
    DeploymentContract,
    Stage3Policy,
    WaypointPlanner,
)
import numpy as np
import pytest

BUNDLE = Path(__file__).parents[1] / 'models' / 'stage3_model_600' / 'stage3_policy.yaml'


@pytest.fixture
def contract() -> DeploymentContract:
    return DeploymentContract.load(BUNDLE)


def test_observation_layout(contract: DeploymentContract) -> None:
    joint_position = contract.q_default + np.arange(7, dtype=np.float32) * 0.1
    joint_velocity = np.arange(7, dtype=np.float32) * 0.2
    target = np.asarray((0.5, 0.1, 0.1, 1.0, 0.0, 0.0, 0.0), dtype=np.float32)
    previous_action = np.asarray((0.2, -0.3), dtype=np.float32)
    tcp = np.asarray((0.4, -0.1, 0.1), dtype=np.float32)
    center = np.asarray((0.45, 0.0), dtype=np.float32)
    waypoint = np.asarray((0.5, 0.0), dtype=np.float32)
    observation = contract.build_observation(
        joint_position, joint_velocity, target, previous_action, tcp, center, 0.05, True, waypoint
    )
    np.testing.assert_allclose(observation[:7], np.arange(7) * 0.1, atol=1.0e-6)
    np.testing.assert_allclose(observation[7:14], joint_velocity)
    np.testing.assert_allclose(observation[14:21], target)
    np.testing.assert_allclose(observation[21:23], previous_action)
    np.testing.assert_allclose(observation[23:26], tcp)
    np.testing.assert_allclose(observation[26:30], (0.05, 0.1, 0.05, 1.0), atol=1.0e-7)
    np.testing.assert_allclose(observation[30:32], (0.1, 0.1), atol=1.0e-7)


def test_action_processing_preserves_previous_component_clipped_action(
    contract: DeploymentContract,
) -> None:
    previous_action, delta = contract.process_action(np.asarray((2.0, -1.0)), 0.001)
    np.testing.assert_allclose(previous_action, (1.0, -1.0))
    np.testing.assert_allclose(delta, np.asarray((1.0, -1.0)) / np.sqrt(2.0) * 0.001)


def test_reference_advance_accumulates_and_limits_tcp_lead(
    contract: DeploymentContract,
) -> None:
    tcp = np.asarray((0.45, 0.0))
    reference = tcp.copy()
    delta = np.asarray((0.003, 0.004))

    reference = contract.advance_reference(reference, tcp, delta, 0.015)
    np.testing.assert_allclose(reference, (0.453, 0.004))
    reference = contract.advance_reference(reference, tcp, delta, 0.015)
    np.testing.assert_allclose(reference, (0.456, 0.008))
    reference = contract.advance_reference(reference, tcp, delta, 0.015)
    np.testing.assert_allclose(reference, (0.459, 0.012))
    reference = contract.advance_reference(reference, tcp, delta, 0.015)
    np.testing.assert_allclose(np.linalg.norm(reference - tcp), 0.015)


def test_reference_advance_respects_action_workspace(contract: DeploymentContract) -> None:
    reference = np.asarray((contract.action_workspace_x[1], contract.action_workspace_y[1]))
    advanced = contract.advance_reference(
        reference,
        reference,
        np.asarray((0.01, 0.01)),
        0.015,
    )
    np.testing.assert_allclose(advanced, reference)


def test_waypoint_planner_uses_target_for_clear_path(contract: DeploymentContract) -> None:
    plan = WaypointPlanner(contract).plan(
        np.asarray((0.4, -0.1)), np.asarray((0.6, -0.1)), np.asarray((0.5, 0.1)), 0.04, True
    )
    assert not plan.direct_path_blocked
    assert plan.waypoint_valid
    np.testing.assert_allclose(plan.waypoint_xy, (0.6, -0.1))


def test_waypoint_planner_detours_around_blocked_path(contract: DeploymentContract) -> None:
    center = np.asarray((0.5, 0.0))
    radius = 0.04
    plan = WaypointPlanner(contract).plan(
        np.asarray((0.4, 0.0)), np.asarray((0.6, 0.0)), center, radius, True
    )
    assert plan.direct_path_blocked
    assert plan.waypoint_valid
    assert (
        np.linalg.norm(plan.waypoint_xy - center) >= radius + contract.waypoint_clearance - 1.0e-9
    )


def test_onnx_matches_exported_golden_vectors(contract: DeploymentContract) -> None:
    pytest.importorskip('onnxruntime')
    policy = Stage3Policy(contract)
    vectors = np.load(contract.bundle_dir / 'stage3_golden_vectors.npz')
    actual = np.stack([policy.infer(observation) for observation in vectors['observations']])
    np.testing.assert_allclose(actual, vectors['actor_actions'], atol=1.0e-6, rtol=0.0)
