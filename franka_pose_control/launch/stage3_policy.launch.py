"""Launch the Stage 3 policy bridge without starting robot hardware."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    share_dir = Path(get_package_share_directory('franka_pose_control'))
    config = share_dir / 'config' / 'stage3_policy.yaml'
    contract = share_dir / 'models' / 'stage3_model_600' / 'stage3_policy.yaml'
    dry_run = LaunchConfiguration('dry_run')
    return LaunchDescription(
        [
            DeclareLaunchArgument('dry_run', default_value='true'),
            Node(
                package='franka_pose_control',
                executable='stage3_policy_node',
                name='fr3_stage3_policy',
                output='screen',
                parameters=[
                    str(config),
                    {
                        'contract_path': str(contract),
                        'dry_run': ParameterValue(dry_run, value_type=bool),
                    },
                ],
            ),
        ]
    )
