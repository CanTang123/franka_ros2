from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    parameters = PathJoinSubstitution(
        [FindPackageShare('franka_pose_control'), 'config', 'pose_target.yaml']
    )

    return LaunchDescription(
        [
            Node(
                package='franka_pose_control',
                executable='pose_target_node',
                name='franka_pose_target',
                parameters=[parameters],
                output='screen',
                emulate_tty=True,
            )
        ]
    )
