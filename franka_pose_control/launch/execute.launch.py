"""向现有控制节点提交一次执行请求；不重启保存轨迹的节点。"""

import os

from ament_index_python.packages import get_package_prefix
from launch import LaunchDescription
from launch.actions import ExecuteProcess


def generate_launch_description():
    executable = os.path.join(get_package_prefix('franka_pose_control'),
                              'lib', 'franka_pose_control', 'pose')
    return LaunchDescription([
        ExecuteProcess(cmd=[executable, 'execute'], output='screen'),
    ])
