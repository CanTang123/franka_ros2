"""启动规划节点并发送 YAML 中选择的目标，保持运行以保留轨迹缓存。"""

import math
import os

import yaml
from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def plan(context):
    value = lambda name: LaunchConfiguration(name).perform(context)
    with open(value('targets_file'), encoding='utf-8') as stream:
        document = yaml.safe_load(stream)
    targets = document.get('targets', {}) if isinstance(document, dict) else {}
    name = value('target')
    if name not in targets:
        raise ValueError(f'目标 {name!r} 不存在，可选目标：{list(targets)}')
    target = targets[name]
    mode = value('mode') or target.get('mode', 'line')
    if mode not in ('line', 'ptp', 'home'):
        raise ValueError('mode 必须为 line、ptp 或 home')
    executable = os.path.join(get_package_prefix('franka_pose_control'),
                              'lib', 'franka_pose_control', 'pose')
    command = [executable, '--timeout', '15', mode]
    if mode != 'home':
        position = target.get('position')
        orientation = target.get('orientation')
        for field, values, length in [('position', position, 3), ('orientation', orientation, 4)]:
            if field == 'orientation' and values is None:
                continue
            if (not isinstance(values, list) or len(values) != length or
                    any(isinstance(v, bool) or not isinstance(v, (int, float)) or
                        not math.isfinite(v) for v in values)):
                raise ValueError(f'{field} 必须包含 {length} 个有限数值')
        command += [str(v) for v in position]
        if orientation is not None:
            if math.hypot(*orientation) < 1e-6:
                raise ValueError('orientation 不能为零四元数')
            command += ['--quat'] + [str(v) for v in orientation]
        relative = target.get('relative', False)
        if not isinstance(relative, bool):
            raise ValueError('relative 必须为 true 或 false')
        if relative:
            command += ['--relative']
    start = value('start_node').lower()
    if start not in ('true', 'false'):
        raise ValueError('start_node 必须为 true 或 false')
    actions = []
    if start == 'true':
        actions.append(Node(package='franka_pose_control', executable='pose_target_node',
                            name='franka_pose_target', parameters=[value('params_file')],
                            output='screen'))
    actions.append(ExecuteProcess(cmd=command, output='screen'))
    return actions


def generate_launch_description():
    share = get_package_share_directory('franka_pose_control')
    return LaunchDescription([
        DeclareLaunchArgument('targets_file', default_value=os.path.join(share, 'config', 'targets.yaml'),
                              description='命名目标配置文件'),
        DeclareLaunchArgument('target', default_value='up_1cm', description='选择一个命名目标'),
        DeclareLaunchArgument('mode', default_value='', description='可覆盖目标的模式：line/ptp/home'),
        DeclareLaunchArgument('start_node', default_value='true',
                              description='首次启动为 true；已有控制节点时设为 false'),
        DeclareLaunchArgument('params_file', default_value=os.path.join(share, 'config', 'pose_target.yaml')),
        OpaqueFunction(function=plan),
    ])
