# FR3 Stage 3 真机策略桥接

本功能将 Isaac Lab 导出的 Stage 3 waypoint 避障策略接入 ROS 2。策略以 30 Hz 读取真实机械臂状态，构造与仿真一致的 32 维观测，输出二维 TCP 增量，再向独立的笛卡尔阻抗控制器发送小步参考位姿。

## 安全边界

- 节点默认 `dry_run:=true` 且未使能，不会向控制器发送运动目标。
- 真机首次运行前，必须确认 `/franka_robot_state_broadcaster/current_pose` 与仿真 `fr3_hand_tcp` 表示同一 TCP，且坐标系 `base`、固定高度 0.10 m、朝下四元数 `[1,0,0,0]` 一致。
- 模型在仿真中的最大参考步长为 20 mm；真机默认额外限制为每个 30 Hz 周期 1 mm。
- 节点检查状态超时、关节速度、工作区、非法数值和禁区安全余量，但不能替代 Franka 自身安全机制、现场急停、碰撞监控或人工监护。
- 节点同时检查完整 Franka 状态；机器人不在 `MOVE` 模式、存在任何当前错误或控制命令成功率低于阈值时会自动取消使能。
- 本代码只完成离线构建与接口验证，尚未在真实机械臂上验证。

## 构建

容器中先安装 ONNX Runtime，再构建两个修改过的包：

```bash
cd /ros2_ws
python3 -m pip install -r src/franka_pose_control/requirements-stage3.txt
colcon build --symlink-install \
  --packages-select franka_example_controllers franka_bringup franka_pose_control \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

## 启动与 dry-run

先用已有 MoveIt 点到点功能将 TCP 移到训练平面附近，再停止 `fr3_arm_controller`。启动阻抗控制器前，
确认机械臂静止、现场有人监护且急停可用。以下两个结果必须表示同一物理 TCP；若位置或姿态不一致，
不要关闭 dry-run：

```bash
ros2 topic echo /franka_robot_state_broadcaster/current_pose --once \
  --qos-reliability best_effort
ros2 run tf2_ros tf2_echo base fr3_hand_tcp
```

不要同时运行 MoveIt 的 `fr3_arm_controller`。使用专门的外部目标阻抗控制器启动硬件：

```bash
ros2 launch franka_bringup example.launch.py \
  controller_names:=cartesian_impedance_policy_controller robot_ips:=172.16.0.2
```

另一个终端启动策略节点；默认只计算和发布调试参考：

```bash
source /ros2_ws/install/setup.bash
ros2 launch franka_pose_control stage3_policy.launch.py dry_run:=true
```

确认状态 topic 的关节顺序、TCP 坐标、目标和禁区配置后，发布一条原子命令：

```bash
ros2 run franka_pose_control stage3_command trial_001 0.55 0.10 \
  --avoidance --forbidden_x 0.48 --forbidden_y 0.02 --forbidden_radius 0.05
ros2 service call /fr3_stage3_policy/enable std_srvs/srv/SetBool "{data: true}"
ros2 topic echo /fr3_stage3_policy/status
ros2 topic echo /fr3_stage3_policy/reference_pose
```

停止策略：

```bash
ros2 service call /fr3_stage3_policy/stop std_srvs/srv/Trigger "{}"
```

完成 dry-run、低速空载和现场安全检查后，才可改为 `dry_run:=false`。节点不会自动切换控制器，也不会自动使能。
执行前可用 `ros2 control list_controllers` 确认只有预期的命令控制器处于 active 状态。

## 话题与服务

| 接口 | 类型 | 说明 |
|---|---|---|
| `~/command` | `Stage3PolicyCommand` | 最终目标、圆形禁区和实验编号的原子命令 |
| `~/reference_pose` | `PoseStamped` | 无论 dry-run 与否均发布的策略参考位姿 |
| `~/status` | `Stage3PolicyStatus` | 动作、误差、waypoint 与安全状态 |
| `~/enable` | `SetBool` | 在状态与命令有效后使能或取消使能 |
| `~/stop` | `Trigger` | 立即停止发送新目标并取消使能 |

策略节点发布到 `/cartesian_impedance_policy_controller/equilibrium_pose`。该控制器复用 Franka 示例笛卡尔阻抗控制器，但通过 `external_target_mode` 禁用了原示例的内部周期运动，避免覆盖策略目标。
控制器的零空间投影使用固定尺寸阻尼最小二乘求解，避免在 1 kHz 实时循环中执行动态尺寸 SVD 和堆内存分配。状态 broadcaster 以 100 Hz 更新，便利状态话题以 50 Hz 发布，足以支持 30 Hz 策略并减少实时循环负担。
策略增量会累积为阻抗平衡参考，避免相对当前 TCP 的固定小偏移在摩擦或模型误差下形成稳态停滞；累积参考与实测 TCP 的距离由 `max_reference_lead_m` 限制，默认 15 mm，且始终受训练工作空间约束。发布新命令、重新使能、停止或切换绕行 waypoint 时会重置累积参考。

设置控制器参数 `publish_diagnostics:=true` 时，控制器以 10 Hz 提供三个诊断话题：`~/accepted_equilibrium_pose` 是订阅回调实际接受的外部目标，`~/internal_equilibrium_pose` 是控制循环低通滤波后的内部平衡位姿，`~/commanded_joint_torques` 的 `effort` 是最终写入七轴 effort 接口的命令力矩。正式实时运行默认关闭这些诊断。
