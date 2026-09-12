# Franka 笛卡尔位姿控制

Stage 3 强化学习策略的真机桥接、dry-run 和安全启停说明见 `README_STAGE3.md`。

`pose_target_node` 为 FR3 提供带安全确认机制的普通点到点、笛卡尔直线和 Home 控制。使用本节点前，机器人应已通过
`franka_fr3_moveit_config/moveit.launch.py` 启动，并且 MoveIt 和真实机械臂连接正常。

节点刻意将规划和执行分为两个阶段，避免发布一条错误位姿后机器人立即运动：

1. 向 `~/target_pose` 发布普通点到点目标，或向 `~/linear_target_pose` 发布直线目标；节点校验目标并且**只进行规划**。
2. 规划成功后，轨迹会显示在 RViz 中，并在短时间内缓存。
3. 检查轨迹正确后，显式调用 `~/execute` 服务才会执行运动。
4. 调用 `~/stop`、按 `Ctrl+C` 或向进程发送 `SIGTERM`，都会请求取消规划或执行，并清除缓存轨迹。

## 编译

在容器内执行：

```bash
cd /ros2_ws
colcon build --symlink-install --packages-select franka_pose_control
source install/setup.bash
```

## 用两个 launch 规划和执行（推荐）

先保持 MoveIt 真机 launch 运行。编辑 `config/targets.yaml` 设置各个命名目标，每次选择一个目标规划，
不自动连续执行目标列表。`mode` 为 `line`（直线）、`ptp`（点到点）或 `home`；
`position` 为米，`relative: true` 表示沿 base 坐标轴的相对位移；省略 `orientation` 时保持当前姿态。
如需指定姿态，`orientation` 为四元数 `[x, y, z, w]`。

终端一启动规划（默认示例是向上 1 cm）：

```bash
ros2 launch franka_pose_control plan.launch.py target:=up_1cm
```

此 launch 同时启动位姿控制节点，并保持运行以保存缓存。不要同时另开 `pose_target.launch.py`。
发送端会先通过 `planning_ready` 服务等待控制节点发现 MoveIt 的规划接口（launch 默认等待上限
15 秒），就绪后才发送目标。若超时，请检查 MoveIt launch；目标不会自动重试发送。
等待日志出现 `Complete Cartesian line ready`（直线）或 `Plan ready`（点到点/Home），在 RViz 检查轨迹。
默认缓存有效期为 10 秒；过期应重新规划。需要更长检查时间时，可在启动前修改 `pose_target.yaml` 中
`max_plan_age`，执行前的起点漂移检查仍保留。

终端二启动执行：

```bash
ros2 launch franka_pose_control execute.launch.py
```

这会请求执行当前缓存的轨迹，执行 launch 的客户端结束不代表机器人已运动完成，最终结果看规划节点日志。
保持终端一运行；关闭它会清除缓存，并请求取消正在执行的运动。

已有位姿控制节点时，后续规划使用 `start_node:=false`，避免创建同名节点：

```bash
ros2 launch franka_pose_control plan.launch.py start_node:=false target:=point_a
ros2 launch franka_pose_control plan.launch.py start_node:=false target:=point_b
ros2 launch franka_pose_control plan.launch.py start_node:=false target:=home
```

以上为三个独立用法示例，不要连续发送；每次规划成功并检查后，再单独启动执行 launch。
可用 `mode:=ptp` 或 `mode:=line` 覆盖该目标的模式，也可用 `targets_file:=/绝对路径/targets.yaml`
加载自己的目标文件。`start_node:=false` 的规划客户端退出后，已有控制节点继续保存轨迹。

运行中需要停止时，在保持运行的规划节点终端按 Ctrl+C，或使用 `pose stop`。

## 启动

终端一启动并保持标准 MoveIt 真机 launch：

```bash
ros2 launch franka_fr3_moveit_config moveit.launch.py \
  robot_ip:=172.16.0.2 \
  load_gripper:=true
```

终端二启动位姿控制节点：

```bash
cd /ros2_ws
source install/setup.bash
ros2 launch franka_pose_control pose_target.launch.py
```

本节点不会启动第二个硬件接口，也不能与另一个 `ros2_control_node` 配合启动。它复用已经运行的
MoveIt `move_group` 和 `fr3_arm_controller`。

## 简化命令（推荐）

编译并启动上述节点后，在另一个容器终端执行：

```bash
source /ros2_ws/install/setup.bash
alias pose='ros2 run franka_pose_control pose'
```

随后可以用以下短命令。位置单位为米；省略 `--quat` 时自动读取新鲜 TCP 位姿并保持当前姿态。

```bash
# 绝对位置目标，只规划，数值请按当前工作区修改
pose line 0.55 0.0 0.52
pose ptp 0.55 0.0 0.52

# 沿 base 的 Z 轴向上 1 cm，保持当前姿态，只规划
pose line 0 0 0.01 --relative

# 明确指定目标四元数，顺序 x y z w
pose line 0.55 0.0 0.52 --quat 1 0 0 0

pose home       # 只规划回 Home
pose execute    # 在 RViz 检查规划成功后，显式执行
pose stop       # 请求停止并清除缓存
```

`--relative` 的位移沿 `base` 轴计算，不是沿工具自身坐标轴。目标发布成功并不代表规划成功，
请查看控制节点日志。所有目标仍受到节点原有的距离、工作区、忙碌状态和轨迹有效期限制。
`pose execute` 的服务响应只表示执行请求状态，最终运动结果查看控制节点日志。
短命令退出或按 Ctrl+C 只结束命令客户端；要取消控制节点中的运动，请运行 `pose stop`。

此 alias 仅在当前终端有效；不设置 alias 时使用 `ros2 run franka_pose_control pose` 替代 `pose`。
自定义接口可通过全局参数指定，例如 `pose --frame base --node /franka_pose_target line 0 0 0.01 --relative`。

## 读取当前位姿

读取 Franka 硬件广播的末端执行器位姿：

```bash
ros2 topic echo \
  /franka_robot_state_broadcaster/current_pose \
  --once \
  --qos-reliability best_effort
```

或者查询 MoveIt 使用的夹爪 TCP：

```bash
ros2 run tf2_ros tf2_echo base fr3_hand_tcp
```

目标位置的单位为米，目标姿态使用四元数，字段顺序为 `x, y, z, w`。

## 发送目标位姿

目标必须使用配置文件中指定的 `base` 坐标系。下面的示例保持一次实测姿态不变，仅将 Z 方向向上
移动 20 mm：

```bash
ros2 topic pub --once \
  /franka_pose_target/target_pose \
  geometry_msgs/msg/PoseStamped \
"{header: {frame_id: base},
  pose: {
    position: {
      x: 0.644517,
      y: -0.144972,
      z: 0.487255
    },
    orientation: {
      x: 0.992201,
      y: -0.097227,
      z: 0.016096,
      w: -0.076320
    }
  }}"
```

发布目标后机械臂不会运动。节点只会请求 MoveIt 规划，并在规划成功时输出：

```text
Plan ready and displayed in RViz
```

请先在 RViz 中检查规划轨迹，确认轨迹平滑、方向正确且不会发生碰撞。

## 发送笛卡尔直线目标

向 `linear_target_pose` 发布目标时，MoveIt 会从当前实测 TCP 位姿到目标位姿进行笛卡尔插值和逐点
逆运动学计算。下面示例保持姿态不变并沿 Z 方向向上直线移动 20 mm：

```bash
ros2 topic pub --once \
  /franka_pose_target/linear_target_pose \
  geometry_msgs/msg/PoseStamped \
"{header: {frame_id: base},
  pose: {
    position: {
      x: 0.644517,
      y: -0.144972,
      z: 0.487255
    },
    orientation: {
      x: 0.992201,
      y: -0.097227,
      z: 0.016096,
      w: -0.076320
    }
  }}"
```

发布后仍然只规划、不运动。默认要求整条直线路径 `fraction = 1.0`，并且要求：

- 所有插值点均能求出连续的关节解。
- 所有插值点和连接段均通过 MoveIt 碰撞检查。
- 相邻插值点的任何旋转关节变化不超过 0.20 rad。
- 返回轨迹包含有效且递增的时间戳。

任一条件不满足，整条轨迹都会被拒绝，不允许执行部分路径。规划成功后，先在 RViz 检查，再调用
`/franka_pose_target/execute`。笛卡尔直线规划不会绕开障碍物；直线上存在障碍物时应当规划失败。

## 规划回到 Home 关节位姿

配置文件已加入以下 Home 关节角，单位为弧度：

```text
J1 = 0.0
J2 = 0.0
J3 = 0.0
J4 = -1.5708
J5 = 0.0
J6 = 1.5708
J7 = 0.7854
```

调用以下服务，只规划回到 Home 的轨迹：

```bash
ros2 service call \
  /franka_pose_target/plan_home \
  std_srvs/srv/Trigger \
  "{}"
```

该服务不会立即移动机械臂。规划成功后先在 RViz 检查整条轨迹，再在默认 10 秒有效期内调用下面的
`/franka_pose_target/execute` 服务。Home 是关节空间目标，不受笛卡尔目标的 50 mm 单次平移限制；
因此从当前位置到 Home 可能是一段较长运动，必须确认现场无障碍物且 Planning Scene 与现场一致。

Home 数值和允许误差可以在 `config/pose_target.yaml` 中通过 `home_joint_names`、
`home_joint_positions` 和 `home_joint_tolerance` 修改。两个数组必须长度相同且关节名称不能重复。

## 执行已检查的轨迹

默认情况下，规划结果只保留 10 秒。在有效期内显式调用：

```bash
ros2 service call \
  /franka_pose_target/execute \
  std_srvs/srv/Trigger \
  "{}"
```

如果计划已经过期、机器人偏离规划起点，或者没有可用计划，执行请求会被拒绝。此时需要重新发布目标并
检查新轨迹。

## 停止和安全退出

运行期间可以随时请求取消规划或轨迹执行：

```bash
ros2 service call \
  /franka_pose_target/stop \
  std_srvs/srv/Trigger \
  "{}"
```

在节点终端按 `Ctrl+C` 时，节点会：

1. 清除尚未执行的缓存计划。
2. 向正在运行的 MoveIt action 发送取消请求。
3. 最多等待 2 秒接收取消结果。
4. 停止 ROS executor 并退出进程。

软件取消不能代替机器人的安全功能。如果机器人没有按预期停止，应立即使用机器人规定的安全控制装置。

## 默认安全限制

默认参数位于 `config/pose_target.yaml`：

- 关节最大速度比例：5%。
- 关节最大加速度比例：5%。
- 普通点到点目标最大平移距离：50 mm。
- 普通点到点目标最大旋转角度：0.35 rad。
- 笛卡尔直线最大长度：200 mm，最大姿态变化：0.35 rad。
- 笛卡尔插值间距：5 mm，TCP 最大速度：0.05 m/s。
- 笛卡尔路径必须完整，并启用碰撞检查和关节跳变限制。
- 当前 TCP 状态最大允许延迟：1 秒。
- 规划结果有效期：10 秒。
- 执行前最大位置漂移：10 mm。
- 执行前最大姿态漂移：0.10 rad。
- 同一时刻只允许一个规划或执行操作。
- 目标必须位于配置的 TCP 工作空间包围盒内。
- Home 使用上述 7 个关节角和 0.005 rad 的目标容差。

可以编辑参数文件调整限制，但建议先在仿真或很小的真机运动范围内验证。工作空间包围盒只是粗略边界，
不能替代 MoveIt 碰撞模型。

## 重要安全说明

MoveIt 默认只能检测机器人模型中已有的碰撞对象。现场桌面、夹具、相机、工具、线缆和其他障碍物必须
加入 Planning Scene，否则 MoveIt 不知道它们存在。

真机执行前至少确认：

- 机器人工作区内无人。
- 急停和规定的安全控制装置可随时触及。
- Desk 中 FCI 模式保持激活。
- `fr3_arm_controller`、`joint_state_broadcaster` 和
  `franka_robot_state_broadcaster` 均为 `active`。
- 已在 RViz 中检查本次规划轨迹。
- 目标位姿使用正确的基坐标系和 TCP。
