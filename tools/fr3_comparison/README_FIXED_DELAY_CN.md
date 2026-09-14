# FR3 固定轨迹 + 延迟模拟（实机调试模式）

这是**延迟模拟实验**：只读取固定数据，不加载或调用模型。原始输入优先使用
`fr3_joint_sequences/A..D/debug_slow5x/joint_trajectory_points.json`。由于原始 slow5x 数据的离散
折点加速度超过当前 FR3 MoveIt 限位。推荐的 `fixed-generate-continuous` 会生成独立的
`fr3_joint_sequences_continuous`：全程 C2 连续、总时长 44 s，相对 slow5x 分段线性参考的最大
逐关节偏差不超过 0.0035 rad，并满足现场限位及 20% 加速度余量。旧的 `fixed-generate` 则生成
每 0.2 s 停稳一次的 132 s 保守版本。A/B/C/D 仅是轨迹来源标签，
派生数据不是原方法新结果；结果不得称为四种方法的真实推理速度或在线闭环结果。

## 安全与索引规则

- 节点默认 dry-run 且未使能；`fixed-run` 也必须另行调用 `enable` 才可能发送 action。
- 请求周期固定 0.2 s。每个请求只保留一个，串行等待设定的 0/10/50/100 ms；不补发错过的周期。
- 序列点 0 是起点。请求 1 只授权索引 1..4，请求 2 授权 5..8，以此类推直到最终四点。
  索引只在完整通过限位、速度、碰撞和控制器检查后提交。迟到、旧响应、重复预留、不完整末段、
  不跳点、不重复执行、不堆积并发请求。
- 连续派生序列每段以固定边界点为 t=0，四个固定目标位于 0.05/0.10/0.15/0.20 s；消息同时
  携带位置、速度和加速度，使 Jazzy `joint_trajectory_controller` 使用五次插值。正常情况下，
  下一 action 在完全相同的 q/dq/ddq 边界受控替换旧 action。旧 action 自带额外 0.2 s 的平滑
  制动尾段；下一请求迟到时不会盲目继续回放后续索引。
  设定延迟只推迟第一段及各次“请求到发送”，不按墙钟跳过轨迹点。
- 保留实测状态新鲜度、共同起点、URDF 位置/速度、0.03 rad 单步、0.6 rad/s、现场加速度限位、
  MoveIt 路径采样碰撞检查、规划总超时、控制器跟踪容差、取消和操作员停止。程序不会自动调大门限。
- 碰撞检查阶段经过 20 ms 后仅重发尚未完成的样本一次；请求总预算仍为 150 ms。重发也未在
  总截止时间前完成时才执行安全停止。

## 现场配置（必须确认，不能猜）

复制模板并填写现场事实：

```bash
cp tools/fr3_comparison/site_config.example.json /path/to/fr3_site.json
export FR3_SITE_CONFIG=/path/to/fr3_site.json
```

核对关节名、joint state/robot description/MoveIt 服务、控制器和 move group；填写七轴加速度限位。
`tool_transform`、`payload`、`collision_model` 必须分别写明夹爪/相机工具变换、机器人负载参数以及
MoveIt 中的夹爪、相机、支架、桌面等碰撞模型，并把 `verified` 改为 `true`。未填写时允许只做
dry-run 排查接口，但 `fixed-run` 会在启动参数检查阶段拒绝。

仓库模板反映当前单臂 FR3 配置，不代表现场确认。离线检查显示原始四条 slow5x 序列按仓库
`fr3_joint_limits.yaml` 的加速度限位均有折点超限（全部 J4，A/B 另有 J2），执行入口会硬性阻断。
请使用下面的连续安全派生序列，不要提高限位绕过。

## 可复制命令

构建并加载：

```bash
cd /ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select franka_pose_control
source /ros2_ws/install/setup.bash
cd /ros2_ws/src
export FR3_SITE_CONFIG=/ros2_ws/src/fr3_site.json
```

生成连续安全序列（不需要 ROS、不运动）。当前仓库已经生成好此目录；只有目录不存在或换了
现场限位时才需要运行生成命令。生成依赖 `python3-scipy`，输出目录必须尚不存在以防误覆盖：

```bash
export FR3_SEQUENCES=/ros2_ws/src/fr3_joint_sequences
export FR3_CONTINUOUS_SEQUENCES=/ros2_ws/src/fr3_joint_sequences_continuous
bash tools/fr3_comparison/fr3_test.sh fixed-generate-continuous
```

切换到安全序列并离线校验；成功返回 0：

```bash
export FR3_SEQUENCES=/ros2_ws/src/fr3_joint_sequences_continuous
bash tools/fr3_comparison/fr3_test.sh fixed-check --output /tmp/fr3_fixed_check.json
echo $?
```

启动硬件/MoveIt，并规划回固定序列共同起点：

```bash
export FR3_ROBOT_IP=192.168.0.1  # 必须替换成现场地址
bash tools/fr3_comparison/fr3_test.sh robot

FR3_TRIAL="/ros2_ws/src/fr3_joint_sequences/source_trial.json" \
  bash tools/fr3_comparison/fr3_test.sh home-plan
FR3_TRIAL="/ros2_ws/src/fr3_joint_sequences/source_trial.json" \
  bash tools/fr3_comparison/fr3_test.sh home
```

另一个终端做固定 A 轨迹、50 ms 延迟的 dry-run。启动后仍需 enable，dry-run 只发布预览并执行
现场状态/限位/速度/碰撞检查，不发送运动 action：

```bash
source /opt/ros/jazzy/setup.bash
source /ros2_ws/install/setup.bash
cd /ros2_ws/src
export FR3_SITE_CONFIG=/ros2_ws/src/fr3_site.json
export FR3_SEQUENCES=/ros2_ws/src/fr3_joint_sequences_continuous
bash tools/fr3_comparison/fr3_test.sh fixed-dry-run A 50
ros2 service call /fr3_comparison/enable std_srvs/srv/SetBool '{data: true}'
```

只有安全派生序列的离线检查、现场配置和 dry-run 全部通过后，才启动允许实机的入口：

```bash
bash tools/fr3_comparison/fr3_test.sh fixed-run A 50
ros2 service call /fr3_comparison/enable std_srvs/srv/SetBool '{data: true}'
```

停止/急停软件请求：

```bash
ros2 service call /fr3_comparison/stop std_srvs/srv/Trigger '{}'
```

这条服务会禁用客户端并请求取消当前 action；现场硬件急停仍使用机器人急停装置，不能由软件替代。

## 日志与图

原始 JSONL 记录实验类型、设定延迟、请求至发送耗时、固定索引、实测 q/dq、控制器反馈、末端
XYZ/四元数和停止原因。停止后自动生成 `*_reports/run_NNN/`，其中包括关节/末端 CSV、
`delay_timing.csv`、跟踪图、速度图、末端图和 `delay_comparison.png/.pdf`。不同延迟必须使用同一条
轨迹、分别回共同起点并分别运行；图中的延迟是模拟等待加安全检查/调度耗时，不是模型推理测速。
