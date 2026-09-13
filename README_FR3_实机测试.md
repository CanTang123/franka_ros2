# FR3 四方法实机测试：从部署、回起点到记录作图

本指南对应 `franka_ros2` 完整部署包。服务器运行原生 Ubuntu 24.04；同机运行 GPU/CPU 推理、ROS 2 Jazzy、MoveIt 和 FR3 硬件接口。所有命令从**解压后的 `franka_ros2` 仓库根目录**执行，不要求仓库必须位于 `/ros2_ws`。

本版补齐了回起点功能：根据 trial 中的七关节起点调用 MoveIt 规划，执行后以实测状态确认到位。回起点与模型执行分开计时、分别记日志；不会把准备运动算作方法结果。

## 1. 包内有什么，服务器还需要什么

- `local_models/`：两份原始权重、归一化参数、物理模型/网格、当前数学源码、示例 trial 及数值核对证据。
- `tools/fr3_comparison/fr3_test.sh`：硬件启动、回起点、检查、方法客户端、使能/停止的统一入口。
- `requirements-fr3-gpu.txt`：与原 5090 环境一致的核心 JAX/CUDA 包版本；引用 `local_models/requirements-cpu.txt`。
- `validation/`：本次软件验证证据。CPU/GPU 实际设备在报告的 `device_info` 中。
- Python/Conda 环境、ROS、libfranka、驱动和第三方 ROS 源码依赖不在压缩包中，首次安装仍需要软件源。

直接连接 FR3 的电脑需已有可用 FCI、PREEMPT_RT 内核、实时权限和控制柜 LAN 有线连接。**GPU 网络推理约每 0.2 秒调用一次；1 kHz 底层控制仍由 CPU 运行。** RT 内核与 NVIDIA 驱动组合需现场验证，不能用 GPU 推理通过代替实时控制验证。

官方依据：[Franka 系统要求](https://frankarobotics.github.io/docs/doc/libfranka/docs/system_requirements.html)、[实时内核与权限](https://frankarobotics.github.io/docs/doc/libfranka/docs/real_time_kernel.html)。

## 2. ROS 编译：选择符合当前服务器情况的一种

### 已有能运行 FR3 的工作区

保留现场机器人 IP、工具、负载、namespace、控制器增益等配置。把本版 `tools/fr3_comparison/` 和 `franka_pose_control` 的 CMake/package.xml 更新合入原仓库，保留完整 `local_models/`。在**原工作区根目录**执行：

```bash
source /opt/ros/jazzy/setup.bash
# 如果原工作区已经编译，先加载其 install/setup.bash
source install/setup.bash
sudo apt install python3-matplotlib
colcon build --symlink-install --packages-select franka_pose_control
source install/setup.bash
ros2 pkg executables franka_pose_control
```

应能看到 `fr3_comparison_node` 和 `fr3_return_to_start`。在同一 colcon 工作区只能保留一份同名 ROS 包。

### 新服务器，直接把 franka_ros2 仓库作为工作区根目录

先安装 [ROS 2 Jazzy](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html)、`ros-dev-tools`，以及与机器人系统版本兼容的 libfranka/Franka 依赖。仓库的原始 `dependency.repos` 提供依赖版本；已有现场验证的版本应优先沿用，不能盲目升级机器人软件。

在当前 `franka_ros2` 根目录执行以下命令。它们会联网导入仓库依赖，不会控制机械臂：

```bash
source /opt/ros/jazzy/setup.bash
vcs import . < dependency.repos --recursive --skip-existing
# 仅在这台电脑还未初始化 rosdep 时执行一次：sudo rosdep init
rosdep update
rosdep install --from-paths . --ignore-src --rosdistro jazzy -y --skip-keys=zed_wrapper
sudo apt install python3-matplotlib
colcon build --symlink-install \
  --packages-up-to franka_pose_control franka_fr3_moveit_config franka_bringup \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
ros2 pkg executables franka_pose_control
```

新机依赖安装/ROS 编译必须在目标机完成；此部署包不包含已经编译好的 ROS 二进制。

`fr3_test.sh` 会自动查找仓库或上级目录的 `install/setup.bash`。如果代码包与原工作区分开放置，在每个 ROS 终端设置：

```bash
export FR3_ROS_SETUP=/实际工作区/install/setup.bash
```

## 3. 推理终端：安装并验证 GPU 环境

在一个独立终端进入仓库根目录：

```bash
nvidia-smi
conda create -n fr3-inference python=3.11 pip -y
conda activate fr3-inference
python -m pip install -r requirements-fr3-gpu.txt
python -m pip check

export JAX_PLATFORMS=cuda,cpu
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python -c 'import jax; print(jax.__version__, jax.devices()); assert jax.default_backend() == "gpu"'

python tools/fr3_comparison/verify_runtime.py \
  --bundle local_models --output server_gpu_readiness.json
python tools/fr3_comparison/verify_transport.py \
  --bundle local_models --trial local_models/trials/A_000_s0.json \
  --output server_gpu_transport.json
```

两份报告必须 `passed: true`，并显示 `device_info.backend: gpu`。输出文件必须不存在；重复测试时换文件名保留旧记录。这里核对一组条件的完整 H16 输出、重复一致性、实测输入响应以及本机 HTTP，不能据此推断全部实机任务成功。

**`cuda,cpu` 必须同时启用：CUDA 排第一，网络仍默认在 GPU；CPU backend 供 B 方法的 MuJoCo 回调使用。** 单独设置 `cuda` 曾导致 B 方法报 `pure_callback failed to find a local CPU device`，本版启动入口也会把单独的 `cuda` 自动补为 `cuda,cpu`。

如果只使用 CPU，安装 `local_models/requirements-cpu.txt`，将 `JAX_PLATFORMS` 改成 `cpu`，其余验证相同。不能通过升级 JAX 或放宽数值门限跳过验证。GPU 模式下 B 方法的 native MuJoCo rollout 仍在 CPU。

验证通过后，在**同一推理终端**启动并保持运行：

```bash
JAX_PLATFORMS=cuda,cpu bash tools/fr3_comparison/run_local_inference.sh
```

等待 `"ready": true`，检查 `device_info`。服务监听 `127.0.0.1:8765`。不要把 GPU/Conda 环境加载进下面的 ROS 终端。

## 4. 硬件终端：启动 MoveIt 与真实机械臂

另开系统 ROS 终端进入仓库根目录。若自动激活 Conda，先 `conda deactivate`，直到没有激活的 Conda 环境。

```bash
export FR3_ROBOT_IP=172.16.0.2  # 改为实际控制柜 IP
export FR3_LOAD_GRIPPER=true   # 与现场夹爪/工具配置一致
bash tools/fr3_comparison/fr3_test.sh robot
```

已有正常运行的硬件/MoveIt launch 就保留它，不再启动第二份。无图形桌面时可追加 `use_rviz:=false`；初次规划预览需要在已连接 ROS 的 RViz 中查看。

本工具默认无 namespace，使用 `/joint_states`、`/robot_description`、`/move_action`、`/execute_trajectory`、`/check_state_validity`、`/controller_manager` 和 `/fr3_arm_controller`。有 namespace 时必须按第 9 节配置。

## 5. 操作终端：查看并回到 trial 起点

另开系统 ROS 终端，进入仓库根目录。默认试验使用 `local_models/trials/A_000_s0.json`，A/B/C/D 共用它。

先只规划、不执行：

```bash
bash tools/fr3_comparison/fr3_test.sh home-plan
```

检查 RViz `/display_planned_path` 的路径以及桌面、工具、负载、障碍物是否与现场 Planning Scene 一致。日志里的 `planned` 记录了七关节规划点和时间。规划不是简单地把当前关节角直线插到起点。

确认现场路径可执行后，运行下面的**运动命令**：

```bash
bash tools/fr3_comparison/fr3_test.sh home
```

`home` 会从当前实测位置**重新规划**，检查路径并经 MoveIt ExecuteTrajectory 执行；不是回放上一次 `home-plan` 的缓存。它采用最多 10% 的速度/加速度缩放，另检查关节速度不超过 0.3 rad/s。执行后必须检测到各关节距 trial 起点不超过 0.01 rad、实测速度不超过 0.05 rad/s，并持续 0.3 秒，才返回成功。

单独复查是否到位：

```bash
bash tools/fr3_comparison/fr3_test.sh start-check
```

如果起点碰撞、规划失败、关节状态超过 100 ms 未更新、机器人在规划后移动、控制器未激活或执行失败，工具报错退出，不修改 trial 起点来让检查通过。回起点日志在 `experiment_logs/home_*.jsonl`，不会混入方法结果。

## 6. 先做 dry-run，再运行 A

仍在操作终端，启动 dry-run：

```bash
bash tools/fr3_comparison/fr3_test.sh dry-run A
```

客户端启动后仍处于未使能状态。在**另一个系统 ROS 终端**进入仓库根目录，执行：

```bash
bash tools/fr3_comparison/fr3_test.sh enable
```

dry-run 会读取真实状态、调用模型并检查目标和碰撞，但不发送运动 action。可查看 `/fr3_comparison/reference_trajectory`。机械臂静止而 trial 相位继续推进，后续目标可能被步长限制拒绝；此模式是接口检查，不是完整闭环任务成功验证。

完成 dry-run 后，在客户端终端按 Ctrl+C，等待保存结果并退出。**比较客户端整个存活期间都持有控制锁，即使尚未 enable；回起点前必须先退出它。**

如有需要先 `home`，然后启动允许实机执行的客户端：

```bash
bash tools/fr3_comparison/fr3_test.sh home
bash tools/fr3_comparison/fr3_test.sh run A
```

在另一系统 ROS 终端调用：

```bash
bash tools/fr3_comparison/fr3_test.sh enable
```

**这一次 enable 会开始模型控制的机械臂运动。** 运行前保留现场急停与既有 Franka 安全设置。不要同时运行 RViz Execute、pose_target 或其他发送同一控制器目标的程序；本工具的进程锁只约束同用户的本工具，不锁定所有 ROS 应用。

运行按 trial 时长自动停止并记录。为开始下一次试验，先 Ctrl+C 退出客户端，再回起点；不要直接重新 enable 从终点继续试验。

## 7. B/C/D 怎么比较，如何停止

每种方法都执行同一顺序：退出上一客户端 → `home` → `run 方法` → 另一终端 `enable`。例如：

```bash
bash tools/fr3_comparison/fr3_test.sh home
bash tools/fr3_comparison/fr3_test.sh run B
# 在另一 ROS 终端执行 fr3_test.sh enable
```

之后把 B 改成 C 或 D。每次命令自动生成唯一日志文件名；记录试验顺序、重复次数、控制器设置和失败原因，保持起点、trial、工具和超时门限一致。

|方法|网络/计算|
|---|---|
|A DeMoFlow|mean25k 状态网络 + decoder10k，J=1、t*=0.5|
|B 动力学对照|共享 decoder + native MuJoCo rollout + decoder，J=1、t*=0.5|
|C Direct|共享 decoder，初始噪声在 t=0 解码|
|D Bypass|共享 decoder，初始噪声在 t=0.5 解码|

B 不是原生 DynaFlow 默认设置；当前包仍是原冻结 seed9/mean25k/decoder10k，不是后续 miniscan 中自动挑选的新模型。

随时在另一系统 ROS 终端请求停止：

```bash
bash tools/fr3_comparison/fr3_test.sh stop       # 停止方法客户端
bash tools/fr3_comparison/fr3_test.sh home-stop  # 停止回起点
```

也可以在对应进程按 Ctrl+C。软件停止会请求取消目标，不等于已确认机械制动；需要紧急停止时使用现场硬件急停。

## 8. 结果在哪里

默认保存到仓库根目录 `experiment_logs/`，可在 ROS 终端通过 `FR3_LOG_DIR` 改成持久数据盘路径。日志名包含方法、run/dry-run 和时间，不覆盖旧结果。

```text
experiment_logs/
  home_时间.jsonl
  A_run_时间.jsonl
  A_run_时间_reports/run_001/
    joint_states.csv          # 实测 q1..q7、dq1..dq7、时间戳
    ee_pose.csv               # 实测 XYZ、四元数 xyzw、坐标系、时间戳
    controller_feedback.csv   # 控制器目标、实测与误差（若收到）
    plans.jsonl
    summary.json
    joint_positions.png/.pdf
    joint_velocities.png/.pdf
    ee_trajectory.png/.pdf
    ee_position.png/.pdf
    ee_orientation.png/.pdf
```

停止后继续记录 1 秒再自动作图。保持客户端运行或正常 Ctrl+C 等待它完成保存。缺末端话题时仍记录关节，并在摘要中报告缺失，不会生成伪造的末端轨迹。默认末端话题 `/franka_robot_state_broadcaster/current_pose` 对应硬件 `O_T_EE`，不保证是你的工具 TCP。

`summary.json` 给出实际起点、停止前和最终关节/末端值以及停止原因；它不把到达时长当成任务成功，也不把末端位移当作轨迹跟踪误差。任务误差需结合现场 TCP 和参考路径标定另行定义。

手动作图（系统 Python 安装了 Matplotlib 时）：

```bash
python3 tools/fr3_comparison/report_results.py \
  experiment_logs/实际日志名.jsonl --output experiment_logs/新报告目录
```

## 9. 常见配置与报错

|现象|处理|
|`Another comparison/home process owns...`|Ctrl+C 退出另一个比较或回起点进程；进程结束锁自动释放，不要删锁文件绕过互斥|
|找不到 `fr3_return_to_start`|重新编译 franka_pose_control 并加载正确 install/setup.bash|
|缺关节状态/robot_description/MoveIt|检查硬件 launch、ROS_DOMAIN_ID、namespace 和话题名|
|`Measured state differs...`|退出客户端后执行 `home`，不要放宽起点门限|
|回起点碰撞/规划拒绝|核对现场场景、工具及目标可达性；不要绕过校验|
|推理连接失败|确认推理终端 ready、同机端口 8765 和相同 trial|
|GPU 载入失败|检查驱动、固定依赖和设备；不要误把 CPU 回退当成 GPU 验证|
|周期超时|保留日志，测量推理/MoveIt/调度耗时；不提高 150 ms 门限掩盖问题|
|末端 CSV 空|检查实际 PoseStamped 话题，使用 `--ee-pose /实际话题`|

示例：修改比较客户端接口：

```bash
bash tools/fr3_comparison/fr3_test.sh run A \
  --joint-states /robot/joint_states \
  --robot-description /robot/robot_description \
  --state-validity /robot/check_state_validity \
  --controller-manager /robot/controller_manager \
  --controller /robot/fr3_arm_controller \
  --ee-pose /robot/franka_robot_state_broadcaster/current_pose
```

回起点也要传递同样的关节/描述/控制器参数，并按现场设置 `--move-action /robot/move_action --execute-action /robot/execute_trajectory`；`--controller` 必须与比较客户端完全一致。客户端自身仍是根 namespace 的 `/fr3_comparison`，enable/stop 命令不变。

换 trial 时，在推理和 ROS 两侧终端设置同一个 `export FR3_TRIAL=/绝对路径/trial.json`，然后重启推理服务。不要只改一侧。默认包只提供配对示例条件；任意目标路径需要相应有效 descriptor/trial，并非随意输入末端点即可运行。

## 10. 已验证范围

本版回起点工具使用 MoveIt MoveGroup（仅规划）和 ExecuteTrajectory（显式执行），包括限位、轨迹时间、速度、首末点、当前场景采样检查和实测到位校验。路径采样不是连续碰撞证明；执行依赖现场 MoveIt、控制器和正确 Planning Scene。

本版已通过 **31 项离线测试**。在开发服务器 Ubuntu 24.04、RTX 5090、Python 3.11/JAX 0.6.2 环境中，从工作区外的完整部署目录运行：GPU 四方法 H16 数值核对、本机 HTTP 请求与错误 trial 拒绝、CPU 数值回归均通过。GPU 报告最大差异为 1.19e-7 rad；它是单条件首段核对，不是完整实机性能测试。首次 CUDA-only 失败及修复后的记录都保存在 `validation/`。环境使用已有验证环境，未声称在全新系统完成过依赖安装。

新增离线测试使用模拟 ROS transport，覆盖计划/执行分离、动作失败、迟到接受取消、规划后机器人移动、场景拒绝、控制互斥与输入校验；不会把它们标成实机测试。本服务器没有安装 ROS 2，尚未完成真实 ROS 编译或回起点硬件执行。软件验证明细见包内 `validation/`；迁移后必须先完成本指南中的目标机验证和 dry-run。
