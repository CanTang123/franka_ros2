# FR3 四方法实机闭环接入

**完整中文测试步骤：[回起点 → 起点检查 → dry-run → A/B/C/D → 记录作图](../../README_FR3_实机测试.md)。**
新增 `return_to_start.py`（ROS 可执行名 `fr3_return_to_start`）；默认只规划，`--execute` 才经 MoveIt 执行，`--check-only` 只检查实测到位。`fr3_test.sh` 在仓库根目录提供 `home-plan/home/start-check/run/dry-run/enable/stop/home-stop` 命令。
比较客户端与回起点工具在同用户、同控制器下进程互斥；回起点前须退出比较客户端（仅 stop 不会释放进程锁）。新包为 `fr3_full_workflow_20260912.tar.gz`，GPU 依赖见仓库根目录 `requirements-fr3-gpu.txt`。

**当前版本在实机电脑本地运行 A/B/C/D 推理和 ROS 控制，不依赖本服务器或 SSH 隧道。** 使用包含权重的 `fr3_local_inference_complete.tar.gz`。之前约 5 MB 的 `fr3_deployment_with_recording.tar.gz` 只有源码，不能用于独立本地推理。

同一台实机电脑启动两个进程：Python 3.11/JAX 推理进程与 ROS 2 系统 Python 客户端。两者只通过 `127.0.0.1:8765` 通信，避免把 JAX 依赖混入 ROS 环境。默认 CPU；首次安装 Python/ROS 软件依赖仍需要可用的软件源或预先安装的环境。

实现入口：`inference_server.py`（实机电脑本地推理）、`fr3_comparison_node`（ROS 2 实机端）。默认 dry-run、未使能。记录导入与 RViz 预览为辅助功能，不能替代实测状态重规划。

|方法|含义|FR3 冻结推理|
|---|---|---|
|A|DeMoFlow 完整方法|独立状态网络更新一次，然后动作解码；J=1，t*=0.5|
|B|共享解码器的动力学对照|decoder + native MuJoCo rollout 给出状态端点，再解码；J=1，t*=0.5；不是原生 DynaFlow 默认配置|
|C|Direct|不更新初始噪声，在 t=0 解码一次|
|D|Bypass 消融|不更新初始噪声，在 t=0.5 解码一次|

四组共用 seed9 的 decoder10k EMA；A 使用 mean25k EMA。此入口没有接入旧 action-only CFM / joint-CFM、原生 DynaFlow 或其他训练 seed，不应把 B/C/D 重命名为这些方法。

## 1. 包内文件与推理环境

完整包顶层为 `franka_ros2/`，包含以下本地推理数据：

```text
local_models/
  checkpoints/mean25k.pkl        # A 使用的状态网络，原始 checkpoint
  checkpoints/decoder10k.pkl     # A/B/C/D 共用的动作解码器，原始 checkpoint
  data/stats.npz                 # 冻结归一化参数、关节映射参数
  model/execution.xml           # B rollout 的原始物理模型
  assets/link*.stl               # 同一模型引用的八个碰撞网格
  source/                       # 所需 JAX 网络、FR3 数学代码与 native rollout
  trials/                       # 配对示例任务 A/B/C/D、INDEX.json
  evidence/                     # 用于离线数值核对的原始计划记录
  FROZEN_CANDIDATES.json         # 原始训练身份，不改写
  bundle.json                   # 相对路径映射与所有文件 SHA256
  requirements-cpu.txt           # 冻结 Python 依赖
```

两个原始 checkpoint 共约 329 MB（未压缩），还保留训练元数据和优化器内容，推理仅加载其中 EMA。无需另外下载四份独立权重。包内历史 `/home/wqf/...` 字符串只用于来源追溯，运行时通过相对路径映射解析本地文件；XML 网格目录仅在内存中重定位，物理参数不变。加载器不会读取训练 episodes.npz。

在实机电脑解压后进入 `franka_ros2` 目录。推理环境要求 Python 3.11；例如有 Conda 时：

```bash
conda create -n fr3-inference python=3.11 pip
conda activate fr3-inference
python -m pip install -r local_models/requirements-cpu.txt
```

用包内证据在这台电脑进行验证（输出文件必须不存在）：

```bash
JAX_PLATFORMS=cpu python tools/fr3_comparison/verify_runtime.py \
  --bundle local_models --output local_readiness.json
```

验证要求四方法完整 H16 与原始计划的最大差异不超过 2e-6rad、重复请求一致，且修改实测输入后输出发生变化。它还会报告该电脑的一次预热后推理耗时；这不是实时性能保证。CPU 型号不同可能影响 0.2s 周期是否满足，保持原有超时门限，不通过增大门限掩盖问题。

## 2. 构建 ROS 客户端并启动本地推理

保留实机电脑原有 IP、namespace、控制器增益及其他现场配置，将源码更新合入现有工作区。不要在同一工作区保留两个同名 ROS 包。ROS 环境只需新增 Matplotlib 作图依赖：

```bash
sudo apt install python3-matplotlib
cd /ros2_ws
colcon build --symlink-install --packages-select franka_pose_control
source install/setup.bash
```

推理终端使用 Python 3.11 环境，在解压后的 `franka_ros2` 目录运行：

```bash
conda activate fr3-inference
bash tools/fr3_comparison/run_local_inference.sh
```

等待 `"ready": true`，表示包内文件哈希检查及四种模型接口预热完成。默认加载 `local_models/trials/A_000_s0.json`。可追加 `--trial /absolute/path/another_trial.json`，并让 ROS 客户端使用完全相同的 trial。服务仅监听本机端口，**无需本服务器 IP 或 SSH 连接**。

另一个终端使用已有 ROS 2 Jazzy 系统环境；保留 MoveIt/hardware launch，客户端复用 `/fr3_arm_controller/follow_joint_trajectory`，不启动第二套硬件接口。例如先按下一节启动 dry-run，trial 指向 `local_models/trials/A_000_s0.json` 的实际绝对路径。

依赖 `control_msgs`、`trajectory_msgs`、`controller_manager_msgs`、`moveit_msgs` 和 `rclpy` 已声明在 package.xml。ROS 终端无需激活 JAX 环境；模型权重由同机的独立推理进程读取。

## 3. 从实测状态做 dry-run

```bash
ros2 run franka_pose_control fr3_comparison_node \
  --trial /absolute/path/fr3_trial.json --method A --log /absolute/path/A_dryrun_001.jsonl
```

节点启动后仍未使能：

```bash
ros2 service call /fr3_comparison/enable std_srvs/srv/SetBool '{data: true}'
ros2 topic echo /fr3_comparison/reference_trajectory --once
```

dry-run 使用新鲜实测状态向模型请求计划、检查关节限位和碰撞、发布调试轨迹，但不发送控制 action。机械臂静止时任务相位仍按墙钟推进，因此后续目标可能被步长限制拒绝；这是接口联调，不是闭环运动成功。日志中的 `stopped` 给出具体原因。

默认接口是 `/joint_states`、`/robot_description`、`/check_state_validity`、`/controller_manager/list_controllers`、规划组 `fr3_arm`。有 namespace 时用命令行 `--joint-states`、`--robot-description`、`--state-validity`、`--controller-manager`、`--controller`、`--move-group` 指定实际名称。关节名称固定为 fr3_joint1..7，接收时按名字重排，不按 topic 的数组顺序猜测。

## 4. 现场执行与停止

完成 dry-run、现场模型/工具/障碍物核对后，退出旧客户端，以新日志名启动同一命令并添加 `--execute`。依然需要调用 `enable` 才会发送目标。各方法分别运行，将 `--method` 改为 B、C、D，trial、部署限制和控制器参数保持一致；每次回到同一起点，实验顺序和重复次数单独记录。

停止：

```bash
ros2 service call /fr3_comparison/stop std_srvs/srv/Trigger '{}'
```

Ctrl+C/SIGTERM 也请求取消 action；如果停止后才收到迟到的 action 接受回复，客户端会取消该目标。软件停止并不代表机械制动已被测量确认，现场仍使用既有急停和 Franka 安全设置。

## 5. 自动记录末端与关节，运行结束后作图

正常到达运行时长、调用 stop、控制异常停止和 Ctrl+C 都会保留当前运行的数据。默认停止后继续接收 1 秒数据，再由独立进程生成 CSV/JSON 与 PNG/PDF 图，不在控制回调里画图。节点保持运行时也会自动出图，无需关闭节点。新一次 enable 会在停止过程记录完后开始，并获得新的 run ID。

例如运行：

```bash
ros2 run franka_pose_control fr3_comparison_node \
  --trial /absolute/path/fr3_trial.json --method A \
  --log /absolute/path/results/A_trial001.jsonl
```

仍然默认 dry-run、需要 enable；现场执行时按前文添加 `--execute`。运行停止后生成：

```text
results/
  A_trial001.jsonl                     # 持续写入的原始日志
  A_trial001_reports/
    run_001_report.log                # 作图进程状态/错误
    run_001/
      joint_states.csv                # 七关节 q(rad)、dq(rad/s)、时间戳
      ee_pose.csv                     # 末端 XYZ(m)、四元数xyzw、frame、时间戳
      controller_feedback.csv         # 控制器目标/实测/误差（若收到 action feedback）
      plans.jsonl                     # 每次规划原始记录，不冒充实际运动
      summary.json                    # 初始/停止前/最终位置、范围、峰值速度、缺失记录
      joint_positions.png / .pdf
      joint_velocities.png / .pdf
      ee_trajectory.png / .pdf         # 三维轨迹和XY投影
      ee_position.png / .pdf           # XYZ随时间变化
      ee_orientation.png / .pdf        # 四元数随时间变化
```

末端实测默认订阅 `/franka_robot_state_broadcaster/current_pose`（`PoseStamped`）；有 namespace 时用 `--ee-pose /实际话题`。数据来自硬件广播的 `O_T_EE`，按消息中的 `frame_id` 保存，不擅自改成 `fr3_hand_tcp` 或用仿真轨迹代替。四元数 CSV 保留收到的 xyzw；图中只做归一化及正负号对齐便于阅读。坐标系变化时分别作图，超过 100ms 的观测空缺处断开曲线。

关节曲线为订阅收到的实测 q/dq；如果有控制器 feedback，位置图另外用虚线画其 desired 目标。默认按收到的消息频率记录，不额外下采样，不保证网络丢包时仍有完整硬件频率。CSV 同时保留本机接收时间和源 ROS 时间戳，各传感器序列不强行插值对齐。图的时间原点为 enable，竖虚线表示 disable，后续数据展示停止过程。

`summary.json` 包含停止原因、dry-run 标记、末端最终 XYZ/四元数、位移和观测轨迹长度，以及各关节变化范围和峰值速度。末端位移、姿态变化是相对于本次起点的变化量，不是任务跟踪误差或成功判定。无末端消息时仍保存关节 CSV/图，并在 summary 中报告缺失，不伪造末端曲线。

可用 `--post-stop-seconds 2` 延长停止后记录，或 `--no-auto-report` 仅记录、稍后作图。原始 JSONL 持续写盘；进程异常退出后可手动恢复报告（最后一行截断会被标记并跳过）：

```bash
ros2 run franka_pose_control report_results.py \
  /absolute/path/results/A_trial001.jsonl \
  --run-id 1 --output /absolute/path/results/A_trial001_replot
```

也可以把日志带回本服务器，只依赖 Python/Matplotlib，无需 ROS：

```bash
python tools/fr3_comparison/report_results.py A_trial001.jsonl --output A_trial001_report
```

加 `--no-plots` 时只导出数据和摘要。输出目录必须是新目录；不同方法/重复实验分别命名，保留失败和中止记录。自动作图缺少 Matplotlib 时 CSV/摘要仍可导出，错误写入 `run_001_report.log` 和 summary 的 warnings。

## 执行协议与日志解释

- 规划器返回 H16（0.8s）绝对关节目标；实机一次只授权前 4 个目标，每个 0.05s。请求按 0.2s 周期发出，任务相位使用实机端实际经过的墙钟时间，随机源按 control_step=0/4/8/... 折叠。
- 每段 ROS 轨迹以校验过的实测起点为 t=0，四个目标放在 0.05/0.10/0.15/0.20s；未添加平滑或逐方法调参。已有 JointTrajectoryController 会插值位置并通过 effort PID 执行，**这与仿真零阶保持位置伺服不是相同底层协议**。需记录控制器版本、增益、现场工具和负载；这里的实机结果作为独立实验报告。
- 默认整个网络/推理/MoveIt 校验预算 150ms，状态最大延迟 100ms，关节起点容差 0.03rad，单目标步长 0.03rad，测得及目标段平均速度上限 0.6rad/s。超时、状态异常、目标跳变、限位、碰撞检查拒绝、控制器失败均停止并请求取消，不裁剪模型动作掩盖失败。
- MoveIt 在当前起点与四个关节线性目标之间每至多 0.01rad 采样检查；这是有限采样，不能当作连续碰撞证明，且依赖现场 Planning Scene 正确包含桌面、工具等对象。
- 网络和校验引入实际动作起始延迟，段间可能出现保持或目标替换；没有把这段延迟假定为零。每次 `plan_ready` 记录请求状态、物理相位、完整 H16、执行前缀、权重身份和端到端耗时；`measured_state` 记录时间戳和 q/dq，`controller_result` 记录控制器结果。`trial duration reached` 仅表示达到时长，不是任务成功。
- dry-run/超时/中止必须保留。实机末端误差、成功判据、姿态标定及跨试验统计尚需按现场 TCP/参考轨迹定义，不能直接复用仿真表格数字。

官方接口依据：[Jazzy JointTrajectoryController 文档](https://control.ros.org/jazzy/doc/ros2_controllers/joint_trajectory_controller/doc/userdoc.html)。

## 已验证与尚未验证

最新工作流版：31 项离线测试通过；部署目录移到原工作区外后，RTX 5090 GPU 四方法数值与本机 HTTP 通过，CPU 数值回归通过。GPU 设置需 `JAX_PLATFORMS=cuda,cpu`，B 方法的 native MuJoCo 回调需要 CPU backend；本版也兼容将单独 `cuda` 自动补全。回起点尚未进行真实 ROS 构建/硬件执行。以下为先前各版本的验证记录，最新证据见完整包 `validation/`。

已做：四组源记录导入；协议/异常输入测试；模拟 ROS transport 的状态超时、网络超时、碰撞拒绝、停止后迟到接受等路径；CPU 上一组冻结 condition/source 的完整 H16 与历史计划核对；真实本地 HTTP 服务的四方法请求与错误任务拒绝。

新增记录/作图检查覆盖：停止后的末端采样、重复时间戳、多次运行隔离、坐标系变化、末端缺失、截断日志恢复、端点数值、无窗口 PNG/PDF 输出，以及停止后自动启动真实作图子进程。相关测试使用明确标注的合成数据，不是实机测量。记录/作图版为 16 项通过；加入本地包路径和完整性校验后，共 18 项离线测试通过，最新日志在研究项目 `outputs/fr3_local_isolation_20260912/`。

完整包的验证证据在 `validation/`；原始源码接口验证记录还保存在开发服务器研究项目的 `outputs/fr3_deployment_*_20260912/`。第一组 H16 的最大差异为约 1.2e-7rad，重复请求一致，实测状态输入变化能影响四组输出。这些是有界软件检查，不是多任务性能评估。

本服务器未安装 ROS 2，尚未完成真实 ROS 消息/服务/action 联调、colcon 构建、跨机器 SSH 链路测量或硬件执行。以上实机端命令是部署步骤，不表示已经执行过。正式运行前需在实机电脑完成该部分验证。
