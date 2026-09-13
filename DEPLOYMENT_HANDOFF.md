# FR3 完整本地推理包：权重、控制、记录与作图

**最新完整流程请先阅读 [README_FR3_实机测试.md](README_FR3_实机测试.md)。** 新增 `fr3_return_to_start` 回起点、同控制器进程互斥、`fr3_test.sh` 统一入口、GPU 固定依赖及真实设备记录。新版发布包名为 `fr3_full_workflow_20260912.tar.gz`。下面保留原有本地推理与记录说明；新机按中文指南的 GPU/CPU 路径选择安装。

如果使用 Windows/WSL 笔记本，先阅读 [WSL 本地推理与实机控制的区别](WSL_LOCAL.md)。本包可用于先做本地 CPU 推理测试，普通 WSL 不应直接视为已满足 FR3 实时控制要求。

此完整包包含两份原始模型 checkpoint（mean25k、decoder10k）、归一化参数、B 方法的物理模型和网格、当前推理源码、ROS 接口、末端/关节记录、自动作图和示例 trial。推理与 ROS 控制都在实机电脑运行，无需本服务器或 SSH 隧道，也无需 GitHub 登录。旧的约 5 MB 源码包不包含权重，请使用本版。包中不含 ROS 系统安装或 Python 环境；首次软件依赖安装需要软件源或预先装好的环境。

## 更新实机代码

先解压到临时目录检查：

```bash
mkdir -p ~/fr3_deployment_review
tar -xzf fr3_local_inference_complete.tar.gz -C ~/fr3_deployment_review
```

顶层目录为 `franka_ros2/`，`PACKAGED_FILES_SHA256.json` 记录包内各文件的哈希。保留实机电脑当前硬件 IP、namespace、控制器配置和其他自定义改动，再合入代码。此功能主要需要更新 `tools/fr3_comparison/`、`franka_pose_control/CMakeLists.txt` 和 `franka_pose_control/package.xml`；后两项包含脚本安装与依赖声明。权重与物理模型目录 `local_models/` 必须完整保留；示例任务在 `local_models/trials/`，也要复制到可访问位置。不要在同一个 colcon 工作区放置两个同名 ROS 包。

在原有 ROS 2 Jazzy 环境中安装作图依赖并编译（工作区路径按现场修改）：

```bash
sudo apt install python3-matplotlib
cd /ros2_ws
colcon build --symlink-install --packages-select franka_pose_control
source install/setup.bash
```

## 启动与结果文件

先在实机电脑的独立 Python 3.11 环境中安装冻结推理依赖并核对输出。以下命令从解压后的 `franka_ros2` 目录执行：

```bash
conda create -n fr3-inference python=3.11 pip
conda activate fr3-inference
python -m pip install -r local_models/requirements-cpu.txt
JAX_PLATFORMS=cpu python tools/fr3_comparison/verify_runtime.py --bundle local_models --output local_readiness.json
bash tools/fr3_comparison/run_local_inference.sh
```

等待 `"ready": true` 后，保留已有 MoveIt/hardware launch，在另一个 ROS 系统环境终端启动客户端。两进程通过本机 `127.0.0.1:8765` 通信，ROS 终端不需要激活推理 Conda 环境。例如：

```bash
ros2 run franka_pose_control fr3_comparison_node \
  --trial /ros2_ws/src/franka_ros2/local_models/trials/A_000_s0.json \
  --method A --log /tmp/fr3_results/A_trial001.jsonl
```

这仍是未使能的 dry-run。现场执行前按完整说明完成检查；运行时添加 `--execute`，并显式调用 enable。B/C/D 使用相同 trial，分别更换 `--method` 和日志名。原有启动位置、状态新鲜度、控制器与碰撞校验仍生效，不会为了记录而自动移动机械臂。

停止后继续记录 1 秒，并自动在 `/tmp/fr3_results/A_trial001_reports/run_001/` 生成：

- `joint_states.csv`：实测七关节角度、速度及时间戳。
- `ee_pose.csv`：硬件广播的末端 XYZ、四元数 xyzw、坐标系及时间戳。
- `controller_feedback.csv`：控制器目标、实测与误差，缺失时保留空表。
- `summary.json`：停止原因、记录数量、初始/停止前/最终末端与关节结果、缺失信息。
- 末端三维轨迹、XYZ、四元数、七关节位置和速度图，各有 PNG 和 PDF。

原始 JSONL 持续保存。作图进程的输出在 `run_001_report.log`。重启电脑可能清理 `/tmp`，正式实验请把 `--log` 指向持久数据目录。如果只想记录，增加 `--no-auto-report`；之后可把日志带回服务器作图：

```bash
python tools/fr3_comparison/report_results.py A_trial001.jsonl --output A_trial001_report
```

默认末端话题是 `/franka_robot_state_broadcaster/current_pose`，可用 `--ee-pose` 修改。它表示硬件配置中的 O_T_EE，不保证等于 fr3_hand_tcp。没有末端消息时明确报告缺失，不使用仿真轨迹或 FK 替代。

## 验证范围

18 项离线测试通过，包括包内文件完整性/路径校验、异常停止、停止后记录、数据缺失/截断恢复和真实作图子进程。合成数据生成的图已检查；未执行真实 ROS 构建、跨机器连接或硬件实验。本地载入四种方法的首段数值与本地 HTTP 核对记录随包附带；另进行移目录测试，禁止读取开发服务器工作区。权重保留原始 SHA256 与训练身份，绝对路径仅为来源元数据。CPU 包尚需在实际实机电脑上测量耗时，以上测试不能替代现场验证。
