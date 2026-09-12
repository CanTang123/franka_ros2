# FR3 四方法实机闭环接入

本服务器编辑、校验并运行冻结模型；实机电脑运行本仓库已有 MoveIt 和 `fr3_arm_controller`，订阅实测关节状态，每 0.2 秒请求新计划。Git 用于同步代码，SSH 隧道用于实验时通信，模型权重留在服务器。

实现入口：`inference_server.py`（服务器）、`fr3_comparison_node`（ROS 2 实机端）。默认 dry-run、未使能。记录导入与 RViz 预览为辅助功能，不能替代实测状态重规划。

|方法|含义|FR3 冻结推理|
|---|---|---|
|A|DeMoFlow 完整方法|独立状态网络更新一次，然后动作解码；J=1，t*=0.5|
|B|共享解码器的动力学对照|decoder + native MuJoCo rollout 给出状态端点，再解码；J=1，t*=0.5；不是原生 DynaFlow 默认配置|
|C|Direct|不更新初始噪声，在 t=0 解码一次|
|D|Bypass 消融|不更新初始噪声，在 t=0.5 解码一次|

四组共用 seed9 的 decoder10k EMA；A 使用 mean25k EMA。此入口没有接入旧 action-only CFM / joint-CFM、原生 DynaFlow 或其他训练 seed，不应把 B/C/D 重命名为这些方法。

## 代码与数据在哪里

本地仓库：`/home/wqf/Projects/franka_ros2`，分支 `feat/fr3-comparison-integration`，基于远端 `my-local-changes`。所有新增代码都在此仓库；原始研究代码、结果、数据与权重保持原样。

`comparison_data/` 和验证日志被 Git 忽略。示例导入了一组配对的 condition/source，包含四方法的 8.8 秒记录。在线客户端使用其中的初始状态、任务 descriptor、condition ID、source 和时长；**不会把记录的 actions 当作在线模型输出**。记录中的历史仿真成功标记也不用于判断实机成功。

代码提交到独立分支。完成 GitHub 登录后，在本服务器推送：

```bash
git push -u origin feat/fr3-comparison-integration
```

推送成功后，在实机电脑已有 checkout 中拉取（先保留原有未提交修改）：

```bash
git fetch origin
git switch --track origin/feat/fr3-comparison-integration
```

以后在这个分支使用 `git pull --ff-only` 更新。首次部署也可用 `git clone --branch feat/fr3-comparison-integration https://github.com/CanTang123/franka_ros2.git`。无法使用 GitHub 时，仍可传输 patch 并应用到兼容的 checkout。权重、trial JSON 和运行日志独立传输。

## 1. 本服务器准备一次配对实验

现成示例：`comparison_data/demo/A_000_s0.json`。它记录了一条八字任务，并非自动选择的现场可行任务；先核对工具、基座、任务姿态和现场障碍物，使用已有 MoveIt 流程将机械臂放到声明的起点。本客户端不会自动移动到起点。

重新导出（输出目录必须不存在）：

```bash
cd /home/wqf/Projects/franka_ros2
/home/wqf/miniconda3/envs/dm-cfm/bin/python tools/fr3_comparison/export_results.py \
  --results-root /home/wqf/Projects/dynamics_manifold_cfm/outputs/demoflow_paper_followup_20260911_171203/evaluation/test \
  --output comparison_data/another_trial --source 0
```

可加 `--condition-id ID`，或用 `--all` 导出全部配对记录。导入检查来源 SHA256、方法配对、单位、形状、有限值和时间网格；不会筛掉失败样本。四种方法必须用同一份 trial JSON，以保持任务和随机源一致。

启动服务器（CPU 已完成数值核对；该命令运行一小时，无自动重启）：

```bash
cd /home/wqf/Projects/dynamics_manifold_cfm
scripts/task.sh launch --gpu cpu \
  --log-dir /home/wqf/Projects/dynamics_manifold_cfm/outputs/fr3_live_trial_001 \
  --name fr3-live-trial-001 --timeout 3600 -- \
  python /home/wqf/Projects/franka_ros2/tools/fr3_comparison/inference_server.py \
    --project /home/wqf/Projects/dynamics_manifold_cfm \
    --frozen-manifest /home/wqf/Projects/dynamics_manifold_cfm/outputs/demoflow_paper_followup_20260911_171203/FROZEN_CANDIDATES.json \
    --extension /home/wqf/Projects/fr3_native_adjoint_20260909 \
    --trial /home/wqf/Projects/franka_ros2/comparison_data/demo/A_000_s0.json
```

查看任务 `stdout.log`，出现 `"ready": true` 才表示哈希验证与 A/B/C/D 预热完成。服务仅监听 `127.0.0.1:8765`，只接受启动时声明的 condition/source/descriptor。源代码从当前 `dynamics_manifold_cfm/src` 加载，并与冻结训练中用到的 FR3 数学依赖逐文件核对；不加载历史源码快照。冻结身份中额外记录的 Go1 adapter 不在 FR3 导入路径内，其身份仍原样保留。

## 2. 实机电脑连接与构建

在实机电脑上替换下面的 `USER@SERVER` 为当前服务器 SSH 地址：

```bash
ssh -N -o ExitOnForwardFailure=yes -L 8765:127.0.0.1:8765 USER@SERVER
```

另一个终端复制 trial JSON（代码可通过 Git 或 patch 同步，不必复制权重）：

```bash
scp USER@SERVER:/home/wqf/Projects/franka_ros2/comparison_data/demo/A_000_s0.json ./fr3_trial.json
```

在已有 ROS 2 Jazzy 工作区更新代码后构建：

```bash
cd /ros2_ws
colcon build --symlink-install --packages-select franka_pose_control
source install/setup.bash
```

保留已有 MoveIt/hardware launch。客户端复用 `/fr3_arm_controller/follow_joint_trajectory`，不启动另一套硬件接口。依赖 `control_msgs`、`trajectory_msgs`、`controller_manager_msgs`、`moveit_msgs` 和 `rclpy`，已经声明在 package.xml；实机不需要 JAX、PyTorch 或模型权重。

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

## 执行协议与日志解释

- 规划器返回 H16（0.8s）绝对关节目标；实机一次只授权前 4 个目标，每个 0.05s。请求按 0.2s 周期发出，任务相位使用实机端实际经过的墙钟时间，随机源按 control_step=0/4/8/... 折叠。
- 每段 ROS 轨迹以校验过的实测起点为 t=0，四个目标放在 0.05/0.10/0.15/0.20s；未添加平滑或逐方法调参。已有 JointTrajectoryController 会插值位置并通过 effort PID 执行，**这与仿真零阶保持位置伺服不是相同底层协议**。需记录控制器版本、增益、现场工具和负载；这里的实机结果作为独立实验报告。
- 默认整个网络/推理/MoveIt 校验预算 150ms，状态最大延迟 100ms，关节起点容差 0.03rad，单目标步长 0.03rad，测得及目标段平均速度上限 0.6rad/s。超时、状态异常、目标跳变、限位、碰撞检查拒绝、控制器失败均停止并请求取消，不裁剪模型动作掩盖失败。
- MoveIt 在当前起点与四个关节线性目标之间每至多 0.01rad 采样检查；这是有限采样，不能当作连续碰撞证明，且依赖现场 Planning Scene 正确包含桌面、工具等对象。
- 网络和校验引入实际动作起始延迟，段间可能出现保持或目标替换；没有把这段延迟假定为零。每次 `plan_ready` 记录请求状态、物理相位、完整 H16、执行前缀、权重身份和端到端耗时；`measured_state` 记录时间戳和 q/dq，`controller_result` 记录控制器结果。`trial duration reached` 仅表示达到时长，不是任务成功。
- dry-run/超时/中止必须保留。实机末端误差、成功判据、姿态标定及跨试验统计尚需按现场 TCP/参考轨迹定义，不能直接复用仿真表格数字。

官方接口依据：[Jazzy JointTrajectoryController 文档](https://control.ros.org/jazzy/doc/ros2_controllers/joint_trajectory_controller/doc/userdoc.html)。

## 已验证与尚未验证

已做：四组源记录导入；协议/异常输入测试；模拟 ROS transport 的状态超时、网络超时、碰撞拒绝、停止后迟到接受等路径；CPU 上一组冻结 condition/source 的完整 H16 与历史计划核对；真实本地 HTTP 服务的四方法请求与错误任务拒绝。

验证证据在 `comparison_data/runtime_verification.json`、`comparison_data/transport_verification.json` 以及研究项目的 `outputs/fr3_deployment_*_20260912/`。第一组 H16 的最大差异为约 1.2e-7rad，重复请求一致，实测状态输入变化能影响四组输出。这些是有界软件检查，不是多任务性能评估。

本服务器未安装 ROS 2，尚未完成真实 ROS 消息/服务/action 联调、colcon 构建、跨机器 SSH 链路测量或硬件执行。以上实机端命令是部署步骤，不表示已经执行过。正式运行前需在实机电脑完成该部分验证。
