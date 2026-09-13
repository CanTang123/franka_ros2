#!/usr/bin/env bash
# Run from anywhere; payload and logs are resolved from this repository.
set -euo pipefail
FR3_SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FR3_REPO_DIR=$(cd "$FR3_SCRIPT_DIR/../.." && pwd)
FR3_TRIAL=${FR3_TRIAL:-$FR3_REPO_DIR/local_models/trials/A_000_s0.json}
FR3_LOG_DIR=${FR3_LOG_DIR:-$FR3_REPO_DIR/experiment_logs}
FR3_ACTION=${1:-help}
if [[ $# -gt 0 ]]; then shift; fi
if [[ "$FR3_ACTION" == help ]]; then
  cat <<'HELP'
用法：bash tools/fr3_comparison/fr3_test.sh <命令> [参数]
  robot                 启动硬件+MoveIt（必须设置 FR3_ROBOT_IP）
  home-plan             仅规划到 trial 起点，不执行
  home                  规划并执行回起点，检查实测到位后退出
  start-check           仅检查起点位置、静止状态及碰撞有效性
  dry-run A|B|C|D       启动未使能的 dry-run 客户端
  run A|B|C|D           启动允许执行的客户端；仍须另行 enable
  enable                使能当前比较客户端（run 模式下开始运动）
  stop                  请求停止当前比较客户端
  home-stop             请求停止回起点
环境：FR3_ROS_SETUP（工作区 install/setup.bash），FR3_TRIAL，FR3_LOG_DIR。
home/run/dry-run 后可追加节点参数，例如 --controller /fr3_arm_controller。
回起点前必须退出比较客户端；同一用户、同一控制器的本工具互斥。
HELP
  exit 0
fi
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  echo "请在系统 ROS 终端运行（退出 Conda 环境）；推理进程使用另一终端。" >&2
  exit 1
fi
# ROS setup files may reference unset variables.
set +u
if [[ -f /opt/ros/jazzy/setup.bash ]]; then source /opt/ros/jazzy/setup.bash; fi
if [[ -n "${FR3_ROS_SETUP:-}" ]]; then
  source "$FR3_ROS_SETUP"
else
  FR3_SCAN_DIR="$FR3_REPO_DIR"
  while [[ "$FR3_SCAN_DIR" != / ]]; do
    if [[ -f "$FR3_SCAN_DIR/install/setup.bash" ]]; then
      source "$FR3_SCAN_DIR/install/setup.bash"
      break
    fi
    FR3_SCAN_DIR=$(dirname "$FR3_SCAN_DIR")
  done
fi
set -u
command -v ros2 >/dev/null || { echo "找不到 ROS 2；先安装并加载 Jazzy。" >&2; exit 1; }
case "$FR3_ACTION" in
  robot)
    : "${FR3_ROBOT_IP:?请设置 FR3_ROBOT_IP 为实际机器人 IP}"
    exec ros2 launch franka_fr3_moveit_config moveit.launch.py \
      robot_ip:="$FR3_ROBOT_IP" use_fake_hardware:=false \
      load_gripper:="${FR3_LOAD_GRIPPER:-true}" "$@" ;;
  enable) exec ros2 service call /fr3_comparison/enable std_srvs/srv/SetBool '{data: true}' ;;
  stop) exec ros2 service call /fr3_comparison/stop std_srvs/srv/Trigger '{}' ;;
  home-stop) exec ros2 service call /fr3_return_to_start/stop std_srvs/srv/Trigger '{}' ;;
  home|home-plan|start-check|run|dry-run) ;;
  *) echo "未知命令；运行 fr3_test.sh help 查看。" >&2; exit 2 ;;
esac
[[ -f "$FR3_TRIAL" ]] || { echo "找不到 trial: $FR3_TRIAL；设置 FR3_TRIAL。" >&2; exit 1; }
mkdir -p "$FR3_LOG_DIR"
FR3_RUN_ID=$(date +%Y%m%d_%H%M%S_%N)
case "$FR3_ACTION" in
  home|home-plan|start-check)
    FR3_MODE=()
    [[ "$FR3_ACTION" != home ]] || FR3_MODE=(--execute)
    [[ "$FR3_ACTION" != start-check ]] || FR3_MODE=(--check-only)
    exec ros2 run franka_pose_control fr3_return_to_start --trial "$FR3_TRIAL" \
      --log "$FR3_LOG_DIR/${FR3_ACTION}_${FR3_RUN_ID}.jsonl" "${FR3_MODE[@]}" "$@" ;;
  run|dry-run)
    FR3_METHOD=${1:?请指定 A、B、C 或 D}
    shift
    [[ "$FR3_METHOD" =~ ^[ABCD]$ ]] || { echo "方法必须是 A/B/C/D" >&2; exit 2; }
    FR3_MODE=()
    [[ "$FR3_ACTION" != run ]] || FR3_MODE=(--execute)
    exec ros2 run franka_pose_control fr3_comparison_node --trial "$FR3_TRIAL" \
      --method "$FR3_METHOD" --log "$FR3_LOG_DIR/${FR3_METHOD}_${FR3_ACTION}_${FR3_RUN_ID}.jsonl" \
      "${FR3_MODE[@]}" "$@" ;;
esac
