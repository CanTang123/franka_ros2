#!/usr/bin/env bash
# Run from anywhere; payload and logs are resolved from this repository.
set -euo pipefail
FR3_SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FR3_REPO_DIR=$(cd "$FR3_SCRIPT_DIR/../.." && pwd)
FR3_TRIAL=${FR3_TRIAL:-$FR3_REPO_DIR/local_models/trials/A_000_s0.json}
FR3_LOG_DIR=${FR3_LOG_DIR:-$FR3_REPO_DIR/experiment_logs}
FR3_SEQUENCES=${FR3_SEQUENCES:-$FR3_REPO_DIR/fr3_joint_sequences}
FR3_SAFE_SEQUENCES=${FR3_SAFE_SEQUENCES:-$FR3_REPO_DIR/fr3_joint_sequences_safe}
FR3_CONTINUOUS_SEQUENCES=${FR3_CONTINUOUS_SEQUENCES:-$FR3_REPO_DIR/fr3_joint_sequences_continuous}
FR3_SITE_CONFIG=${FR3_SITE_CONFIG:-$FR3_SCRIPT_DIR/site_config.example.json}
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
  fixed-check            离线校验四条固定序列（不需要 ROS，不运动）
  fixed-generate         从 debug_slow5x 生成满足现场限位的 C2 平滑固定序列
  fixed-generate-continuous  生成约44秒、最大偏差0.0035rad的全局C2连续序列
  fixed-dry-run A DELAY  固定序列延迟模拟 dry-run；DELAY=0/10/50/100 ms
  fixed-run A DELAY      允许固定序列实机入口；仍须 enable，且要求现场配置已确认
  enable                使能当前比较客户端（run 模式下开始运动）
  stop                  请求停止当前比较客户端
  home-stop             请求停止回起点
环境：FR3_ROS_SETUP、FR3_TRIAL、FR3_LOG_DIR、FR3_SEQUENCES、FR3_SAFE_SEQUENCES、FR3_CONTINUOUS_SEQUENCES、FR3_SITE_CONFIG。
home/run/dry-run 后可追加节点参数，例如 --controller /fr3_arm_controller。
回起点前必须退出比较客户端；同一用户、同一控制器的本工具互斥。
HELP
  exit 0
fi
if [[ "$FR3_ACTION" == fixed-check ]]; then
  exec python3 "$FR3_SCRIPT_DIR/validate_fixed_replay.py" --sequences "$FR3_SEQUENCES" \
    --site-config "$FR3_SITE_CONFIG" "$@"
fi
if [[ "$FR3_ACTION" == fixed-generate ]]; then
  exec python3 "$FR3_SCRIPT_DIR/generate_safe_sequences.py" --source "$FR3_SEQUENCES" \
    --output "$FR3_SAFE_SEQUENCES" --site-config "$FR3_SITE_CONFIG" "$@"
fi
if [[ "$FR3_ACTION" == fixed-generate-continuous ]]; then
  exec python3 "$FR3_SCRIPT_DIR/generate_continuous_sequences.py" --source "$FR3_SEQUENCES" \
    --output "$FR3_CONTINUOUS_SEQUENCES" --site-config "$FR3_SITE_CONFIG" "$@"
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
elif [[ "$(basename "$FR3_REPO_DIR")" == src && -f "$FR3_REPO_DIR/../install/setup.bash" ]]; then
  # Docker Compose mounts this repository at /ros2_ws/src.  Its workspace
  # overlay is /ros2_ws/install; never source host build artifacts mounted as
  # /ros2_ws/src/install (they may belong to another ROS/Python version).
  source "$FR3_REPO_DIR/../install/setup.bash"
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
if [[ "${ROS_DISTRO:-}" != jazzy || "${PYTHONPATH:-}" == *'/opt/ros/humble/'* ||
      "${PYTHONPATH:-}" == *'/ros2_ws/src/install/'* ]]; then
  echo "ROS 环境被 Humble 或源码目录中的旧 install 污染；请开全新 shell，仅加载 /opt/ros/jazzy 和 /ros2_ws/install。" >&2
  exit 1
fi
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
  home|home-plan|start-check|run|dry-run|fixed-run|fixed-dry-run) ;;
  *) echo "未知命令；运行 fr3_test.sh help 查看。" >&2; exit 2 ;;
esac
if [[ "$FR3_ACTION" =~ ^(home|home-plan|start-check|run|dry-run)$ ]]; then
  [[ -f "$FR3_TRIAL" ]] || { echo "找不到 trial: $FR3_TRIAL；设置 FR3_TRIAL。" >&2; exit 1; }
fi
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
  fixed-run|fixed-dry-run)
    FR3_METHOD=${1:?请指定 A、B、C 或 D}; shift
    FR3_DELAY=${1:?请指定延迟 0、10、50 或 100 ms}; shift
    [[ "$FR3_METHOD" =~ ^[ABCD]$ ]] || { echo "方法标签必须是 A/B/C/D" >&2; exit 2; }
    [[ "$FR3_DELAY" =~ ^(0|10|50|100)$ ]] || { echo "延迟必须是 0/10/50/100 ms" >&2; exit 2; }
    [[ -f "$FR3_SITE_CONFIG" ]] || { echo "找不到现场配置: $FR3_SITE_CONFIG" >&2; exit 1; }
    FR3_MODE=()
    [[ "$FR3_ACTION" != fixed-run ]] || FR3_MODE=(--execute)
    exec ros2 run franka_pose_control fr3_comparison_node --method "$FR3_METHOD" \
      --fixed-sequences "$FR3_SEQUENCES" --simulated-delay-ms "$FR3_DELAY" \
      --site-config "$FR3_SITE_CONFIG" \
      --log "$FR3_LOG_DIR/fixed_${FR3_METHOD}_${FR3_DELAY}ms_${FR3_ACTION}_${FR3_RUN_ID}.jsonl" \
      "${FR3_MODE[@]}" "$@" ;;
esac
