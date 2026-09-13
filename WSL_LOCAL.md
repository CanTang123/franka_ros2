# Windows / WSL 笔记本的使用范围

用户提供的配置为 Intel Core Ultra 7 356H、GeForce RTX 5060 Laptop GPU 8GB。当前完整包按 CPU 推理准备，不需要先安装 CUDA，也未在该笔记本上测量性能或验证 GPU。两份原始 checkpoint、归一化参数和物理模型均在 `local_models/` 中。

## 先在 WSL 中验证本地推理

Windows PowerShell 检查 WSL 版本：

```powershell
wsl -l -v
```

建议用 WSL2 的 Ubuntu 环境准备 Python 3.11 推理环境。进入解压后的 `franka_ros2` 目录，按 `DEPLOYMENT_HANDOFF.md` 安装 `local_models/requirements-cpu.txt`，然后运行：

```bash
JAX_PLATFORMS=cpu python tools/fr3_comparison/verify_runtime.py \
  --bundle local_models --output laptop_readiness.json
bash tools/fr3_comparison/run_local_inference.sh
```

这些命令不连接机器人。`laptop_readiness.json` 核对四方法 H16 输出，并记录这台电脑的预热后单次推理耗时。完整控制周期还包括 ROS 调度、状态读取和碰撞校验，不能只用这里的单次推理时间判断实机实时性。修改到 GPU 运行前，应另行核对 RTX 5060、Windows 驱动、WSL CUDA 与冻结 JAX 版本的兼容性；本包没有宣称该 GPU 配置已验证。

## WSL 推理通过不等于可以直接控制 FR3

Franka 官方要求执行 libfranka 控制程序的电脑使用实时优先级及 PREEMPT_RT 内核。普通 WSL 安装不能据此视为已满足实机实时控制要求。参考 [Franka 实时内核文档](https://frankarobotics.github.io/docs/doc/libfranka/docs/real_time_kernel.html)。

- 如果现场另有符合要求的 Linux 控制电脑：可先在 WSL 检查包，再把完整包放到现场电脑本地运行。
- 如果只有这台笔记本且要直接连接 FR3：先准备并验证原生 Linux/实时内核和机器人有线网络；不要把 WSL 中网络可达或推理成功当作控制链路已经合格。CPU 推理是当前包的默认方案。

查看当前 Linux 内核和实时标志可用 `uname -r`、`cat /sys/kernel/realtime`。这些只是环境检查，不能代替现场网络抖动和控制周期验证。此交付没有更改 libfranka 的实时要求，也没有开启机械臂运动。
