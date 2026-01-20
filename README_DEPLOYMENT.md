# ACT-100-v30 模型部署使用指南

## 快速开始

### 1. 环境准备

确保你的环境已安装所有依赖：
```bash
# 激活 lerobot 环境
conda activate lerobot

# 确认 CUDA 可用
nvidia-smi

# 确认所需的 Python 包
python3 -c "import torch; import cv2; import numpy as np; print('✓ Dependencies OK')"
```

### 2. 启动机器人系统

**按顺序启动以下节点：**

```bash
# Terminal 1: 启动机器人控制器
roslaunch your_robot_package robot_control.launch

# Terminal 2: 启动相机
roslaunch realsense2_camera rs_camera.launch  # 根据你的设置调整

# Terminal 3: 启动末端位姿计算节点（如果有单独的节点）
rosrun your_package end_pose_publisher.py
```

### 3. 检查 ROS Topics

运行部署代码前，确认所有必需的 topics 正在发布：

```bash
# 检查所有 topics
rostopic list

# 检查关键 topics 的发布频率
rostopic hz /realsense_top/color/image_raw/compressed
rostopic hz /realsense_left/color/image_raw/compressed
rostopic hz /realsense_right/color/image_raw/compressed
rostopic hz /robot/arm_left/joint_states_single
rostopic hz /robot/arm_right/joint_states_single
rostopic hz /robot/arm_left/end_pose
rostopic hz /robot/arm_right/end_pose

# 检查数据内容（示例）
rostopic echo /robot/arm_left/end_pose --noarr
```

**必需的 Topics 清单：**
- ✓ `/realsense_top/color/image_raw/compressed` (CompressedImage)
- ✓ `/realsense_left/color/image_raw/compressed` (CompressedImage)
- ✓ `/realsense_right/color/image_raw/compressed` (CompressedImage)
- ✓ `/robot/arm_left/joint_states_single` (JointState, 7维)
- ✓ `/robot/arm_right/joint_states_single` (JointState, 7维)
- ✓ `/robot/arm_left/end_pose` (JointState, 7维: x,y,z,qx,qy,qz,qw)
- ✓ `/robot/arm_right/end_pose` (JointState, 7维: x,y,z,qx,qy,qz,qw)

### 4. 运行部署代码

```bash
# 进入项目目录
cd /home/zeno/NPM-VLA-Project/NPM-VLA

# 运行部署脚本
python3 IL_policies/pipier_act_main_v2.py
```

### 5. 观察日志输出

正常启动后，你会看到类似的日志：

```
[INFO] Loading ACT Policy from: /home/zeno/NPM-VLA-Project/NPM-VLA/IL_policies/checkpoints/ACT-100-v30/checkpoints/080000/pretrained_model
======================================================================
[INFO] Policy loaded successfully
  chunk_size: 100
  n_action_steps: 100
======================================================================
[FIX-4] Publishers: pub_left -> arm_left, pub_right -> arm_right
======================================================================
[CONFIG] Normalization enabled:
  State normalization: ENABLED (mean_std)
  EE Pose normalization: ENABLED (mean_std)
  Image normalization: ENABLED (ImageNet, 224x224)
  Action unnormalization: ENABLED (mean_std)
  Control rate: 10 Hz
  EMA smoothing: True, alpha=0.3
  Action clipping: True, max_delta=0.5
======================================================================
Waiting for sensor data...
✓ All sensors ready
======================================================================
[DIAG] Step 1
  Raw state LEFT:        [-0.260  0.755 -0.903 ...]
  Raw state RIGHT:       [ 0.092  0.828 -0.949 ...]
  Raw ee_pose LEFT:      [ 0.154 -0.042  0.255 ...]
  Raw ee_pose RIGHT:     [ 0.171  0.014  0.244 ...]
  Normalized state:      [ 0.005  0.001 -0.002 ...]
  Normalized ee_pose:    [-0.010  0.015 -0.004 ...]
  Model output (norm):   [ 0.123 -0.456  0.789 ...]
  Unnormalized action L: [-0.242  0.802 -0.856 ...]
  Unnormalized action R: [ 0.105  0.873 -0.902 ...]
  Delta LEFT:  [ 0.018  0.047  0.047 ...]
  Delta RIGHT: [ 0.013  0.045  0.047 ...]
======================================================================
✓ Actions sent to robot
```

## 参数调优

### 安全参数

编辑 `pipier_act_main_v2.py` 的以下部分：

```python
# Safety parameters (第215-217行附近)
MAX_JOINT_DELTA = 0.5        # 单步最大角度变化(弧度)
ENABLE_ACTION_CLIPPING = True # 启用安全限幅

# EMA smoothing (第219-221行附近)
ENABLE_SMOOTHING = True      # 启用EMA平滑
SMOOTHING_ALPHA = 0.3        # 平滑系数 (0-1, 越小越平滑)
```

**调优建议：**

| 场景 | MAX_JOINT_DELTA | SMOOTHING_ALPHA | 说明 |
|------|----------------|-----------------|------|
| 初次测试 | 0.3 | 0.2 | 非常保守，动作缓慢平滑 |
| 正常使用 | 0.5 | 0.3 | 默认设置，平衡性能和安全 |
| 快速响应 | 0.8 | 0.5 | 更快的响应，降低平滑 |
| 精细操作 | 0.4 | 0.25 | 平滑且精确 |

## 故障排除

### 问题 1: 代码卡在 "Waiting for sensor data..."

**原因：** 某个或某些 topic 没有数据

**排查步骤：**
```bash
# 1. 检查所有 topics 是否存在
rostopic list | grep -E "(realsense|robot|end_pose)"

# 2. 检查每个 topic 的发布频率
for topic in /realsense_top/color/image_raw/compressed \
             /realsense_left/color/image_raw/compressed \
             /realsense_right/color/image_raw/compressed \
             /robot/arm_left/joint_states_single \
             /robot/arm_right/joint_states_single \
             /robot/arm_left/end_pose \
             /robot/arm_right/end_pose; do
  echo "Checking $topic..."
  timeout 2 rostopic hz $topic 2>&1 | head -n 3
done

# 3. 检查 topic 数据格式
rostopic echo /robot/arm_left/end_pose -n 1
```

**解决方案：**
- 确保所有相机节点已启动
- 确保机器人控制器已启动
- 确保末端位姿计算节点已启动（或在控制器中已启用）

### 问题 2: end_pose topic 格式不正确

**症状：** 日志显示 ee_pose 数据为 None 或维度不对

**检查数据格式：**
```bash
rostopic echo /robot/arm_left/end_pose -n 1
```

**期望格式 (JointState)：**
```yaml
header:
  seq: 123
  stamp: ...
  frame_id: ''
name: []
position: [x, y, z, qx, qy, qz, qw]  # 7个元素
velocity: []
effort: []
```

**如果格式不对：**
- 检查你的 end_pose 发布节点
- 确认是否使用 JointState 消息类型
- 确认 position 字段包含 7 个元素 [x, y, z, qx, qy, qz, qw]

### 问题 3: 动作抖动或不稳定

**可能原因：**
1. 光照变化导致图像不稳定
2. SMOOTHING_ALPHA 太大
3. 传感器数据噪声

**解决方案：**
```python
# 增加平滑度
SMOOTHING_ALPHA = 0.2  # 从 0.3 降到 0.2

# 降低最大变化
MAX_JOINT_DELTA = 0.3  # 从 0.5 降到 0.3
```

### 问题 4: CUDA out of memory

**解决方案：**
```bash
# 方案1: 释放其他GPU占用
pkill -f python  # 小心：这会杀掉所有python进程

# 方案2: 使用CPU运行（较慢）
# 编辑 pipier_act_main_v2.py 第187行
device = "cpu"  # 改为强制使用CPU
```

### 问题 5: 机器人运动不符合预期

**检查清单：**
1. 归一化值检查（日志中的 "Normalized state" 和 "Normalized ee_pose"）
   - 应该在 [-3, 3] 范围内
   - 如果超出很多，说明归一化参数可能不对

2. Delta 值检查
   - 应该平滑变化
   - 不应该有突变
   - 如果频繁触发 "Clipping" 警告，考虑增大 MAX_JOINT_DELTA

3. 相机位置和角度
   - 确保与训练时的相机布置一致
   - 检查光照条件

### 问题 6: ImportError 或 ModuleNotFoundError

**解决方案：**
```bash
# 确认激活了正确的环境
conda activate lerobot

# 检查 lerobot 是否正确安装
python3 -c "from lerobot.policies.act.modeling_act import ACTPolicy; print('✓ LeRobot OK')"

# 如果失败，重新安装 lerobot
cd /path/to/lerobot
pip install -e .
```

## 高级使用

### 录制测试数据

运行时可以录制 ROS bag 用于调试：

```bash
rosbag record -O test_deployment.bag \
  /realsense_top/color/image_raw/compressed \
  /realsense_left/color/image_raw/compressed \
  /realsense_right/color/image_raw/compressed \
  /robot/arm_left/joint_states_single \
  /robot/arm_right/joint_states_single \
  /robot/arm_left/end_pose \
  /robot/arm_right/end_pose \
  /robot/arm_left/vla_joint_cmd \
  /robot/arm_right/vla_joint_cmd
```

### 性能监控

```bash
# 监控 GPU 使用
watch -n 1 nvidia-smi

# 监控 topic 频率
rostopic hz /robot/arm_left/vla_joint_cmd

# 监控延迟
rostopic delay /robot/arm_left/joint_states_single
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `pipier_act_main_v2.py` | 主部署脚本 |
| `CHANGELOG_v30.md` | 详细的修改记录 |
| `README_DEPLOYMENT.md` | 本文件 - 部署使用指南 |

## 模型信息

- **Checkpoint**: ACT-100-v30, step 080000
- **训练数据**: 100 episodes
- **图像尺寸**: 224×224 (ImageNet normalization)
- **控制频率**: 10 Hz
- **Action Chunk**: 100 steps
- **Architecture**: ACT with ResNet18 backbone

## 技术支持

如果遇到问题：
1. 检查日志输出中的 [DIAG] 信息
2. 验证所有 topic 数据正常
3. 参考故障排除部分
4. 检查训练配置是否与部署环境一致

---

**祝部署顺利！**
