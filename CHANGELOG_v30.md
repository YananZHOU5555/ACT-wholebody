# ACT-100-v30 部署代码更新日志

## 概述
将部署代码从旧的 `hanger_act_v1` checkpoint 更新到新的 `ACT-100-v30` checkpoint 080000。

## 主要修改

### 1. Checkpoint 路径更新
- **旧路径**: `/home/zeno/NPM-VLA-Project/NPM-VLA/IL_policies/checkpoints/hanger_act_v1`
- **新路径**: `/home/zeno/NPM-VLA-Project/NPM-VLA/IL_policies/checkpoints/ACT-100-v30/checkpoints/080000/pretrained_model`

### 2. 图像尺寸修改
- **旧尺寸**: 256x256
- **新尺寸**: 224x224
- 修改位置: `preprocess_image()` 函数中的 `cv2.resize()`

### 3. 归一化参数更新
从新的 checkpoint safetensors 文件中提取的归一化参数:

#### State 归一化参数 (observation.state)
```python
STATE_MEAN = [-0.2606237, 0.7552671, -0.9028265, ...]  # 14维
STATE_STD = [0.14581202, 1.0039188, 0.62520707, ...]   # 14维
```

#### EE Pose 归一化参数 (observation.ee_pose) - **新增**
```python
EE_POSE_MEAN = [0.15446927, -0.04177966, 0.25474164, ...]  # 14维
EE_POSE_STD = [0.1985284, 0.0829908, 0.06982857, ...]      # 14维
```

#### Action 反归一化参数
```python
ACTION_MEAN = [-0.25858715, 0.745404, -0.9228722, ...]  # 14维
ACTION_STD = [0.14637406, 0.9804746, 0.62880504, ...]    # 14维
```

### 4. 新增 End Effector Pose 输入特征

#### 新增全局缓冲区
```python
latest_ee_pose = {
    "left": None,
    "right": None,
}
```

#### 新增 ROS 订阅器
```python
rospy.Subscriber("/robot/arm_left/ee_pose", JointState, cb_ee_pose_left, queue_size=1)
rospy.Subscriber("/robot/arm_right/ee_pose", JointState, cb_ee_pose_right, queue_size=1)
```

#### 新增归一化函数
```python
def normalize_ee_pose(ee_pose: np.ndarray) -> np.ndarray:
    """Normalize end effector pose: (ee_pose - mean) / std"""
    return (ee_pose - EE_POSE_MEAN) / EE_POSE_STD
```

#### 在观测字典中添加 ee_pose
```python
obs = {
    "observation.images.main": main_img,
    "observation.images.secondary_0": wrist_l,
    "observation.images.secondary_1": wrist_r,
    "observation.state": state,
    "observation.ee_pose": ee_pose,  # 新增
}
```

### 5. 控制频率修正
- **旧频率**: 20 Hz
- **新频率**: 10 Hz
- **依据**: 数据集 timestamp 统计 (72.8s / 728 frames = 0.1s per frame)

### 6. 日志增强
在诊断日志中新增 ee_pose 相关信息:
```python
rospy.loginfo(f"  Raw ee_pose LEFT:      {np.array2string(latest_ee_pose['left'], precision=3)}")
rospy.loginfo(f"  Raw ee_pose RIGHT:     {np.array2string(latest_ee_pose['right'], precision=3)}")
rospy.loginfo(f"  Normalized ee_pose:    {np.array2string(ee_pose_normalized[:7], precision=3)}...")
```

## 模型配置信息

### 网络架构
- **类型**: ACT (Action Chunking Transformer)
- **Vision Backbone**: ResNet18 (ImageNet pretrained)
- **Transformer**: 
  - dim_model: 512
  - n_heads: 8
  - n_encoder_layers: 4
  - n_decoder_layers: 1
- **VAE**: 
  - latent_dim: 32
  - n_vae_encoder_layers: 4

### 输入输出
- **输入特征**:
  - observation.state: [14] (关节角度)
  - observation.ee_pose: [14] (左右臂末端位姿 x7)
  - observation.images.main: [3, 224, 224]
  - observation.images.secondary_0: [3, 224, 224] (左腕相机)
  - observation.images.secondary_1: [3, 224, 224] (右腕相机)
  - observation.images.secondary_2: [3, 224, 224] (未使用)

- **输出特征**:
  - action: [14] (目标关节角度)

### Action Chunking
- **chunk_size**: 100
- **n_action_steps**: 100
- **n_obs_steps**: 1

## 注意事项

1. **ee_pose topic 名称**: 代码中假设 topic 为 `/robot/arm_left/ee_pose` 和 `/robot/arm_right/ee_pose`，如果实际 topic 名称不同，需要修改订阅器。

2. **ee_pose 数据格式**: 假设格式为 `[x, y, z, qx, qy, qz, qw]` (7维)，左右臂各7维，总共14维。

3. **数据采集同步**: 确保所有传感器数据(images, joint_states, ee_pose)都能正常发布，否则控制循环会一直等待。

4. **归一化参数来源**: 所有归一化参数均从 checkpoint 的 safetensors 文件中提取，确保与训练时保持一致。

## 测试建议

1. 首先在仿真环境中测试，确认所有 topic 都能正确订阅
2. 检查日志输出，确认归一化值在合理范围内（normalized state 和 ee_pose 应该在 [-3, 3] 范围内）
3. 观察 action delta，确保不会有突变（应该在 MAX_JOINT_DELTA=0.5 以内）
4. 逐步提高 MAX_JOINT_DELTA 和调整 SMOOTHING_ALPHA 来优化性能

## 文件位置
- **部署脚本**: `/home/zeno/NPM-VLA-Project/NPM-VLA/IL_policies/pipier_act_main_v2.py`
- **Checkpoint**: `/home/zeno/NPM-VLA-Project/NPM-VLA/IL_policies/checkpoints/ACT-100-v30/checkpoints/080000/pretrained_model`
