# ACT-wholebody 快速开始指南

## 🎯 一句话总结

用**一个数据集**训练**四种不同的 policy**，对比力矩和底座速度各自的贡献。

---

## ⚡ 3步上手

### 第1步：转换数据（5分钟）

```bash
cd /home/zeno/NPM-VLA-Project/NPM-VLA/IL_policies/ACT-wholebody

# 检查配置（确认路径正确）
# DATA_ROOT = Path("/home/zeno/piper_ros/data_collect/ACT-100")
# REPO_NAME = "ACT-100-wholebody-v17"

# 运行转换
python convert_bag_wholebody.py
```

**生成的数据集：**
- 位置：`/home/zeno/piper_ros/data_collect/ACT-100-wholebody-v17/`
- 包含：observation.state (14D), observation.effort (14D torque), observation.base_velocity (3D), action (17D)

---

### 第2步：训练 Policy（选一个或全跑）

```bash
cd /home/zeno/NPM-VLA-Project/NPM-VLA/IL_policies/ACT-wholebody

# 🔥 全开模式（推荐先跑这个）
bash piper_act_wholebody.sh --use_torque --mix

# 📊 消融实验（4个都跑，对比贡献）
bash piper_act_wholebody.sh --use_torque --mix   # 1. 全开
bash piper_act_wholebody.sh --use_torque         # 2. 仅力矩
bash piper_act_wholebody.sh --mix                # 3. 仅底座
bash piper_act_wholebody.sh                      # 4. 基线
```

**训练输出：**
```
/home/zeno/NPM-VLA-Project/NPM-VLA/IL_policies/checkpoints/ACT-wholebody/
├── torqueTrue_mixTrue/     # 1. 全开
├── torqueTrue_mixFalse/    # 2. 仅力矩
├── torqueFalse_mixTrue/    # 3. 仅底座
└── torqueFalse_mixFalse/   # 4. 基线
```

---

### 第3步：查看结果

- **W&B Dashboard**: 自动上传，查看训练曲线
- **本地 Checkpoints**: `checkpoints/ACT-wholebody/torque*_mix*/`

---

## 🧪 参数说明（2个关键开关）

| 参数 | 作用 | True 时输入 | False 时输入 |
|------|------|-------------|--------------|
| `--use_torque` | 是否使用力矩 | `qtor_17 = [0,0,0, 左臂力矩, 右臂力矩]` | `qtor_17 = [0,0,...,0]` (17个0) |
| `--mix` | 是否使用底座速度 | `qpos_17 = [vx,vy,ω, 左臂,  右臂]` | `qpos_17 = [0,0,0, 左臂, 右臂]` |

**底层逻辑：**
- 数据集保存**完整信息**（力矩 + 底座速度）
- 训练时通过参数**动态控制**使用哪些信息
- 未使用的部分填充为 0

---

## 📁 文件说明

| 文件 | 作用 | 何时使用 |
|------|------|----------|
| `convert_bag_wholebody.py` | ROS bag → LeRobot 数据集 | 第一次转换数据时 |
| `piper_act_wholebody.sh` | 训练启动脚本 | 每次训练时 |
| `train_wholebody.py` | Python 训练脚本 | 被 shell 调用，一般不直接用 |
| `modeling_act_wholebody.py` | 核心模型代码 | 无需手动修改 |
| `configuration_act_wholebody.py` | 配置类 | 无需手动修改 |

---

## ⚙️ 自定义配置

**修改数据集路径：**
```bash
bash piper_act_wholebody.sh \
  --dataset_repo_id="your-dataset-name" \
  --dataset_root="/path/to/your/dataset"
```

**调整训练参数：**
```bash
bash piper_act_wholebody.sh \
  --use_torque --mix \
  --batch_size=16 \
  --steps=100000
```

**禁用 W&B：**
```bash
bash piper_act_wholebody.sh --use_torque --mix --no_wandb
```

---

## 🐛 常见问题

**Q1: 转换脚本报错 "Missing odom data"**
- A: 你的 bag 没有 `/ranger_base_node/odom`，检查 bag 内容
- 解决：`rosbag info your.bag` 确认有这个 topic

**Q2: 训练脚本报错 "KeyError: observation.effort"**
- A: 数据集没有力矩数据
- 解决：重新用 `convert_bag_wholebody.py` 转换数据

**Q3: 想用原14维数据训练怎么办？**
- A: 用原来的转换脚本和训练脚本即可，这是额外的17维版本

**Q4: 训练太慢了**
- A: 降低 batch_size 或使用更大的 GPU
- 或者减少 steps（默认80000步约需10小时）

---

## 💡 Tips

1. **先跑全开模式**，确认流程正确
2. **再跑4个消融实验**，对比结果
3. **查看 W&B**，对比4条训练曲线
4. **保存配置文件**：每次训练会自动保存 `config.yaml` 到输出目录

---

## 📊 预期效果

训练完成后，你会得到：
- ✅ 4个不同的 policy checkpoints
- ✅ 对比实验结果（哪个模式效果最好）
- ✅ 理解力矩和底座速度各自的贡献

**下一步：**
- 用最佳 policy 进行 deployment
- 或者继续调整超参数
