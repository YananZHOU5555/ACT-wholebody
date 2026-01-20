#!/usr/bin/env python3
"""
ACT Policy Deployment for Piper Dual-Arm Robot (LeRobot)
Model: ACT-100-v30 checkpoint 080000

关键修复：
1. [FIX-1] State 输入归一化
2. [FIX-2] Image 输入 ImageNet 归一化 (224x224)
3. [FIX-3] Action 输出反归一化
4. [FIX-4] Publisher topic 左右修正
5. [FIX-5] 控制频率匹配数据集 fps (10 Hz)
6. [NEW] 新增 ee_pose 输入特征
"""
from pathlib import Path

import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage, JointState
from geometry_msgs.msg import PoseStamped
import cv2
import torch

from lerobot.policies.act.modeling_act import ACTPolicy


# ====== [FIX-1, FIX-2, FIX-3] 归一化统计量（从 checkpoint 080000 中提取）======
# State 归一化参数 (observation.state)
STATE_MEAN = np.array([
    -0.2606237,   0.7552671,  -0.9028265,   0.02707627,  0.8637471,  -0.12004092,
     0.05240162,  0.09250281,  0.82755446, -0.94922143, -0.04218172,  0.88446194,
    -0.06515904,  0.06623472
], dtype=np.float32)

STATE_STD = np.array([
    0.14581202, 1.0039188,  0.62520707, 0.1590158,  0.15923156, 0.15031408,
    0.01653986, 0.21557309, 1.0577756,  0.70970774, 0.07519264, 0.12686406,
    0.16542834, 0.00788918
], dtype=np.float32)

# EE Pose 归一化参数 (observation.ee_pose)
EE_POSE_MEAN = np.array([
     0.15446927, -0.04177966,  0.25474164, -0.03061543,  0.03641266,  0.02460553,
     0.00939875,  0.170788,    0.01410458,  0.24435385, -0.06805381,  0.5084636,
    -0.00386543,  0.23324111
], dtype=np.float32)

EE_POSE_STD = np.array([
    0.1985284,  0.0829908,  0.06982857, 0.08557645, 0.87311316, 0.08806045,
    0.46865588, 0.2036056,  0.09856424, 0.06983946, 0.12701693, 0.7124043,
    0.03690327, 0.3967393
], dtype=np.float32)

# Action 反归一化参数
ACTION_MEAN = np.array([
    -0.25858715,  0.745404,   -0.9228722,   0.02938369,  0.7823795,  -0.12291837,
     0.06786283,  0.09204281,  0.8162567,  -0.9696943,  -0.04376967,  0.8052044,
    -0.06240808,  0.0679277
], dtype=np.float32)

ACTION_STD = np.array([
    0.14637406, 0.9804746,  0.62880504, 0.17202471, 0.1568117,  0.15103714,
    0.0284012,  0.21647085, 1.0325869,  0.7129219,  0.08253058, 0.14118974,
    0.17439136, 0.02257736
], dtype=np.float32)

# Image 归一化参数（ImageNet 标准）
IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ----------------- Global buffers -----------------

def decode_compressed_image(msg: CompressedImage) -> np.ndarray:
    np_arr = np.frombuffer(msg.data, dtype=np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img_bgr


latest_imgs = {
    "main": None,
    "wrist_l": None,
    "wrist_r": None,
}
latest_q = {
    "left": None,
    "right": None,
}
latest_ee_pose = {
    "left": None,
    "right": None,
}

smoothed_action = {
    "left": None,
    "right": None,
}


# ----------------- Image callbacks -----------------
def cb_main(msg: CompressedImage):
    latest_imgs["main"] = decode_compressed_image(msg)

def cb_wrist_l(msg: CompressedImage):
    latest_imgs["wrist_l"] = decode_compressed_image(msg)

def cb_wrist_r(msg: CompressedImage):
    latest_imgs["wrist_r"] = decode_compressed_image(msg)


# ----------------- Joint callbacks -----------------
def cb_joints_left(msg: JointState):
    latest_q["left"] = np.array(msg.position, dtype=np.float32)

def cb_joints_right(msg: JointState):
    latest_q["right"] = np.array(msg.position, dtype=np.float32)


# ----------------- End effector pose callbacks -----------------
def cb_ee_pose_left(msg: PoseStamped):
    """Callback for left arm end effector pose (x, y, z, qx, qy, qz, qw)"""
    latest_ee_pose["left"] = np.array([
        msg.pose.position.x,
        msg.pose.position.y,
        msg.pose.position.z,
        msg.pose.orientation.x,
        msg.pose.orientation.y,
        msg.pose.orientation.z,
        msg.pose.orientation.w
    ], dtype=np.float32)

def cb_ee_pose_right(msg: PoseStamped):
    """Callback for right arm end effector pose (x, y, z, qx, qy, qz, qw)"""
    latest_ee_pose["right"] = np.array([
        msg.pose.position.x,
        msg.pose.position.y,
        msg.pose.position.z,
        msg.pose.orientation.x,
        msg.pose.orientation.y,
        msg.pose.orientation.z,
        msg.pose.orientation.w
    ], dtype=np.float32)


# ----------------- [FIX-2] Image preprocessing with ImageNet normalization -----------------
def preprocess_image(img_bgr: np.ndarray) -> torch.Tensor:
    """
    Convert BGR uint8 (H,W,3) to normalized float32 torch tensor (1,3,224,224).
    Applies ImageNet normalization: (img - mean) / std
    """
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)  # Updated to 224x224
    img = img.astype(np.float32) / 255.0  # [H,W,3] in [0,1]

    # ====== [FIX-2] ImageNet 归一化 ======
    img = (img - IMAGE_MEAN) / IMAGE_STD

    img = np.transpose(img, (2, 0, 1))    # [3,H,W]
    img = np.expand_dims(img, 0)          # [1,3,H,W]
    return torch.from_numpy(img)


# ----------------- [FIX-1] State normalization -----------------
def normalize_state(state: np.ndarray) -> np.ndarray:
    """Normalize state: (state - mean) / std"""
    return (state - STATE_MEAN) / STATE_STD


# ----------------- [NEW] EE Pose normalization -----------------
def normalize_ee_pose(ee_pose: np.ndarray) -> np.ndarray:
    """Normalize end effector pose: (ee_pose - mean) / std"""
    return (ee_pose - EE_POSE_MEAN) / EE_POSE_STD


# ----------------- [FIX-3] Action unnormalization -----------------
def unnormalize_action(action: np.ndarray) -> np.ndarray:
    """Unnormalize action: action * std + mean"""
    return action * ACTION_STD + ACTION_MEAN


# ----------------- Load ACT policy -----------------
def load_policy(ckpt_dir: str, device: str) -> ACTPolicy:
    ckpt_path = Path(ckpt_dir).expanduser().resolve()
    rospy.loginfo(f"Loading ACT Policy from: {ckpt_path}")
    
    policy = ACTPolicy.from_pretrained(pretrained_name_or_path=str(ckpt_path))
    policy = policy.to(device)
    
    rospy.loginfo("=" * 70)
    rospy.loginfo("[INFO] Policy loaded successfully")
    rospy.loginfo(f"  chunk_size: {policy.config.chunk_size}")
    rospy.loginfo(f"  n_action_steps: {policy.config.n_action_steps}")
    rospy.loginfo("=" * 70)
    
    policy.eval()
    return policy


def main():
    rospy.init_node("piper_act_main")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rospy.loginfo(f"Using device: {device}")

    ckpt_dir = "/home/zeno/NPM-VLA-Project/NPM-VLA/IL_policies/checkpoints/ACT-100-v30/checkpoints/080000/pretrained_model"
    policy = load_policy(ckpt_dir, device)
    policy.reset()

    # ----------------- ROS subs / pubs -----------------
    rospy.Subscriber("/realsense_top/color/image_raw/compressed", CompressedImage, cb_main, queue_size=1)
    rospy.Subscriber("/realsense_left/color/image_raw/compressed", CompressedImage, cb_wrist_l, queue_size=1)
    rospy.Subscriber("/realsense_right/color/image_raw/compressed", CompressedImage, cb_wrist_r, queue_size=1)

    rospy.Subscriber("/robot/arm_left/joint_states_single", JointState, cb_joints_left, queue_size=1)
    rospy.Subscriber("/robot/arm_right/joint_states_single", JointState, cb_joints_right, queue_size=1)

    # ====== [NEW] Subscribe to end effector poses ======
    rospy.Subscriber("/robot/arm_left/end_pose", PoseStamped, cb_ee_pose_left, queue_size=1)
    rospy.Subscriber("/robot/arm_right/end_pose", PoseStamped, cb_ee_pose_right, queue_size=1)

    # ====== [FIX-4] 修复 Publisher topic 名称 ======
    pub_left = rospy.Publisher("/robot/arm_left/vla_joint_cmd", JointState, queue_size=1)
    pub_right = rospy.Publisher("/robot/arm_right/vla_joint_cmd", JointState, queue_size=1)

    rospy.loginfo("[FIX-4] Publishers: vla_joint_cmd (7 values: 6 joints + 1 gripper)")

    # ====== [FIX-5] 控制频率匹配数据集 fps ======
    rate = rospy.Rate(10)  # 数据集是 10 fps (72.8s max timestamp / 728 frames = 0.1s)

    # Safety parameters
    MAX_JOINT_DELTA = 1.5  # 不使用时此参数无效
    ENABLE_ACTION_CLIPPING = False  # 禁用安全限幅，直接使用模型输出

    # EMA smoothing
    ENABLE_SMOOTHING = True
    SMOOTHING_ALPHA = 0.3

    rospy.loginfo("=" * 70)
    rospy.loginfo("[CONFIG] Normalization enabled:")
    rospy.loginfo(f"  State normalization: ENABLED (mean_std)")
    rospy.loginfo(f"  EE Pose normalization: ENABLED (mean_std)")
    rospy.loginfo(f"  Image normalization: ENABLED (ImageNet, 224x224)")
    rospy.loginfo(f"  Action unnormalization: ENABLED (mean_std)")
    rospy.loginfo(f"  Control rate: 10 Hz")
    rospy.loginfo(f"  EMA smoothing: {ENABLE_SMOOTHING}, alpha={SMOOTHING_ALPHA}")
    rospy.loginfo(f"  Action clipping: {ENABLE_ACTION_CLIPPING}, max_delta={MAX_JOINT_DELTA}")
    rospy.loginfo("=" * 70)
    rospy.loginfo("Waiting for sensor data...")

    data_ready_logged = False
    step_count = 0

    while not rospy.is_shutdown():
        # Check if all required data is available
        if (any(v is None for v in latest_imgs.values()) or
            any(v is None for v in latest_q.values()) or
            any(v is None for v in latest_ee_pose.values())):
            rate.sleep()
            continue

        if not data_ready_logged:
            rospy.loginfo("✓ All sensors ready")
            data_ready_logged = True

        # --------------- Build observation dict ---------------
        # [FIX-2] Image preprocessing with ImageNet normalization (224x224)
        main_img = preprocess_image(latest_imgs["main"]).to(device)
        wrist_l = preprocess_image(latest_imgs["wrist_l"]).to(device)
        wrist_r = preprocess_image(latest_imgs["wrist_r"]).to(device)
        # secondary_2 复用顶部相机图像（与训练时一致）
        secondary_2 = main_img

        # [FIX-1] State normalization
        state_raw = np.concatenate([latest_q["left"], latest_q["right"]], axis=0).astype(np.float32)
        state_normalized = normalize_state(state_raw)
        state = torch.from_numpy(state_normalized[None, :]).to(device)

        # [NEW] EE Pose normalization
        ee_pose_raw = np.concatenate([latest_ee_pose["left"], latest_ee_pose["right"]], axis=0).astype(np.float32)
        ee_pose_normalized = normalize_ee_pose(ee_pose_raw)
        ee_pose = torch.from_numpy(ee_pose_normalized[None, :]).to(device)

        obs = {
            "observation.images.main": main_img,
            "observation.images.secondary_0": wrist_l,
            "observation.images.secondary_1": wrist_r,
            "observation.images.secondary_2": secondary_2,  # 复用顶部相机
            "observation.state": state,
            "observation.ee_pose": ee_pose,  # NEW
        }

        # --------------- Policy inference ---------------
        with torch.no_grad():
            action_tensor = policy.select_action(obs)

        if action_tensor.dim() == 2:
            action_normalized = action_tensor[0, :].cpu().numpy()
        else:
            action_normalized = action_tensor.cpu().numpy()

        # [FIX-3] Action unnormalization - 这是关键！
        action = unnormalize_action(action_normalized)

        if len(action) != 14:
            rospy.logwarn(f"Invalid action dim: {len(action)}")
            rate.sleep()
            continue

        action_left = action[:7].copy()
        action_right = action[7:14].copy()

        # --------------- Diagnostic logging ---------------
        step_count += 1
        if step_count <= 5 or step_count % 50 == 0:
            rospy.loginfo("=" * 70)
            rospy.loginfo(f"[DIAG] Step {step_count}")
            rospy.loginfo(f"  Raw state LEFT:        {np.array2string(latest_q['left'], precision=3)}")
            rospy.loginfo(f"  Raw state RIGHT:       {np.array2string(latest_q['right'], precision=3)}")
            rospy.loginfo(f"  Raw ee_pose LEFT:      {np.array2string(latest_ee_pose['left'], precision=3)}")
            rospy.loginfo(f"  Raw ee_pose RIGHT:     {np.array2string(latest_ee_pose['right'], precision=3)}")
            rospy.loginfo(f"  Normalized state:      {np.array2string(state_normalized[:7], precision=3)}...")
            rospy.loginfo(f"  Normalized ee_pose:    {np.array2string(ee_pose_normalized[:7], precision=3)}...")
            rospy.loginfo(f"  Model output (norm):   {np.array2string(action_normalized[:7], precision=3)}...")
            rospy.loginfo(f"  Unnormalized action L: {np.array2string(action_left, precision=3)}")
            rospy.loginfo(f"  Unnormalized action R: {np.array2string(action_right, precision=3)}")

            delta_left = action_left - latest_q["left"][:7]
            delta_right = action_right - latest_q["right"][:7]
            rospy.loginfo(f"  Delta LEFT:  {np.array2string(delta_left, precision=3)}")
            rospy.loginfo(f"  Delta RIGHT: {np.array2string(delta_right, precision=3)}")
            rospy.loginfo("=" * 70)

        # --------------- EMA smoothing ---------------
        global smoothed_action
        if ENABLE_SMOOTHING:
            if smoothed_action["left"] is None:
                smoothed_action["left"] = action_left
                smoothed_action["right"] = action_right
            else:
                smoothed_action["left"] = SMOOTHING_ALPHA * action_left + (1.0 - SMOOTHING_ALPHA) * smoothed_action["left"]
                smoothed_action["right"] = SMOOTHING_ALPHA * action_right + (1.0 - SMOOTHING_ALPHA) * smoothed_action["right"]
            action_left = smoothed_action["left"]
            action_right = smoothed_action["right"]

        # --------------- Safety clipping ---------------
        if ENABLE_ACTION_CLIPPING:
            cur_left = latest_q["left"][:7]
            cur_right = latest_q["right"][:7]

            delta_left = action_left - cur_left
            delta_right = action_right - cur_right

            if np.abs(delta_left).max() > MAX_JOINT_DELTA:
                rospy.logwarn(f"[SAFETY] Clipping left delta")
                delta_left = np.clip(delta_left, -MAX_JOINT_DELTA, MAX_JOINT_DELTA)
                action_left = cur_left + delta_left

            if np.abs(delta_right).max() > MAX_JOINT_DELTA:
                rospy.logwarn(f"[SAFETY] Clipping right delta")
                delta_right = np.clip(delta_right, -MAX_JOINT_DELTA, MAX_JOINT_DELTA)
                action_right = cur_right + delta_right

        # --------------- Publish ---------------
        # 发送完整的 7 个值：6个关节 + 1个夹爪
        msg_left = JointState()
        msg_left.header.stamp = rospy.Time.now()
        msg_left.position = action_left.tolist()  # 全部7个值

        msg_right = JointState()
        msg_right.header.stamp = rospy.Time.now()
        msg_right.position = action_right.tolist()  # 全部7个值

        pub_left.publish(msg_left)
        pub_right.publish(msg_right)

        rospy.loginfo_throttle(2.0, "✓ Actions sent to robot")

        rate.sleep()


if __name__ == "__main__":
    main()