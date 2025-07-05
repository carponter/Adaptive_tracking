import os
import sys
import time
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import random
import wandb
import cv2
import traceback
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, TensorDataset

from models.generative import CVAE

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# 创建保存模型的目录
def create_save_dirs():
    current_dir = os.getcwd()  # 获取当前工作目录
    dirs = ["saves", "logs"]
    for dir_name in dirs:
        dir_path = os.path.join(current_dir, dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"创建目录: {dir_path}")

# 定义不同视角对应的目标图片目录
goal_dirs = {
    "human": "/nfs/dataset/hjz/rand/ran_goal_human",
    "robotdog": "/nfs/dataset/hjz/rand/ran_goal_robotdog",
    "default": "/nfs/dataset/hjz/rand/ran_goal_animal1",
    "low": "/nfs/dataset/hjz/rand/ran_goal_animal2"
}

def get_random_goal(view):
    """根据视角，从对应目录随机返回一张图片"""
    if view in goal_dirs:
        goal_dir = goal_dirs[view]
    else:
        goal_dir = goal_dirs["default"]
    
    goal_files = [f for f in os.listdir(goal_dir) if os.path.isfile(os.path.join(goal_dir, f))]
    if not goal_files:
        return None
    
    goal_file = random.choice(goal_files)
    goal_path = os.path.join(goal_dir, goal_file)
    
    # 尝试不同方法加载图像
    try:
        # 如果是numpy数组
        if goal_file.endswith('.npy'):
            goal = np.load(goal_path)
        # 如果是图像文件
        else:
            goal = cv2.imread(goal_path)
            if goal is None:
                print(f"无法使用cv2.imread加载图像: {goal_path}")
                return None
    except Exception as e:
        print(f"加载目标图像时出错: {e}")
        return None
    
    return goal

def get_bounding_box(mask_image):
    """获取掩码图像中目标的边界框"""
    # 确保mask_image是2D或3D数组
    if len(mask_image.shape) > 2:
        if mask_image.shape[2] == 3:  # 如果是RGB图像
            # 检查是否为白色(255,255,255)
            white_pixels = np.where(np.all(mask_image == 255, axis=2))
            if len(white_pixels[0]) == 0:
                return None
            
            # 计算边界框
            min_y, max_y = np.min(white_pixels[0]), np.max(white_pixels[0])
            min_x, max_x = np.min(white_pixels[1]), np.max(white_pixels[1])
        else:
            # 如果不是RGB图像，使用第一个通道
            mask_binary = mask_image[:, :, 0] > 0
            if not np.any(mask_binary):
                return None
                
            # 找到非零区域的索引
            y_indices, x_indices = np.where(mask_binary)
            if len(y_indices) == 0:
                return None
                
            # 计算边界框
            min_y, max_y = np.min(y_indices), np.max(y_indices)
            min_x, max_x = np.min(x_indices), np.max(x_indices)
    else:
        # 如果是2D图像，直接使用阈值
        mask_binary = mask_image > 0
        if not np.any(mask_binary):
            return None
            
        # 找到非零区域的索引
        y_indices, x_indices = np.where(mask_binary)
        if len(y_indices) == 0:
            return None
            
        # 计算边界框
        min_y, max_y = np.min(y_indices), np.max(y_indices)
        min_x, max_x = np.min(x_indices), np.max(x_indices)
    
    return min_x, min_y, max_x, max_y

def reward_cal(state, goal):
    """计算状态和目标之间的IoU奖励"""
    # 获取边界框
    state_bbox = get_bounding_box(state)
    goal_bbox = get_bounding_box(goal)
    
    # 如果任一边界框无效，返回0
    if state_bbox is None or goal_bbox is None:
        return 0
    
    # 解析边界框
    x_min_1, y_min_1, x_max_1, y_max_1 = state_bbox
    x_min_2, y_min_2, x_max_2, y_max_2 = goal_bbox
    
    # 计算交集边界
    x_min_i = max(x_min_1, x_min_2)
    y_min_i = max(y_min_1, y_min_2)
    x_max_i = min(x_max_1, x_max_2)
    y_max_i = min(y_max_1, y_max_2)
    
    # 检查是否有交集
    if x_max_i < x_min_i or y_max_i < y_min_i:
        return 0
    
    # 计算面积
    area_1 = (x_max_1 - x_min_1) * (y_max_1 - y_min_1)
    area_2 = (x_max_2 - x_min_2) * (y_max_2 - y_min_2)
    area_i = (x_max_i - x_min_i) * (y_max_i - y_min_i)
    
    # 计算IoU
    iou = area_i / float(area_1 + area_2 - area_i)
    return iou

def sample_goal_from_data(state_data, current_view, num_samples=2, max_attempts=100):
    """
    从原数据 state_data 中获取前两帧作为goal
    如果数据不足两帧，则使用随机抽取的方式补充
    """
    goals = []
    
    # 检查state_data的长度是否足够
    if len(state_data) >= 2:
        # 取前两帧作为goal，但仍需确认它们包含目标
        for i in range(min(2, len(state_data))):
            if state_data[i].max() == 255 and get_bounding_box(state_data[i]) is not None:
                goals.append(state_data[i])
    
    # 如果没有获取到足够的goal，则使用随机抽取补充
    attempts = 0
    while len(goals) < num_samples and attempts < max_attempts:
        candidate = random.choice(state_data)
        # 检查 candidate 是否含有白色目标区域
        if candidate.max() == 255 and get_bounding_box(candidate) is not None:
            goals.append(candidate)
        attempts += 1
    
    # 如果仍然没有足够的goal，则从外部目录获取
    if len(goals) == 0:
        goals = [get_random_goal(current_view) for _ in range(num_samples)]
    
    return goals
def get_config():
    """获取配置参数"""
    parser = argparse.ArgumentParser(description='CVAE Training for Task Inference')
    
    # 必要的参数
    parser.add_argument("--run_name", type=str, default="CVAE-Training-Delta-", help="运行名称")
    parser.add_argument("--data_path", type=str, default="/nfs/dataset/hjz/test", help="数据集路径")
    parser.add_argument("--seed", type=int, default=1, help="随机种子")
    parser.add_argument("--save_interval", type=int, default=50, help="每X个epoch保存一次模型")
    parser.add_argument("--batch_size", type=int, default=32, help="批大小")
    parser.add_argument("--hidden_size", type=int, default=256, help="隐藏层大小")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--z_dim", type=int, default=8, help="CVAE潜变量维度")
    parser.add_argument("--n_epochs", type=int, default=1000, help="训练的epoch数")
    parser.add_argument("--log_interval", type=int, default=10, help="日志记录间隔")
    
    # CVAE特定参数
    parser.add_argument("--use_tabular_encoder", action="store_true", help="使用表格编码器")
    parser.add_argument("--predict_state_difference", action='store_true', help="预测状态差异而非绝对状态")
    parser.add_argument("--output_variance", type=str, default='output',
                        choices=['zero', 'parameter', 'output', 'output_raw', 'reference'], 
                        help="CVAE输出方差类型")
    parser.add_argument("--logvar_min", type=float, default=-5.0, help="最小对数方差")
    parser.add_argument("--logvar_max", type=float, default=2.0, help="最大对数方差")
    parser.add_argument("--merge_reward_next_state", action='store_true', default=False, help="合并奖励和下一状态")
    
    # 设备和保存相关
    parser.add_argument("--use_gpu", action="store_true", default=True, help="是否使用GPU")
    
    # 仅保留原始输入类型参数
    parser.add_argument("--input_type", type=str, default='deva', help="输入类型(deva, devadepth等)")
    
    args = parser.parse_args()
    return args

class MyDataset(TensorDataset):
    """
    数据集类，用于处理多任务数据
    """
    def __init__(self, dataset, goals):
        # 首先打印每个任务数据的原始形状，以便调试
        print("任务数据原始形状:")
        for i, task_data in enumerate(dataset):
            states_shape = task_data[0].shape
            actions_shape = task_data[1].shape
            rewards_shape = task_data[2].shape
            next_states_shape = task_data[3].shape
            print(f"任务 {i} - 状态: {states_shape}, 动作: {actions_shape}, 奖励: {rewards_shape}, 下一状态: {next_states_shape}")
        
        # 重新组织数据，确保形状正确
        processed_data = []
        task_indices = []
        
        # 处理每个任务的数据
        for task_idx, task_data in enumerate(dataset):
            states = task_data[0]
            actions = task_data[1]
            rewards = task_data[2]
            next_states = task_data[3]
            
            # 归一化状态数据（如果尚未归一化）
            if states.max() > 1.0:
                states = states / 255.0
            
            # 归一化下一状态数据（如果尚未归一化）
            if next_states.max() > 1.0:
                next_states = next_states / 255.0
            
            # 归一化动作数据
            min_val = np.array([-30, -100]).astype(np.float32)
            max_val = np.array([30, 100]).astype(np.float32)
            
            # 将动作归一化到0到1的范围
            actions_normalized = ((actions - min_val) / (max_val - min_val)).astype(np.float32)
            
            # 将动作归一化到-1到1的范围
            actions_normalized = (2 * actions_normalized - 1).astype(np.float32)
            actions = actions_normalized
            
            # 确保扁平化前数据维度正确
            n_samples = states.shape[0]
            
            # 检查并调整数据形状
            if len(states.shape) > 2:  # 如果状态是图像数据 (samples, channels, height, width)
                # 扁平化状态和下一状态
                states_flat = states.reshape(n_samples, -1)
                next_states_flat = next_states.reshape(n_samples, -1)
            else:
                states_flat = states
                next_states_flat = next_states
                
            # 确保动作和奖励是二维的 (samples, feature_dim)
            if len(actions.shape) == 1:
                actions = actions.reshape(-1, 1)
            if len(rewards.shape) == 1:
                rewards = rewards.reshape(-1, 1)
                
            # 添加到处理后的数据
            processed_data.append((states_flat, actions, rewards, next_states_flat))
            task_indices.extend([task_idx] * n_samples)
        
        # 将处理后的数据转换为张量
        states_tensor = torch.cat([torch.FloatTensor(data[0]) for data in processed_data])
        actions_tensor = torch.cat([torch.FloatTensor(data[1]) for data in processed_data])
        rewards_tensor = torch.cat([torch.FloatTensor(data[2]) for data in processed_data])
        next_states_tensor = torch.cat([torch.FloatTensor(data[3]) for data in processed_data])
        task_idx_tensor = torch.LongTensor(task_indices)
        
        # 打印处理后的张量形状
        print(f"处理后的张量形状:")
        print(f"状态张量: {states_tensor.shape}")
        print(f"动作张量: {actions_tensor.shape}")
        print(f"奖励张量: {rewards_tensor.shape}")
        print(f"下一状态张量: {next_states_tensor.shape}")
        print(f"任务索引张量: {task_idx_tensor.shape}")
        
        # 初始化父类
        super().__init__(states_tensor, actions_tensor, rewards_tensor, next_states_tensor, task_idx_tensor)

def get_cvae_loss(model, obs, action, reward, next_obs, task_idx, detach_encoder=False):
    """
    计算CVAE损失函数
    """
    # 将数据移动到正确的设备
    device = next(model.parameters()).device
    obs = obs.to(device)
    action = action.to(device)
    reward = reward.to(device)
    next_obs = next_obs.to(device)
    task_idx = task_idx.to(device)
    
    # 根据是否分离编码器梯度来前向传播
    with torch.no_grad() if detach_encoder else torch.enable_grad():
        mean, logvar, z_sample = model.forward_encoder(obs, action, reward, next_obs, task_idx)

    # KL散度损失设为0（已移除）
    kl_loss = torch.tensor(0.0, device=device)
    print("reward", reward)

    decoder_output = model.forward_decoder(obs, action, z_sample)
    next_obs_pred, reward_pred, logvar_s, logvar_r = decoder_output
    # print(f"Decoder输出: {next_obs_pred.shape}, {reward_pred.shape}, {logvar_s.shape}, {logvar_r.shape}")
    print(f"Decoder输出: {next_obs_pred}, {reward_pred}, {logvar_s}, {logvar_r}")
    
    # 计算重构损失并返回
    return kl_loss, *model.losses(obs, action, reward, next_obs, z_sample)

def load_training_data(data_path, config):
    """加载训练数据，并按任务划分"""
    data_list = os.listdir(data_path)
    data_list.sort()
    print('Loading dataset...')
    
    # 记录任务类型和数据
    task_data = {}
    task_ids = {}
    task_id_counter = 0
    
    for d in range(len(data_list)):
        print('Loading:', data_list[d])
        # 从文件名推断任务类型
        file_name = data_list[d].lower()
        
        # 确定FOV和视角
        if "75" in file_name:
            fov = "75"
        elif "90" in file_name:
            fov = "90"
        elif "105" in file_name:
            fov = "105"
        else:
            fov = "90"
            
        if "rot30" in file_name:
            view = "lookdown"
        else:
            view = "level"
            
        # 确定追踪者-目标组合
        if "human" in file_name:
            tracker_target = "human_human"
        elif "robotdog" in file_name:
            tracker_target = "robotdog_human"
        elif "animal1" in file_name:
            tracker_target = "human_human"  # 默认
        elif "animal2" in file_name:
            tracker_target = "human_human"  # 默认
        elif "animal3" in file_name:
            tracker_target = "robotdog_human"  # 默认
        else:
            tracker_target = "human_human"  # 默认
            
        # 组合成任务标识
        task_name = f"{fov}_{view}_{tracker_target}"
        
        # 为任务分配唯一ID
        if task_name not in task_ids:
            task_ids[task_name] = task_id_counter
            task_id_counter += 1
            
        current_task_id = task_ids[task_name]
            
        # 加载 .pt 文件
        loaded_data = torch.load(os.path.join(data_path, data_list[d]), weights_only=False)
        if isinstance(loaded_data, list):
            frames = [frame for frame in loaded_data if frame]  
        else:
            frames = [loaded_data]
        
        # 处理图像数据
        if ('deva' in config.input_type.lower() or 'image' in config.input_type.lower() or 'mask' in config.input_type.lower()):
            state_tmp = np.array([np.array(frame['mask'][:, :, 0:3]) for frame in frames])[:-1]
            next_state_tmp = np.array([np.array(frame['mask'][:, :, 0:3]) for frame in frames])[1:]
            if 'devadepth' in config.input_type.lower() or 'rgbd' in config.input_type.lower():
                state_tmp = np.array([np.array(frame['image'][:, :, 0:4]) for frame in frames])[:-1]
                next_state_tmp = np.array([np.array(frame['image'][:, :, 0:4]) for frame in frames])[1:]
        
        # 获取动作与奖励信息
        act_tmp = np.array([np.array(frame['action']) for frame in frames])[:-1].squeeze(axis=1)
        
        # 计算IOU奖励
        re_iou = np.array([
            np.mean([reward_cal(state_tmp[i], goal) 
                     for goal in sample_goal_from_data(state_tmp, tracker_target.split('_')[0], num_samples=2, max_attempts=60)]) 
            for i in range(len(state_tmp))
        ])
        re_tmp = re_iou
        
        # 确保数据长度一致
        assert state_tmp.shape[0] == next_state_tmp.shape[0] and re_tmp.shape[0] == next_state_tmp.shape[0] and \
            next_state_tmp.shape[0] == act_tmp.shape[0], "数据长度不匹配，请检查每一帧的数据是否齐全！"
        
        # 将当前任务的数据添加到字典中
        if task_name not in task_data:
            task_data[task_name] = []
            
        # 遍历所有时间步，处理数据
        for i in range(state_tmp.shape[0]):
            # 设置 done 标志
            if i == state_tmp.shape[0] - 1:
                done = True
            else:
                done = False
            
            # 重塑状态数据
            current_state = cv2.resize(state_tmp[i], (64, 64)).transpose(2, 0, 1)
            next_state = cv2.resize(next_state_tmp[i], (64, 64)).transpose(2, 0, 1)
            
            # 调整动作形状，确保是二维的
            action = act_tmp[i].reshape(-1)
            if len(action.shape) == 0:  # 如果是标量，转换为数组
                action = np.array([action])
            
            # 调整奖励形状，确保是标量
            reward = np.array([re_tmp[i]])
                
            # 记录任务和相应的训练样本
            task_data[task_name].append({
                'state': current_state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'done': np.array(done),
                'task_id': current_task_id
            })
    
    print('Dataset loading finished.')
    print(f'识别到的任务类型: {list(task_ids.keys())}')
    print(f'任务ID映射: {task_ids}')
    print(f'任务数量: {len(task_ids)}')
    
    # 显示任务数据的统计信息
    print("\n任务数据统计信息:")
    for task_name, samples in task_data.items():
        print(f"任务 '{task_name}' (ID: {task_ids[task_name]}): {len(samples)} 个样本")
        # 打印第一个样本的形状，以便调试
        if samples:
            first_sample = samples[0]
            print(f"  样本形状 - 状态: {first_sample['state'].shape}, 动作: {first_sample['action'].shape}, "
                  f"奖励: {first_sample['reward'].shape}, 下一状态: {first_sample['next_state'].shape}")
    
    return task_data, task_ids

def prepare_training_data(data_path, config):
    """
    加载并预处理数据，不使用k折交叉验证
    返回训练数据集
    """
    print(f"加载数据准备训练...")
    
    # 从task_data和task_ids加载原始数据
    task_data, task_ids = load_training_data(data_path, config)
    
    # 将任务名称按照ID排序
    sorted_task_names = sorted(task_ids.keys(), key=lambda x: task_ids[x])
    
    print(f"数据准备中，共有 {len(sorted_task_names)} 个任务...")
    
    # 准备训练数据
    train_dataset = []
    train_goals = []
    
    # 所有任务数据都用于训练
    for task_name in sorted_task_names:
        samples = task_data[task_name]
        
        states = np.array([sample['state'] for sample in samples])
        actions = np.array([sample['action'] for sample in samples])
        rewards = np.array([sample['reward'].reshape(-1) for sample in samples])
        next_states = np.array([sample['next_state'] for sample in samples])
        
        # 打印数据形状信息
        print(f"  任务 {task_name} 数据形状:")
        print(f"    状态: {states.shape}")
        print(f"    动作: {actions.shape}")
        print(f"    奖励: {rewards.shape}")
        print(f"    下一状态: {next_states.shape}")
        
        # 添加到训练数据集
        train_dataset.append([states, actions, rewards, next_states])
        train_goals.append(task_name)
    
    print(f"数据准备完成")
    return train_dataset, train_goals, task_ids

def main():
    # 获取配置
    config = get_config()
    
    # 创建保存目录
    create_save_dirs()
    
    # 设置随机种子
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() and config.use_gpu else "cpu")
    print(f"使用设备: {device}")
    
    # 初始化wandb
    wandb.init(
        project="cvae-training",
        name=config.run_name + time.strftime("%Y%m%d-%H%M%S"),
        config=vars(config)
    )
    
    # 加载训练数据
    train_dataset, train_goals, task_ids = prepare_training_data(config.data_path, config)

    task_ids_file = os.path.join(current_dir, "saves", "map.ptrom")
    torch.save(task_ids, task_ids_file)
    print(f"保存任务ID映射到 {task_ids_file}")    
    # 更新wandb配置
    wandb.config.update({
        "model_type": "CVAE Training",
        "num_tasks": len(task_ids),
        "tasks": list(task_ids.keys())
    }, allow_val_change=True)
    
    print("\n============= 开始训练 =============")
    
    # 创建数据集对象
    train_set = MyDataset(train_dataset, train_goals)
    print(f"训练集大小: {len(train_set)}")
    
    # 确定输入通道数
    if 'deva' in config.input_type.lower() or 'image' in config.input_type.lower() or 'mask' in config.input_type.lower():
        state_channels = 3
    if 'devadepth' in config.input_type.lower() or 'rgbd' in config.input_type.lower():
        state_channels = 4
    
    # 计算扁平化状态大小
    input_shape = (state_channels, 64, 64)
    flat_state_size = np.prod(input_shape)
    
    # 初始化CVAE模型
    model = CVAE(
        hidden_size=config.hidden_size,
        num_hidden_layers=2,
        z_dim=config.z_dim,
        action_size=2,  # 动作维度为2
        state_size=flat_state_size,
        reward_size=1,
        tabular_encoder_entries=len(task_ids) if config.use_tabular_encoder else None,
        predict_state_difference=config.predict_state_difference,
        output_variance=config.output_variance,
        merge_reward_next_state=config.merge_reward_next_state,
        logvar_min=config.logvar_min,
        logvar_max=config.logvar_max,
    ).to(device)
    
    # 初始化优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # 更新wandb配置
    wandb.config.update({
        "num_train_tasks": len(train_goals),
        "train_tasks": train_goals,
        "state_channels": state_channels,
        "tabular_encoder": config.use_tabular_encoder
    }, allow_val_change=True)
    
    # 训练循环
    print(f"开始训练，共{config.n_epochs}个epochs...")
    model.train()
    
    # 创建DataLoader进行批量加载，避免一次性将所有数据加载到GPU
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    
    prev_epoch_total_loss = None
    
    for epoch in range(config.n_epochs):
        epoch_start_time = time.time()
        
        n_iter = 0
        epoch_losses = defaultdict(float)
        
        # 使用DataLoader批量加载数据
        for batch_data in train_loader:
            # 将批次数据移到设备上
            train_data = tuple(t.to(device) for t in batch_data)
            
            if len(train_data[0]) == 0:
                continue
                
            # 训练数据的损失计算
            kl_loss, obs_recon_loss, reward_recon_loss, unscaled_obs_loss, unscaled_rew_loss, ref_obs_loss, ref_rew_loss \
                = get_cvae_loss(model, *train_data)
            

                
            # 计算总损失
            total_loss = obs_recon_loss + reward_recon_loss
            if ref_obs_loss is not None:
                total_loss = total_loss + ref_obs_loss + ref_rew_loss
            
            # 反向传播和优化
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # 更新计数器
            n_iter += 1
            
            # 累积损失
            epoch_losses["kl_loss"] += kl_loss.item()
            epoch_losses["obs_recon_loss"] += obs_recon_loss.item()
            epoch_losses["reward_recon_loss"] += reward_recon_loss.item()
            epoch_losses["unscaled_obs_loss"] += unscaled_obs_loss.item()
            epoch_losses["unscaled_rew_loss"] += unscaled_rew_loss.item()
            epoch_losses["total_loss"] += total_loss.item()
            
            if ref_obs_loss is not None:
                epoch_losses["ref_obs_loss"] += ref_obs_loss.item()
                epoch_losses["ref_rew_loss"] += ref_rew_loss.item()
            
            # 每隔一定批次打印和记录损失
            if n_iter % config.log_interval == 0:
                print(f'Epoch {epoch}/{config.n_epochs}, Batch {n_iter}/{len(train_loader)}, '
                      f'KL Loss: {kl_loss.item():.4f}, Obs Recon: {obs_recon_loss.item():.4f}, '
                      f'Reward Recon: {reward_recon_loss.item():.4f}, Total: {total_loss.item():.4f}')
                
                # 记录到wandb
                batch_metrics = {
                    "epoch": epoch + (n_iter / len(train_loader)),  # 使用epoch作为主要横坐标，并添加批次进度作为小数部分
                    "batch_kl_loss": kl_loss.item(),
                    "batch_obs_recon_loss": obs_recon_loss.item(),
                    "batch_reward_recon_loss": reward_recon_loss.item(),
                    "batch_total_loss": total_loss.item()
                }
                wandb.log(batch_metrics)
        
        # 计算每个epoch的平均损失
        avg_losses = {k: v/max(1, n_iter) for k, v in epoch_losses.items()}
        
        # 计算epoch耗时
        epoch_time = time.time() - epoch_start_time
        
        # 监控损失值异常变化
        if epoch > 0 and prev_epoch_total_loss is not None:
            loss_change = avg_losses["total_loss"] - prev_epoch_total_loss
            loss_percent_change = (loss_change / prev_epoch_total_loss) * 100 if prev_epoch_total_loss != 0 else 0
            
            # 如果损失变化过大，记录警告
            if abs(loss_percent_change) > 20:  # 如果损失变化超过20%
                print(f"⚠️ 警告: Epoch {epoch} 损失变化较大: {loss_percent_change:.2f}%")
                # 记录异常损失变化到wandb
                wandb.log({"epoch": epoch, "loss_change_warning": loss_percent_change})
        
        # 保存当前损失值以便下一个epoch比较
        prev_epoch_total_loss = avg_losses["total_loss"]
        
        # 打印epoch摘要
        print(f"\nEpoch {epoch}/{config.n_epochs} 完成, 耗时: {epoch_time:.2f}秒")
        print(f"平均损失: KL={avg_losses['kl_loss']:.4f}, Obs={avg_losses['obs_recon_loss']:.4f}, "
              f"Reward={avg_losses['reward_recon_loss']:.4f}, Total={avg_losses['total_loss']:.4f}")
        
        # 记录epoch摘要到wandb
        wandb.log({
            "epoch": epoch,  # 使用整数epoch作为主要横坐标
            "epoch_time": epoch_time,
            "epoch_avg_kl_loss": avg_losses["kl_loss"],
            "epoch_avg_obs_recon_loss": avg_losses["obs_recon_loss"],
            "epoch_avg_reward_recon_loss": avg_losses["reward_recon_loss"],
            "epoch_avg_total_loss": avg_losses["total_loss"]
        })
        
        # 按轮次保存模型，每个保存间隔都保存
        if (epoch + 1) % config.save_interval == 0 or epoch == config.n_epochs - 1:
            # 保存CVAE模型
            current_dir = os.getcwd()
            save_path = os.path.join(current_dir, "saves", f"cvae_ep{epoch+1}.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'task_ids': task_ids,
                'avg_losses': avg_losses,
                'config': vars(config)
            }, save_path)
            print(f"保存CVAE模型到 {save_path}")
    
    print("CVAE模型训练完成!")
    wandb.finish()

if __name__ == "__main__":
    main()
