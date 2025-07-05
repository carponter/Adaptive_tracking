import os
import time
import numpy as np
from collections import deque
import torch
import wandb
import argparse
from buffer import ReplayBuffer
import random
import cv2
torch.autograd.set_detect_anomaly = True
from agent_CNN_LSTM_with_CVAE import CQLSAC_CNN_LSTM_with_CVAE
from task_embedding_utils import load_task_embeddings

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 创建保存模型的目录
def create_save_dirs():
    dirs = ["saves", "logs"]
    for dir_name in dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"创建目录: {dir_name}")

# 定义不同视角对应的目标图片目录
goal_dirs = {
    "human": "/nfs/dataset/hjz/rand/ran_goal_human",
    "robotdog": "/nfs/dataset/hjz/rand/ran_goal_robotdog",
    "default": "/nfs/dataset/hjz/rand/ran_goal_animal1",
    "low": "/nfs/dataset/hjz/rand/ran_goal_animal2"
}

def get_random_goal(view):
    """根据视角，从对应目录随机返回一张图片"""
    directory = goal_dirs[view]
    # 列出目录下所有文件（可加过滤：例如只选png/jpg文件）
    files = [f for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not files:
        raise ValueError(f"No image files found in directory {directory}")
    selected_file = random.choice(files)
    file_path = os.path.join(directory, selected_file)
    img = cv2.imread(file_path)
    return img

def get_bounding_box(mask_image):
    # 找到目标区域 (255, 255, 255) 的所有位置
    target_pixels = np.where(np.all(mask_image == [255, 255, 255], axis=-1))

    if len(target_pixels[0]) == 0:
        return None  # 没有找到目标区域

    # 计算边界框的坐标
    y_min = np.min(target_pixels[0])
    y_max = np.max(target_pixels[0])
    x_min = np.min(target_pixels[1])
    x_max = np.max(target_pixels[1])

    return x_min, y_min, x_max, y_max

def reward_cal(state, goal):
    if state.max()==255:
        boxA = get_bounding_box(state)
        if boxA is None:
            return 0
        boxB = get_bounding_box(goal)
        if boxB is None:
            return 0

        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # compute the area of both rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # compute the intersection over union
        iou = interArea / float(boxAArea + boxBArea - interArea)
        total_reward = iou
    else:
        total_reward = 0

    return total_reward

# def sample_goal_from_data(state_data, current_view, num_samples=3, max_attempts=100):
#     """
#     从原数据 state_data 中随机抽取 num_samples 张含有白色目标的图像
#     若在 max_attempts 内未能抽到足够的，则返回已有的候选
#     """
#     goals = []
#     attempts = 0
#     while len(goals) < num_samples and attempts < max_attempts:
#         candidate = random.choice(state_data)
#         # 检查 candidate 是否含有白色目标区域
#         if candidate.max() == 255 and get_bounding_box(candidate) is not None:
#             goals.append(candidate)
#         attempts += 1
#     if len(goals) == 0:
#         goals = [get_random_goal(current_view) for _ in range(num_samples)]
#     return goals
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
    parser = argparse.ArgumentParser(description='RL with CVAE for Task Inference')
    parser.add_argument("--run_name", type=str, default="CQL-SAC-CVAE-GT2-", help="Run name")
    parser.add_argument("--buffer_path", type=str, default="/nfs/dataset/hjz/min_data")
    parser.add_argument("--episodes", type=int, default=1500, help="Number of episodes")
    parser.add_argument("--buffer_size", type=int, default=1000000, help="Maximal training dataset size")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--save_every", type=int, default=50, help="Saves the network every x epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden size for networks")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate for RL")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature parameter")
    parser.add_argument("--cql_weight", type=float, default=1.0, help="CQL weight")
    parser.add_argument("--target_action_gap", type=float, default=10, help="Target action gap")
    parser.add_argument("--with_lagrange", type=int, default=0, help="Whether to use Lagrange")
    parser.add_argument("--tau", type=float, default=5e-3, help="Tau parameter")
    parser.add_argument("--eval_every", type=int, default=1, help="Evaluation frequency")
    parser.add_argument("--lstm_seq_len", type=int, default=20, help="LSTM sequence length")
    parser.add_argument("--lstm_out", type=int, default=64, help="LSTM output size")
    parser.add_argument("--lstm_layer", type=int, default=1, help="Number of LSTM layers")
    parser.add_argument("--stack_frames", type=int, default=1, help="Number of stacked frames")
    parser.add_argument("--input_type", type=str, default='deva_cnn_lstm', help="Input type")
    parser.add_argument("--z_dim", type=int, default=8, help="CVAE latent dimension")  # 从10改为5
    
    # 简化后的参数
    parser.add_argument("--task_embeddings_dir", type=str, default='/nfs/dataset/task_embeddings', 
                        help="Path to pre-generated task embeddings")
    parser.add_argument("--use_task_encoding", type=int, default=1, 
                        help="Whether to use task encoding (1) or not (0)")
    parser.add_argument("--num_tasks", type=int, default=8, 
                        help="Number of tasks for tabular encoder")
    
    args = parser.parse_args()
    return args

def load_Buffer(buffer, path, config):
    data_list = os.listdir(path)
    data_list.sort()
    print('loading dataset buffer...')
    
    # 记录任务类型和数据
    task_data = {}
    task_ids = {}
    task_id_counter = 0
    
    # 加载预生成的任务嵌入
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    task_embeddings_dict = None
    
    if os.path.exists(config.task_embeddings_dir):
        print(f"从 {config.task_embeddings_dir} 加载预生成的任务嵌入...")
        try:
            task_embeddings_dict, name_to_id_map = load_task_embeddings(config.task_embeddings_dir, device)
            if task_embeddings_dict is not None:
                print(f"成功加载 {len(task_embeddings_dict)} 个任务嵌入")
            else:
                print("无法加载预生成的任务嵌入，将在添加到缓冲区时传递任务ID")
        except Exception as e:
            print(f"加载预生成任务嵌入时出错: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"未找到任务嵌入目录 {config.task_embeddings_dir}，请确保已运行 generate_task_embeddings.py 生成任务嵌入")
        return None, None, None
    
    for d in range(len(data_list)):
        print('loading :', data_list[d])
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
        try:
            # 直接加载数据文件
            loaded_data = torch.load(os.path.join(path, data_list[d]))
        except Exception as e:
            print(f"标准加载失败: {e}，尝试使用pickle_module=None")
            loaded_data = torch.load(os.path.join(path, data_list[d]), pickle_module=None)
            
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
        # re_iou = np.array([
        #     np.mean([reward_cal(state_tmp[i], goal) 
        #              for goal in sample_goal_from_data(state_tmp, tracker_target.split('_')[0], num_samples=3, max_attempts=60)]) 
        #     for i in range(len(state_tmp))
        # ])
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
            
        # 遍历所有时间步，将数据加入 buffer
        for i in range(state_tmp.shape[0]):
            # 设置 done 标志
            if i % state_tmp.shape[0] == 0 and i > 0:
                done = True
            else:
                done = False
                
            # 记录任务和相应的训练样本
            task_data[task_name].append({
                'state': np.array(cv2.resize(state_tmp[i], (64, 64)).transpose(2, 0, 1)),
                'action': act_tmp[i],
                'reward': np.array(re_tmp[i]),
                'next_state': np.array(cv2.resize(next_state_tmp[i], (64, 64)).transpose(2, 0, 1)),
                'done': np.array(done),
                'task_id': current_task_id,
                'task_name': task_name
            })
            
            # 获取当前任务的嵌入（通过任务ID直接从预生成的任务嵌入表中查找）
            task_embedding = None
            if task_embeddings_dict is not None and current_task_id in task_embeddings_dict:
                task_embedding = task_embeddings_dict[current_task_id]
            
            # 添加到回放缓冲区 - 直接使用任务嵌入
            buffer.add(
                torch.from_numpy(np.array(cv2.resize(state_tmp[i], (64, 64)).transpose(2, 0, 1))).float().cuda(),
                torch.from_numpy(act_tmp[i]).float().cuda(),
                torch.from_numpy(np.array(re_tmp[i])).float().cuda(),
                torch.from_numpy(np.array(cv2.resize(next_state_tmp[i], (64, 64)).transpose(2, 0, 1))).float().cuda(),
                torch.from_numpy(np.array(done)).float().cuda(),
                task_embedding=task_embedding  # 直接传递任务嵌入
            )
    
    print('loading dataset buffer finished.')
    print(f'识别到的任务类型: {list(task_ids.keys())}')
    print(f'任务ID映射: {task_ids}')
    print(f'任务数量: {len(task_ids)}')
    
    # 显示任务数据的统计信息
    print("\n任务数据统计信息:")
    for task_name, samples in task_data.items():
        print(f"任务 '{task_name}' (ID: {task_ids[task_name]}): {len(samples)} 个样本")
    
    # 保存任务ID映射，以便后续使用
    task_id_file = "saves/task_ids.pt"
    torch.save(task_ids, task_id_file)
    print(f"任务ID映射已保存到: {task_id_file}")
    
    return buffer, task_data, task_ids

def train_rl_with_cvae(config):
    """使用预训练的CVAE模型进行RL训练"""
    # 设置随机种子
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 创建模型保存目录
    create_save_dirs()
    
    # 初始化缓冲区
    buffer = ReplayBuffer(buffer_size=config.buffer_size, batch_size=config.batch_size, device=device,
                          lstm_seq_len=config.lstm_seq_len, config=config)  # 设置不按任务ID过滤
    
    # 加载数据和任务嵌入
    buffer, task_data, task_ids = load_Buffer(buffer, config.buffer_path, config)
    if buffer is None:
        print("加载数据失败，确保已运行 generate_task_embeddings.py 生成任务嵌入")
        return
        
    # 确保使用正确的任务数量
    num_tasks = max(len(task_ids), config.num_tasks)
    print(f"使用任务ID映射: {task_ids}，共 {num_tasks} 个任务")
        
    with wandb.init(project="CQL", name=config.run_name, config=config) as run:
        # 记录额外配置信息到wandb
        wandb.config.update({
            "num_tasks": num_tasks,
            "tasks": list(task_ids.keys()) if task_ids else "Unknown",
        })
        
        # 初始化任务感知RL Agent
        if 'deva' in config.input_type.lower() or 'image' in config.input_type.lower() or 'mask' in config.input_type.lower():
            state_channels = 3
            if 'devadepth' in config.input_type.lower() or 'rgbd' in config.input_type.lower():
                state_channels = 4
                
            if 'cnn' in config.input_type.lower() and 'lstm' in config.input_type.lower():
                agent = CQLSAC_CNN_LSTM_with_CVAE(
                    state_size=(state_channels, 64, 64),
                    action_size=2,
                    tau=config.tau,
                    hidden_size=config.hidden_size,
                    learning_rate=config.learning_rate,
                    temp=config.temperature,
                    with_lagrange=config.with_lagrange,
                    cql_weight=config.cql_weight,
                    target_action_gap=config.target_action_gap,
                    device=device,
                    stack_frames=config.stack_frames,
                    lstm_seq_len=config.lstm_seq_len,
                    lstm_layer=config.lstm_layer,
                    lstm_out=config.lstm_out,
                    z_dim=config.z_dim,
                    reward_size=1,
                    use_task_encoding=config.use_task_encoding,
                    num_tasks=num_tasks
                )
                
                # 设置任务数量
                if hasattr(agent, 'num_tasks'):
                    agent.num_tasks = num_tasks
                elif hasattr(agent, 'CNN_LSTM') and hasattr(agent.CNN_LSTM, 'num_tasks'):
                    agent.CNN_LSTM.num_tasks = num_tasks
                
        agent.to(device)
        wandb.watch(agent, log="gradients", log_freq=10)
        
        # 加载任务嵌入
        print(f"从 {config.task_embeddings_dir} 加载预生成的任务嵌入...")
        task_embeddings_dict, task_names_dict = load_task_embeddings(config.task_embeddings_dir, device)
        if task_embeddings_dict is None:
            print("无法加载任务嵌入，退出训练")
            return
        
        # 记录所有可用任务ID
        all_task_ids = list(task_ids.values())
        print(f"可用任务ID: {all_task_ids}")
        
        # 开始训练
        steps = 0
        average10 = deque(maxlen=10)
        total_steps = 0
        
        # 创建模型保存目录
        checkpoint_dir = "saves/checkpoints_gt"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            
        for i in range(1, config.episodes + 1):
            episode_start_time = time.time()
            episode_steps = 0
            rewards = 0
            
            # 训练循环 - 不再按任务ID筛选，让缓冲区随机采样
            while True:
                # 从缓冲区随机采样数据
                experiences = buffer.sample()
                
                # 训练Agent
                agent.train()  # 设置为训练模式
                results = agent.learn(experiences)
                
                train_q1, train_q2, policy_loss, alpha_loss, bellmann_error1, bellmann_error2, cql1_loss, cql2_loss, current_alpha, lagrange_alpha_loss, lagrange_alpha, *extra = results
                
                steps += 1
                if steps >= 200:  # 每个episode训练200个批次
                    episode_steps += 1
                    steps = 0
                    break
            
            # 计算本episode的耗时
            episode_time = time.time() - episode_start_time
            
            average10.append(rewards)
            total_steps += episode_steps
            print(f"Episode: {i} | Policy Loss: {policy_loss} | Time: {episode_time:.2f}s")
            
            # 记录训练指标 - 移除特定任务相关的指标，因为我们现在是随机混合任务训练
            log_dict = {
                "Steps": total_steps,
                "train_q1": train_q1,
                "train_q2": train_q2,
                "Policy Loss": policy_loss,
                "Alpha Loss": alpha_loss,
                "Lagrange Alpha Loss": lagrange_alpha_loss,
                "CQL1 Loss": cql1_loss,
                "CQL2 Loss": cql2_loss,
                "Bellman error 1": bellmann_error1,
                "Bellman error 2": bellmann_error2,
                "Alpha": current_alpha,
                "Lagrange Alpha": lagrange_alpha,
                "Episode": i,
                "Buffer size": buffer.__len__(),
                "Episode time (s)": episode_time,
            }
            
            wandb.log(log_dict)
            
            # 定期保存模型
            if i % config.save_every == 0 or i == config.episodes:
                # 保存模型checkpoint
                checkpoint_path = f"{checkpoint_dir}/CQL-SAC-CVAE_ep{i}.pt"
                
                torch.save({
                    'model_state_dict': agent.state_dict(),
                    'task_ids': task_ids,
                    'num_tasks': num_tasks,
                    'episode': i,
                    'config': vars(config),
                }, checkpoint_path)
                print(f"已保存模型checkpoint到: {checkpoint_path}")
                
                # 保存到wandb
                wandb.save(checkpoint_path)
        
        print("训练完成!")
        
        # 保存最终模型
        final_model_path = f"{checkpoint_dir}/final_CQL-SAC-CVAE.pt"
        torch.save({
            'model_state_dict': agent.state_dict(),
            'task_ids': task_ids,
            'num_tasks': num_tasks,
            'episode': config.episodes,
            'config': vars(config),
        }, final_model_path)
        
        print(f"最终模型已保存到: {final_model_path}")
        wandb.save(final_model_path)

if __name__ == "__main__":
    config = get_config()
    train_rl_with_cvae(config)
