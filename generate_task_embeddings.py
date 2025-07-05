"""
生成并存储任务嵌入

这个脚本用于从预训练的CVAE模型中提取任务嵌入，并将其保存到文件中。
这样在训练和评估RL模型时，可以直接使用预生成的任务嵌入，
而不需要每次都通过任务ID查表或推断。
"""

import os
import torch
import numpy as np
import argparse
from models.generative import CVAE
from networks_with_CVAE import TaskAwareCNN_LSTM
import json

def parse_args():
    parser = argparse.ArgumentParser(description='生成任务嵌入')
    parser.add_argument('--cvae_model_path', type=str, default='/home/hjz/EVT/Agent/cvae_ep500.pt',
                        help='预训练CVAE模型路径')
    parser.add_argument('--output_dir', type=str, default='/home/hjz/EVT/Offline_RL_Active_Tracking/Tracking-Anything-with-DEVA/task',
                        help='输出目录')
    parser.add_argument('--task_ids_file', type=str, default='/home/hjz/EVT/Agent/map.ptrom',
                        help='任务ID映射文件路径')
    parser.add_argument('--z_dim', type=int, default=8,
                        help='潜在空间维度')
    return parser.parse_args()

def load_cvae_model(model_path, device, z_dim=8):
    """
    加载预训练的CVAE模型
    
    Args:
        model_path: 模型路径
        device: 设备
        z_dim: 潜在空间维度
        
    Returns:
        加载的CVAE模型和任务ID映射
    """
    print(f"加载CVAE模型: {model_path}")
    try:
        # 不使用add_safe_globals，直接加载
        import numpy as np
        
        # 尝试加载模型
        try:
            checkpoint = torch.load(model_path, map_location=device)
        except Exception as load_err:
            print(f"简单加载失败，错误: {load_err}")
            # 尝试其他加载方式
            checkpoint = torch.load(model_path, map_location=device, pickle_module=None)
        
        # 提取参数
        if 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
        else:
            model_state_dict = checkpoint
            
        if 'task_ids' in checkpoint:
            task_ids = checkpoint['task_ids']
            print(f"从模型中加载任务ID映射: {task_ids}")
        else:
            task_ids = None
            print("模型中没有任务ID映射")
            
        if 'num_tasks' in checkpoint:
            num_tasks = checkpoint['num_tasks']
            print(f"任务数量: {num_tasks}")
        else:
            num_tasks = 8  # 默认任务数量
            print(f"使用默认任务数量: {num_tasks}")
            
        # 创建CVAE模型
        cvae = CVAE(
        hidden_size=256,
            num_hidden_layers=2,
            z_dim=8,
            action_size=2,  # 动作维度为2
            state_size=12288,
            reward_size=1,
            tabular_encoder_entries=num_tasks,
            predict_state_difference=False,
            output_variance='output',
            logvar_min=-10,
            logvar_max=2,
        )
        
        # 加载权重
        cvae.load_state_dict(model_state_dict, strict=False)
        
        # 确保是表格编码器
        if hasattr(cvae, 'tabular_encoder') and not cvae.tabular_encoder:
            cvae.tabular_encoder = True
            print("已将CVAE的tabular_encoder设置为True")
        
        return cvae, task_ids, num_tasks
        
    except Exception as e:
        print(f"加载模型出错: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def generate_task_embeddings(cvae, num_tasks, device):
    """
    为每个任务ID生成嵌入
    
    Args:
        cvae: CVAE模型
        num_tasks: 任务数量
        device: 设备
        
    Returns:
        任务嵌入字典，key为任务ID，value为嵌入张量
    """
    task_embeddings = {}
    
    print(f"为{num_tasks}个任务生成嵌入...")
    with torch.no_grad():
        for task_id in range(num_tasks):
            try:
                # 使用CVAE模型生成嵌入
                mean, logvar, z_sample = cvae.forward_encoder(None, None, None, None, task_id)
                
                # 保存嵌入
                task_embeddings[task_id] = z_sample.cpu().numpy()
                print(f"成功生成任务ID {task_id} 的嵌入，形状: {z_sample.shape}")
                
            except Exception as e:
                print(f"生成任务ID {task_id} 的嵌入时出错: {e}")
                
    return task_embeddings

def save_embeddings(task_embeddings, task_ids, output_dir):
    """
    保存任务嵌入到文件
    
    Args:
        task_embeddings: 任务嵌入字典
        task_ids: 任务ID映射
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存任务嵌入
    embedding_path = os.path.join(output_dir, 'task_embeddings.pt')
    torch.save(task_embeddings, embedding_path)
    print(f"任务嵌入已保存到: {embedding_path}")
    
    # 保存任务名称到ID的映射
    if task_ids is not None:
        task_ids_path = os.path.join(output_dir, 'task_ids.json')
        with open(task_ids_path, 'w') as f:
            # 将字典键转换为字符串
            task_ids_str = {str(k): v for k, v in task_ids.items()}
            json.dump(task_ids_str, f, indent=4)
        print(f"任务ID映射已保存到: {task_ids_path}")
        
        # 创建任务名称到嵌入的映射
        name_to_embedding = {}
        for task_name, task_id in task_ids.items():
            if task_id in task_embeddings:
                name_to_embedding[task_name] = task_embeddings[task_id]
        
        # 保存任务名称到嵌入的映射
        name_embedding_path = os.path.join(output_dir, 'task_name_embeddings.pt')
        torch.save(name_to_embedding, name_embedding_path)
        print(f"任务名称到嵌入的映射已保存到: {name_embedding_path}")

def main():
    args = parse_args()
    
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载CVAE模型
    cvae, task_ids, num_tasks = load_cvae_model(args.cvae_model_path, device, args.z_dim)
    if cvae is None:
        print("加载模型失败，退出")
        return
    
    # 如果传入了任务ID映射文件，从文件加载
    if args.task_ids_file and os.path.exists(args.task_ids_file) and task_ids is None:
        try:
            # 直接加载任务ID映射
            import numpy as np
            
            # 尝试加载任务ID映射
            try:
                task_ids = torch.load(args.task_ids_file)
            except Exception as load_err:
                print(f"简单加载失败，错误: {load_err}")
                # 尝试其他加载方式
                task_ids = torch.load(args.task_ids_file, pickle_module=None)
            print(f"从文件加载任务ID映射: {task_ids}")
            # 更新任务数量
            num_tasks = max(num_tasks, len(task_ids))
        except Exception as e:
            print(f"加载任务ID映射文件出错: {e}")
    
    # 生成任务嵌入
    task_embeddings = generate_task_embeddings(cvae, num_tasks, device)
    
    # 保存任务嵌入
    save_embeddings(task_embeddings, task_ids, args.output_dir)
    
    print("任务嵌入生成完成!")

if __name__ == "__main__":
    main()
