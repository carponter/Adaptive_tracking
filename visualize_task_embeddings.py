"""
可视化任务嵌入

这个脚本用于可视化从CVAE模型中提取的任务嵌入。
它可以使用t-SNE或PCA降维技术将高维任务嵌入降到2维，便于可视化。
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='可视化任务嵌入')
    parser.add_argument('--embeddings_dir', type=str, default='/home/hjz/EVT/Offline_RL_Active_Tracking/Tracking-Anything-with-DEVA/task',
                        help='任务嵌入目录')
    parser.add_argument('--method', type=str, default='tsne', choices=['tsne', 'pca'],
                        help='降维方法：tsne或pca')
    parser.add_argument('--output_dir', type=str, default='./visualization',
                        help='输出目录，用于保存可视化结果')
    return parser.parse_args()

def load_task_embeddings(embeddings_dir):
    """
    加载任务嵌入和任务ID映射
    
    Args:
        embeddings_dir: 任务嵌入目录
        
    Returns:
        任务嵌入、任务ID映射和任务名称到嵌入的映射
    """
    # 加载任务嵌入
    embedding_path = os.path.join(embeddings_dir, 'task_embeddings.pt')
    if not os.path.exists(embedding_path):
        raise FileNotFoundError(f"任务嵌入文件 {embedding_path} 不存在")
    
    task_embeddings = torch.load(embedding_path)
    print(f"加载了 {len(task_embeddings)} 个任务嵌入")
    
    # 加载任务ID映射
    task_ids_path = os.path.join(embeddings_dir, 'task_ids.json')
    task_ids = None
    if os.path.exists(task_ids_path):
        with open(task_ids_path, 'r') as f:
            task_ids = json.load(f)
            # 将键转换为字符串
            task_ids = {k: int(v) for k, v in task_ids.items()}
        print(f"加载了 {len(task_ids)} 个任务ID映射")
    
    # 加载任务名称到嵌入的映射
    name_embedding_path = os.path.join(embeddings_dir, 'task_name_embeddings.pt')
    name_to_embedding = None
    if os.path.exists(name_embedding_path):
        name_to_embedding = torch.load(name_embedding_path)
        print(f"加载了 {len(name_to_embedding)} 个任务名称到嵌入的映射")
    
    return task_embeddings, task_ids, name_to_embedding

def visualize_embeddings(task_embeddings, task_ids, method='tsne'):
    """
    可视化任务嵌入
    
    Args:
        task_embeddings: 任务嵌入字典
        task_ids: 任务ID映射
        method: 降维方法，可选：'tsne'或'pca'
        
    Returns:
        任务嵌入的2D表示和标签
    """
    # 收集所有嵌入和对应的标签
    embeddings = []
    labels = []
    
    # 创建ID到任务名称的映射
    id_to_name = {}
    if task_ids is not None:
        # 反转映射，使得可以通过ID查找任务名称
        for name, idx in task_ids.items():
            id_to_name[idx] = name
    
    # 排序，确保可视化顺序一致性
    sorted_ids = sorted(task_embeddings.keys())
    
    for task_id in sorted_ids:
        # 获取嵌入
        embedding = task_embeddings[task_id]
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.cpu().numpy()
        if len(embedding.shape) > 1 and embedding.shape[0] == 1:
            embedding = embedding.flatten()
            
        embeddings.append(embedding)
        
        # 获取标签
        if task_id in id_to_name:
            labels.append(id_to_name[task_id])
        else:
            labels.append(f"Task {task_id}")
    
    # 将嵌入转换为numpy数组
    embeddings = np.array(embeddings)
    
    # 降维
    if method == 'tsne':
        print("使用t-SNE进行降维...")
        embeddings_2d = TSNE(n_components=2, perplexity=min(len(embeddings)-1, 5), 
                             random_state=42).fit_transform(embeddings)
    elif method == 'pca':
        print("使用PCA进行降维...")
        embeddings_2d = PCA(n_components=2).fit_transform(embeddings)
    else:
        raise ValueError(f"不支持的降维方法: {method}")
    
    return embeddings_2d, labels

def plot_embeddings(embeddings_2d, labels, method, output_dir):
    """
    绘制嵌入的2D表示
    
    Args:
        embeddings_2d: 嵌入的2D表示
        labels: 标签
        method: 降维方法
        output_dir: 输出目录
    """
    plt.figure(figsize=(10, 8))
    
    # 绘制散点图
    for i, (x, y) in enumerate(embeddings_2d):
        plt.scatter(x, y, marker='o', s=100)
        plt.annotate(labels[i], (x, y), xytext=(5, 5), textcoords='offset points', fontsize=12)
    
    plt.title(f'Task Embedding Visualization (using {method.upper()})')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图像
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'task_embeddings_{method}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    # 显示图像
    plt.show()
    
    # 额外生成一个HSV彩色版本
    plt.figure(figsize=(10, 8))
    
    # 创建颜色映射
    colors = plt.cm.hsv(np.linspace(0, 1, len(embeddings_2d)))
    
    # 绘制散点图
    for i, (x, y) in enumerate(embeddings_2d):
        plt.scatter(x, y, color=colors[i], marker='o', s=100)
        plt.annotate(labels[i], (x, y), xytext=(5, 5), textcoords='offset points', fontsize=12)
    
    plt.title(f'Task Embedding Visualization (using {method.upper()} - Color Version)')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图像
    output_path = os.path.join(output_dir, f'task_embeddings_{method}_color.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Color visualization saved to: {output_path}")

def visualize_heatmap(task_embeddings, task_ids, output_dir):
    """
    生成任务嵌入的热力图
    
    Args:
        task_embeddings: 任务嵌入字典
        task_ids: 任务ID映射
        output_dir: 输出目录
    """
    # 收集所有嵌入和对应的标签
    embeddings = []
    labels = []
    
    # 创建ID到任务名称的映射
    id_to_name = {}
    if task_ids is not None:
        # 反转映射，使得可以通过ID查找任务名称
        for name, idx in task_ids.items():
            id_to_name[idx] = name
    
    # 排序，确保可视化顺序一致性
    sorted_ids = sorted(task_embeddings.keys())
    
    for task_id in sorted_ids:
        # 获取嵌入
        embedding = task_embeddings[task_id]
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.cpu().numpy()
        if len(embedding.shape) > 1 and embedding.shape[0] == 1:
            embedding = embedding.flatten()
            
        embeddings.append(embedding)
        
        # 获取标签
        if task_id in id_to_name:
            labels.append(id_to_name[task_id])
        else:
            labels.append(f"Task {task_id}")
    
    # 将嵌入转换为numpy数组
    embeddings = np.array(embeddings)
    
    # 创建热力图
    plt.figure(figsize=(12, 8))
    plt.imshow(embeddings, aspect='auto', cmap='viridis')
    plt.colorbar(label='Value')
    plt.title('Task Embedding Heatmap')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Task')
    plt.yticks(np.arange(len(labels)), labels)
    plt.xticks(np.arange(embeddings.shape[1]), [f'Dim {i+1}' for i in range(embeddings.shape[1])])
    
    # 保存图像
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'task_embeddings_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to: {output_path}")

def main():
    args = parse_args()
    
    # 加载任务嵌入
    task_embeddings, task_ids, name_to_embedding = load_task_embeddings(args.embeddings_dir)
    
    # 可视化嵌入
    embeddings_2d, labels = visualize_embeddings(task_embeddings, task_ids, args.method)
    
    # 绘制嵌入
    plot_embeddings(embeddings_2d, labels, args.method, args.output_dir)
    
    # 生成热力图
    visualize_heatmap(task_embeddings, task_ids, args.output_dir)
    
    print(f"Task embedding visualization completed! Results saved in {args.output_dir} directory")

if __name__ == "__main__":
    main()
