import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks import FlattenMlp
import models.pytorch_utils as ptu
import mbrl.models as models
import mbrl
import mbrl.util.math

def unreduced_loss(mlp: models.GaussianMLP, model_in, target):
    """
    计算mbrl.models.GaussianMLP的非减少损失
    
    参数:
        mlp: GaussianMLP模型
        model_in: 模型输入
        target: 目标值
        
    返回:
        非减少损失
    """
    assert not mlp.deterministic
    assert model_in.ndim == target.ndim
    if model_in.ndim == 2:  # 添加ensemble维度
        model_in = model_in.unsqueeze(0)
        target = target.unsqueeze(0)
    pred_mean, pred_logvar = mlp.forward(model_in, use_propagation=False)
    if target.shape[0] != mlp.num_members:
        target = target.repeat(mlp.num_members, 1, 1)
    nll = mbrl.util.math.gaussian_nll(pred_mean, pred_logvar, target, reduce=False).squeeze(0).mean(dim=-1)
    return nll

class FOCALDecoder(nn.Module):
    def __init__(self, 
                 obs_size,
                 action_size,
                 task_embedding_size,
                 task_embedd_is_deterministic,
                 device, 
                 num_layers, 
                 ensemble_size, 
                 hidden_size, 
                 ) -> None:
        super(FOCALDecoder, self).__init__()
        input_size = obs_size + action_size + task_embedding_size
        if task_embedd_is_deterministic:
            input_size += task_embedding_size
            
        output_dynamic_size = obs_size
        output_reward_size = 1
        self.ensemble_size = ensemble_size
        self.device = device
        
        # 使用mbrl库中的GaussianMLP实现
        self.dynamic_decoder = models.GaussianMLP(
            in_size=input_size,
            out_size=output_dynamic_size,
            device=device,
            num_layers=num_layers,
            ensemble_size=ensemble_size,
            hid_size=hidden_size,
            learn_logvar_bounds=True,
            deterministic=False
        ).requires_grad_(True)
        
        self.reward_decoder = models.GaussianMLP(
            in_size=input_size,
            out_size=output_reward_size,
            device=device,
            num_layers=num_layers,
            ensemble_size=ensemble_size,
            hid_size=hidden_size,
            learn_logvar_bounds=True,
            deterministic=False
        ).requires_grad_(True)

    def forward(self, task_embedding, state, action):
        """
        接收任务嵌入、状态和动作，输出下一状态和奖励的预测（均值和对数方差）
        
        参数:
            task_embedding: 任务嵌入/潜变量，维度为 [batch_size, z_dim]
            state: 当前状态，维度为 [batch_size, state_size]
            action: 动作，维度为 [batch_size, action_size]
            
        返回:
            mean_state: 下一状态预测的均值, [batch_size, state_size]
            logvar_state: 下一状态预测的对数方差, [batch_size, state_size]
            mean_reward: 奖励预测的均值, [batch_size, 1]
            logvar_reward: 奖励预测的对数方差, [batch_size, 1]
        """
        # 将输入连接起来
        input_tensor = torch.cat((task_embedding, state, action), dim=-1)
        
        # 添加ensemble维度并重复
        input_tensor = input_tensor.repeat(self.ensemble_size, 1, 1)
        
        # 获取动态预测和奖励预测
        mean_state, logvar_state = self.dynamic_decoder(input_tensor)
        mean_reward, logvar_reward = self.reward_decoder(input_tensor)
        
        # 去除ensemble维度
        mean_state = mean_state.squeeze(0)
        logvar_state = logvar_state.squeeze(0)
        mean_reward = mean_reward.squeeze(0)
        logvar_reward = logvar_reward.squeeze(0)
        
        return mean_state, logvar_state, mean_reward, logvar_reward

    def loss(self, task_embedding, state, action, target_reward, target_state):
        """
        计算模型损失
        
        参数:
            task_embedding: 任务嵌入/潜变量
            state: 当前状态
            action: 动作
            target_reward: 目标奖励
            target_state: 目标下一状态
            
        返回:
            损失值（状态预测损失 + 奖励预测损失）
        """
        input_tensor = torch.cat((task_embedding, state, action), dim=-1)
        input_tensor = input_tensor.repeat(self.ensemble_size, 1, 1)
        
        # 计算状态差异，而不是直接预测下一个状态
        state_target = (target_state - state).repeat(self.ensemble_size, 1, 1)
        reward_target = target_reward.repeat(self.ensemble_size, 1, 1)
        
        # 使用GaussianMLP的内置损失函数
        state_loss, _ = self.dynamic_decoder.loss(input_tensor, state_target)
        reward_loss, _ = self.reward_decoder.loss(input_tensor, reward_target)
        
        return state_loss + reward_loss

    def unreduced_loss(self, task_embedding, state, action, target_reward, target_state):
        """
        计算非归约损失（每个样本一个损失值）
        
        参数:
            task_embedding: 任务嵌入/潜变量
            state: 当前状态
            action: 动作
            target_reward: 目标奖励
            target_state: 目标下一状态
            
        返回:
            每个样本的损失值
        """
        input_tensor = torch.cat((task_embedding, state, action), dim=-1)
        input_tensor = input_tensor.repeat(self.ensemble_size, 1, 1)
        
        # 计算状态差异
        state_target = target_state - state
        
        # 使用辅助函数计算非归约损失
        state_loss = unreduced_loss(self.dynamic_decoder, input_tensor, state_target.repeat(self.ensemble_size, 1, 1))
        reward_loss = unreduced_loss(self.reward_decoder, input_tensor, target_reward.repeat(self.ensemble_size, 1, 1))
        
        return state_loss + reward_loss
