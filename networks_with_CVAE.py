import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from utils_basic import weights_init
from models.generative import CVAE, reparameterize
import os
# import clip
class CNN_simple(nn.Module):
    def __init__(self, obs_shape, stack_frames):
        super(CNN_simple, self).__init__()
        c,w,h = obs_shape
        # self.conv1 = nn.Conv2d(obs_shape[0], 32, 5, stride=1, padding=2)
        self.conv1 = nn.Conv2d(c, 32, 5, stride=1, padding=2)

        self.maxp1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        self.maxp2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
        self.maxp3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.maxp4 = nn.MaxPool2d(2, 2)

        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.conv4.weight.data.mul_(relu_gain)

        dummy_state = Variable(torch.rand(stack_frames, c, w, h))
        out = self.forward(dummy_state)
        self.outshape = out.shape
        out = out.view(stack_frames, -1)
        cnn_dim = out.size(-1)
        self.outdim = cnn_dim
        self.apply(weights_init)
        self.train()

    def forward(self, x, batch_size=1, fc=False):
        x = F.relu(self.maxp1(self.conv1(x)))
        x = F.relu(self.maxp2(self.conv2(x)))
        x = F.relu(self.maxp3(self.conv3(x)))
        x = F.relu(self.maxp4(self.conv4(x)))
        return x

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class CNN_LSTM(nn.Module):
    def __init__(self, state_size, action_size, hidden_size,stack_frames,lstm_out,lstm_layer):
        super(CNN_LSTM, self).__init__()
        self.input_shape = state_size
        self.action_size = action_size
        self.stack_frames = stack_frames
        self.CNN_Simple = CNN_simple(self.input_shape,self.stack_frames)
        self.cnn_dim = self.CNN_Simple.outdim
        self.lstm_layer=lstm_layer
        self.lstm_out = lstm_out
        # self.outdim = layer_size
        self.outdim=self.lstm_out
        self.lstm = nn.LSTM(input_size=self.cnn_dim, hidden_size=self.lstm_out, num_layers=self.lstm_layer,batch_first=True)

        self.ht = None
        self.ct = None

        # self.head_1 = nn.Linear(self.lstm_out, layer_size)
        #
        # self.ff_1 = nn.Linear(layer_size, layer_size)
    def forward(self, input):
        """

        """


        # if input.shape[1]>1:
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        input = input.reshape(-1, input.shape[-3], input.shape[-2], input.shape[-1])
        x=self.CNN_Simple(input)
        x=x.reshape(x.shape[0],-1)
        x=x.reshape(batch_size,seq_len,-1)
        h0=torch.rand(self.lstm_layer*1,batch_size,self.lstm_out).cuda()
        c0=torch.rand(self.lstm_layer*1,batch_size,self.lstm_out).cuda()

        # if self.ht == None or self.ct == None:
        #     x, (ht, ct) = self.lstm(x)
        # else:
        x, (ht,ct) = self.lstm(x,(h0,c0))
        # self.ht=ht
        # self.ct=ct
        # x = torch.relu(self.head_1(x))
        # out = torch.relu(self.ff_1(x))

        return x
    def inference(self,input,ht=None,ct=None):
        # if input.shape[1]>1:
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        input = input.reshape(-1, input.shape[-3], input.shape[-2], input.shape[-1])
        x = self.CNN_Simple(input)
        x = x.reshape(x.shape[0], -1)
        x = x.reshape(batch_size, seq_len, -1)
        if ht ==None or ct==None:
            x, (ht, ct) = self.lstm(x)

        else:
            x, (ht, ct) = self.lstm(x,(ht,ct))
        # x = torch.relu(self.head_1(x))
        # out = torch.relu(self.ff_1(x))

        return x,ht,ct
class LSTM(nn.Module):
    def __init__(self, input_dim, action_size, hidden_size,lstm_out,lstm_layer):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.action_size = action_size



        self.lstm_layer=lstm_layer
        self.lstm_out = lstm_out
        # self.outdim = layer_size
        self.outdim=self.lstm_out
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.lstm_out, num_layers=self.lstm_layer,batch_first=True)

        self.ht = None
        self.ct = None

        # self.head_1 = nn.Linear(self.lstm_out, layer_size)
        #
        # self.ff_1 = nn.Linear(layer_size, layer_size)
    def forward(self, input):
        """

        """


        # if input.shape[1]>1:
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        # input = input.reshape(-1, input.shape[-3], input.shape[-2], input.shape[-1])
        # x=self.CNN_Simple(input)
        # x=x.reshape(x.shape[0],-1)
        x=input.reshape(batch_size,seq_len,-1)

        h0=torch.rand(self.lstm_layer*1,batch_size,self.lstm_out).cuda()
        c0=torch.rand(self.lstm_layer*1,batch_size,self.lstm_out).cuda()

        # if self.ht == None or self.ct == None:
        #     x, (ht, ct) = self.lstm(x)
        # else:
        x, (ht,ct) = self.lstm(x,(h0,c0))
        # self.ht=ht
        # self.ct=ct
        # x = torch.relu(self.head_1(x))
        # out = torch.relu(self.ff_1(x))

        return x
    def inference(self,input,ht=None,ct=None):
        # if input.shape[1]>1:
        batch_size = input.shape[0]
        seq_len = input.shape[1]

        x = input.reshape(batch_size, seq_len, -1)
        if ht ==None or ct==None:
            x, (ht, ct) = self.lstm(x)

        else:
            x, (ht, ct) = self.lstm(x,(ht,ct))
        # x = torch.relu(self.head_1(x))
        # out = torch.relu(self.ff_1(x))

        return x,ht,ct

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, hidden_size=32, init_w=3e-3, log_std_min=-20, log_std_max=2):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.mu = nn.Linear(hidden_size, action_size)
        self.log_std_linear = nn.Linear(hidden_size, action_size)

    def forward(self, state):

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)

        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std
    
    def evaluate(self, state, epsilon=1e-6):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        e = dist.rsample().to(state.device)
        action = torch.tanh(e)
        log_prob = (dist.log_prob(e) - torch.log(1 - action.pow(2) + epsilon)).sum(1, keepdim=True)
        # log_prob = (dist.log_prob(e) - torch.log(1 - action.pow(2) + epsilon)).sum(2, keepdim=True)

        return action, log_prob
        
    
    def get_action(self, state):
        """
        returns the action based on a squashed gaussian policy. That means the samples are obtained according to:
        a(s,e)= tanh(mu(s)+sigma(s)+e)
        """
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        e = dist.rsample().to(state.device)
        action = torch.tanh(e)
        return action.detach().cpu()
    
    def get_det_action(self, state):
        mu, log_std = self.forward(state)
        return torch.tanh(mu).detach().cpu()


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, hidden_size=32, seed=1):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_size (int): Number of nodes in the network layers
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size+action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = torch.cat((state, action), dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    


class TaskAwareCNN_LSTM(nn.Module):
    """
    集成CVAE的任务感知CNN_LSTM网络
    在CNN提取特征后、LSTM处理之前拼接任务嵌入
    """
    def __init__(self, state_size, action_size, hidden_size, stack_frames, lstm_out, lstm_layer, 
                 z_dim=8, reward_size=1, use_task_encoding=True, tabular_encoder_entries=None):
        super(TaskAwareCNN_LSTM, self).__init__()
        self.input_shape = state_size
        self.action_size = action_size
        self.stack_frames = stack_frames
        self.CNN_Simple = CNN_simple(state_size, stack_frames)
        self.cnn_dim = self.CNN_Simple.outdim
        self.lstm_layer = lstm_layer
        self.lstm_out = lstm_out
        self.z_dim = z_dim
        self.use_task_encoding = use_task_encoding
        
        # 任务嵌入字典和任务名称映射
        self.task_embeddings_dict = None
        self.task_names_dict = None
        
        # CVAE for task embedding
        self.cvae = CVAE(
            hidden_size=hidden_size,
            num_hidden_layers=2,
            z_dim=z_dim,
            action_size=action_size,
            state_size=np.prod(state_size),
            reward_size=reward_size,
            predict_state_difference=False,  # 修改为与训练时一致
            merge_reward_next_state=False,   # 修改为与训练时一致
            output_variance='output',
            logvar_min=-5.0,                # 明确指定，与训练时一致
            logvar_max=2.0,                  # 明确指定，与训练时一致
            tabular_encoder_entries=tabular_encoder_entries  # bool -> whether to use tabular encoder
        ).to(device='cuda' if torch.cuda.is_available() else 'cpu')
        
        if tabular_encoder_entries is not None:
            print(f"Using tabular encoder, entries: {tabular_encoder_entries}")
            self.cvae.tabular_encoder = True # bool -> encoder uses tabular encoder
        
        # expand the dimension of lstm input if using task encoding
        lstm_input_dim = self.cnn_dim
        if use_task_encoding:
            lstm_input_dim = self.cnn_dim + z_dim
        
        self.lstm = nn.LSTM(input_size=lstm_input_dim, hidden_size=lstm_out, num_layers=lstm_layer, batch_first=True)
        self.ht = None
        self.ct = None
        self.outdim = lstm_out

        self.num_tasks = tabular_encoder_entries if tabular_encoder_entries is not None else 8
        
        # 默认设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, input, task_embeddings=None):
        """
        前向传播，在CNN提取特征后、LSTM处理之前拼接任务嵌入
        
        Args:
            input: 输入状态序列 [batch_size, seq_len, c, h, w]
            task_embeddings: 任务嵌入向量 [batch_size, z_dim]，必须提供
        """
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        input_reshaped = input.reshape(-1, input.shape[-3], input.shape[-2], input.shape[-1])
        cnn_features = self.CNN_Simple(input_reshaped)
        cnn_features = cnn_features.reshape(cnn_features.shape[0], -1)
        cnn_features = cnn_features.reshape(batch_size, seq_len, -1)
        
        if self.use_task_encoding:
            if task_embeddings is None:
                raise ValueError("Using task embedding mode, task_embeddings must be provided")
            task_encoding = task_embeddings
            
            if len(task_encoding.shape) == 1:
                task_encoding = task_encoding.unsqueeze(0)
            
            if task_encoding.shape[0] != batch_size:
                task_encoding = task_encoding.expand(batch_size, -1)
            
            task_encoding = task_encoding.unsqueeze(1).expand(batch_size, seq_len, self.z_dim)
            lstm_input = torch.cat([cnn_features, task_encoding], dim=-1)
        else:
            lstm_input = cnn_features
            
        h0 = torch.rand(self.lstm_layer, batch_size, self.lstm_out).to(lstm_input.device)
        c0 = torch.rand(self.lstm_layer, batch_size, self.lstm_out).to(lstm_input.device)
        lstm_output, (ht,ct) = self.lstm(lstm_input, (h0, c0))
        
        return lstm_output
    
    def _normalize_image_for_cvae(self, image_tensor):
        """
        data normalization for CVAE
        """
        if image_tensor.max() <= 1.0 and image_tensor.min() >= 0.0:
            return image_tensor
        
        normalized = image_tensor.clone()          
        if normalized.max() > 1.0:
            normalized = normalized / 255.0
            
        return normalized
    
    def load_task_embeddings(self, embeddings_dir):
        """
        加载预生成的任务嵌入表
        
        Args:
            embeddings_dir: 任务嵌入目录
            
        Returns:
            bool: 是否成功加载
        """
        if not os.path.exists(embeddings_dir):
            print(f"任务嵌入目录不存在: {embeddings_dir}")
            return False
        
        try:
            # 直接加载模型，不使用 add_safe_globals
                        
            # 加载任务嵌入表
            embeddings_path = os.path.join(embeddings_dir, 'task_embeddings.pt')
            if os.path.exists(embeddings_path):
                try:
                    self.task_embeddings_dict = torch.load(embeddings_path, map_location=self.device)
                except Exception as load_err:
                    print(f"加载任务嵌入文件时出错: {load_err}")
                    return False
                print(f"成功加载任务嵌入表，包含 {len(self.task_embeddings_dict)} 个任务")
            else:
                print(f"任务嵌入文件不存在: {embeddings_path}")
                return False
            
            # 加载任务名称映射
            task_ids_path = os.path.join(embeddings_dir, 'task_ids.json')
            if os.path.exists(task_ids_path):
                with open(task_ids_path, 'r') as f:
                    import json
                    self.task_names_dict = json.load(f)
                print(f"成功加载任务名称映射，包含 {len(self.task_names_dict)} 个任务")
            else:
                print(f"任务名称映射文件不存在: {task_ids_path}")
                
            # 将NumPy数组转换为PyTorch张量
            for task_id, embedding in self.task_embeddings_dict.items():
                if isinstance(embedding, np.ndarray):
                    self.task_embeddings_dict[task_id] = torch.tensor(embedding, device=self.device)
            
            return True
        except Exception as e:
            print(f"加载任务嵌入表失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_cvae_model(self, model_path):
        """
        加载预训练的CVAE模型
        
        Args:
            model_path: CVAE模型路径
            
        Returns:
            bool: 是否成功加载
        """
        if not os.path.exists(model_path):
            print(f"CVAE模型文件不存在: {model_path}")
            return False
        
        try:
            # 直接加载模型，不使用 add_safe_globals
                        
            # 加载模型
            try:
                # 首先尝试简单加载
                checkpoint = torch.load(model_path, map_location=self.device)
            except Exception as load_err:
                print(f"简单加载失败，错误: {load_err}")
                # 尝试其他加载方式
                checkpoint = torch.load(model_path, map_location=self.device, pickle_module=None)
            
            # 提取模型状态字典
            if 'model_state_dict' in checkpoint:
                model_state_dict = checkpoint['model_state_dict']
            else:
                model_state_dict = checkpoint
                
            # 尝试从checkpoint中读取配置信息，并更新模型参数
            if 'config' in checkpoint:
                config = checkpoint['config']
                print("从checkpoint读取到配置信息：")
                for key in ['predict_state_difference', 'merge_reward_next_state', 'logvar_min', 'logvar_max']:
                    if key in config:
                        print(f"  {key}: {config[key]}")
                        # 更新CVAE的属性
                        if hasattr(self.cvae, key):
                            setattr(self.cvae, key, config[key])
            
            # 加载模型状态
            self.cvae.load_state_dict(model_state_dict, strict=False)
            
            # 确保是表格编码器
            if hasattr(self.cvae, 'tabular_encoder') and not self.cvae.tabular_encoder:
                self.cvae.tabular_encoder = True
                print("已将CVAE的tabular_encoder设置为True")
            
            print(f"成功加载CVAE模型: {model_path}")
            return True
        except Exception as e:
            print(f"加载CVAE模型失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_task_embedding_by_id(self, task_id):
        """
        根据任务ID获取任务嵌入向量
        
        Args:
            task_id: 任务ID
            
        Returns:
            torch.Tensor: 任务嵌入向量，如果找不到则返回None
        """
        if self.task_embeddings_dict is None:
            print("错误: 任务嵌入表未加载，请先调用load_task_embeddings方法")
            return None
        
        # 确保任务ID是整数
        task_id = int(task_id)
        
        if task_id in self.task_embeddings_dict:
            embedding = self.task_embeddings_dict[task_id]
            if not isinstance(embedding, torch.Tensor):
                embedding = torch.tensor(embedding, device=self.device)
            return embedding
        else:
            print(f"警告: 任务ID {task_id} 在任务嵌入表中不存在")
            return None
    
    def get_task_embedding_by_name(self, task_name):
        """
        根据任务名称获取任务嵌入向量
        
        Args:
            task_name: 任务名称
            
        Returns:
            torch.Tensor: 任务嵌入向量，如果找不到则返回None
        """
        if self.task_names_dict is None:
            print("错误: 任务名称映射未加载，请先调用load_task_embeddings方法")
            return None
        
        if self.task_embeddings_dict is None:
            print("错误: 任务嵌入表未加载，请先调用load_task_embeddings方法")
            return None
        
        # 查找任务ID
        task_id = None
        for name, id_val in self.task_names_dict.items():
            if name == task_name:
                task_id = int(id_val)
                break
        
        if task_id is None:
            print(f"警告: 任务名称 '{task_name}' 在任务名称映射中不存在")
            return None
        
        # 获取任务嵌入
        return self.get_task_embedding_by_id(task_id)
    
    def get_task_embedding_from_cvae(self, task_id):
        """
        从CVAE中获取任务嵌入向量
        
        Args:
            task_id: 任务ID
            
        Returns:
            torch.Tensor: 任务嵌入向量，如果找不到则返回None
        """
        if not hasattr(self.cvae, 'tabular_encoder') or not self.cvae.tabular_encoder:
            print("错误: CVAE不是表格编码器，无法根据任务ID获取嵌入")
            return None
        
        with torch.no_grad():
            try:
                mean, logvar, z_sample = self.cvae.forward_encoder(None, None, None, None, task_id)
                return z_sample
            except Exception as e:
                print(f"从CVAE获取任务ID {task_id} 的嵌入时出错: {e}")
                return None
            
    def inference(self, input, ht=None, ct=None, task_id=None, task_name=None, inference_mode='direct'):
        """
        支持旧的接口方式调用，同时保留任务嵌入功能
        
        Args:
            input: 输入状态序列 [batch_size, seq_len, c, h, w]
            ht: LSTM的隐藏状态
            ct: LSTM的单元状态
            task_id: 任务ID，直接模式使用
            task_name: 任务名称，直接模式使用
            inference_mode: 推断模式，'direct'或'infer'
        
        Returns:
            tuple: (x, ht, ct) - 传统CNN_LSTM风格的返回
            或者 tuple: (x, ht, ct, best_task_id, best_embedding) - 任务嵌入的详细返回
        """
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        device = input.device
        
        # 根据不同模式获取任务嵌入
        if inference_mode == 'direct':
            # 直接模式：使用预先生成的任务嵌入
            # print(f"使用直接模式获取任务嵌入，任务ID: {task_id}，任务名称: {task_name}")
            
            # 检查是否提供了任务ID或任务名称
            if task_id is not None:
                best_task_id = task_id
                best_embedding = self.get_task_embedding_by_id(task_id)
                if best_embedding is None:
                    print(f"警告: 任务ID {task_id} 在任务嵌入表中不存在，使用随机嵌入")
                    best_task_id = -1
                    best_embedding = torch.randn(1, self.z_dim, device=device)
                # else:
                    # print(f"成功获取任务ID {task_id} 的嵌入")
            elif task_name is not None:
                best_embedding = self.get_task_embedding_by_name(task_name)
                if best_embedding is None:
                    print(f"警告: 任务名称 '{task_name}' 在任务名称映射中不存在，使用随机嵌入")
                    best_task_id = -1
                    best_embedding = torch.randn(1, self.z_dim, device=device)
                else:
                    # 查找任务ID
                    best_task_id = -1
                    for name, id_val in self.task_names_dict.items():
                        if name == task_name:
                            best_task_id = int(id_val)
                            break
                    # print(f"成功获取任务名称 '{task_name}' (ID: {best_task_id}) 的嵌入")
            else:
                print("错误: 直接模式下必须提供任务ID或任务名称")
                best_task_id = -1
                best_embedding = torch.randn(1, self.z_dim, device=device)
        elif inference_mode == 'infer':
            # 推断模式暂不支持旧接口
            print("警告: 推断模式在旧接口中不支持，使用随机嵌入")
            best_task_id = -1
            best_embedding = torch.randn(1, self.z_dim, device=device)
        else:
            # 不支持的模式
            print(f"错误: 不支持的推断模式 '{inference_mode}'，必须是 'direct' 或 'infer'")
            best_task_id = -1
            best_embedding = torch.randn(1, self.z_dim, device=device)
        
        # 处理输入状态和LSTM
        input_reshaped = input.reshape(-1, input.shape[-3], input.shape[-2], input.shape[-1])
        cnn_features = self.CNN_Simple(input_reshaped)
        cnn_features = cnn_features.reshape(cnn_features.shape[0], -1)
        cnn_features = cnn_features.reshape(batch_size, seq_len, -1)
        
        # 拼接任务嵌入
        if self.use_task_encoding:
            if best_embedding.shape[0] == 1:
                task_encoding = best_embedding.expand(batch_size, -1)
            else:
                task_encoding = best_embedding
                
            task_encoding = task_encoding.unsqueeze(1).expand(batch_size, seq_len, self.z_dim)
            lstm_input = torch.cat([cnn_features, task_encoding], dim=-1)
        else:
            lstm_input = cnn_features
        
        # 使用LSTM处理序列，与CNN_LSTM类的inference方法兼容
        if ht is None or ct is None:
            lstm_output, (new_ht, new_ct) = self.lstm(lstm_input)
        else:
            lstm_output, (new_ht, new_ct) = self.lstm(lstm_input, (ht, ct))
            
        # 返回与旧接口兼容的输出
        return lstm_output, new_ht, new_ct

    def inference_with_task_embedding(self, raw_states, actions=None, rewards=None, raw_next_states=None, 
                task_id=None, task_name=None, inference_mode='direct'):
        """
        从原始状态轨迹推断任务表示并处理输入状态
        支持两种模式：
        1. 'direct'模式：直接使用提供的任务ID或任务名称获取预生成的任务嵌入
        2. 'infer'模式：通过CVAE推断最佳任务ID和任务嵌入
        
        Args:
            raw_states: 输入状态序列 [batch_size, seq_len, c, h, w]
            actions: 动作序列 [batch_size, seq_len, action_dim]，推断模式需要
            rewards: 奖励序列 [batch_size, seq_len]，推断模式需要
            raw_next_states: 下一状态序列 [batch_size, seq_len, c, h, w]，推断模式需要
            task_id: 任务ID，直接模式使用
            task_name: 任务名称，直接模式使用
            inference_mode: 推断模式，'direct'或'infer'
        
        Returns:
            tuple: (x, ht, ct, best_task_id, best_embedding)
            - x: LSTM的输出特征
            - ht, ct: LSTM的隐藏状态
            - best_task_id: 推断出的最佳任务ID
            - best_embedding: 对应的任务嵌入向量
        """
        batch_size = raw_states.shape[0]
        seq_len = raw_states.shape[1]
        device = raw_states.device
        
        # 根据不同模式获取任务嵌入
        if inference_mode == 'direct':
            # 直接模式：使用预先生成的任务嵌入
            print("Using direct mode to get task embedding...")
            
            # 检查是否提供了任务ID或任务名称
            if task_id is not None:
                best_task_id = task_id
                best_embedding = self.get_task_embedding_by_id(task_id)
                if best_embedding is None:
                    print(f"警告: 任务ID {task_id} 在任务嵌入表中不存在，使用随机嵌入")
                    best_task_id = -1
                    best_embedding = torch.randn(1, self.z_dim, device=device)
                else:
                    print(f"成功获取任务ID {task_id} 的嵌入")
            elif task_name is not None:
                best_embedding = self.get_task_embedding_by_name(task_name)
                if best_embedding is None:
                    print(f"警告: 任务名称 '{task_name}' 在任务名称映射中不存在，使用随机嵌入")
                    best_task_id = -1
                    best_embedding = torch.randn(1, self.z_dim, device=device)
                else:
                    # 查找任务ID
                    best_task_id = -1
                    for name, id_val in self.task_names_dict.items():
                        if name == task_name:
                            best_task_id = int(id_val)
                            break
                    print(f"成功获取任务名称 '{task_name}' (ID: {best_task_id}) 的嵌入")
            else:
                print("错误: 直接模式下必须提供任务ID或任务名称")
                best_task_id = -1
                best_embedding = torch.randn(1, self.z_dim, device=device)
        
        elif inference_mode == 'infer':
            # 推断模式：使用CVAE推断最佳任务嵌入
            print("Executing task embedding inference (for evaluation phase)...")
            
            # 检查必要参数
            if actions is None or rewards is None or raw_next_states is None:
                print("错误: 推断模式下必须提供actions, rewards和raw_next_states")
                best_task_id = -1
                best_embedding = torch.randn(1, self.z_dim, device=device)
            else:
                # 使用完整的轨迹数据进行任务推断，不进行采样
                print(f"Using full length of {seq_len} trajectory for task inference")
                
                # 直接使用原始轨迹数据
                # flat_states = raw_states.reshape(-1, *raw_states.shape[2:])
                # flat_next_states = raw_next_states.reshape(-1, *raw_next_states.shape[2:])
                # flat_actions = actions.reshape(-1, actions.shape[-1])
                # flat_rewards = rewards.reshape(-1, 1)

                flat_states = raw_states.reshape(-1, *raw_states.shape[2:])
                flat_next_states = raw_next_states.reshape(-1, *raw_next_states.shape[2:])
                flat_actions = actions.reshape(-1, actions.shape[-1])
                flat_rewards = rewards.reshape(-1, 1)

                # print("rewards:", flat_rewards)
                
                flat_states_normalized = self._normalize_image_for_cvae(flat_states.clone())
                flat_next_states_normalized = self._normalize_image_for_cvae(flat_next_states.clone())
                
                # flat_actions_np = np.array(flat_actions.cpu())
                # min_val = np.array([-30, -100]).astype(np.float32)
                # max_val = np.array([30, 100]).astype(np.float32)
                # normalized_actions = ((flat_actions_np - min_val) / (max_val - min_val)).astype(np.float32)
                # normalized_actions = (2 * normalized_actions - 1).astype(np.float32)
                # normalized_flat_actions = torch.from_numpy(normalized_actions).to(device)
                normalized_flat_actions = flat_actions.clone().to(device)
                # print("actions:", flat_actions_np)
                
                obs = flat_states_normalized.reshape(flat_states.shape[0], -1)
                next_obs = flat_next_states_normalized.reshape(flat_next_states.shape[0], -1)
                
                if not hasattr(self.cvae, 'tabular_encoder') or not self.cvae.tabular_encoder:
                    print("错误: CVAE没有表格编码器，使用随机任务嵌入")
                    best_task_id = -1
                    best_embedding = torch.randn(1, self.z_dim, device=device)
                    
                elif not hasattr(self.cvae, 'encoder') or not isinstance(self.cvae.encoder, nn.Parameter):
                    print("错误: CVAE没有表格形式的encoder参数，使用随机任务嵌入")
                    best_task_id = -1
                    best_embedding = torch.randn(1, self.z_dim, device=device)
                else:
                    task_losses = {}
                    best_task_id = -1
                    best_loss = float('inf')
                    best_z = None
                    
                    print(f"Evaluating all {self.num_tasks} tasks to find the best match...")
                    
                    with torch.no_grad():
                        for task_id in range(self.num_tasks):
                            try:
                                mean, logvar, z_sample = self.cvae.forward_encoder(None, None, None, None, task_id)
                                
                                if z_sample.size(0) == 1 and obs.size(0) > 1:
                                    z_sample = z_sample.expand(obs.size(0), -1)
                                
                                # 修改这里：处理forward_decoder返回4个值的情况
                                decoder_output = self.cvae.forward_decoder(obs, normalized_flat_actions, z=z_sample)
                                
                                # forward_decoder返回的顺序是 mean_s_, mean_r, logvar_s_, logvar_r
                                next_obs_pred, logvar_s, reward_pred, logvar_r = decoder_output
                                
                                # # 打印调试信息
                                # print(f"任务 {task_id} 方差统计:")
                                # print(f"状态方差范围: {logvar_s.min().item():.4f} 到 {logvar_s.max().item():.4f}, 均值: {logvar_s.mean().item():.4f}")
                                # print(f"奖励方差范围: {logvar_r.min().item():.4f} 到 {logvar_r.max().item():.4f}, 均值: {logvar_r.mean().item():.4f}")
                                
                                # # 计算重构误差（不使用方差加权）
                                # raw_state_error = ((next_obs_pred - next_obs)**2).mean().item()
                                # raw_reward_error = ((reward_pred - flat_rewards)**2).mean().item()
                                # print(f"原始重构误差: 状态={raw_state_error:.6f}, 奖励={raw_reward_error:.6f}")
                                # print("output:", next_obs_pred, logvar_s, reward_pred, logvar_r)
                                    
                                # 计算损失
                                state_loss, reward_loss, unscaled_obs_loss, unscaled_rew_loss, ref_obs_loss, ref_rew_loss = self.cvae.losses(obs, normalized_flat_actions, flat_rewards, next_obs, z_sample)
                                
                                total_loss = state_loss + reward_loss
                                
                                # 打印损失组成
                                print(f"损失组成: 状态损失={state_loss.item():.6f}, 奖励损失={reward_loss.item():.6f}, 总损失={total_loss.item():.6f}")
                                print("----------")
                                
                                task_losses[task_id] = {
                                    'state_loss': state_loss.item(),
                                    'reward_loss': reward_loss.item(),
                                    'total_loss': total_loss.item(),
                                    'z_sample': z_sample[0].clone() if z_sample.size(0) > 0 else None  # 仅保存第一个样本的编码
                                }
                                
                                if total_loss < best_loss:
                                    best_loss = total_loss.item()
                                    best_task_id = task_id
                                    best_z = z_sample[0].clone() if z_sample.size(0) > 0 else z_sample.clone()
                            
                            except Exception as e:
                                print(f"评估任务ID {task_id} 时出错: {e}")
                                continue
                    
                    # 获取最佳任务嵌入
                    if best_task_id >= 0:
                        print(f"推断出最佳任务ID: {best_task_id}, 重构损失: {best_loss:.6f}")
                        
                        sorted_tasks = sorted(task_losses.items(), key=lambda x: x[1]['total_loss'])
                        for i, (task_id, loss_info) in enumerate(sorted_tasks[:5]):  # 只显示前5名
                            print(f"  #{i+1}: 任务ID {task_id}, 总损失: {loss_info['total_loss']:.6f}, "
                                  f"状态损失: {loss_info['state_loss']:.6f}, 奖励损失: {loss_info['reward_loss']:.6f}")
                            
                        best_embedding = best_z.unsqueeze(0) if best_z.dim() == 1 else best_z
                    else:
                        # 未找到有效编码，使用随机编码作为后备
                        print("警告: 未能找到有效的任务编码，使用随机编码")
                        best_task_id = -1
                        best_embedding = torch.randn(1, self.z_dim, device=device)
        
        else:
            # 不支持的模式
            print(f"错误: 不支持的推断模式 '{inference_mode}'，必须是 'direct' 或 'infer'")
            best_task_id = -1
            best_embedding = torch.randn(1, self.z_dim, device=device)
        
        # 处理输入状态和LSTM
        input_reshaped = raw_states.reshape(-1, raw_states.shape[-3], raw_states.shape[-2], raw_states.shape[-1])
        cnn_features = self.CNN_Simple(input_reshaped)
        cnn_features = cnn_features.reshape(cnn_features.shape[0], -1)
        cnn_features = cnn_features.reshape(batch_size, seq_len, -1)
        
        # 拼接任务嵌入
        if self.use_task_encoding:
            if best_embedding.shape[0] == 1:
                task_encoding = best_embedding.expand(batch_size, -1)
            else:
                task_encoding = best_embedding
                
            task_encoding = task_encoding.unsqueeze(1).expand(batch_size, seq_len, self.z_dim)
            lstm_input = torch.cat([cnn_features, task_encoding], dim=-1)
        else:
            lstm_input = cnn_features
            
        # 初始化LSTM隐藏状态
        h0 = torch.rand(self.lstm_layer, batch_size, self.lstm_out).to(device)
        c0 = torch.rand(self.lstm_layer, batch_size, self.lstm_out).to(device)
        
        # 处理LSTM
        lstm_output, (ht, ct) = self.lstm(lstm_input, (h0, c0))
        
        # 返回全部所需信息，包括最佳任务的损失值（用于任务稳定性判断）
        best_loss_val = best_loss if 'best_loss' in locals() and isinstance(best_loss, float) else float('inf')
        return lstm_output, ht, ct, best_task_id, best_embedding, best_loss_val

    def inference_with_task_embedding_single_step(self, state, action, reward, next_state, task_id):
        """
        只对单步数据和指定task_id算loss和embedding，返回loss和embedding。
        state: [1, C, H, W]
        action: [1, action_dim]
        reward: [1, 1]
        next_state: [1, C, H, W]
        task_id: int
        """
        device = state.device
        obs = self._normalize_image_for_cvae(state).reshape(1, -1)
        next_obs = self._normalize_image_for_cvae(next_state).reshape(1, -1)
        normalized_action = action.clone().to(device)
        mean, logvar, z_sample = self.cvae.forward_encoder(None, None, None, None, task_id)
        decoder_output = self.cvae.forward_decoder(obs, normalized_action, z=z_sample)
        next_obs_pred, logvar_s, reward_pred, logvar_r = decoder_output
        # print("next_obs:", next_obs_pred)
        # print("logvar_s:",logvar_s)
        # print("reward:", reward_pred)
        # print("logvar_r:", logvar_r)
        state_loss, reward_loss, *_ = self.cvae.losses(obs, normalized_action, reward, next_obs, z_sample)
        total_loss = state_loss + reward_loss
        embedding = z_sample[0].clone() if z_sample.size(0) > 0 else z_sample.clone()
        return total_loss.item(), embedding

    def inference_with_given_embedding(self, input, ht=None, ct=None, embedding=None):
        """
        LSTM inference using a provided embedding tensor (shape: [1, z_dim] or [batch, z_dim]).
        This bypasses task_id/task_name logic and directly uses the given embedding.
        """
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        device = input.device
        input_reshaped = input.reshape(-1, input.shape[-3], input.shape[-2], input.shape[-1])
        cnn_features = self.CNN_Simple(input_reshaped)
        cnn_features = cnn_features.reshape(cnn_features.shape[0], -1)
        cnn_features = cnn_features.reshape(batch_size, seq_len, -1)
        # print("CNN features shape:", cnn_features.shape)
        if self.use_task_encoding:
            if embedding is None:
                raise ValueError("Embedding must be provided for inference_with_given_embedding.")
            if embedding.shape[0] != batch_size:
                embedding = embedding.expand(batch_size, -1)
            task_encoding = embedding.unsqueeze(1).expand(batch_size, seq_len, self.z_dim)
            lstm_input = torch.cat([cnn_features, task_encoding], dim=-1)
            # print("LSTM input shape with task encoding:", lstm_input.shape)
        else:
            lstm_input = cnn_features
        if ht is None or ct is None:
            lstm_output, (new_ht, new_ct) = self.lstm(lstm_input)
        else:
            lstm_output, (new_ht, new_ct) = self.lstm(lstm_input, (ht, ct))
        return lstm_output, new_ht, new_ct

    def inference_with_task_embedding_batch(self, states, actions, rewards, next_states, task_id):
        """
        对一段轨迹（N步）和指定task_id算平均loss和embedding，返回平均loss和embedding。
        states: [N, C, H, W]
        actions: [N, action_dim]
        rewards: [N, 1]
        next_states: [N, C, H, W]
        task_id: int
        """
        device = states.device
        N = states.shape[0]
        obs = self._normalize_image_for_cvae(states).reshape(N, -1)
        next_obs = self._normalize_image_for_cvae(next_states).reshape(N, -1)
        normalized_actions = actions.clone().to(device)
        mean, logvar, z_sample = self.cvae.forward_encoder(None, None, None, None, task_id)
        if z_sample.size(0) == 1 and N > 1:
            z_sample = z_sample.expand(N, -1)
        decoder_output = self.cvae.forward_decoder(obs, normalized_actions, z=z_sample)
        next_obs_pred, logvar_s, reward_pred, logvar_r = decoder_output
        # print("next_obs:", next_obs_pred)
        # print("logvar_s:",logvar_s)
        # print("reward:", reward_pred)
        # print("logvar_r:", logvar_r)
        state_loss, reward_loss, *_ = self.cvae.losses(obs, normalized_actions, rewards, next_obs, z_sample)
        total_loss = state_loss + reward_loss
        embedding = z_sample[0].clone() if z_sample.size(0) > 0 else z_sample.clone()
        return total_loss.item(), embedding
