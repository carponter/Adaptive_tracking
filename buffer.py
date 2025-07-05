import numpy as np
import random
import torch
from collections import deque, namedtuple

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, device,lstm_seq_len,config):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.device = device
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        # 修改Experience类，添加task_embedding字段用于存储任务嵌入向量
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "task_embedding"])
        self.batch_id=[]
        self.st_id =[]
        self.lstm_seq_len = lstm_seq_len
        self.input_type = config.input_type


    def add(self, state, action, reward, next_state, done, task_embedding=None):
        """Add a new experience to memory.
        
        Args:
            state: 状态
            action: 动作
            reward: 奖励
            next_state: 下一状态
            done: 是否结束
            task_embedding: 任务嵌入向量，默认为None
        """
        e = self.experience(
            state,
            action,
            reward,
            next_state,
            done,
            task_embedding)
        self.memory.append(e)
    
    def sample(self):
        """从缓冲区中采样一批数据，无需指定任务ID"""
        # for lstm,seq frame input
        if ('cnn'in self.input_type.lower() and 'lstm' in self.input_type.lower()) or 'mlp' in self.input_type.lower() or 'clip' in self.input_type.lower():
            states_test = []
            actions_test = []
            rewards_test = []
            next_states_test = []
            done_test = []
            self.batch_id = []
            self.st_id = []
            for i in range(self.batch_size):
                self.st_id.append(random.randint(0, 349- self.lstm_seq_len))
                self.batch_id.append(random.randint(0, int(len(self.memory)/349))-1)

            for i in range(self.batch_size):
                experiences_test = []
                for j in range(0, self.lstm_seq_len):
                    experiences_test.append(self.memory[self.batch_id[i]*349+self.st_id[i]])
                    self.st_id[i]+=1


                # states_test.append(np.concatenate([e.state for e in experiences_test if e is not None]))
                states_test.append([e.state for e in experiences_test if e is not None])
                actions_test.append([e.action for e in experiences_test if e is not None])
                rewards_test.append([e.reward for e in experiences_test if e is not None])
                next_states_test.append([e.next_state for e in experiences_test if e is not None])
                done_test.append([e.done for e in experiences_test if e is not None])
                
                # 提取任务嵌入信息，如果存在的话
                if hasattr(experiences_test[0], 'task_embedding') and experiences_test[0].task_embedding is not None:
                    # 假设同一序列内任务嵌入相同，只取第一个
                    task_emb = experiences_test[0].task_embedding
                    if not isinstance(task_emb, torch.Tensor):
                        task_emb = torch.tensor(task_emb, device=self.device)
                else:
                    # 如果没有任务嵌入，创建一个值为None的占位符
                    task_emb = None

            states=torch.stack([torch.stack(s) for s in states_test])
            actions=torch.stack([torch.stack(a) for a in actions_test])
            rewards=torch.stack([torch.stack(r) for r in rewards_test])
            next_states=torch.stack([torch.stack(n) for n in next_states_test])
            dones=torch.stack([torch.stack(d) for d in done_test])
            
            # 处理任务嵌入
            task_embeddings = []
            for i in range(self.batch_size):
                if hasattr(self.memory[self.batch_id[i]*349+self.st_id[i]-self.lstm_seq_len], 'task_embedding'):
                    emb = self.memory[self.batch_id[i]*349+self.st_id[i]-self.lstm_seq_len].task_embedding
                    if emb is not None:
                        if not isinstance(emb, torch.Tensor):
                            emb = torch.tensor(emb, device=self.device)
                        task_embeddings.append(emb)
                    else:
                        task_embeddings.append(None)
                else:
                    task_embeddings.append(None)
            
            # 检查是否所有的任务嵌入都是None
            if all(emb is None for emb in task_embeddings):
                # 如果都是None，则不返回任务嵌入信息
                return (states, actions, rewards, next_states, dones)
            else:
                # 否则，在结果元组中添加任务嵌入信息
                # 对于None的任务嵌入，使用零向量替代
                valid_embs = [emb for emb in task_embeddings if emb is not None]
                if valid_embs:
                    emb_dim = valid_embs[0].shape[-1]
                    batch_task_embeddings = torch.zeros((self.batch_size, emb_dim), device=self.device)
                    for i, emb in enumerate(task_embeddings):
                        if emb is not None:
                            batch_task_embeddings[i] = emb
                    return (states, actions, rewards, next_states, dones, batch_task_embeddings)
                else:
                    return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)