import torch
import torch.nn as nn
import numpy as np

class FlattenMlp(nn.Module):
    """
    扁平化输入并通过MLP处理
    """
    def __init__(
            self,
            input_size,
            output_size,
            hidden_sizes,
            hidden_activation=nn.ReLU(),
            output_activation=None,
    ):
        super(FlattenMlp, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        # 构建网络层
        self.layers = []
        in_size = input_size
        
        # 隐藏层
        for next_size in hidden_sizes:
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            self.layers.append(fc)
            self.layers.append(hidden_activation)
            
        # 输出层
        fc = nn.Linear(in_size, output_size)
        self.layers.append(fc)
        if output_activation is not None:
            self.layers.append(output_activation)
            
        # 创建Sequential网络
        self.model = nn.Sequential(*self.layers)

    def forward(self, *inputs):
        # 扁平化并连接所有输入
        flat_inputs = torch.cat([x.view(x.size(0), -1) for x in inputs], dim=1)
        return self.model(flat_inputs)