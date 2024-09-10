import torch
import torch.nn as nn
import torch.nn.functional as F

# # 加载模型权重
# model_weights = torch.load('/home/zhx/Project/ml-agents/ml-agents/mlagents/trainers/results/co-collection_7_17/Pour/Pour-21999900_actor_model.pth')
#
# # 查看模型权重的结构
# print(model_weights)


class SimpleVisualEncoder(nn.Module):
    def __init__(self, in_channels, dense_in_features):
        super(SimpleVisualEncoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=(8, 8), stride=(4, 4)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(16, 32, kernel_size=(4, 4), stride=(2, 2)),
            nn.LeakyReLU(negative_slope=0.01)
        )
        self.dense = nn.Sequential(
            nn.Linear(dense_in_features, 256),
            nn.LeakyReLU(negative_slope=0.01)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dense(x)
        return x


class VectorInput(nn.Module):
    def forward(self, x):
        return x


class LinearEncoder(nn.Module):
    def __init__(self):
        super(LinearEncoder, self).__init__()
        self.seq_layers = nn.Sequential(
            nn.Linear(525, 256),
            nn.SiLU(),  # Swish activation function
            nn.Linear(256, 256),
            nn.SiLU()
        )

    def forward(self, x):
        return self.seq_layers(x)


class GaussianDistribution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GaussianDistribution, self).__init__()
        self.mu = nn.Linear(in_features, out_features)

    def forward(self, x):
        mu = self.mu(x)
        # Add other components if necessary for your distribution
        return mu


class ActionModel(nn.Module):
    def __init__(self):
        super(ActionModel, self).__init__()
        self._continuous_distribution = GaussianDistribution(256, 6)

    def forward(self, x):
        return self._continuous_distribution(x)


class SimpleActor(nn.Module):
    def __init__(self):
        super(SimpleActor, self).__init__()
        self.network_body = nn.Sequential(
            SimpleVisualEncoder(4, 32),
            SimpleVisualEncoder(3, 6048),
            VectorInput()
        )
        self._body_endoder = LinearEncoder()
        self.action_model = ActionModel()

    def forward(self, x):
        x = self.network_body(x)
        x = self._body_endoder(x)
        actions = self.action_model(x)
        return actions

# 假设你的模型有一个输入层、一个隐藏层和一个输出层
# input_size = 10  # 根据你的实际输入大小设置
# hidden_size = 20  # 根据你的实际隐藏层大小设置
# output_size = 1  # 根据你的实际输出大小设置

actor_model = SimpleActor()
# 加载预训练权重
model_file_path = '/home/zhx/Project/ml-agents/ml-agents/mlagents/trainers/results/test4/Collection/Collection-128_policy_model.pth'
model_data = torch.load(model_file_path)
# 检查加载的数据类型
print(type(model_data))

# 如果是字典，检查其键
