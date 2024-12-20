import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, data):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(data.num_features, 6)
        self.conv2 = GCNConv(6, 4)
        self.fc = torch.nn.Linear(4, 1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.pc1_score = torch.tensor(pc1_score(data.x[:, :-2].cpu().detach().numpy()),
        #                               dtype=torch.float32, requires_grad=False, device=self.device)[:, np.newaxis]
        self.channel_attention1 = ChannelAttention(16)
        self.channel_attention2 = ChannelAttention(16)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = x.relu()
        # x = self.channel_attention1(x)
        x = self.conv2(x, edge_index)
        x = x.relu()
        # x = self.channel_attention2(x)
        x = self.fc(x)
        # x = x * self.pc1_score
        x = torch.sigmoid(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction

        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)

    def forward(self, x):
        # 对每列（即每个特征维度）进行池化操作
        avg_out = torch.mean(x, dim=0, keepdim=True)
        max_out, _ = torch.max(x, dim=0, keepdim=True)

        # MLP操作：通过两个全连接层生成注意力权重
        avg_out = self.fc2(F.relu(self.fc1(avg_out)))
        max_out = self.fc2(F.relu(self.fc1(max_out)))

        # 将两个池化的结果相加得到最终的通道注意力
        out = avg_out + max_out

        # 使用sigmoid激活函数，得到每个通道的注意力权重
        out = torch.sigmoid(out)  # 这里得到的是 (1, n) 的注意力权重

        # 将注意力权重乘到原始特征矩阵上
        return out * x  # 乘法会广播到每个特征维度
