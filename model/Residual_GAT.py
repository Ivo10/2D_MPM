import torch

from torch_geometric.nn import GATConv
from model.GCN_PCA import ChannelAttention


class Residual_GAT(torch.nn.Module):
    def __init__(self, data):
        super(Residual_GAT, self).__init__()
        self.conv1 = GATConv(data.num_features, 32)
        self.conv2 = GATConv(32, 64)
        self.fc = torch.nn.Linear(64, 1)
        self.res_fc = torch.nn.Linear(data.num_features, 64)
        self.channel_attention1 = ChannelAttention(32)
        self.channel_attention2 = ChannelAttention(64)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        residual = x
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.channel_attention1(x)
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.channel_attention2(x)
        residual = self.res_fc(residual)
        x = x + residual
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x
