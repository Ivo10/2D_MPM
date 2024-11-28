import torch
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, data):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(data.num_features, 16)
        self.conv2 = GCNConv(16, 64)
        self.fc = torch.nn.Linear(64, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = x.relu()
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = x.relu()
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x
