import torch
import numpy as np

from torch_geometric.nn import GCNConv
from tools.pca import pc1_score


class GCN(torch.nn.Module):
    def __init__(self, data):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(data.num_features, 32)
        self.conv2 = GCNConv(32, 64)
        self.fc = torch.nn.Linear(64, 1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pc1_score = torch.tensor(pc1_score(data.x[:, :-2].cpu().detach().numpy()), dtype=torch.float32, requires_grad=False)[:, np.newaxis].to(self.device)
        print(self.pc1_score.shape)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = x * self.pc1_score
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x
