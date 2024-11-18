import torch
import torch.nn as nn

class FusionAndClassifier(nn.Module):
    def __init__(self, gcn_feature, cnn_feature):
        super(FusionAndClassifier, self).__init__()
        self.fusion_layer = nn.Linear(gcn_feature + cnn_feature, 64)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(64, 1)

    def forward(self, gcn_out, cnn_out):
        x = torch.cat((gcn_out, cnn_out), dim=1)
        x = self.fusion_layer(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.classifier(x)
        x = torch.sigmoid(x)
        return x