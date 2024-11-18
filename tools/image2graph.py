import random

from torch_geometric.data import Data
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import numpy as np
import torch

prefix_path = '../datasets/npy/'
npy_suffix = '.npy'
mask = np.load(prefix_path + 'Mask' + npy_suffix)
target = np.load(prefix_path + 'Target' + npy_suffix)

knn_graph = 4

def build_node_edge(height, width, degree):
    name_attr = ['Anticline_Buffer', 'Godenville_Formation_Buffer']
    attributes = [np.load(prefix_path + name + npy_suffix).reshape(-1) for name in name_attr]

    # scaler = MinMaxScaler()
    # attributes = [scaler.fit_transform(item.reshape(-1, 1)).flatten() for item in attributes]
    encoder = OneHotEncoder(sparse=False)
    attributes = [encoder.fit_transform(attribute.reshape(-1, 1)) for attribute in attributes]

    current_index = 0
    node_features = []
    indices_map = {}

    for i in range(height):
        for j in range(width):
            if mask[i, j] == 1:
                index = i * width + j
                node_feature = []
                for attribute in attributes:
                    # print(attribute[index])
                    node_feature.extend(attribute[index])
                node_features.append(node_feature)
                indices_map[index] = current_index
                current_index += 1
    print(current_index)

    node_features = torch.tensor(node_features, dtype=torch.float).contiguous()

    edge_index = []
    for i in range(height):
        for j in range(width):
            if mask[i, j] == 1:
                index = i * width + j
                if i < height - 1 and mask[i + 1, j] == 1:
                    down_index = (i + 1) * width + j
                    edge_index.append([indices_map[index], indices_map[down_index]])
                    edge_index.append([indices_map[down_index], indices_map[index]])
                if j < width - 1 and mask[i, j + 1] == 1:
                    right_index = i * width + j + 1
                    edge_index.append([indices_map[index], indices_map[right_index]])
                    edge_index.append([indices_map[right_index], indices_map[index]])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return node_features, edge_index

def build_mask():
    deposit = target.reshape(-1)[mask.reshape(-1) == 1]
    zero_indices = np.where(deposit == 0)[0]
    deposit_num = np.sum(deposit == 1)
    random_indices = np.random.choice(zero_indices, size=deposit_num, replace=False)  # 随机选择非矿点位置
    no_deposit = np.zeros_like(deposit)
    no_deposit[random_indices] = 1

    train_mask = torch.tensor(deposit + no_deposit, dtype=torch.bool)

    true_indices = torch.where(train_mask == True)[0]
    val_num = (int)(0.2 * len(true_indices))
    val_indices = random.sample(true_indices.tolist(), val_num)
    val_mask = torch.zeros_like(train_mask, dtype=torch.bool)
    val_mask[val_indices] = True
    train_mask[val_indices] = False

    train_mask = train_mask.unsqueeze(1)
    y = torch.tensor(deposit, dtype=torch.float).unsqueeze(1)

    return train_mask, val_mask, y


if __name__ == '__main__':
    node_features, edge_index = build_node_edge(2220, 1826, 3)
    train_mask, val_mask, y = build_mask()
    data = Data(x=node_features, edge_index=edge_index, y=y,
                train_mask=train_mask, val_mask=val_mask)
    print(data)