import random

from torch_geometric.data import Data
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import os

mask = np.load('../datasets/npy/label/Mask.npy')
target = np.load('../datasets/npy/label/Target.npy')


def build_node_edge(degree, attribute_path):
    current_index = 0
    node_features = []
    indices_map = {}
    layer = np.load(os.path.join(attribute_path, os.listdir(attribute_path)[0]))
    height = layer.shape[0]
    width = layer.shape[1]
    attributes = []

    # 检查每个证据层维度和大小
    for file in os.listdir(attribute_path):
        path = os.path.join(attribute_path, file)
        attribute = np.load(path)
        if len(attribute.shape) != 2 or attribute.shape[0] != height or attribute.shape[1] != width:
            raise ValueError(f"文件{file}数据格式错误！请检查数据格式")
        attributes.append(attribute.reshape(-1))

    for i in range(height):
        for j in range(width):
            if mask[i, j] == 1:
                index = i * width + j
                node_feature = []
                for attribute in attributes:
                    node_feature.append(attribute[index])
                node_features.append(node_feature)
                indices_map[index] = current_index
                current_index += 1

    node_features = np.array(node_features)
    scaler = MinMaxScaler()
    node_features = scaler.fit_transform(node_features)
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
    print(torch.sum(train_mask))
    print(torch.sum(val_mask))
    print(torch.sum(y))

    return train_mask, val_mask, y


if __name__ == '__main__':
    node_features, edge_index = build_node_edge(4, '../datasets/npy/geochemical/')
    train_mask, val_mask, y = build_mask()
    data = Data(x=node_features, edge_index=edge_index, y=y,
                train_mask=train_mask, val_mask=val_mask)
    print(data)
