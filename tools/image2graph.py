import random

import numpy as np
import torch
from scipy.spatial.distance import cdist
from torch_geometric.data import Data

from tools.preprocess import preprocessing

mask = np.load('./datasets/npy/label/Mask.npy')
target = np.load('./datasets/npy/label/Target.npy')
anticline = np.load('./datasets/npy/geology/Anticline_Buffer.npy')
godenville = np.load('./datasets/npy/geology/Godenville_Formation_Buffer.npy')
height = mask.shape[0]
width = mask.shape[1]


def build_edge():
    '''
    构造图结构邻接关系
    :return:
    '''
    node_features, indices_map = preprocessing([
        '../datasets/npy/geochemical',
        '../datasets/npy/geology'
    ])
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
    '''
    生成train_mask、val_mask、y
    :return: 
    '''
    deposit = target.reshape(-1)[mask.reshape(-1) == 1]
    valid_indices = np.where(mask.reshape(-1) == 1)[0]

    zero_indices = np.where(deposit == 0)[0]
    deposit_num = np.sum(deposit == 1)
    random_indices = np.random.choice(zero_indices, size=deposit_num, replace=False)  # 随机选择非矿点位置
    no_deposit = np.zeros_like(deposit)
    no_deposit[random_indices] = 1

    # no_deposit_coords = create_no_deposit_coords() # 二维坐标系下的负样本坐标
    # no_deposit_coords_flatten = np.ravel_multi_index(no_deposit_coords.T, mask.shape)
    # no_deposit_indices = np.where(np.isin(no_deposit_coords_flatten, valid_indices))[0]
    #
    # no_deposit = np.zeros_like(deposit)
    # no_deposit[no_deposit_indices] = 1

    train_mask = torch.tensor(deposit + no_deposit, dtype=torch.bool)

    true_indices = torch.where(train_mask == True)[0]
    val_num = (int)(0.2 * len(true_indices))
    val_indices = random.sample(true_indices.tolist(), val_num)
    val_mask = torch.zeros_like(train_mask, dtype=torch.bool)
    val_mask[val_indices] = True
    train_mask[val_indices] = False
    y = torch.tensor(deposit, dtype=torch.float).unsqueeze(1)

    return train_mask, val_mask, y


def create_no_deposit_coords(min_deposit_distance=100, max_buffer_level=8, min_sample_distance=10):
    '''
    生成负样本坐标
    :param min_deposit_distance: 负样本与已知矿点的最小距离(保证负样本远离已知正样本)
    :param max_buffer_level: 负样本的最大构造等级(保证负样本远离背斜、断层等构造因素)
    :param min_sample_distance: 负样本之间的最小距离(保证负样本尽可能分散)
    :return:
    '''
    negative_mask = np.zeros_like(target, dtype=bool)

    # 矿点坐标
    mineral_coords = np.column_stack(np.where(target == 1))

    index = 0
    for i in range(height):
        for j in range(width):
            if mask[i, j] == 1 and target[i, j] == 0:
                dist_map = np.sqrt((mineral_coords[:, 0] - i) ** 2 + (mineral_coords[:, 1] - j) ** 2)
                if np.min(dist_map) > min_deposit_distance and anticline[i, j] < max_buffer_level and godenville[
                    i, j] < max_buffer_level:
                    negative_mask[i, j] = 1
            print(f'第{index}样本完成计算')
            index += 1

    n_samples = int(np.sum(target))
    negative_indices = np.column_stack(np.where(negative_mask))
    sampled_coords = [negative_indices[random.randint(0, len(negative_indices) - 1)]]

    for _ in range(n_samples - 1):
        # 计算所有负样本和采样点的距离
        dist_matrix = cdist(negative_indices, sampled_coords)
        # 找出negative_indices中，与所有已采样点(sampled_coords)距离大于min_distance的点
        valid_indices = np.where(np.min(dist_matrix, axis=1) > min_sample_distance)[0]

        if valid_indices.size > 0:
            # 随机选择一个负样本的坐标
            new_sample = negative_indices[random.choice(valid_indices)]
            sampled_coords.append(new_sample)
        else:
            raise ValueError('无满足条件的负样本')

    np.save('./temp/sample_coords.npy', np.array(sampled_coords))

    return np.array(sampled_coords)


if __name__ == '__main__':
    node_features, edge_index = build_edge()
    train_mask, val_mask, y = build_mask()
    data = Data(x=node_features, edge_index=edge_index, y=y,
                train_mask=train_mask, val_mask=val_mask)
    print(data)
