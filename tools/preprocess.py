import datetime
import os

import numpy as np

mask = np.load('./datasets/npy/label/Mask.npy')


def preprocessing(attribute_paths):
    '''
    整理所有属性为二维ndarray格式
    :param attribute_paths:geochemical和geology数据路径
    :return:
    '''
    layer = np.load(os.path.join(attribute_paths[0], os.listdir(attribute_paths[0])[0]))
    height = layer.shape[0]
    width = layer.shape[1]
    attributes = []

    # 检查每个数据层的维度
    for attribute_path in attribute_paths:
        for file in os.listdir(attribute_path):
            path = os.path.join(attribute_path, file)
            attribute = np.load(path)
            if len(attribute.shape) != 2 or attribute.shape[0] != height or attribute.shape[1] != width:
                raise ValueError(f"文件{file}数据格式错误！请检查数据格式")
            attributes.append(attribute.reshape(-1))

    node_features = []
    indices_map = {}
    current_index = 0
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

    return node_features, indices_map


def save_csv(data, csv_path):
    TIMESTAMP = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S").replace("'", "")
    data.to_csv(os.path.join(csv_path, 'mineral_data_{}.csv'.format(TIMESTAMP)), encoding='utf-8')


if __name__ == '__main__':
    node_features, indices_map = preprocessing([
        '../datasets/npy/geochemical',
        '../datasets/npy/geology'
    ])

    # save_csv(node_features, '../datasets/csv/')
