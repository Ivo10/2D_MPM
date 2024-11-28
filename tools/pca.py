import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from evaluation.CreateHeatingImage import create_heating_image


def preprocessing():
    npy_files = [file for file in os.listdir('../datasets/npy/geochemical/')]
    attributes = [np.load('../datasets/npy/geochemical/' + name).reshape(-1) for name in npy_files]
    mask = np.load('../datasets/npy/label/Mask.npy')

    current_index = 0
    node_features = []
    indices_map = {}
    height = 2220
    width = 1826

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
    data = np.array(node_features)
    print(data.shape)

    return data

def pca(data):
    npy_files = [file for file in os.listdir('../datasets/npy/geochemical/')]

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    pca = PCA(n_components=4)
    pca.fit(data_scaled)

    loadings = pca.components_.T
    pc1_loadings = loadings[:, 0]

    variables = [file[:-4] for file in npy_files]

    # 绘制水平柱状图
    plt.figure(figsize=(8, 12))
    plt.barh(variables, pc1_loadings, color='skyblue')
    plt.axvline(0, color='black', linewidth=0.8)
    plt.xlabel('Loadings on PC1')
    plt.title('PCA Loadings (PC1)')
    plt.show()

    print(pca.explained_variance_ratio_)

def pc1_score(data):
    # 1. 标准化数据
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(data)

    # 2. 计算协方差矩阵
    cov_matrix = np.cov(data_standardized, rowvar=False)

    # 3. 计算特征值和特征向量
    eigvals, eigvecs = np.linalg.eig(cov_matrix)

    # 4. 选择最大特征值对应的特征向量作为第一主成分
    pc1 = eigvecs[:, np.argmax(eigvals)]

    # 5. 计算每个数据点在 PC1 上的得分
    pc1_scores = data_standardized.dot(pc1)

    return pc1_scores

if __name__ == '__main__':
    data = preprocessing()
    # pca(data)
    pc1_scores = pc1_score(data)
    mask = np.load('../datasets/npy/label/Mask.npy')
    create_heating_image(pc1_scores, mask)