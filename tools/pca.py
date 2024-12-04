import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from evaluation.CreateHeatingImage import create_heating_image
from tools.preprocess import preprocessing


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
    data = data - np.mean(data, axis=0)

    # 2. 计算协方差矩阵
    cov_matrix = np.cov(data, rowvar=False)
    # 3. 计算特征值和特征向量
    eigvals, eigvecs = np.linalg.eig(cov_matrix)
    # 4. 选择最大特征值对应的特征向量作为第一主成分
    pc1 = eigvecs[:, np.argmax(eigvals)]
    # 5. 计算每个数据点在 PC1 上的得分
    pc1_scores = data.dot(pc1)

    scaler1 = MinMaxScaler()
    pc1_scores = np.array(pc1_scores).reshape(-1, 1)
    pc1_scores = scaler1.fit_transform(pc1_scores).flatten().tolist()

    return pc1_scores


if __name__ == '__main__':
    node_features, _ = preprocessing([
        '../datasets/npy/geochemical',
        '../datasets/npy/geology'
    ])
    # pca(data)
    pc1_scores = pc1_score(node_features[:, :-2])
    print(len(pc1_scores))
    mask = np.load('../datasets/npy/label/Mask.npy')
    create_heating_image(pc1_scores, mask)
