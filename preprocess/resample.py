import numpy as np
import os

from scipy.ndimage import zoom


def resample(input_path, output_path, target_shape):
    '''
    对构造证据层和矿点、Mask数据加密
    :param input_path: 输入文件夹路径
    :param output_path: 输出文件夹路径
    :param target_shape: 重采样后大小
    :return:
    '''
    for file in os.listdir(input_path):
        data = np.load(os.path.join(input_path, file))
        original_shape = data.shape
        print(f"原始形状: {original_shape}")

        # 计算缩放因子
        zoom_factors = (target_shape[0] / original_shape[0], target_shape[1] / original_shape[1])

        # 最临界插值
        resampled_data = zoom(data, zoom_factors, order=0)

        np.save(os.path.join(output_path, file), resampled_data)
        print(f"重采样完成，新形状: {resampled_data.shape}")
        print(f"新文件已保存为: {file}")


if __name__ == '__main__':
    target_shape = (2220, 1826)

    resample('../datasets/npy/label/', '../datasets/npy/label/', target_shape)
    resample('../datasets/npy/geology/', '../datasets/npy/geology/', target_shape)
