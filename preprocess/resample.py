import numpy as np
from scipy.ndimage import zoom

# 输入和输出文件路径
input_npy = "../datasets/npy/Target.npy"
output_npy = "../datasets/npy/Target.npy"

# 加载原始 .npy 数据
data = np.load(input_npy)

# 原始数据的形状
original_shape = data.shape  # (100, 220)
print(f"原始形状: {original_shape}")

# 目标形状
target_shape = (2220, 1826)

# 计算缩放因子
zoom_factors = (target_shape[0] / original_shape[0], target_shape[1] / original_shape[1])

# 使用最近邻插值进行重采样
resampled_data = zoom(data, zoom_factors, order=0)  # order=0 表示最近邻插值

# 保存为新的 .npy 文件
np.save(output_npy, resampled_data)

print(f"重采样完成，新形状: {resampled_data.shape}")
print(f"新文件已保存为: {output_npy}")
