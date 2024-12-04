import datetime

import matplotlib.pyplot as plt
import numpy as np

timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S").replace("'", "")

def create_heating_image(out, mask):
    '''
    生成成矿远景区图
    :param out:模型对所有数据运行结果(一维数组)
    :param mask:从mask.npy获取的mask数组
    :return:
    '''
    result = np.full(mask.shape, np.nan)
    indices = np.argwhere(mask == 1)
    for idx, val in zip(indices, out):
        result[tuple(idx)] = val
    plt.imshow(result, cmap='jet')
    plt.colorbar()
    plt.savefig('../figs/heatmap_' + timestamp + '.png')
    plt.show()
    plt.close()