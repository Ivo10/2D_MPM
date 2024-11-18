import numpy as np
import gdal
import os


def raster_to_npy(file_path, output_npy):
    print('栅格数据目录为：', str(file_path))
    print('npy输出目录为：', str(output_npy))
    value = gdal.Open(file_path)
    value = value.GetRasterBand(1).ReadAsArray()
    np.save(output_npy, value)
    print('Raster to Npy successful running')

if __name__ == '__main__':
    for file in os.listdir('../datasets/tif'):
        if file.endswith('tif'):
            raster_to_npy('../datasets/tif/' + file,
                          '../datasets/npy/' + file[:file.find('.')] + '.npy')