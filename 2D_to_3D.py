import os
import numpy as np
#from libtiff import TIFF
import tifffile

volume = []
Files_Path = './volume'  # 把你的所有图片放在py文件同目录下的volume文件夹
Res_path = './res'
imgList = os.listdir(Files_Path)  # 读取文件目录下的所有文件名
imgList.sort(key=lambda x: int(x.split('.')[0]))  # 按照数字顺序排列图片，图片名示例：1.tif, 2.tif, 3.tif, 4.tif, 5.tif ...
for count in range(0, len(imgList)):
    #tif = TIFF.open(Files_Path + '/' + imgList[count], mode='r')
    tif = tifffile.imread(Files_Path + '/' + imgList[count])
    tif = np.array(tif)
    if tif.shape[0] == 1:
        tif = tif.squeeze(0)
    # else:
    #     assert tif.shape[0] == 1
    volume.append(tif)
volume = np.array(volume)
print('Read success.')
tifffile.imsave(os.path.join(Res_path,'res.tif'),volume)
#tifffile.imsave(Files_Path + '.tif', volume)