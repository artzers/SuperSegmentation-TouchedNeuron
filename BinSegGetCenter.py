import os, tifffile
import numpy as np
import SimpleITK as sitk

image = tifffile.imread('./test/orig15.tif_clstm.tif')
# hist = np.histogram(image,bins=256)
# avaList = []
# for i in range(1,256):
#     if hist[0][i] > 10:
#         avaList.append(i)
thre =128
posList = []

curImage = image * (image > thre)
cca = sitk.ConnectedComponentImageFilter()
cca.SetFullyConnected(True)
_input = sitk.GetImageFromArray(curImage.astype(np.uint8))
output_ex = cca.Execute(_input)
stats = sitk.LabelShapeStatisticsImageFilter()
stats.Execute(output_ex)
num_label = cca.GetObjectCount()
num_list = [i for i in range(1, num_label+1)]

for i in num_list:
    if stats.GetNumberOfPixels(i) > 15:
        posList.append(stats.GetCentroid(i))

id = 0
with open('orig15_clstm',"w") as fp:
    for k in range(len(posList)):
        line = '%d 1 %lf %lf %lf 1 -1\n'%(k+1,posList[k][0],posList[k][1],posList[k][2] / 3)
        fp.write(line)