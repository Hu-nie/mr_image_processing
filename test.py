from operator import le
from posixpath import normpath
from sys import path
import numpy as np
import SimpleITK as sitk
import os
from matplotlib import pyplot as plt
from py import log
from util import img_norm
import glob
import tqdm
import seaborn as sns
import cv2

path = 'D:/jeonbuk university/TOF_MR/JSK/TOF_1/'
normal = []

# print(file_list)
## 이미지 해상도 확인 후 데이터 3D 배열로 결합
image = sitk.ReadImage(glob.glob(os.path.join(path,'*.dcm'))[112])
image_array = sitk.GetArrayFromImage(image)
whole_array = np.expand_dims(np.empty(((image_array[0].shape)[0],(image_array[0].shape)[1])),axis=0)
# image_array = img_norm(image_array)
# img = np.squeeze(image_array)

for filename in glob.glob(os.path.join(path,'*.dcm')):
    print(filename)
    image = sitk.ReadImage(filename)
    image_array = sitk.GetArrayFromImage(image)
    img = img_norm(image_array) #SI Value Convert to 0~255
    t, t_otsu = cv2.threshold(img, -1, 255,  cv2.THRESH_BINARY | cv2.THRESH_OTSU )
    t_otsu_1=np.where(t_otsu == 255, 1, t_otsu)
    a = img * t_otsu_1
    # print(set(t_otsu))
    normal = normal + (a.flatten()).tolist()
    
    # print(set(tuple(t_otsu)))
    # foreResion = foreResion + (img[np.where(img<= t)].tolist())

    # whole_array = np.concatenate((whole_array,image_array),axis=0)

normal =  [item for item in normal if item != 0]



# whole_array = whole_array[1:]
# print(whole_array.shape)

# t, t_otsu = cv2.threshold(img, -1, 255,  cv2.THRESH_BINARY | cv2.THRESH_OTSU )
# print(img)

# img[img<=t]=0
# img[img>t]=255

# normal = np.array(foreResion)

# ## z-score를 통한 정규화 진행후 분포 표현
# # normal = (whole_array - whole_array.mean()) / whole_array.std()
# # normal = whole_array.flatten()

# # # normal = image_array.flatten()
cut_off = np.percentile(normal, 99.775)
# # # normal= np.sort(normal)

# sns.displot(normal,log=True,kde=True)

print(min(normal))
#Create Graph
plt.subplot(1,2,1)
plt.hist(normal,bins =150, label='normal',color = 'midnightblue')
plt.axvline(cut_off, color='red',label = 'line at x ={:.3f}'.format(cut_off), linestyle='dashed', linewidth=1)
plt.xlabel('Signal intensity')
plt.ylabel('n')
plt.legend(loc='upper left')



plt.subplot(1,2,2)
plt.hist(normal,bins =150, label='log',log=True,color = 'midnightblue')
plt.axvline(cut_off, color='red',label = 'line at x ={:.3f}'.format(cut_off), linestyle='dashed', linewidth=1)
plt.xlabel('Signal intensity')
plt.ylabel('log(n)')
plt.legend(loc='upper left')


#히스토그램 임의 Cut off 확인을 위한 axvline

# plt.axvline(Z.mean(), color='red',label = 'line at x ={:.3f}'.format(Z.mean()), linestyle='dashed', linewidth=1)
# plt.axvline(1, color='black',label = 'line at x ={:.3f}'.format(1), linestyle='dashed', linewidth=1)
# plt.axvline(2, color='yellow',label = 'line at x ={:.3f}'.format(2), linestyle='dashed', linewidth=1)
# plt.axvline(2.5, color='black',label = 'line at x ={:.3f}'.format(2.5), linestyle='dashed', linewidth=1)
# plt.axvline(3.04, color='blue',label = 'line at x ={:.3f}'.format(3.04), linestyle='dashed', linewidth=1)
# plt.axvline(4, color='yellow',label = 'line at x ={:.3f}'.format(4), linestyle='dashed', linewidth=1)

plt.grid()
plt.legend()
plt.show()

