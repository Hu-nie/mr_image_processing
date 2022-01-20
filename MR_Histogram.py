from sys import path
import numpy as np
import SimpleITK as sitk
import os
from matplotlib import pyplot as plt
from util import img_norm
from scipy.stats import gumbel_r
import glob
import time
start = time.time() 

path = 'D:/jeonbuk university/TOF_MR/Experiment/Normal/50_20 tof/'
file_list = os.listdir(path)
cut_off = 488

# print(file_list)
## 이미지 해상도 확인 후 데이터 3D 배열로 결합
image = sitk.ReadImage(path + file_list[0])
image_array = sitk.GetArrayFromImage(image).astype('float16')
whole_array = np.expand_dims(np.empty(((image_array[0].shape)[0],(image_array[0].shape)[1])),axis=0)



for filename in glob.glob(os.path.join(path,'*.dcm')):
    print(filename)
    image = sitk.ReadImage(filename)
    image_array = sitk.GetArrayFromImage(image).astype('float16')
    # image_array = img_norm(image_array) #SI Value Convert to 0~255
    whole_array = np.concatenate((whole_array,image_array),axis=0)
    
whole_array = whole_array[1:]
print(whole_array.shape)


## z-score를 통한 정규화 진행후 분포 표현
# normal = (whole_array - whole_array.mean()) / whole_array.std()
normal = whole_array.flatten()
# normal= np.sort(normal)


#Create Graph
plt.subplot(1,2,1)
plt.hist(normal,bins =100, label='normal',color = 'midnightblue')
plt.xlabel('Signal intensity')
plt.ylabel('n')
plt.legend(loc='upper left')



plt.subplot(1,2,2)
plt.hist(normal,bins =100, label='log',log=True,color = 'midnightblue')
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
print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
plt.grid()
plt.legend()
plt.show()


