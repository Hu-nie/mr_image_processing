import numpy as np
import os
from matplotlib import pyplot as plt
from util import getResolution,imageNormalization, normal_dist, gumbel_dist
import glob
import cv2
from tqdm import tqdm
path = 'D:/3_jeonbuk university/TOF_MR/SDH/TOF_1/'
normal = list()


whole_array = getResolution(path)

for filename in tqdm(glob.glob(os.path.join(path,'*.dcm'))):
    img = imageNormalization(filename) #SI Value Convert to 0~255
    _ , t_otsu = cv2.threshold(img, -1, 255,  cv2.THRESH_BINARY | cv2.THRESH_OTSU )
    foresion = img *(np.where(t_otsu == 255, 1, t_otsu))
    normal = normal + (foresion.flatten()).tolist()

    

normal =  np.array([item for item in tqdm(normal) if item != 0])
normal = np.sort(normal)

mean = np.mean(normal)
std = np.std(normal)


n_pdf = normal_dist(normal,mean,std)
g_pdf = gumbel_dist(normal,mean,std)

# idx = np.argwhere(np.diff(np.sign(5*n_pdf - g_pdf))).flatten()
# print(idx)
# plt.plot(normal,2*n_pdf , color = 'red')
plt.plot(normal,g_pdf , color = 'black')
plt.plot(normal,10*n_pdf , color = 'blue')
plt.xlabel('Data points')
plt.ylabel('Probability Density')

# print(min(normal))
#Create Graph
# plt.subplot(1,2,1)
# plt.hist(normal,bins =150, label='normal',color = 'midnightblue')
# plt.axvline(cut_off, color='red',label = 'line at x ={:.3f}'.format(cut_off), linestyle='dashed', linewidth=1)
# plt.xlabel('Signal intensity')
# plt.ylabel('n')
# plt.legend(loc='upper left')

# # print('1')

# plt.subplot(1,2,2)
# plt.hist(normal,bins =150, label='log',log=True,color = 'midnightblue')
# plt.axvline(cut_off, color='red',label = 'line at x ={:.3f}'.format(cut_off), linestyle='dashed', linewidth=1)
# plt.xlabel('Signal intensity')
# plt.ylabel('log(n)')
# plt.legend(loc='upper left')


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


