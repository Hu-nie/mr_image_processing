from sys import path
import numpy as np
import seaborn as sns
import os
from matplotlib import pyplot as plt
import glob
from tqdm import tqdm
from util import getResolution,imageNormalization



path = 'D:/3_jeonbuk university/TOF_MR/JSK/TOF_1/'
whole_array = getResolution(path)


for filename in tqdm(glob.glob(os.path.join(path,'*.dcm'))):
    image_array = imageNormalization(filename)
    image_array = np.expand_dims(image_array, axis=0)
    whole_array = np.concatenate((whole_array,image_array),axis=0)
    
whole_array = whole_array[1:]

print(whole_array.shape)


normal = whole_array.flatten()

cut_off = np.percentile(normal, 99.775)
# normal= np.sort(normal)

# sns.kdeplot(normal,log=True,kde=True,hist=False)a


#Create Graph
plt.subplot(1,2,1)
plt.hist(normal,bins =500, label='normal',color = 'midnightblue')
# plt.axvline(cut_off, color='red',label = 'line at x ={:.3f}'.format(cut_off), linestyle='dashed', linewidth=1)
plt.xlabel('Signal intensity')
plt.ylabel('n')
plt.legend(loc='upper left')



plt.subplot(1,2,2)
plt.hist(normal,bins =500, label='log',log=True,color = 'midnightblue')
# plt.axvline(cut_off, color='red',label = 'line at x ={:.3f}'.format(cut_off), linestyle='dashed', linewidth=1)
plt.xlabel('Signal intensity')
plt.ylabel('log(n)')
plt.legend(loc='upper left')



plt.grid()
plt.legend()
plt.show()


