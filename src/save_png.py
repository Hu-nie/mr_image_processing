import numpy as np
import os
from matplotlib import pyplot as plt
from util import *
import glob
import cv2
from tqdm import tqdm
import seaborn as sns

# path = 'C:/Users/Hoon/Desktop/a'
# mng = plt.get_current_fig_manager()
# mng.frame.Maximize(True)


path = 'C:/Users/Hoon/Desktop/intra/'


print(path)
normal = list()
set_v = 5
whole_arr = getResolution(path)

for filename in tqdm(glob.glob(os.path.join(path,'*.dcm'))):
    img_arr = np.expand_dims(dicomToarray(filename), axis=0)
    whole_arr = np.concatenate((whole_arr, img_arr), axis=0)


norm_arr, min_v, max_v = image_norm3D(whole_arr[1:])
whole_arr_f = np.sort(whole_arr.flatten())
cut_off = np.percentile(whole_arr_f, 99.5)



for arr in tqdm(norm_arr):
    _ , t_otsu = cv2.threshold(arr, -1, 255,  cv2.THRESH_BINARY | cv2.THRESH_OTSU )
    foresion = arr *(np.where(t_otsu == 255, 1, t_otsu))
    normal = normal + (foresion.flatten()).tolist()


normal, n_pdf, g_pdf, inter = getIntersection(set_v,normal)

print(*inter.xy)
print(min_v,max_v)
print(deNormalization((inter.xy)[0][0],min_v,max_v))
# value = deNormalization(a)
deNorm_v =deNormalization((inter.xy)[0][0],min_v,max_v)



plt.clf()

plt.figure(figsize=(20,8))
#Create Graph
plt.subplot(1,3,1)
plt.hist(whole_arr_f,bins =500, label='normal',color = 'midnightblue')
plt.axvline(cut_off, color='red',label = 'x of percentile 99.5  ={:.3f}'.format(cut_off), linestyle='dashed', linewidth=1)
plt.axvline(deNorm_v, color='blue',label = 'x of new_method  ={:.3f}'.format(deNorm_v), linestyle='dashed', linewidth=1)
plt.xlabel('Signal intensity')
plt.ylabel('n')
plt.legend(loc='upper left')



plt.subplot(1,3,2)
plt.hist(whole_arr_f,bins =500, label='log',log=True,color = 'midnightblue')
plt.axvline(cut_off, color='red',label = 'x of percentile 99.5 ={:.3f}'.format(cut_off), linestyle='dashed', linewidth=1)
plt.axvline(deNorm_v, color='blue',label = 'x of new_method  ={:.3f}'.format(deNorm_v), linestyle='dashed', linewidth=1)
plt.xlabel('Signal intensity')
plt.ylabel('log(n)')
plt.legend(loc='upper left')

plt.subplot(1,3,3)
plt.plot(normal,g_pdf , color = 'black')
plt.plot(normal,set_v*n_pdf , color = 'blue')
plt.plot(*inter.xy,'ro',label = 'point at x ={:.3f}'.format((inter.xy)[0][0]))
plt.xlabel('Data points')
plt.ylabel('Probability Density')



plt.grid()
plt.legend()
# plt.show()
plt.savefig('./src/result/'+'a.png')