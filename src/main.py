import numpy as np
import os
from matplotlib import pyplot as plt
from util import *
import glob
import cv2
from tqdm import tqdm
import seaborn as sns
import pandas as pd


# path = 'C:/Users/Hoon/Desktop/dcom_intra/'
# mng = plt.get_current_fig_manager()
# mng.frame.Maximize(True)
# C:\Users\Hoon\Desktop\

path_dir = 'C:/Users/Hoon/Desktop/dcom_intra/'
 
file_list = os.listdir(path_dir)
df = pd.DataFrame(index=range(0), columns=['patient', 'Distribution', 'Distribution_Norm',
                                              'Percentile','Percentile_Norm'])

for file in file_list:
    path = path_dir + file+'/'
    print(path)
    normal = list()
    set_v = 20
    
    
    whole_arr = getResolution(path)
    
    for filename in tqdm(glob.glob(os.path.join(path,'*.dcm'))):
        img_arr = np.expand_dims(dicomToarray(filename), axis=0)
        whole_arr = np.concatenate((whole_arr, img_arr), axis=0)


    # whole_arr = whole_arr[1:]
    norm_arr, min_v, max_v = image_norm3D(whole_arr)
    
    whole_arr_f=  whole_arr.flatten()
    whole_arr_f = np.sort(whole_arr)
    
    arr_avg = average(whole_arr_f)
    arr_mean =  np.mean(whole_arr_f)
    arr_std =  np.std(whole_arr_f)
    
    print(arr_avg,arr_mean,arr_std)
    
    print(whole_arr_f.shape)
    p_cut = np.percentile(whole_arr_f, 99.775)
    p_norm = (p_cut - arr_mean) / arr_std
    print(p_cut,p_norm)
    for arr in tqdm(norm_arr):
        _ , t_otsu = cv2.threshold(arr, -1, 255,  cv2.THRESH_BINARY | cv2.THRESH_OTSU )
        foresion = arr *(np.where(t_otsu == 255, 1, t_otsu))
        normal = normal + (foresion.flatten()).tolist()


    normal, n_pdf, g_pdf, inter = getIntersection(set_v,normal)

    # value = deNormalization(a)
    deNorm_v =deNormalization((inter.xy)[0][0],min_v,max_v)
    si_norm = (deNorm_v - whole_arr_f.mean()) / whole_arr_f.std()
    


    data_to_insert = {'patient': file , 'Distribution': si_cut, 'Distribution_Norm': si_norm,
                      'Percentile': p_cut , 'Percentile_Norm': p_norm,'Min' :min_v,'Max' :max_v,
                      'Mean' :Voxel_mean,'Std' :Voxel_std }

#     # 데이터 추가해서 원래 데이터프레임에 저장하기
    df = df.append(data_to_insert, ignore_index=True)

    print(df)

# df.to_csv("ica.csv", mode='a', header=False)

    # plt.clf()
    
    
    # plt.figure(figsize=(20,8))
    # #Create Graph
    # plt.subplot(1,3,1)
    # plt.hist(whole_arr_f,bins =500, label='normal',color = 'midnightblue')
    # plt.axvline(p_cut, color='red',label = 'x of percentile 99.5  ={:.3f}'.format(p_cut), linestyle='dashed', linewidth=1)
    # plt.axvline(deNorm_v, color='blue',label = 'x of new_method  ={:.3f}'.format(deNorm_v), linestyle='dashed', linewidth=1)
    # plt.xlabel('Signal intensity')
    # plt.ylabel('n')
    # plt.legend(loc='upper left')



    # plt.subplot(1,3,2)
    # plt.hist(whole_arr_f,bins =500, label='log',log=True,color = 'midnightblue')
    # plt.axvline(p_cut, color='red',label = 'x of percentile 99.5 ={:.3f}'.format(p_cut), linestyle='dashed', linewidth=1)
    # plt.axvline(deNorm_v, color='blue',label = 'x of new_method  ={:.3f}'.format(deNorm_v), linestyle='dashed', linewidth=1)
    # plt.xlabel('Signal intensity')
    # plt.ylabel('log(n)')
    # plt.legend(loc='upper left')

    # plt.subplot(1,3,3)
    # plt.plot(normal,g_pdf ,  color = 'black')
    # plt.plot(normal,set_v*n_pdf,  color = 'blue')
    # plt.plot(*inter.xy,'ro',label = 'point at x ={:.3f}'.format((inter.xy)[0][0]))
    # plt.xlabel('Data points')
    # plt.ylabel('Probability Density')
    # plt.xlim([((inter.xy)[0][0])-2, ((inter.xy)[0][0])+2])
    # plt.ylim([0.0,0.05])  


    # plt.grid()
    # plt.legend()
    # # plt.show()
    # plt.savefig('./src/ECA_2/'+file+'.png')