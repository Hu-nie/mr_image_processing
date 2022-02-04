import SimpleITK as sitk
import numpy as np
import os
import glob
import pydicom
from tqdm import tqdm
from shapely.geometry import LineString
## 이미지 해상도 확인 후 데이터 3D 배열로 결합

def getResolution(path):
    image = sitk.ReadImage(glob.glob(os.path.join(path,'*.dcm'))[0])
    image_array = sitk.GetArrayFromImage(image)
    whole_array = np.expand_dims(np.empty(((image_array[0].shape)[0],(image_array[0].shape)[1])),axis=0)

    return whole_array

## 

def dicomToarray(filename):
    image = pydicom.read_file(filename)
    image_array = image.pixel_array
    
    return image_array

def imageNormalization(filename):
    image_array = dicomToarray(filename)
    img = np.squeeze(image_array)
    copy_img = img.copy()
    min_v = np.min(copy_img)
    max_v = np.max(copy_img)
    norm_img = ((copy_img - min_v) * (1/(max_v - min_v) * 255))
    norm_img = norm_img.astype(np.uint16)
    # copy_img = np.expand_dims(copy_img, axis=0)

    return norm_img

def image_norm3D(whole_arr):
    min_v = np.min(whole_arr)
    max_v = np.max(whole_arr)

    for n, arr in enumerate(whole_arr):

        whole_arr[n] = ((arr - min_v) * (1/(max_v - min_v) * 255))
        whole_arr = whole_arr.astype(np.uint16)
        
    return whole_arr, min_v, max_v

def deNormalization(normalized, min_v, max_v):
    value = (normalized/255 )* (max_v - min_v) + min_v
    
    return value    


def normal_dist(x , mean , sd):
    prob_density = (1/np.sqrt(2*(np.pi*sd))) * np.exp(-0.5*((x-mean)/sd)**2)

    return prob_density


def gumbel_dist(x,mean,sd):
    prob_density = 1/sd*np.exp(-((x-mean)/sd)-np.exp(-((x-mean)/sd)))

    return prob_density


def getIntersection(set_v, normal):
    normal =  np.array([item for item in normal if item != 0])
    normal = np.sort(normal)

    mean = np.mean(normal)
    std = np.std(normal)


    n_pdf = normal_dist(normal,mean,std)
    g_pdf = gumbel_dist(normal,mean,std)




    line_1 = LineString(np.column_stack((normal,g_pdf)))
    line_2 = LineString(np.column_stack((normal,set_v*n_pdf)))
    inter = line_1.intersection(line_2)

    return normal, n_pdf, g_pdf, inter


def createMIP(np_img, slices_num):
    ''' create the mip image from original image, slice_num is the number of 
    slices for maximum intensity projection'''
    img_shape = np_img.shape
    np_mip = np.zeros(img_shape)
    for i in tqdm(range(img_shape[0])):
        start = max(0, i-slices_num)
        np_mip[i,:,:] = np.amax(np_img[start:i+1],0)
    return np_mip


# ## z-score를 통한 정규화 진행후 분포 표현
# # normal = (whole_array - whole_array.mean()) / whole_array.std()
# # normal = whole_array.flatten()

#히스토그램 임의 Cut off 확인을 위한 axvline

# plt.axvline(Z.mean(), color='red',label = 'line at x ={:.3f}'.format(Z.mean()), linestyle='dashed', linewidth=1)
# plt.axvline(1, color='black',label = 'line at x ={:.3f}'.format(1), linestyle='dashed', linewidth=1)
# plt.axvline(2, color='yellow',label = 'line at x ={:.3f}'.format(2), linestyle='dashed', linewidth=1)
# plt.axvline(2.5, color='black',label = 'line at x ={:.3f}'.format(2.5), linestyle='dashed', linewidth=1)
# plt.axvline(3.04, color='blue',label = 'line at x ={:.3f}'.format(3.04), linestyle='dashed', linewidth=1)
# plt.axvline(4, color='yellow',label = 'line at x ={:.3f}'.format(4), linestyle='dashed', linewidth=1)


# from nibabel import nifti2dicom_1file

# nifti2dicom_1file('./data/mip.nii','./data')

# import dicom2nifti

# dicom2nifti.convert_directory( 'D:/3_jeonbuk university/TOF_MR/JSK/TOF_1/','./data')