import SimpleITK as sitk
import numpy as np
import os
import glob
import pydicom
from tqdm import tqdm

## 이미지 해상도 확인 후 데이터 3D 배열로 결합
def getResolution(path):
    image = sitk.ReadImage(glob.glob(os.path.join(path,'*.dcm'))[0])
    image_array = sitk.GetArrayFromImage(image)
    whole_array = np.expand_dims(np.empty(((image_array[0].shape)[0],(image_array[0].shape)[1])),axis=0)

    return whole_array


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

    # copy_img1 = copy_img - np.min(copy_img)
    # copy_img = copy_img1 /np.max(copy_img1)

    # copy_img *= 2**8-1
    norm_img = norm_img.astype(np.uint16)
    # copy_img = np.expand_dims(copy_img, axis=0)
    return norm_img
# def imageNormalization(filename):
#     image_array = dicomToarray(filename)
#     img = np.squeeze(image_array)
#     img_copy = img.copy()
#     min_v = np.min(img_copy)
#     max_v = np.max(img_copy)
#     norm_img = ((img_copy - min_v) * (1/(max_v - min_v) * 255))
        
#     return norm_img, min_v, max_v



def deNormalization(normalized, max, min):
    denormalized_d = normalized * (max - min) + min
    
    return denormalized_d    




def normal_dist(x , mean , sd):
    prob_density = (1/np.sqrt(2*(np.pi*sd))) * np.exp(-0.5*((x-mean)/sd)**2)

    return prob_density


def gumbel_dist(x,mean,sd):
    prob_density = 1/sd*np.exp(-((x-mean)/sd)-np.exp(-((x-mean)/sd)))

    return prob_density


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
