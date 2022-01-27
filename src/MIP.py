import numpy as np
import os
from matplotlib import pyplot as plt
from util import getResolution,imageNormalization, createMIP
import glob
import cv2
from tqdm import tqdm
import SimpleITK as sitk
from os.path import isfile, join
path = 'D:/3_jeonbuk university/TOF_MR/SDH/TOF_1/'
normal = list()


whole_array = getResolution(path)
print(whole_array.shape)
sitk_img = sitk.ReadImage(glob.glob(os.path.join(path,'*.dcm'))[0])

for filename in tqdm(glob.glob(os.path.join(path,'*.dcm'))):
    img = imageNormalization(filename) #SI Value Convert to 0~255
    _, mask = cv2.threshold(img, 94, 255, cv2.THRESH_BINARY)
    print(mask.shape)
    print(img.shape)
    res = cv2.bitwise_and(img,img, mask= mask)
    copy_img = np.expand_dims(res, axis=0)
    whole_array = np.concatenate((whole_array,copy_img),axis=0)
    
whole_array = whole_array[1:]
print(whole_array.shape)
