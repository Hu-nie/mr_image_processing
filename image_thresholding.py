from sys import path
import numpy as np
import cv2, pydicom
from sklearn.preprocessing import MinMaxScaler
import SimpleITK as sitk
import os
from numpy.core.fromnumeric import shape, size
from PIL import Image
from matplotlib import pyplot as plt
# path = './intracranial tof/'
# file_list = os.listdir(path)



# whole_array = np.expand_dims(np.empty((512,512)),axis=0)

# print(shape(whole_array))

# for i in file_list:
#     print(i)
#     image = sitk.ReadImage(path + i)
#     image_array = sitk.GetArrayFromImage(image).astype('float64')
#     whole_array = np.concatenate((whole_array,image_array),axis=0)
    
# whole_array = whole_array[1:]

# Z = (whole_array[1:2] - whole_array[1:2].mean()) / whole_array[1:2].std()

file_name  = './intracranial tof/181094_tof_001.dcm'
image = sitk.ReadImage(file_name)
image_array = sitk.GetArrayFromImage(image).astype('float32')

img = np.squeeze(image_array)
copy_img = img.copy()
min = np.min(copy_img)
max = np.max(copy_img)

copy_img1 = copy_img - np.min(copy_img)
copy_img = copy_img1 /np.max(copy_img1)

copy_img *= 2**8-1
copy_img = copy_img.astype(np.uint8)


# copy_img = np.expand_dims(copy_img, axis=-1)
# copy_img = cv2.cvtColor(copy_img, cv2.COLOR_GRAY2BGR)
print(shape(copy_img))


copy_img = cv2.GaussianBlur(copy_img, (5,5), 0)
# t, t_otsu = cv2.threshold(copy_img, -1, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU )



# copy_img[copy_img > 16] = 255
# copy_img[copy_img <= 16] = 125

# dst3 = cv2.subtract(copy_img, t_otsu)


# print(copy_img-t_otsu)
plt.imshow(copy_img,cmap='gray')
plt.show()




