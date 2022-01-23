import imp
import cv2
from matplotlib import pyplot as plt
from util import load_dicome,img_norm
import numpy as np
import SimpleITK as sitk
import os
import glob


## image load & normalilzation
path = 'D:/jeonbuk university/TOF_MR/JSK/TOF_1/'
image = sitk.ReadImage(glob.glob(os.path.join(path,'*.dcm'))[112])
image_array = sitk.GetArrayFromImage(image)
# image_array = load_dicome('D:/jeonbuk university/TOF_MR/JSK/TOF_1/')
copy_img = img_norm(image_array)
copy_img = np.squeeze(copy_img)

# print(copy_img.shape)

# c = np.percentile(copy_img,[99.775],interpolation='nearest')
# print(c)

_, t_130 = cv2.threshold(copy_img, 65, 255, cv2.THRESH_BINARY)
t, t_otsu = cv2.threshold(copy_img, -1, 255,  cv2.THRESH_BINARY | cv2.THRESH_OTSU )


print('otsu threshold:', t_otsu)

imgs = {'Original': copy_img, 't:130':t_130, f'otsu:{t:.0f}': t_otsu}
for i, (key, value) in enumerate(imgs.items()):
    plt.subplot(1, 3 , i+1)
    plt.title(key)
    plt.imshow(value, cmap='gray')
    plt.xticks([])
    plt.yticks([])



plt.show()

# _, t_130 = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)
# # otsu algorithm을 적용한 이미지
# t, t_otsu = cv2.threshold(img, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# print('otsu threshold:', t)

# imgs = {'Original': img, 't:130':t_130, f'otsu:{t:.0f}': t_otsu}
# for i, (key, value) in enumerate(imgs.items()):
#     plt.subplot(1, 3 , i+1)
#     plt.title(key)
#     plt.imshow(value, cmap='gray')
#     plt.xticks([])
#     plt.yticks([])