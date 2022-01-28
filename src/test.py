import numpy as np
import os
from matplotlib import pyplot as plt
from util import getResolution,imageNormalization
import glob
import cv2
from tqdm import tqdm
import seaborn as sns



path = 'D:/3_jeonbuk university/TOF_MR/JSK/TOF_1/'

img = imageNormalization(glob.glob(os.path.join(path,'*.dcm'))[69]) #SI Value Convert to 0~255


# m , mask = cv2.threshold(img, -1, 255,  cv2.THRESH_BINARY | cv2.THRESH_OTSU )

# img = img.astype(np.uint8)
# mask = mask.astype(np.uint8)
# res =  cv2.bitwise_and(img,mask)

_, mask = cv2.threshold(img, 94, 255, cv2.THRESH_BINARY)

res_1 =  cv2.bitwise_and(img,mask)

print(img.dtype,mask.dtype)
# contours, hierarchy = cv2.findContours(t_94, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


# for cnt in contours:
#     cv2.drawContours(res_1, [cnt], 0, (255, 0, 0), 1)  # blue




imgs = {'Original': img, 'mask':mask, 'img&&mask': res_1}


for i, (key, value) in enumerate(imgs.items()):
    plt.subplot(1, 3 , i+1)
    plt.title(key)
    plt.imshow(value, cmap='gray')
    plt.xticks([])
    plt.yticks([])

plt.show()



