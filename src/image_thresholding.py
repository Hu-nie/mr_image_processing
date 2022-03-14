
import cv2
from matplotlib import pyplot as plt
from util import imageNormalization
import numpy as np
import SimpleITK as sitk
import os
import glob
from skimage.metrics import structural_similarity as ssim

## image load & normalilzation
path =  'D:/3_jeonbuk university/TOF_MR/Experiment/Normal/50_20 tof/'
# image = sitk.ReadImage(glㅍob.glob(os.path.join(path,'*.dcm'))[112])
# image_array = sitk.GetArrayFromImage(image)
# image_array = load_dicome('D:/jeonbuk university/TOF_MR/JSK/TOF_1/')
copy_img = imageNormalization(glob.glob(os.path.join(path,'*.dcm'))[60])
# copy_img = np.squeeze(copy_img)

# print(copy_img.shape)

# c = np.percentile(copy_img,[99.775],interpolation='nearest')
# print(c)


_, t_130 = cv2.threshold(copy_img, 10, 255, cv2.THRESH_BINARY)
t, t_otsu = cv2.threshold(copy_img, -1, 255,  cv2.THRESH_BINARY | cv2.THRESH_OTSU )



# 구조화 요소 커널, 사각형 (3x3) 생성 ---①
k = cv2.getStructuringElement(cv2. cv2.MORPH_RECT, (3,3))
# 침식 연산 적용 ---②
erosion = cv2.erode(t_130, k)

kernel = np.ones((11, 11), np.uint8)
result = cv2.morphologyEx(t_130, cv2.MORPH_CLOSE, kernel)

tempDiff = cv2.subtract(t_130, erosion)
tempDiff2 = cv2.subtract(t_130, result)
# diff = (diff * 255).astype("uint8")
# thresh = cv2.threshold(diff, 0, 255,
#                        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]


# # 차이점 빨간색으로 칠하기
# tempDiff[thresh == 255] = [0, 0, 255]
# imageC[thresh == 255] = [0, 0, 255]

# print(t_130.shpae)
print('otsu threshold:', t)

imgs = {'Original': t_130, 'erosion':erosion, 'diff': t_otsu}
for i, (key, value) in enumerate(imgs.items()):
    plt.subplot(1, 3 , i+1)
    plt.title(key)
    plt.imshow(value, cmap='gray')
    plt.xticks([])
    plt.yticks([])



plt.show()
