import cv2
from matplotlib import pyplot as plt
from util import load_dicome,img_norm
import numpy as np

## image load & normalilzation
image_array = load_dicome('D:\\samsung\\red\\00201534\\Intracranial TOF source\\ser801img00056.dcm')
copy_img = img_norm(image_array)

# print(copy_img.shape)

# c = np.percentile(copy_img,[99.775],interpolation='nearest')
# print(c)

_, t_130 = cv2.threshold(copy_img, 65, 255, cv2.THRESH_BINARY)
t, t_otsu = cv2.threshold(copy_img, -1, 255,  cv2.THRESH_BINARY | cv2.THRESH_OTSU )


print('otsu threshold:', t)

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