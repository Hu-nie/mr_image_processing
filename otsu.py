import cv2
import matplotlib.pyplot as plt

img = cv2.imread('sample.png', cv2.IMREAD_GRAYSCALE)

# 130을 threshold로 하는 이미지
_, t_130 = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)
# otsu algorithm을 적용한 이미지
t, t_otsu = cv2.threshold(img, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
print('otsu threshold:', t)

imgs = {'Original': img, 't:130':t_130, f'otsu:{t:.0f}': t_otsu}
for i, (key, value) in enumerate(imgs.items()):
    plt.subplot(1, 3 , i+1)
    plt.title(key)
    plt.imshow(value, cmap='gray')
    plt.xticks([])
    plt.yticks([])
plt.show()