import cv2
from matplotlib import pyplot as plt
from util import load_dicome,img_norm

## image load & normalilzation
image_array = load_dicome('D:\\samsung\\red\\00201534\\Intracranial TOF source\\ser801img00016.dcm')
copy_img = img_norm(image_array)



copy_img = cv2.GaussianBlur(copy_img, (5,5), 0)
t, t_otsu = cv2.threshold(copy_img, -1, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU )

# copy_img[copy_img > 16] = 255
# copy_img[copy_img <= 16] = 125

plt.imshow(copy_img,cmap='gray')
plt.show()




import iplantuml