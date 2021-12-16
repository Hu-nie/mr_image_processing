import SimpleITK as sitk
import numpy as np
import os

def load_dicome(name):
    file_name  = name
    image = sitk.ReadImage(file_name)
    image_array = sitk.GetArrayFromImage(image).astype('float32')
    
    return image_array


def img_norm(image_array):
    img = np.squeeze(image_array)
    copy_img = img.copy()
    copy_img = img.copy()
    min = np.min(copy_img)
    max = np.max(copy_img)

    copy_img1 = copy_img - np.min(copy_img)
    copy_img = copy_img1 /np.max(copy_img1)

    copy_img *= 2**8-1
    copy_img = copy_img.astype(np.uint8)

    return copy_img


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
