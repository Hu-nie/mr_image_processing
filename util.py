import SimpleITK as sitk
import numpy as np
import os
import glob
import pydicom

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
    copy_img = img.copy()
    min = np.min(copy_img)
    max = np.max(copy_img)

    copy_img1 = copy_img - np.min(copy_img)
    copy_img = copy_img1 /np.max(copy_img1)

    copy_img *= 2**8-1
    copy_img = copy_img.astype(np.uint16)
    # copy_img = np.expand_dims(copy_img, axis=0)
    return copy_img

