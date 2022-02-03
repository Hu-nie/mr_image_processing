import imp
import numpy as np
import glob
from tqdm import tqdm
import os

from util import dicomToarray, getResolution

path = 'D:/3_jeonbuk university/TOF_MR/JSK/TOF_1/'

whole_array = getResolution(path)

for filename in tqdm(glob.glob(os.path.join(path,'*.dcm'))):
    image_array = dicomToarray(filename)
    image_array = np.expand_dims(image_array, axis=0)
    whole_array = np.concatenate((whole_array,image_array),axis=0)
    
    


denormalized_d = normalized_d * (max_d - min_d) + min_d