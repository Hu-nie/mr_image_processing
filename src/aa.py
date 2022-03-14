import matplotlib.pyplot as plt
from util import *
import numpy as np
path_dir = 'C:/Users/Hoon/Desktop/CCA/TOF -강남옥'

# file_list = os.listdir(path_dir)

Union_voxel, ijk_to_xyz = extract_voxel_data(path_dir)

Union_voxel = Union_voxel.T

img = Union_voxel[1]

img1 = img[:, 24:]

img2 = img1[:, :464]

mean_2 = np.mean(img)
Std_2 = np.std(img)
min_2 = np.min(img)
max_2 = np.max(img)

mean_1 = np.mean(img2)
Std_1 = np.std(img2)
min_1 = np.min(img2)
max_1 = np.max(img2)
print(mean_2, Std_2, min_2, max_2)
print(mean_1, Std_1, min_1, max_1)

plt.imshow(img, cmap='gray')

plt.show()