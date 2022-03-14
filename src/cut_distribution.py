import numpy as np
import os
from matplotlib import pyplot as plt
from util import getResolution,imageNormalization, normal_dist,gumbel_dist
import glob
import cv2
from tqdm import tqdm
import seaborn as sns
from shapely.geometry import LineString


path = 'C:/Users/Hoon/Desktop/TOF_1/'
normal = list()


whole_array = getResolution(path)

for filename in tqdm(glob.glob(os.path.join(path,'*.dcm'))):
    img = imageNormalization(filename) #SI Value Convert to 0~255
    _ , t_otsu = cv2.threshold(img, -1, 255,  cv2.THRESH_BINARY | cv2.THRESH_OTSU )
    foresion = img *(np.where(t_otsu == 255, 1, t_otsu))
    normal = normal + (foresion.flatten()).tolist()

    

normal =  np.array([item for item in tqdm(normal) if item != 0])
normal = np.sort(normal)

mean = np.mean(normal)
std = np.std(normal)


n_pdf = normal_dist(normal,mean,std)
g_pdf = gumbel_dist(normal,mean,std)


# idx = np.argwhere(np.diff(np.sign(5*n_pdf - g_pdf))).flatten()
# print(idx)
# plt.plot(normal,2*n_pdf , color = 'red')


plt.plot(normal,g_pdf , color = 'black')
plt.plot(normal,n_pdf , color = 'blue')

line_1 = LineString(np.column_stack((normal,g_pdf)))
line_2 = LineString(np.column_stack((normal,n_pdf)))
inter = line_1.intersection(line_2)

print((inter.xy)[0][0])

plt.plot(*inter.xy,'ro')
plt.xlabel('Data points')
plt.ylabel('Probability Density')


plt.grid()
plt.legend()
plt.show()


