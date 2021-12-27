from sys import path
import numpy as np
from numpy.core.fromnumeric import shape, size
import SimpleITK as sitk
import os
import time
from matplotlib import pyplot as plt

# def dicometoarray():
#     path = 'D:\\samsung\\red\\'
#     file_list = os.listdir(path)
#     whole_array = np.expand_dims(np.empty((880,880)),axis=0)



#     for i in file_list:

#         try:
#             dicome_list =  os.listdir(path + i + '/Intracranial TOF source')
#             for j in dicome_list:
#                image = sitk.ReadImage(path + i + '/Intracranial TOF source/'+j)
#                image_array = sitk.GetArrayFromImage(image).astype('float32')
#                whole_array = np.concatenate((whole_array,image_array),axis=0)

#             print(j)
#             print(shape(image_array))
#         except FileNotFoundError:
#             pass
#         except ValueError:
#             pass

        
#     return whole_array[1:]


# Z = (whole_array - whole_array.mean()) / whole_array.std()
# Z = Z.flatten()

# print(shape(Z))

# min = np.min(Z)
# max = np.max(Z)
# print(min,max)


# #Create Graph
# plt.hist(Z,bins=150,histtype='barstacked')
# plt.axvline(Z.mean(), color='red',label = 'line at x ={:.3f}'.format(Z.mean()), linestyle='dashed', linewidth=1)
# plt.axvline(4, color='blue',label = 'line at x ={:.3f}'.format(4), linestyle='dashed', linewidth=1)

# plt.grid()
# plt.legend()
# plt.show()



#single file analysis


path = './50_20 tof/'
file_list = os.listdir(path)

whole_array = np.expand_dims(np.empty((768,500)),axis=0)

print(shape(whole_array))

for i in file_list:
    print(i)
    image = sitk.ReadImage(path + i)
    image_array = sitk.GetArrayFromImage(image).astype('float64')
    # print(shape(image_array))
    # print(shape(image_array))
    # image_array = np.expand_dims(image_array,axis=0)

    whole_array = np.concatenate((whole_array,image_array),axis=0)
    
whole_array = whole_array[1:]

Z = (whole_array - whole_array.mean()) / whole_array.std()
Z = Z.flatten()
Z= np.sort(Z)
print(shape(Z))


min = np.min(Z)
max = np.max(Z)
print(min,max)


#Create Graph
plt.hist(Z,range=(0.1,2.5),bins =30, histtype='barstacked')




# plt.axvline(Z.mean(), color='red',label = 'line at x ={:.3f}'.format(Z.mean()), linestyle='dashed', linewidth=1)
# plt.axvline(1, color='black',label = 'line at x ={:.3f}'.format(1), linestyle='dashed', linewidth=1)
# plt.axvline(2, color='yellow',label = 'line at x ={:.3f}'.format(2), linestyle='dashed', linewidth=1)
# plt.axvline(2.5, color='black',label = 'line at x ={:.3f}'.format(2.5), linestyle='dashed', linewidth=1)
# plt.axvline(3.04, color='blue',label = 'line at x ={:.3f}'.format(3.04), linestyle='dashed', linewidth=1)
# plt.axvline(4, color='yellow',label = 'line at x ={:.3f}'.format(4), linestyle='dashed', linewidth=1)


plt.grid()
plt.legend()
plt.show()






