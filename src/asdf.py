from numpy import percentile
from util import *

path_dir = 'D:/raw/test/1362429_pbs/'

# print(file_list)/
Weight = 20
percentile = 99.775


# 2D slice to Voxel
Union_voxel, _ = extract_voxel_data(path_dir)

# convert to 0~255
norm_arr, min_v, max_v = voxelNorm(Union_voxel)

# Image Threshold Calculation with OTSU
normal = arrOtsu(norm_arr)

# Calculate the intersection of each distribution graph
normal, _, _, inter = getIntersection(Weight,normal)


# each of cutoff
Voxel_mean, Voxel_std = meanStd(Union_voxel)
p_cut = np.percentile(Union_voxel, percentile)
p_norm = (p_cut - Voxel_mean) / Voxel_std
si_cut = deNorm((inter.xy)[0][0],min_v,max_v)
si_norm = (si_cut - Voxel_mean) / Voxel_std
        
        
dataFrame = {'Distribution': si_cut, 'Distribution_Norm': si_norm,'Percentile': p_cut , 
             'Percentile_Norm': p_norm,'Min' :min_v,'Max' :max_v, 'Mean' :Voxel_mean,'Std' :Voxel_std }



print(dataFrame)       
       
        
# df = pd.DataFrame(index=range(0), columns=['patient', 'Distribution', 'Distribution_Norm',
#                                             'Percentile','Percentile_Norm','Min','Max','Mean','Std'])

# # pr        
# data_to_insert = {'patient': file , 'Distribution': si_cut, 'Distribution_Norm': si_norm,
#                       'Percentile': p_cut , 'Percentile_Norm': p_norm,'Min' :min_v,'Max' :max_v,
#                       'Mean' :Voxel_mean,'Std' :Voxel_std }

# df = df.append(data_to_insert, ignore_index=True)

# file_list = os.listdir(path_dir)

# class Cutoff:
    
#     def __init__(self, path, weight, percentile):
#         self.path = path
#         self.weight = weight
#         self.percentile = percentile
        
#     def getValue(self, path, weight, percentile):
#         normal = []
#         Union_voxel, _ = extract_voxel_data(path)
        
#         norm_arr, min_v, max_v = image_norm3D(Union_voxel)
#         Voxel_mean, Voxel_std = meanStd(Union_voxel)

#         normal = arrOtsu(norm_arr)
#         normal, _, _, inter = getIntersection(weight,normal)

#         # each of cutoff
#         p_cut = np.percentile(Union_voxel, percentile)
#         p_norm = (p_cut - Voxel_mean) / Voxel_std

#         si_cut = deNormalization((inter.xy)[0][0],min_v,max_v)
#         si_norm = (si_cut - (Union_voxel.flatten()).mean()) / (Union_voxel.flatten()).std()
    
#         return si_cut
# path_dir = 'C:/Users/Hoon/Desktop/dcom_intra/'/


        
        
        
        #


      

# df.to_csv("cca2.csv", mode='a', header=True)
    
    
    # print('평균:%f' %Voxel_mean,'표준편차:%f' %Voxel_std,'컷오프:%f' %p_cut,'컷오프(Norm):%f' %p_norm)
