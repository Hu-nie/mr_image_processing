import nibabel as nib
import dicom_numpy
import os
import numpy as np

pathtodicom = './50_20 tof/'
# get list of dicom images from directory that make up the 3D image
dicomlist = [pathtodicom + f for f in os.listdir(pathtodicom)]

# load dicom volume
vol, affine_LPS = dicom_numpy.combine_slices(dicomlist)

# convert the LPS affine to RAS
affine_RAS = np.diagflat([-1,-1,1,1]).dot(affine_LPS)

# create nibabel nifti object
niiimg = nib.Nifti1Image(vol, affine_RAS)
nib.save(niiimg, '/path/to/save')