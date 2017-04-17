import numpy as np
import numpy.linalg as npl
import nibabel as nib

img = nib.load('../../data/filtered_func_data_mni.nii.gz')

vox_to_mm = img.affine

nib.affines.apply_affine(vox_to_mm, [2, 3, 4])

mm_to_vox = npl.inv(img.affine)

nib.affines.apply_affine(mm_to_vox, [0, 0, 10])
