import numpy as np
import torch
import mat73
import scipy.io

from utils import *


## Data path (Train, .mat file) 
FILE_PATH_INPUT1 = ''
FILE_PATH_MASK1 = ''
FILE_PATH_MASKmimi1 = ''
FILE_PATH_LABEL1 = ''


## Patch Constant Variables
matrix_size = ['' ,'' , '']
PS = 64  # Patch Size
sub_num = 1
dir_num = 1
patch_num = [7, 8, 7] # Patch number 
strides = [(matrix_size[i] - PS) // (patch_num[i] - 1) for i in range(3)]
data_min_idx = [0, 0, 0]  # min index of data
data_max_idx = [153, 184, 144]  # max index of data
idx = 0  

# Initialize the lists for storing patches
patches = []
patches_mask = []
patches_maskmimi = []
patches_label = []



# Patch Generation

def load_mat_file(file_path, key):
    data = mat73.loadmat(file_path)
    matrix = data[key]
    if matrix.ndim == 3:
        matrix = np.expand_dims(matrix, axis=3)
    return matrix

def append_patches(patch_num, matrix,data_min_idx, data_max_idx strides, PS, key):
    for i in range(patch_num[0]):
        for j in range(patch_num[1]):
            for k in range(patch_num[2]):
            
            # Starting point
              start_i = min(max(i * strides[0], data_min_idx[0]), data_max_idx[0] - PS)
              start_j = min(max(j * strides[1], data_min_idx[1]), data_max_idx[1] - PS)
              start_k = min(max(k * strides[2], data_min_idx[2]), data_max_idx[2] - PS)
            
            # Ending point
              end_i = start_i + PS
              end_j = start_j + PS
              end_k = start_k + PS
            
            # Patch generation
              patch = matrix[start_i:end_i, start_j:end_j, start_k:end_k, idx]

            # Validation for zero patch
              if np.sum(patch) != 0:
                patches.append(patch)
                print(f"Valid patch at: [{start_i}:{end_i}, {start_j}:{end_j}, {start_k}:{end_k}]")
                print(np.sum(patch))

for dataset_num in range(1, sub_num+1):
    input_mat = load_mat_file(FILE_PATH_INPUT1, 'Lim_up')
    mask_mat = load_mat_file(FILE_PATH_MASK1, 'Mask_lim')
    mask_mimi_mat = load_mat_file(FILE_PATH_MASKmimi1, 'Mask_mimi')
    label_mat = load_mat_file(FILE_PATH_LABEL1, 'Full')
    
    matrix_size = input_mat.shape
    
    strides = [(matrix_size[i] - PS) // (patch_num[i] - 1) for i in range(3)]
    
    matrices_dict = {
        '': input_mat,
        '_mask': mask_mat,
        '_maskmimi': mask_mimi_mat,
        '_label': label_mat
    }

    for idx in range(dir_num):
        for key_suffix, matrix in matrices_dict.items():
            locals()[f'patches{key_suffix}'] = append_patches(locals()[f'patches{key_suffix}'], matrix, strides, PS, idx)


# Save to the train .mat file
scipy.io.savemat('file_train.mat', {
    'patches': patches,
    'patches_mask': patches_mask,
    'patches_maskmimi': patches_maskmimi,
    'patches_label': patches_label
})
