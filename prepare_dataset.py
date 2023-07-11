# %%
import os
import numpy as np
import nibabel as nib
from PIL import Image
from skimage.util import view_as_windows
import json

# Initialize an empty list to store source-target pairs
data_pairs = []

root = '/ifs/loni/faculty/shi/spectrum/zdeng/SynthRAD2023/DATASET/Task1'
save_root = './data/Task1'
types = ['brain'] # , 'pelvis'

num = 160 # dimension of the image that is cropped from the original image
side_num_of_patch = 3  # number of patches in each direction
patch_size = 64 # dimension of each patch, w = h = 64
assert (num - patch_size) % (side_num_of_patch - 1) == 0
step_size =  (num - patch_size) // (side_num_of_patch - 1)

for t in types:
    path = os.path.join(root, t)
    subdirs = os.listdir(path)
    subdirs.sort()
    for idx, sub in enumerate(subdirs):
        if sub == 'overview':
            continue
        files_path = os.path.join(path, sub)
        files = os.listdir(files_path)
        files.sort()
        files = files[::-1] # let mr (source) before ct (target)
        cur_pair = []
        for file in files:
            if file.startswith('mask'):
                continue
            print(t, sub, file)
            file_path = os.path.join(files_path, file)
            img = nib.load(file_path)
            img_data = img.get_fdata()
            # print(img_data.shape)
            # print(img_data.dtype)
            # print(np.min(img_data), np.max(img_data))
            # print('------------------')

            # take the middle 200 slices
            X, Y, Z = img_data.shape
            
            half_num = num // 2
            if X > num:
                img_data = img_data[X//2-half_num:X//2+half_num, :, :]
            if Y > num:
                img_data = img_data[:, Y//2-half_num:Y//2+half_num, :]
            if Z > num:
                img_data = img_data[:, :, Z//2-half_num:Z//2+half_num]
            # normalize
            img_data = ((img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data)) * 255).astype('uint8')
            # create dir for subject
            save_path = os.path.join(save_root, t, sub, file.split('.')[0])
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            total = 0
            # save each single slice in each direction
            for i in range(img_data.shape[0]):
                slice = img_data[i, :, :]
                # rotate 90 degree anti-clockwise
                slice = np.rot90(slice)
                patches = view_as_windows(slice, window_shape=patch_size, step=step_size)
                x, y, _, _ = patches.shape
                for j in range(side_num_of_patch):
                    for k in range(side_num_of_patch):
                        if j >= x or k >= y:
                            continue
                        _img = Image.fromarray(patches[j, k, :, :])
                        img_path = os.path.join(save_path, 'sagittal'+str(i)+'_'+str(j)+'_'+str(k)+'.png')
                        _img.save(img_path)
                        if file.startswith('ct'):
                            cur_pair[total]["target"] = img_path
                        else:
                            cur_pair.append({"source": img_path})
                        total += 1
            for i in range(img_data.shape[1]):
                slice = img_data[:, i, :]
                # rotate 90 degree anti-clockwise
                slice = np.rot90(slice)
                pathes = view_as_windows(slice, window_shape=patch_size, step=step_size)
                x, y, _, _ = patches.shape
                for j in range(side_num_of_patch):
                    for k in range(side_num_of_patch):
                        if j >= x or k >= y:
                            continue
                        _img = Image.fromarray(pathes[j, k, :, :])
                        img_path = os.path.join(save_path, 'axial'+str(i)+'_'+str(j)+'_'+str(k)+'.png')
                        _img.save(img_path)
                        if file.startswith('ct'):
                            cur_pair[total]["target"] = img_path
                        else:
                            cur_pair.append({"source": img_path})
                        total += 1
            for i in range(img_data.shape[2]):
                slice = img_data[:, :, i]
                # rotate 90 degree clockwise
                slice = np.rot90(slice, -1)
                patches = view_as_windows(slice, window_shape=patch_size, step=step_size)
                x, y, _, _ = patches.shape
                for j in range(side_num_of_patch):
                    for k in range(side_num_of_patch):
                        if j >= x or k >= y:
                            continue
                        _img = Image.fromarray(patches[j, k, :, :])
                        img_path = os.path.join(save_path, 'coronal'+str(i)+'_'+str(j)+'_'+str(k)+'.png')
                        _img.save(img_path)
                        if file.startswith('ct'):
                            cur_pair[total]["target"] = img_path
                        else:
                            cur_pair.append({"source": img_path})
                        total += 1

        data_pairs.extend(cur_pair)

# Save data_pairs to a json file
with open('./data/image_pairs.json', 'w') as f:
    json.dump(data_pairs, f)