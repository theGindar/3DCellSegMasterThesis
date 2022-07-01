import numpy as np
import os
import pickle
import copy
import edt
import matplotlib.pyplot as plt
import time
import cv2
import pandas as pd
from sklearn.metrics.cluster import adjusted_rand_score
from skimage.metrics import adapted_rand_error

import torch
from torch import from_numpy as from_numpy
from torchsummary import summary

from func.run_pipeline_super_vox import segment_super_vox_3_channel, semantic_segment_crop_and_cat_3_channel_output, \
    img_3d_erosion_or_expansion, segment_super_vox_2_channel, semantic_segment_crop_and_cat_2_channel_output, \
    generate_super_vox_by_watershed, get_outlayer_of_a_3d_shape, get_crop_by_pixel_val, Cluster_Super_Vox, \
    assign_boudary_voxels_to_cells_with_watershed, \
    delete_too_small_cluster, reassign
from func.run_pipeline import segment, assign_boudary_voxels_to_cells, dbscan_of_seg, semantic_segment_crop_and_cat
from func.cal_accuracy import IOU_and_Dice_Accuracy, VOI
from func.network import VoxResNet, CellSegNet_basic_lite
from func.unet_3d_basic import UNet3D_basic
from func.ultis import save_obj, load_obj

torch.manual_seed(0)
import random
random.seed(0)
np.random.seed(0)


print(f"number of gpus: {torch.cuda.device_count()}")
torch.cuda.set_device(0)
print(f"current gpu: {torch.cuda.current_device()}")


# model=UNet3D_basic(in_channels = 1, out_channels = 3)
# load_path=''
# model=VoxResNet(input_channel=1, n_classes=3, output_func = "softmax")
# load_path=''
model = CellSegNet_basic_lite(input_channel=1, n_classes=3, output_func="softmax")
model_name = "model_Ovules_retrained_control_background"
results_output_path = "output/results_test_model_Ovules_retrained_control_background.csv"
load_path = 'output/model_Ovules_retrained_control_background.pkl'
checkpoint = torch.load(load_path)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


Ovules_data_dict = load_obj("dataset_info/Ovules_dataset_info")
Ovules_data_dict_test = Ovules_data_dict["test"]


### seg one img
# we do not input the whole raw image to the model one time but input raw image crops
crop_cube_size = 128
stride = 64

# hyperparameter for TASCAN, min touching area of two super pixels if they belong to the same cell
min_touching_area = 30

# %%

print("there are test imgs: " + str(Ovules_data_dict['test'].keys()))
case = 'N_435_final_crop_ds2.npz'
print("for test case " + str(case) + " : " + str(Ovules_data_dict['test'][case]))

# %%

hf = np.load(Ovules_data_dict['test'][case])
raw_img = np.array(hf["raw"], dtype=np.float)
hand_seg = np.array(hf["ins"], dtype=np.float)
boundary_gt = np.array(hf["boundary"], dtype=np.float)
foreground_gt = np.array(hf["foreground"], dtype=np.float)

print("raw_img shape: " + str(raw_img.shape))
print("hand_seg shape: " + str(hand_seg.shape))
raw_img_shape = raw_img.shape
hand_seg_shape = hand_seg.shape


# feed the raw img to the model
print('Feed raw img to model. Use different transposes')
raw_img_size = raw_img.shape

seg_foreground_comp = np.zeros(raw_img_size)
seg_boundary_comp = np.zeros(raw_img_size)

transposes = [[0, 1, 2]]  # ,[2,0,1],[0,2,1]]
reverse_transposes = [[0, 1, 2]]  # ,[1,2,0],[0,2,1]]

for idx, transpose in enumerate(transposes):
    print(str(idx + 1) + ": Transpose the image to be: " + str(transpose))
    with torch.no_grad():
        seg_img = \
            semantic_segment_crop_and_cat_3_channel_output(raw_img.transpose(transpose), model, device,
                                                           crop_cube_size=crop_cube_size, stride=stride)
    seg_img_background = seg_img['background']
    seg_img_boundary = seg_img['boundary']
    seg_img_foreground = seg_img['foreground']
    torch.cuda.empty_cache()

    # argmax
    print('argmax', end='\r')
    seg = []
    seg.append(seg_img_background)
    seg.append(seg_img_boundary)
    seg.append(seg_img_foreground)
    seg = np.array(seg)
    seg_argmax = np.argmax(seg, axis=0)
    # probability map to 0 1 segment
    seg_background = np.zeros(seg_img_background.shape)
    seg_background[np.where(seg_argmax == 0)] = 1
    seg_foreground = np.zeros(seg_img_foreground.shape)
    seg_foreground[np.where(seg_argmax == 2)] = 1
    seg_boundary = np.zeros(seg_img_boundary.shape)
    seg_boundary[np.where(seg_argmax == 1)] = 1

    seg_background = seg_background.transpose(reverse_transposes[idx])
    seg_foreground = seg_foreground.transpose(reverse_transposes[idx])
    seg_boundary = seg_boundary.transpose(reverse_transposes[idx])

    seg_foreground_comp += seg_foreground
    seg_boundary_comp += seg_boundary
print("Get model semantic seg by combination")
seg_boundary_comp = np.array(seg_boundary_comp > 0, dtype=np.int)
seg_foreground_comp = np.array(seg_foreground_comp > 0, dtype=np.int)
seg_foreground_comp[seg_boundary_comp > 0] = 0
seg_background_comp = np.array(1 - seg_foreground_comp - seg_boundary_comp > 0, dtype=np.int)

# %%

N = 80



# Generate super vox by watershed
how_close_are_the_super_vox_to_boundary = 2
min_touching_percentage = 0.51

seg_foreground_erosion = 1 - img_3d_erosion_or_expansion(1 - seg_foreground_comp,
                                                         kernel_size=how_close_are_the_super_vox_to_boundary + 1,
                                                         device=device)
seg_foreground_super_voxel_by_ws = generate_super_vox_by_watershed(seg_foreground_erosion,
                                                                   connectivity=min_touching_area)


print("There are " + str(len(np.unique(seg_foreground_super_voxel_by_ws))) + " super voxels")


# Super voxel clustearing
cluster_super_vox = Cluster_Super_Vox(min_touching_area=min_touching_area,
                                      min_touching_percentage=min_touching_percentage)
cluster_super_vox.fit(seg_foreground_super_voxel_by_ws)
seg_foreground_single_cell_with_boundary = cluster_super_vox.output_3d_img

# Delete too small cells
min_cell_size_threshold = 10

seg_foreground_single_cell_with_boundary_delete_too_small = delete_too_small_cluster(
    seg_foreground_single_cell_with_boundary, threshold=min_cell_size_threshold)


# Assign boudary voxels to their nearest cells
seg_final = assign_boudary_voxels_to_cells_with_watershed(seg_foreground_single_cell_with_boundary_delete_too_small,
                                                          seg_boundary_comp, seg_background_comp, compactness=1)

# %%

# Reassign unique numbers
# seg_final=reassign(seg_final)


N = 80

# %%

def colorful_seg(seg):
    unique_vals, val_counts = np.unique(seg, return_counts=True)

    background_val = unique_vals[np.argsort(val_counts)[::-1][0]]

    seg_RGB = []
    for i in range(seg.shape[0]):
        mask_gray = cv2.normalize(src=seg[i, :, :], dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                  dtype=cv2.CV_8UC1)
        seg_slice_RGB = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2RGB)
        seg_RGB.append(seg_slice_RGB)
    seg_RGB = np.array(seg_RGB)

    for idx, unique_val in enumerate(unique_vals):
        print(str(idx / len(unique_vals)), end="\r")
        if unique_val == background_val:
            COLOR = np.array([0, 0, 0], dtype=int)
        else:
            COLOR = np.array(np.random.choice(np.arange(256), size=3, replace=False), dtype=int)

        locs = np.where(seg == unique_val)

        for i in range(3):
            seg_RGB[locs[0], locs[1], locs[2], i] = COLOR[i]

    return seg_RGB


# %%

seg_final_RGB = colorful_seg(seg_final)
hand_seg_RGB = colorful_seg(hand_seg)

# %%

N = 80

fig = plt.figure(figsize=(5, 5))
plt.axis('off')
plt.imshow(seg_final_RGB[N, :, :])
# plt.savefig('Ovules_seg_RGB_'+str(N)+'.png',bbox_inches='tight',dpi=fig.dpi,pad_inches=0.0)

# %% md
# Evaluation

# %%
# %%

ARE = adapted_rand_error(hand_seg.astype(int).flatten(), seg_final.astype(int).flatten())
ARI = adjusted_rand_score(hand_seg.flatten(), seg_final.flatten())
VOI_val = VOI(seg_final.astype(np.int), hand_seg.astype(np.int))

# %%

print("ARI: " + str(ARI))
print("ARE: " + str(ARE))
print("VOI: " + str(VOI_val))


# %%

def img_3d_interpolate(img_3d, output_size, device=torch.device('cpu'), mode='nearest'):
    img_3d = img_3d.reshape(1, 1, img_3d.shape[0], img_3d.shape[1], img_3d.shape[2])
    img_3d = torch.from_numpy(img_3d).float().to(device)
    img_3d = torch.nn.functional.interpolate(img_3d, size=output_size, mode='nearest')
    img_3d = img_3d.detach().cpu().numpy()
    img_3d = img_3d.reshape(img_3d.shape[2], img_3d.shape[3], img_3d.shape[4])

    return img_3d


scale_factor = 0.3
org_shape = seg_final.shape
output_size = (int(org_shape[0] * scale_factor), int(org_shape[1] * scale_factor), int(org_shape[2] * scale_factor))
print(str(org_shape) + " --> " + str(output_size))

# %%

accuracy = IOU_and_Dice_Accuracy(img_3d_interpolate(hand_seg, output_size=output_size),
                                 img_3d_interpolate(seg_final, output_size=output_size))
accuracy_record = accuracy.cal_accuracy_II()

iou = np.array(accuracy_record[:, 1] > 0.7, dtype=np.float)
print('cell count accuracy iou >0.7: ' + str(sum(iou) / len(iou)))

dice = np.array(accuracy_record[:, 2] > 0.7, dtype=np.float)
print('cell count accuracy dice >0.7: ' + str(sum(dice) / len(dice)))

iou = np.array(accuracy_record[:, 1] > 0.5, dtype=np.float)
print('cell count accuracy iou >0.5: ' + str(sum(iou) / len(iou)))

dice = np.array(accuracy_record[:, 2] > 0.5, dtype=np.float)
print('cell count accuracy dice >0.5: ' + str(sum(dice) / len(dice)))

print('avg iou: ' + str(np.mean(accuracy_record[:, 1])))
print('avg dice: ' + str(np.mean(accuracy_record[:, 2])))

# %%
def pipeline(raw_img, hand_seg, model, device,
             crop_cube_size, stride,
             how_close_are_the_super_vox_to_boundary=2,
             min_touching_area=30,
             min_touching_percentage=0.51,
             min_cell_size_threshold=1,
             transposes=[[0, 1, 2]], reverse_transposes=[[0, 1, 2]]):
    seg_final = segment_super_vox_3_channel(raw_img, model, device,
                                             crop_cube_size=crop_cube_size, stride=stride,
                                             how_close_are_the_super_vox_to_boundary=how_close_are_the_super_vox_to_boundary,
                                             min_touching_area=min_touching_area,
                                             min_touching_percentage=min_touching_percentage,
                                             min_cell_size_threshold=min_cell_size_threshold,
                                             transposes=transposes,
                                             reverse_transposes=reverse_transposes)

    ari = adjusted_rand_score(hand_seg.flatten(), seg_final.flatten())
    voi = VOI(seg_final.astype(np.int), hand_seg.astype(np.int))

    scale_factor = 0.3
    org_shape = seg_final.shape
    output_size = (int(org_shape[0] * scale_factor), int(org_shape[1] * scale_factor), int(org_shape[2] * scale_factor))
    print(str(org_shape) + " --> " + str(output_size))

    accuracy = IOU_and_Dice_Accuracy(img_3d_interpolate(hand_seg, output_size=output_size),
                                     img_3d_interpolate(seg_final, output_size=output_size))
    accuracy_record = accuracy.cal_accuracy_II()
    hand_seg_after_accuracy = accuracy.gt
    seg_final_after_accuracy = accuracy.pred

    return accuracy_record, hand_seg_after_accuracy, seg_final_after_accuracy, ari, voi, seg_final



# mass process
seg_final_dict = {}
accuracy_record_dict = {}
ari_dict = {}
voi_dict = {}
for test_file in Ovules_data_dict_test.keys():
    print(test_file)
    hf = np.load(Ovules_data_dict_test[test_file])
    raw_img = np.array(hf["raw"], dtype=np.float)
    hand_seg = np.array(hf["ins"], dtype=np.float)

    accuracy_record, hand_seg_after_accuracy, seg_final_after_accuracy, ari, voi, seg_final = \
        pipeline(raw_img, hand_seg, model, device,
                 crop_cube_size=128,
                 stride=64)

    seg_final_dict[test_file] = seg_final
    accuracy_record_dict[test_file] = accuracy_record
    ari_dict[test_file] = ari
    voi_dict[test_file] = voi

    iou = np.array(accuracy_record[:, 1] > 0.7, dtype=np.float)
    print(".")
    print('cell count accuracy iou >0.7: ' + str(sum(iou) / len(iou)))

    dice = np.array(accuracy_record[:, 2] > 0.7, dtype=np.float)
    print(".")
    print('cell count accuracy dice >0.7: ' + str(sum(dice) / len(dice)))

    iou = np.array(accuracy_record[:, 1] > 0.5, dtype=np.float)
    print(".")
    print('cell count accuracy iou >0.5: ' + str(sum(iou) / len(iou)))

    dice = np.array(accuracy_record[:, 2] > 0.5, dtype=np.float)
    print(".")
    print('cell count accuracy dice >0.5: ' + str(sum(dice) / len(dice)))

    print('avg iou: ' + str(np.mean(accuracy_record[:, 1])))
    print('avg dice: ' + str(np.mean(accuracy_record[:, 2])))
    print("ari: " + str(ari))
    print("voi: " + str(voi))
    print("----------")


results_df = pd.DataFrame({"acc_iou_0_7":[],
                           "acc_iou_0_5":[],
                           "acc_dice_0_7":[],
                           "acc_dice_0_5":[],
                           "avg_iou":[],
                           "avg_dice":[],
                           "voi_1":[],
                           "voi_2":[]})
for item in seg_final_dict.keys():
    print(item)
    accuracy_record = accuracy_record_dict[item]
    ari = ari_dict[item]
    voi = voi_dict[item]
    iou = np.array(accuracy_record[:, 1] > 0.7, dtype=np.float)
    print('cell count accuracy iou >0.7: ' + str(sum(iou) / len(iou)))
    iou_0_7 = sum(iou) / len(iou)

    dice = np.array(accuracy_record[:, 2] > 0.7, dtype=np.float)
    print('cell count accuracy dice >0.7: ' + str(sum(dice) / len(dice)))
    dice_0_7 = sum(dice) / len(dice)

    iou = np.array(accuracy_record[:, 1] > 0.5, dtype=np.float)
    print('cell count accuracy iou >0.5: ' + str(sum(iou) / len(iou)))
    iou_0_5 = sum(iou) / len(iou)

    dice = np.array(accuracy_record[:, 2] > 0.5, dtype=np.float)
    print('cell count accuracy dice >0.5: ' + str(sum(dice) / len(dice)))
    dice_0_5 = sum(dice) / len(dice)

    print('avg iou: ' + str(np.mean(accuracy_record[:, 1])))
    avg_iou = np.mean(accuracy_record[:, 1])

    print('avg dice: ' + str(np.mean(accuracy_record[:, 2])))
    avg_dice = np.mean(accuracy_record[:, 2])
    print("ari: " + str(ari))
    print("voi: " + str(voi))
    print("----------")
    results_df = results_df.append({"acc_iou_0_7":iou_0_7,
                                    "acc_iou_0_5":iou_0_5,
                                    "acc_dice_0_7":dice_0_7,
                                    "acc_dice_0_5":dice_0_5,
                                    "avg_iou":avg_iou,
                                    "avg_dice":avg_dice,
                                    "voi_1":voi[0],
                                    "voi_2":voi[1]}, ignore_index=True)

    results_df.to_csv(results_output_path)
print(f"for model {load_path}")