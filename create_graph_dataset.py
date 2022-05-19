import numpy as np
import time

import torch

from func.run_pipeline_super_vox import segment_super_vox_3_channel, semantic_segment_crop_and_cat_3_channel_output, \
    img_3d_erosion_or_expansion, \
    generate_super_vox_by_watershed
from func.network import VoxResNet, CellSegNet_basic_lite
from func.ultis import save_obj, load_obj


HMS_data_dict = load_obj("dataset_info/HMS_dataset_info")
HMS_data_dict_train = HMS_data_dict["train"]

model = CellSegNet_basic_lite(input_channel=1, n_classes=3, output_func="softmax")
load_path = 'output/model_HMS_delete_fake_cells.pkl'
checkpoint = torch.load(load_path)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# %%

# we do not input the whole raw image to the model one time but input raw image crops
crop_cube_size = 64
stride = 32

# hyperparameter for TASCAN, min touching area of two super pixels if they belong to the same cell
min_touching_area = 30

# %%
img_ws_predictions = {}

for img_name in HMS_data_dict_train.keys():
    print(img_name)
    raw_img_path = f"../../../mnt/HMS_processed/raw/{img_name}.npy"
    print(HMS_data_dict_train[img_name]["raw"])
    hand_seg_path = f"../../../mnt/HMS_processed/segmentation/{img_name}/{img_name}_ins.npy"

    raw_img = np.load(raw_img_path)
    hand_seg = np.load(hand_seg_path)

    start = time.time()

    # feed the raw img to the model
    print('Feed raw img to model')
    raw_img_size = raw_img.shape

    seg_background_comp = np.zeros(raw_img_size)
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

        seg_background_comp += seg_background
        seg_boundary_comp += seg_boundary
    # print("Get model semantic seg by combination")
    seg_background_comp = np.array(seg_background_comp > 0, dtype=float)
    seg_boundary_comp = np.array(seg_boundary_comp > 0, dtype=float)
    seg_foreground_comp = np.array(1 - seg_background_comp - seg_boundary_comp > 0, dtype=float)

    how_close_are_the_super_vox_to_boundary = 2
    min_touching_percentage = 0.51

    seg_foreground_erosion = 1 - img_3d_erosion_or_expansion(1 - seg_foreground_comp,
                                                             kernel_size=how_close_are_the_super_vox_to_boundary + 1,
                                                             device=device)
    seg_foreground_super_voxel_by_ws = generate_super_vox_by_watershed(seg_foreground_erosion,
                                                                       connectivity=min_touching_area)

    img_ws_predictions[img_name] = seg_foreground_super_voxel_by_ws

    end = time.time()
    print("Time elapsed: ", end - start)

np.save("img_ws_predictions.npy", img_ws_predictions)