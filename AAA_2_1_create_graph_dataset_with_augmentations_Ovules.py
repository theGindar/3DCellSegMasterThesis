import numpy as np
import time

import torch

from func.run_pipeline_super_vox import segment_super_vox_3_channel, semantic_segment_crop_and_cat_3_channel_output, \
    img_3d_erosion_or_expansion, \
    generate_super_vox_by_watershed
from func.network import VoxResNet, CellSegNet_basic_lite
from func.ultis import save_obj, load_obj

import torchio as tio
import pickle

from func.graph_learning import SuperVoxToNxGraph

script_num = 2

torch.manual_seed(script_num*10)
import random
random.seed(script_num*10)
np.random.seed(script_num*10)

image_names_to_segment = [
    "N_512_ds2x.npz"
]
save_path_graph_set = f'graphs_dataset_train_with_augmentations_ovules_retrained_skript_{script_num}_1.pkl'
txt_write_file_path = f'flag_folder/AAA_{script_num}_1_FINISHED.txt'


print(f"number of gpus: {torch.cuda.device_count()}")
torch.cuda.set_device(1)
print(f"current gpu: {torch.cuda.current_device()}")

# Data Augmentation

def transform_the_tensor(image_tensors, prob=0.5):
    dict_imgs_tio={}

    for item in image_tensors.keys():
        dict_imgs_tio[item]=tio.ScalarImage(tensor=image_tensors[item])
    subject_all_imgs = tio.Subject(dict_imgs_tio)

    transform_shape = tio.Compose([
        tio.RandomFlip(axes = int(np.random.randint(3, size=1)[0]), p=1)
        #tio.RandomAffine(isotropic=True, degrees=(20,20,20))])#,tio.RandomAffine(p=prob)
        ])

    subject_all_imgs = transform_shape(subject_all_imgs)

    transform_val = tio.Compose([
        tio.RandomBlur(p=prob),
        tio.RandomNoise(p=prob),tio.RandomMotion(p=prob),tio.RandomBiasField(p=prob),tio.RandomSpike(p=prob),tio.RandomGhosting(p=prob)])
    subject_all_imgs['raw'] = transform_val(subject_all_imgs['raw'])

    for item in subject_all_imgs.keys():
        image_tensors[item] = subject_all_imgs[item].data

    return image_tensors

Ovules_data_dict = load_obj("dataset_info/Ovules_dataset_info")
Ovules_data_dict_train = Ovules_data_dict["train"]

model = CellSegNet_basic_lite(input_channel=1, n_classes=3, output_func="softmax")
load_path = 'output/model_Ovules_retrained_control_background_2.pkl'
checkpoint = torch.load(load_path)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

num_augmentations = 10

# %%

# we do not input the whole raw image to the model one time but input raw image crops
crop_cube_size = 128
stride = 64

# hyperparameter for TASCAN, min touching area of two super pixels if they belong to the same cell
min_touching_area = 30

# %%
graphs_list = []
for img_name in image_names_to_segment:
    print(img_name)
    if img_name == ".DS_Store":
        continue
    for img_trans in range(num_augmentations):
        print(f"image to segment: {img_name}, augmentation: {img_trans}")

        hf = np.load(Ovules_data_dict_train[img_name])
        raw_img = np.array(hf["raw"], dtype=np.float)
        hand_seg = np.array(hf["ins"], dtype=np.float)


        output = {
            'raw': np.expand_dims(raw_img, 0),
            'handseg': np.expand_dims(hand_seg, 0)
        }
        output_augmented = transform_the_tensor(output)
        raw_img = output_augmented['raw']
        hand_seg = output_augmented['handseg']

        raw_img = torch.squeeze(raw_img, dim=0).cpu().detach().numpy()
        hand_seg = torch.squeeze(hand_seg, dim=0).cpu().detach().numpy()

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

            print("transforms predicted!")

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

        super_vox_to_graph = SuperVoxToNxGraph()
        graph = super_vox_to_graph.get_nx_graph_from_ws_with_gt(seg_foreground_super_voxel_by_ws, hand_seg)
        graphs_list.append(graph)

        end = time.time()
        with open(save_path_graph_set, 'wb') as f:
            pickle.dump(graphs_list, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Time elapsed: ", end - start)


with open(txt_write_file_path, 'w') as f:
    f.write('1 finished')