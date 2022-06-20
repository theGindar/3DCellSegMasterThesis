import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score

import torch

from func.run_pipeline_super_vox import segment_super_vox_3_channel_edge_gated_model
from func.cal_accuracy import IOU_and_Dice_Accuracy, VOI
from func.network import VoxResNet, CellSegNet_basic_lite, CellSegNet_basic_edge_gated_X
from func.ultis import save_obj, load_obj

torch.manual_seed(0)
import random
random.seed(0)
np.random.seed(0)


print(f"number of gpus: {torch.cuda.device_count()}")
torch.cuda.set_device(0)
print(f"current gpu: {torch.cuda.current_device()}")

model=CellSegNet_basic_edge_gated_X(input_channel=1, n_classes=3, output_func = "softmax")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

load_path='output/model_HMS_edge_gated_24_1.pkl'
checkpoint = torch.load(load_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

HMS_data_dict = load_obj("dataset_info/HMS_dataset_info")
HMS_data_dict_test = HMS_data_dict["test"]

# we do not input the whole raw image to the model one time but input raw image crops
crop_cube_size=64
stride=32

# hyperparameter for TASCAN, min touching area of two super pixels if they belong to the same cell
min_touching_area=30


def img_3d_interpolate(img_3d, output_size, device=torch.device('cpu'), mode='nearest'):
    img_3d = img_3d.reshape(1, 1, img_3d.shape[0], img_3d.shape[1], img_3d.shape[2])
    img_3d = torch.from_numpy(img_3d).float().to(device)
    img_3d = torch.nn.functional.interpolate(img_3d, size=output_size, mode='nearest')
    img_3d = img_3d.detach().cpu().numpy()
    img_3d = img_3d.reshape(img_3d.shape[2], img_3d.shape[3], img_3d.shape[4])

    return img_3d


def img_3d_interpolate(img_3d, output_size, device=torch.device('cpu'), mode='nearest'):
    img_3d = img_3d.reshape(1, 1, img_3d.shape[0], img_3d.shape[1], img_3d.shape[2])
    img_3d = torch.from_numpy(img_3d).float().to(device)
    img_3d = torch.nn.functional.interpolate(img_3d, size=output_size, mode='nearest')
    img_3d = img_3d.detach().cpu().numpy()
    img_3d = img_3d.reshape(img_3d.shape[2], img_3d.shape[3], img_3d.shape[4])

    return img_3d


def pipeline(raw_img, hand_seg, model, device,
             crop_cube_size, stride,
             how_close_are_the_super_vox_to_boundary=2,
             min_touching_area=30,
             min_touching_percentage=0.51,
             min_cell_size_threshold=1,
             transposes=[[0, 1, 2]], reverse_transposes=[[0, 1, 2]]):
    seg_final = segment_super_vox_3_channel_edge_gated_model(raw_img, model, device,
                                                             crop_cube_size=crop_cube_size, stride=stride,
                                                             how_close_are_the_super_vox_to_boundary=how_close_are_the_super_vox_to_boundary,
                                                             min_touching_area=min_touching_area,
                                                             min_touching_percentage=min_touching_percentage,
                                                             min_cell_size_threshold=min_cell_size_threshold,
                                                             transposes=transposes,
                                                             reverse_transposes=reverse_transposes)

    ari = adjusted_rand_score(hand_seg.flatten(), seg_final.flatten())
    voi = VOI(seg_final.astype(np.int), hand_seg.astype(np.int))

    scale_factor = 0.5
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
for test_file in HMS_data_dict_test.keys():
    print(test_file)
    raw_img = np.load(HMS_data_dict_test[test_file]["raw"])
    hand_seg = np.load(HMS_data_dict_test[test_file]["ins"])
    accuracy_record, hand_seg_after_accuracy, seg_final_after_accuracy, ari, voi, seg_final = \
        pipeline(raw_img, hand_seg, model, device,
                 crop_cube_size=64,
                 stride=32)

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

for item in seg_final_dict.keys():
    print(item)
    accuracy_record = accuracy_record_dict[item]
    ari = ari_dict[item]
    voi = voi_dict[item]
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
    print("ari: " + str(ari))
    print("voi: " + str(voi))
    print("----------")

print(f"for model {load_path}")