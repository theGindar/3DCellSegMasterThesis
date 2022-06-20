# train
from func.load_dataset import Cell_Seg_3D_Dataset
from func.network import VoxResNet, CellSegNet_basic_lite, CellSegNet_basic_edge_gated_X
from func.loss_func import dice_accuracy, dice_loss_II, dice_loss_II_weights, dice_loss_org_weights, \
    WeightedCrossEntropyLoss, dice_loss_org_individually, dice_loss_org_individually_with_cellsegloss_and_weights,\
    balanced_cross_entropy, DiceLoss
from func.ultis import save_obj, load_obj

import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import time
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torchvision.ops import sigmoid_focal_loss
import torch.nn.functional as F


import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd

# hyperparameters
# ----------
save_path = 'output/model_HMS_edge_gated_26.pkl'
need_resume = False
load_path = 'output/model_HMS_edge_gated_26.pkl'
loss_save_path = 'output/loss_HMS_edge_gated_26.csv'
learning_rate = 1e-4
max_epoch = 500
model_save_freq = 20
train_file_format = '.npy'
train_img_crop_size = (64, 64, 64)
boundary_importance = 1
batch_size = 5
num_workers = 4
# ----------
torch.manual_seed(0)
import random
random.seed(0)
np.random.seed(0)


print(f"number of gpus: {torch.cuda.device_count()}")
torch.cuda.set_device(1)
print(f"current gpu: {torch.cuda.current_device()}")

# init model
model=CellSegNet_basic_edge_gated_X(input_channel=1, n_classes=3, output_func = "softmax")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

if need_resume and os.path.exists(load_path):
    print("resume model from "+str(load_path))
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

# optimizer
optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)

# dataset and dataloader
HMS_data_dict = load_obj("dataset_info/HMS_dataset_info")
dataset = Cell_Seg_3D_Dataset(HMS_data_dict['train'])
dataset.set_para(file_format=train_file_format, \
    crop_size = train_img_crop_size, \
        boundary_importance = boundary_importance, \
            need_tensor_output = True, need_transform = True)
dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, \
    num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 1), prefetch_factor=1)
print('num of train files: '+str(len(HMS_data_dict['train'].keys())))
print('max epoch: '+str(max_epoch))

start_time = time.time()

loss_df = pd.DataFrame({"epoch":[],
                        "batch": [],
                        "time": [],
                        "total_loss": [],
                        "loss_1": [],
                        "loss_2": [],
                        "accuracy_1": [],
                        "accuracy_2": [],
                        "consistency_weights": []})

#dice_weights = torch.tensor([.2, .4, .4], dtype=torch.float).to(device)
#dice_loss = DiceLoss(weight=dice_weights, normalization='softmax')
#wce_loss = WeightedCrossEntropyLoss()


for ith_epoch in range(0, max_epoch):
    for ith_batch, batch in enumerate(dataset_loader):

        img_input=batch['raw'].to(device)

        seg_groundtruth_f=torch.tensor(batch['foreground']>0, dtype=torch.float).to(device)
        seg_groundtruth_bb=torch.cat((torch.tensor(batch['background']>0, dtype=torch.float), \
            torch.tensor(batch['boundary']>0, dtype=torch.float)), dim=1).to(device)

        seg_edge_groundtruth_f = torch.tensor(batch['edge_foreground'] > 0, dtype=torch.float).to(device)
        seg_edge_groundtruth_bb = torch.cat((torch.tensor(batch['edge_background'] > 0, dtype=torch.float), \
                                        torch.tensor(batch['edge'] > 0, dtype=torch.float)), dim=1).to(device)

        seg_edge_border_groundtruth = torch.tensor(batch['edge']>0, dtype=torch.float).to(device)
        seg_edge_foreground_groundtruth = torch.tensor(batch['edge_foreground'] > 0, dtype=torch.float).to(device)
        seg_edge_background_groundtruth = torch.tensor(batch['edge_background'] > 0, dtype=torch.float).to(device)

        # seg_non_edge = torch.where(
        #     torch.logical_or(seg_edge_border_groundtruth > 1, seg_edge_foreground_groundtruth > 1), 0., 1.).to(device)

        groundtruth_target = torch.cat((seg_edge_background_groundtruth,
                                        seg_edge_border_groundtruth,
                                        seg_edge_foreground_groundtruth), dim=1).to(device)
        
        weights_f=batch['weights_foreground'].to(device)
    
        seg_output, e_output = model(img_input)

        seg_output_f=seg_output[:,2,:,:,:]
        seg_output_bb=torch.cat((seg_output[:,0,:,:,:], seg_output[:,1,:,:,:]), dim=1)

        e_output_f = e_output[:, 2, :, :, :]
        e_output_bb = torch.cat((e_output[:, 0, :, :, :], e_output[:, 1, :, :, :]), dim=1)

        """
        CALCULATE CONSISTENCY WEIGHTS
        """
        # print(f"seg_output_f shape: {seg_output_f.shape}")
        # print(f"e_output_f shape: {e_output_f.shape}")
        # print(f"1: {(seg_output_f * (1 - e_output_f)).to(device).shape}")
        # print(f"2: {(seg_output_f * (1 - e_output_f) * torch.pow((seg_output_f + e_output_f), 2)).to(device).shape}")
        # print(f"3: {(seg_output_f * (1 - e_output_f) * torch.pow((seg_output_f + e_output_f), 2)) * torch.tensor(batch['boundary']>0, dtype=torch.float).to(device).shape}")
        seg_output_f_unsqueezed = torch.unsqueeze(seg_output_f, 1)
        e_output_f_unsqueezed = torch.unsqueeze(e_output_f, 1)
        weights_consistency = 0.9 * ((seg_output_f_unsqueezed * (1 - e_output_f_unsqueezed) * torch.pow((seg_output_f_unsqueezed + e_output_f_unsqueezed), 2)) * torch.tensor(batch['boundary']>0, dtype=torch.float).to(device)).to(device)

        total_weights_consistency = torch.sum(weights_consistency)

        weights_boundary = (batch['weights_boundary'].to(device) - weights_consistency).to(device)
        # print(f"weights_consistency shape: {weights_consistency.shape}")
        # print(f"weights_background shape: {batch['weights_background'].shape}")
        # print(f"weights_boundary shape: {weights_boundary.shape}")
        weights_background = batch['weights_background'].to(device)
        weights_bb = torch.cat((weights_background, weights_boundary), dim=1).to(device)

        weights_bb_old = torch.cat((batch['weights_background'], batch['weights_boundary']), dim=1).to(device)
        # print(f"weights_bb shape: {weights_bb.shape}")
        # print(f"weights_bb_old shape: {weights_bb_old.shape}")

        """
        END CALCULATE CONSISTENCY WEIGHTS
        """


        loss_1=dice_loss_org_weights(seg_output_bb, seg_groundtruth_bb, weights_bb)+\
            dice_loss_II_weights(seg_output_f, seg_groundtruth_f, weights_f)

        # TODO change!
        loss_2 = dice_loss_org_individually(e_output, groundtruth_target) + \
                 .5 * balanced_cross_entropy(e_output, groundtruth_target)
        #loss_2 = balanced_cross_entropy(e_output, groundtruth_target)
        #loss_2 = torch.mean(dice_loss.dice(e_output, groundtruth_target)) + \
        #          .5 * torch.mean(wce_loss.forward(e_output, groundtruth_target))

        loss = loss_1 + loss_2

        accuracy=dice_accuracy(seg_output_f, seg_groundtruth_f)
        accuracy_2 = dice_accuracy(e_output, groundtruth_target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        time_consumption = time.time() - start_time
        
        print(
            "epoch [{0}/{1}]\t"
            "batch [{2}]\t"
            "time(s) {time:.2f}\t"
            "loss {loss:.5f}\t"
            "loss_1 {loss_1:.5f}\t"
            "loss_2 {loss_2:.5f}\t"
            "acc {acc:.5f}\t"
            "consistency {consistency:.5f}\t".format(
                ith_epoch + 1,
                max_epoch,
                ith_batch,
                time = time_consumption,
                loss = loss.item(),
                loss_1=loss_1.item(),
                loss_2=loss_2.item(),
                consistency=total_weights_consistency.item(),
                acc = accuracy.item()))
        """
        loss_df = {"epoch": [],
                   "batch": [],
                   "time": [],
                   "total_loss": [],
                   "loss_1": [],
                   "loss_2": [],
                   "accuracy_1": [],
                   "accuracy_2": []}"""
        loss_df = loss_df.append({"epoch": ith_epoch + 1,
                                  "batch": ith_batch,
                                  "time": time_consumption,
                                  "total_loss": loss.item(),
                                  "loss_1": loss_1.item(),
                                  "loss_2": loss_2.item(),
                                  "accuracy_1": accuracy.item(),
                                  "accuracy_2": accuracy_2.item(),
                                  "consistency_weights": total_weights_consistency.item()}, ignore_index=True)
    
    if (ith_epoch+1)%model_save_freq==0:
        print('epoch: '+str(ith_epoch+1)+' save model')
        model.to(torch.device('cpu'))
        torch.save({'model_state_dict': model.state_dict()}, save_path)
        model.to(device)
        loss_df.to_csv(loss_save_path)
