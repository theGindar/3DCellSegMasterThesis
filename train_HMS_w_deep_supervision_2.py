# train
from func.load_dataset import Cell_Seg_3D_Dataset
from func.network import VoxResNet, CellSegNet_basic_lite_w_groupnorm_deep_supervised_II
from func.loss_func import dice_accuracy, dice_loss_II, dice_loss_II_weights, dice_loss_org_weights
from func.ultis import save_obj, load_obj
import torch.nn.functional as F

import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import time

# hyperparameters
# ----------
save_path = 'output/model_HMS_w_groupnorm.pkl'
need_resume = True
load_path = 'output/model_HMS_w_groupnorm.pkl'
learning_rate = 1e-4
max_epoch = 500
model_save_freq = 50
train_file_format = '.npy'
train_img_crop_size = (64, 64, 64)
boundary_importance = 1
batch_size = 5
num_workers = 4
# ----------
print(f"number of gpus: {torch.cuda.device_count()}")
torch.cuda.set_device(0)
print(f"current gpu: {torch.cuda.current_device()}")
# init model
model=CellSegNet_basic_lite_w_groupnorm_deep_supervised_II(input_channel=1, n_classes=3, output_func = "softmax")
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

for ith_epoch in range(0, max_epoch):
    for ith_batch, batch in enumerate(dataset_loader):

        img_input=batch['raw'].to(device)

        seg_groundtruth_f_64=torch.tensor(batch['foreground']>0, dtype=torch.float).to(device)
        seg_groundtruth_bb_64=torch.cat((torch.tensor(batch['background']>0, dtype=torch.float), \
            torch.tensor(batch['boundary']>0, dtype=torch.float)), dim=1).to(device)

        seg_groundtruth_f_32 = F.interpolate(seg_groundtruth_f_64, size=(32, 32, 32))
        seg_groundtruth_bb_32 = F.interpolate(seg_groundtruth_bb_64, size=(32, 32, 32))

        seg_groundtruth_f_16 = F.interpolate(seg_groundtruth_f_32, size=(16, 16, 16))
        seg_groundtruth_bb_16 = F.interpolate(seg_groundtruth_bb_32, size=(16, 16, 16))

        seg_groundtruth_f_8 = F.interpolate(seg_groundtruth_f_16, size=(8, 8, 8))
        seg_groundtruth_bb_8 = F.interpolate(seg_groundtruth_bb_16, size=(8, 8, 8))
        
        weights_f_64=batch['weights_foreground'].to(device)
        weights_bb_64=torch.cat((batch['weights_background'], batch['weights_boundary']), dim=1).to(device)

        weights_f_32 = F.interpolate(weights_f_64, size=(32, 32, 32))
        weights_bb_32 = F.interpolate(weights_bb_64, size=(32, 32, 32))

        weights_f_16 = F.interpolate(weights_f_32, size=(16, 16, 16))
        weights_bb_16 = F.interpolate(weights_bb_32, size=(16, 16, 16))

        weights_f_8 = F.interpolate(weights_f_8, size=(8, 8, 8))
        weights_bb_8 = F.interpolate(weights_bb_8, size=(8, 8, 8))
    
        seg_output_8, \
            seg_output_16, \
            seg_output_32,\
            seg_output_64 =model(img_input)

        seg_output_f_8=seg_output_8[:,2,:,:,:]
        seg_output_bb_8=torch.cat((seg_output_8[:,0,:,:,:], seg_output_8[:,1,:,:,:]), dim=1)

        seg_output_f_16 = seg_output_16[:, 2, :, :, :]
        seg_output_bb_16 = torch.cat((seg_output_16[:, 0, :, :, :], seg_output_16[:, 1, :, :, :]), dim=1)

        seg_output_f_32 = seg_output_32[:, 2, :, :, :]
        seg_output_bb_32 = torch.cat((seg_output_32[:, 0, :, :, :], seg_output_32[:, 1, :, :, :]), dim=1)

        seg_output_f_64 = seg_output_64[:, 2, :, :, :]
        seg_output_bb_64 = torch.cat((seg_output_64[:, 0, :, :, :], seg_output_64[:, 1, :, :, :]), dim=1)
        
        loss_64 = dice_loss_org_weights(seg_output_bb_64, seg_groundtruth_bb_64, weights_bb_64)+\
            dice_loss_II_weights(seg_output_f_64, seg_groundtruth_f_64, weights_f_64)
        accuracy = dice_accuracy(seg_output_f_64, seg_groundtruth_f_64)

        loss_32 = dice_loss_org_weights(seg_output_bb_32, seg_groundtruth_bb_32, weights_bb_32) + \
                  dice_loss_II_weights(seg_output_f_32, seg_groundtruth_f_32, weights_f_32)
        # accuracy_32 = dice_accuracy(seg_output_f_32, seg_groundtruth_f_32)

        loss_16 = dice_loss_org_weights(seg_output_bb_16, seg_groundtruth_bb_16, weights_bb_16) + \
                  dice_loss_II_weights(seg_output_f_16, seg_groundtruth_f_16, weights_f_16)
        # accuracy_16 = dice_accuracy(seg_output_f_16, seg_groundtruth_f_16)

        loss_8 = dice_loss_org_weights(seg_output_bb_8, seg_groundtruth_bb_8, weights_bb_8) + \
                  dice_loss_II_weights(seg_output_f_8, seg_groundtruth_f_8, weights_f_8)
        # accuracy_8 = dice_accuracy(seg_output_f_8, seg_groundtruth_f_8)

        loss = (loss_64 + loss_32 + loss_16 + loss_8) / 4
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        time_consumption = time.time() - start_time
        
        print(
            "epoch [{0}/{1}]\t"
            "batch [{2}]\t"
            "time(s) {time:.2f}\t"
            "loss {loss:.5f}\t"
            "l_64 {l_64:.5f}\t"
            "l_32 {l_32:.5f}\t"
            "l_16 {l_16:.5f}\t"
            "l_8 {l_8:.5f}\t"
            "acc {acc:.5f}\t".format(
                ith_epoch + 1,
                max_epoch,
                ith_batch,
                time = time_consumption,
                loss = loss.item(),
                l_64=loss_64.item(),
                l_32=loss_32.item(),
                l_16=loss_16.item(),
                l_8=loss_8.item(),
                acc = accuracy.item()))
    
    if (ith_epoch+1)%model_save_freq==0:
        print('epoch: '+str(ith_epoch+1)+' save model')
        model.to(torch.device('cpu'))
        torch.save({'model_state_dict': model.state_dict()}, f'output/model_HMS_w_groupnorm_batchsize5_deep_supervision_II.pkl')
        model.to(device)
