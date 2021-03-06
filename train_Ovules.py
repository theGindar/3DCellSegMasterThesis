# train
from func.load_dataset import Cell_Seg_3D_Dataset_old
from func.network import VoxResNet, CellSegNet_basic_lite
from func.loss_func import dice_accuracy, dice_loss_II, dice_loss_II_weights, dice_loss_org_weights
from func.ultis import save_obj, load_obj

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import os
import time

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# hyperparameters
# ----------
save_path = 'output/model_Ovules_retrained_control_background_2.pkl'
need_resume = True
load_path = 'output/model_Ovules_retrained_control_background_2.pkl'
loss_save_path = 'output/loss_Ovules_retrained_control_background_2.pkl'
learning_rate = 1e-4
max_epoch = 1000
model_save_freq = 20
train_file_format = '.npz'
train_img_crop_size = (64, 64, 64)
boundary_importance = 1
batch_size = 3
num_workers = 4
# ----------
torch.manual_seed(0)
import random
random.seed(0)
np.random.seed(0)


print(f"number of gpus: {torch.cuda.device_count()}")
torch.cuda.set_device(0)
print(f"current gpu: {torch.cuda.current_device()}")

# init model
model=CellSegNet_basic_lite(input_channel=1, n_classes=3, output_func = "softmax")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

if need_resume and os.path.exists(load_path):
    print("resume model from "+str(load_path))
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

# optimizer
optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)

# dataset and dataloader
Ovules_data_dict = load_obj("dataset_info/Ovules_dataset_info")
dataset = Cell_Seg_3D_Dataset_old(Ovules_data_dict['train'])
dataset.set_para(file_format=train_file_format, \
    crop_size = train_img_crop_size, \
        boundary_importance = boundary_importance, \
            need_tensor_output = True, need_transform = True)
dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, \
    num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 1), prefetch_factor=1)
print('num of train files: '+str(len(Ovules_data_dict['train'].keys())))
print('max epoch: '+str(max_epoch))

start_time = time.time()

loss_df = pd.DataFrame({"epoch":[],
                        "batch": [],
                        "time": [],
                        "loss": [],
                        "accuracy": []})

for ith_epoch in range(0, max_epoch):
    for ith_batch, batch in enumerate(dataset_loader):

        img_input=batch['raw'].to(device)

        seg_groundtruth_f=torch.tensor(batch['foreground']>0, dtype=torch.float).to(device)
        seg_groundtruth_ba=torch.tensor(batch['background']>0, dtype=torch.float).to(device)
        seg_groundtruth_bo=torch.tensor(batch['boundary']>0, dtype=torch.float).to(device)

        weights_f=batch['weights_foreground'].to(device)
        weights_ba=batch['weights_background'].to(device)
        weights_bo=batch['weights_boundary'].to(device)
    
        seg_output=model(img_input)
        seg_output_f=seg_output[:,2,:,:,:]
        seg_output_ba=seg_output[:,0,:,:,:]
        seg_output_bo=seg_output[:,1,:,:,:]
        
        loss=10*dice_loss_org_weights(seg_output_bo, seg_groundtruth_bo, weights_bo)+\
            dice_loss_org_weights(seg_output_ba, seg_groundtruth_ba, weights_ba)+\
                dice_loss_II_weights(seg_output_f, seg_groundtruth_f, weights_f)

        # loss = dice_loss_org_weights(seg_output_bo, seg_groundtruth_bo, weights_bo) + \
        #          dice_loss_org_weights(seg_output_ba, seg_groundtruth_ba, weights_ba)+\
        #          dice_loss_II_weights(seg_output_f, seg_groundtruth_f, weights_f)

        accuracy=dice_accuracy(seg_output_bo, seg_groundtruth_bo)

        # debug stuff
        if accuracy == 0.:
            with open('debug_stuff/seg_output.npy', 'wb')as f:
                np.save(f, seg_output.cpu().detach().numpy())
            with open('debug_stuff/seg_groundtruth_f.npy', 'wb')as f:
                np.save(f, seg_groundtruth_f.cpu().detach().numpy())
            with open('debug_stuff/seg_groundtruth_ba.npy', 'wb')as f:
                np.save(f, seg_groundtruth_ba.cpu().detach().numpy())
            with open('debug_stuff/seg_groundtruth_bo.npy', 'wb')as f:
                np.save(f, seg_groundtruth_bo.cpu().detach().numpy())
            print("saved bad results")
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        time_consumption = time.time() - start_time
        
        print(
            "epoch [{0}/{1}]\t"
            "batch [{2}]\t"
            "time(s) {time:.2f}\t"
            "loss {loss:.5f}\t"
            "acc {acc:.5f}\t".format(
                ith_epoch + 1,
                max_epoch,
                ith_batch,
                time = time_consumption,
                loss = loss.item(),
                acc = accuracy.item()))

        loss_df = loss_df.append({"epoch": ith_epoch + 1,
                                  "batch": ith_batch,
                                  "time": time_consumption,
                                  "loss": loss.item(),
                                  "accuracy": accuracy.item()}, ignore_index=True)
    
    if (ith_epoch+1)%model_save_freq==0:
        print('epoch: '+str(ith_epoch+1)+' save model')
        model.to(torch.device('cpu'))
        torch.save({'model_state_dict': model.state_dict()}, save_path)
        model.to(device)
        loss_df.to_csv(loss_save_path)
