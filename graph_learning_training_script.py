#%% md

## pipeline

#%%

import networkx as nx
import numpy as np
import torch

import torch.nn as nn

from statistics import mean

from func.run_pipeline_super_vox import get_outlayer_of_a_3d_shape, get_crop_by_pixel_val
from func.ultis import load_obj

from func.graph_learning import SuperVoxToNxGraph, VoxelGraphDataset


#%%

# load graphs
from func.ultis import load_obj

# graphs = load_obj("graphs_dataset_train")
graphs = load_obj("graphs_dataset_train_with_augmentations")

print(f"number of gpus: {torch.cuda.device_count()}")
torch.cuda.set_device(0)
print(f"current gpu: {torch.cuda.current_device()}")
#%%
import random
random.seed(0)
random.shuffle(graphs)

#%%

dataset = VoxelGraphDataset(graphs)

g = dataset[0]


#%% md

# TODO probably should normalize features!!!!

#%%

from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler

num_examples = len(dataset)
num_train = int(num_examples)

train_sampler = SubsetRandomSampler(torch.arange(num_train))
test_sampler = SubsetRandomSampler(torch.arange(num_train, num_examples))

train_dataloader = GraphDataLoader(
    dataset, sampler=train_sampler, batch_size=5, drop_last=False)

#%%

from func.graph_models import GCN, GCN_2
import torch.nn.functional as F
import random

model = GCN(3, num_classes=2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
CELoss = nn.CrossEntropyLoss()

# training
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
best_val_acc = 0



# features = g.ndata['feat']
# labels = g.ndata['label']
# train_mask = g.ndata['train_mask']
# val_mask = g.ndata['val_mask']

# calculate weights for loss
"""
pos_weights = []
neg_weights = []
for graph_number in range(len(dataset)):
    sample_graph = dataset[graph_number]
    labels = sample_graph.ndata['label']
    # create class weights
    number_positives = torch.count_nonzero(labels)
    positive_weight = 1 - (number_positives / len(labels))
    negative_weight = 1 - positive_weight

    pos_weights.append(positive_weight.item())
    neg_weights.append(negative_weight.item())
weights = torch.tensor([mean(neg_weights), mean(pos_weights)])
print(f"weights: {weights}")
"""
from torchmetrics import F1Score

f1 = F1Score(num_classes=2, average='weighted')

epoch_loss = []
epoch_val_loss = []
epoch_accuracy = []

epoch_f1score = []
epoch_f1score_val = []

epoch_accuracy_val = []
# best_val_acc = 0
best_val_loss = 1000

for e in range(500):
    # get random elements for batch
    #graphs_numbers_list = range(0, len(dataset))
    #rand_graph_numbers = random.sample(graphs_numbers_list, len(dataset))
    for graph_number in range(len(dataset)):
    #for graph_number in range(1):
    #for graph_number in rand_graph_numbers:
        # Forward
        model.train()
        sample_graph = dataset[graph_number].to(device)
        features = sample_graph.ndata['feat'].to(device)
        labels = sample_graph.ndata['label'].to(device)
        train_mask = sample_graph.ndata['train_mask'].to(device)
        val_mask = sample_graph.ndata['val_mask'].to(device)

        # create class weights
        number_positives = torch.count_nonzero(labels)
        percentage_positives = number_positives / len(labels)
        percentage_negatives = 1 - percentage_positives

        weights = torch.tensor([1 - percentage_negatives, 1 - percentage_positives]).to(device)
        #weights = torch.tensor([0.95, 0.05])
        #print(weights)

        CELoss = nn.CrossEntropyLoss(weight=weights)
        train_mask = sample_graph.ndata['train_mask']
        val_mask = sample_graph.ndata['val_mask']
        logits = model(sample_graph, features)

        # Compute prediction
        pred = logits.argmax(1)

        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set.
        loss = CELoss(logits[train_mask], labels[train_mask])



        epoch_loss.append(loss.item())

        #print(loss)
        # Compute accuracy on training/validation/test
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()

        train_f1_score = f1(pred[train_mask], labels[train_mask])
        val_f1_score = f1(pred[val_mask], labels[val_mask])


        epoch_accuracy.append(train_acc.item())
        epoch_accuracy_val.append(val_acc.item())

        epoch_f1score.append(train_f1_score.item())
        epoch_f1score_val.append(val_f1_score.item())


        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(sample_graph, features)
            val_loss = CELoss(logits[val_mask], labels[val_mask])
            epoch_val_loss.append(val_loss.item())
        model.train()

    if e % 5 == 0:
        print('In epoch {}, loss: {:.5f}, val loss: {:.5f}, accuracy: {:.3f}, val accuracy: {:.3f}, f1score: {:.3f}, val f1score: {:.3f}'.format(
            e, mean(epoch_loss), mean(epoch_val_loss), mean(epoch_accuracy), mean(epoch_accuracy_val), mean(epoch_f1score), mean(epoch_f1score_val)))

        #if mean(epoch_accuracy_val) >= best_val_acc:
        if mean(epoch_val_loss) <= best_val_loss:
            print("new best val loss")
            torch.save(model.state_dict(), "output/graph_model.pt")
            best_val_loss = mean(epoch_val_loss)
        epoch_loss = []
        epoch_val_loss = []
        epoch_accuracy = []

        epoch_accuracy_val = []
        epoch_f1score_val = []
