import bz2

import dgl
import torch

import torch.nn as nn

import gc
import pickle5 as pkl

from statistics import mean

from func.graph_learning import VoxelGraphDataset

# load graphs
from func.ultis import load_obj

# load graphs
def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = pkl.load(data)
    return data

def flatten(l):
    return [item for sublist in l for item in sublist]

data_path = "../../../mnt2/"
graph_paths = [
    "graphs_dataset_train_LRP_retrained_SMALL_1_edge_gated.pbz2",
    "graphs_dataset_train_LRP_retrained_SMALL_2_edge_gated.pbz2",
    "graphs_dataset_train_LRP_retrained_SMALL_3_edge_gated.pbz2",
    "graphs_dataset_train_LRP_retrained_SMALL_4_edge_gated.pbz2",
    "graphs_dataset_train_LRP_retrained_SMALL_5_edge_gated.pbz2",
    "graphs_dataset_train_LRP_retrained_SMALL_6_edge_gated.pbz2"
]

graphs = []
for g in graph_paths:
    subset_graphs = decompress_pickle(data_path + g)
    graphs.append(subset_graphs)

graphs = flatten(graphs)

model_save_path = "output/graph_model_focal_LRP_edge_gated.pt"

import random
random.seed(0)
dgl.seed(0)
torch.manual_seed(0)

print(f"number of gpus: {torch.cuda.device_count()}")
torch.cuda.set_device(0)
print(f"current gpu: {torch.cuda.current_device()}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

random.shuffle(graphs)

# remove too big graph
# del graphs[39]

dataset = VoxelGraphDataset(graphs, with_edge_weights=True)

g = dataset[0]

num_examples = len(dataset)
num_train = int(num_examples)

from func.graph_models import GCN

model = GCN(3, num_classes=1)
model.to(device)
CELoss = nn.CrossEntropyLoss()

# training
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
best_val_acc = 0

from torchmetrics import F1Score
from torchvision.ops import sigmoid_focal_loss

f1 = F1Score(num_classes=1, average='weighted').to(device)

epoch_loss = []
epoch_val_loss = []
epoch_accuracy = []

epoch_f1score = []
epoch_f1score_val = []

epoch_accuracy_val = []
# best_val_acc = 0
best_val_loss = 1000

print("ready for training...")
# print('{:.1f} MiB'.format(torch.cuda.max_memory_allocated() / 1000000))



for e in range(500):
    alpha = 0.5
    for graph_number in range(len(dataset)):
        # torch.cuda.empty_cache()
        # print("memory at start:")
        # print('{:.1f} MiB'.format(torch.cuda.memory_allocated() / 1000000))
        # Forward
        model.train()
        sample_graph = dataset[graph_number].to(device)
        features = sample_graph.ndata['feat'].to(device)
        labels = sample_graph.ndata['label'].to(device)
        train_mask = sample_graph.ndata['train_mask'] # .to(device)
        val_mask = sample_graph.ndata['val_mask']# .to(device)

        # create class weights
        number_positives = torch.count_nonzero(labels)
        percentage_positives = number_positives / len(labels)
        percentage_negatives = 1 - percentage_positives

        # CELoss = nn.CrossEntropyLoss(weight=weights)
        train_mask = sample_graph.ndata['train_mask'].to(device)
        val_mask = sample_graph.ndata['val_mask'].to(device)
        logits = model(sample_graph, features)
        model_output = torch.sigmoid(logits)

        # Compute prediction
        pred = (model_output > 0.5).type(torch.FloatTensor).to(device)
        loss = sigmoid_focal_loss(torch.squeeze(logits[train_mask].type(torch.FloatTensor)), labels[train_mask].type(torch.FloatTensor), alpha=alpha, reduction="mean")



        epoch_loss.append(loss.item())

        pred_train = pred[train_mask] # .to('cpu')
        pred_val = pred[val_mask] # .to('cpu')
        labels_train = labels[train_mask] # .to('cpu')
        labels_val = labels[val_mask] # .to('cpu')


        # print(f"pred_train shape: {pred_train.shape}")
        # print(f"labels_train shape: {labels_train.shape}")
        train_acc = (pred_train == labels_train).float().mean()
        # train_acc = torch.tensor([1.])
        val_acc = (pred_val == labels_val).float().mean()

        train_f1_score = f1(pred_train, labels_train)
        val_f1_score = f1(pred_val, labels_val)

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
            # val_loss = CELoss(logits[val_mask], labels[val_mask])
            val_loss = sigmoid_focal_loss(torch.squeeze(logits[val_mask].type(torch.FloatTensor)), labels[val_mask].type(torch.FloatTensor), alpha=alpha, reduction="mean")
            epoch_val_loss.append(val_loss.item())
        model.train()

        torch.cuda.reset_max_memory_allocated()
        """
        del pred
        del labels
        del pred_train
        del pred_val
        del labels_train
        del labels_val
        del sample_graph
        """

        # torch.cuda.empty_cache()
        # torch.cuda.reset_max_memory_allocated()
        # print('{:.1f} MiB'.format(torch.cuda.memory_allocated() / 1000000))


    if e % 5 == 0:
        print('In epoch {}, loss: {:.5f}, val loss: {:.5f}, accuracy: {:.3f}, val accuracy: {:.3f}, f1score: {:.3f}, val f1score: {:.3f}'.format(
            e, mean(epoch_loss), mean(epoch_val_loss), mean(epoch_accuracy), mean(epoch_accuracy_val), mean(epoch_f1score), mean(epoch_f1score_val)))

        #if mean(epoch_accuracy_val) >= best_val_acc:
        if mean(epoch_val_loss) <= best_val_loss:
            print("new best val loss")
            torch.save(model.state_dict(), model_save_path)
            best_val_loss = mean(epoch_val_loss)
        epoch_loss = []
        epoch_val_loss = []
        epoch_accuracy = []

        epoch_accuracy_val = []
        epoch_f1score_val = []



