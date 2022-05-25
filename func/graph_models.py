from dgl.nn.pytorch.conv import GraphConv
import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn
from dgl.nn.pytorch.conv import gatconv

class GCN(nn.Module):
    def __init__(self, in_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, 4)
        self.dropout = nn.Dropout(p=0.5)
        self.conv2 = GraphConv(4, num_classes)
        # self.activation = nn.Softmax()

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.conv2(g, h)
        # h = self.activation(h)
        return h


class GCN_1(nn.Module):
    def __init__(self, in_feats, num_classes):
        super(GCN_1, self).__init__()
        self.conv1 = GraphConv(in_feats, 8)
        self.dropout1 = nn.Dropout(p=0.5)
        self.conv2 = GraphConv(8, 8)
        self.linear1 = nn.Linear(8, num_classes)
        # self.activation = nn.Softmax()

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.dropout1(h)
        h = self.conv2(g, h)
        h = F.relu(h)
        h = self.linear1(h)
        h = F.relu(h)
        return h


class GCN_2(nn.Module):
    def __init__(self, in_feats, num_classes):
        super(GCN_2, self).__init__()
        self.conv1 = GraphConv(in_feats, num_classes)
        # self.activation = nn.Softmax()

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        return h


class GAT(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        if num_layers > 1:
        # input projection (no residual)
            self.gat_layers.append(gatconv(
                in_dim, num_hidden, heads[0],
                feat_drop, attn_drop, negative_slope, False, self.activation))
            # hidden layers
            for l in range(1, num_layers-1):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.gat_layers.append(gatconv(
                    num_hidden * heads[l-1], num_hidden, heads[l],
                    feat_drop, attn_drop, negative_slope, residual, self.activation))
            # output projection
            self.gat_layers.append(gatconv(
                num_hidden * heads[-2], num_classes, heads[-1],
                feat_drop, attn_drop, negative_slope, residual, None))
        else:
            self.gat_layers.append(gatconv(
                in_dim, num_classes, heads[0],
                feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h)
            h = h.flatten(1) if l != self.num_layers - 1 else h.mean(1)
        return h
