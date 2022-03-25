import math
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import SAGEConv, GCNConv

class SAGE_NET(torch.nn.Module):

    def __init__(self, in_feature, out_feature):
        super(SAGE_NET, self).__init__()
        self.gcn = SAGEConv(in_feature, out_feature)

    def forward(self, x, edge_index, active=True):

        x = self.gcn(x, edge_index)
        if active:
            x = F.relu(x)
        return x

class GCN_NET(torch.nn.Module):

    def __init__(self, in_feature, out_feature):
        super(GCN_NET, self).__init__()
        self.gcn = GCNConv(in_feature, out_feature)

    def forward(self, x, edge_index, active=True):

        x = self.gcn(x, edge_index)
        if active:
            x = F.relu(x)
        return x