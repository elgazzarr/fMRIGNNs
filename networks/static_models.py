import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, GINConv,  global_mean_pool,global_max_pool, global_add_pool, TopKPooling
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
from networks.utils import graph_readout
from torch_scatter import scatter_mean


MODELS = ['GCN','GIN', 'GAT']

def get_model_class(model_name):
    """Return the dataset class with the given name."""
    if model_name not in globals():
        raise NotImplementedError("Model not found: {}".format(model_name))
    return globals()[model_name]


class Abstract_GNN(torch.nn.Module):
    """
    An Abstract class for all GNN models
    Subclasses should implement the following:
    - forward()
    - predict()
    """
    def __init__(self, num_nodes, f1, f2, readout):
        super(Abstract_GNN, self).__init__()
        self.readout = readout

    def _reset_parameters(self):
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
                else:
                    nn.init.uniform_(p)

    def forward(self,data):

        raise NotImplementedError


class GCN(Abstract_GNN):
    def __init__(self, num_nodes, f1, f2, readout, **kwargs):
        super().__init__(num_nodes, f1, f2, readout)
        self.readout = readout
        self.conv1 = GCNConv(num_nodes, f1)
        self.conv2 = GCNConv(f1, f2)

        last_dim = 2 if readout=='meanmax' else 1
        self.mlp = nn.Linear(f2*last_dim,2)
        self._reset_parameters()


    def forward(self, data):
        x, edge_index,batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        x = graph_readout(x, batch, self.readout)
        x = self.mlp(x)
        return x



class GIN(Abstract_GNN):
    def __init__(self, num_nodes, f1, f2, readout, extra_fc, **kwargs):
        super().__init__(num_nodes, f1, f2, readout)
        self.ll = extra_fc
        self.conv1 = GINConv(
            Sequential(Linear(num_nodes, f1), BatchNorm1d(f1), ReLU(),
                       Linear(f1, f1), ReLU()))

        self.conv2 = GINConv(
            Sequential(Linear(f1, f2), BatchNorm1d(f2), ReLU(),
                       Linear(f2, f2), ReLU()))

        last_dim = 2 if readout=='meanmax' else 1
        if self.ll:
            self.last = Sequential(Linear(f2*last_dim, f2*last_dim*2), ReLU(),
                                   Linear(f2*last_dim*2, 2))
        else:
            self.last = Linear(f2*last_dim, 2)

        self._reset_parameters()

    def forward(self, data):
        x, edge_index,batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = graph_readout(x, batch, self.readout)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.last(x)
        return x


class GAT(Abstract_GNN):
    def __init__(self, num_nodes, f1, f2, readout, num_heads, concat, **kwargs):
        super().__init__(num_nodes, f1, f2, readout)

        self.conv1 = GATv2Conv(num_nodes, f1, heads=num_heads, concat=concat)
        m = num_heads if concat else 1
        self.conv2 = GATv2Conv(f1*m, f2, heads=1)
        last_dim = 2 if readout=='meanmax' else 1
        self.mlp = nn.Linear(f2*last_dim,2)
        self._reset_parameters()


    def forward(self, data):
        x, edge_index,batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = graph_readout(x, batch, self.readout)
        x = self.mlp(x)
        return x
