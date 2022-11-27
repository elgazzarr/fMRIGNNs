from __future__ import division

import numbers
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn import global_max_pool, global_mean_pool
from networks.utils import STGNNLayer, GraphConstructor, Sparsemax

MODELS = ['ASTGCN','STGCN']

def get_model_class(model_name):
    """Return the dataset class with the given name."""
    if model_name not in globals():
        raise NotImplementedError("Model not found: {}".format(model_name))
    return globals()[model_name]

class ASTGCN(nn.Module):
    """"
    Args:
        num_nodes (int): Number of nodes in the graph.
        kernel_set (list of int): List of kernel sizes.
        dropout (float): Droupout rate.
        dilation_exponential (int): Dilation exponential.
        conv_channels (int): Convolution channels. (Conv, Res)
        in_dim (int): Input dimension.
        num_classes (int): Number of classes.
        layers (int): Number of layers/blocks.
        embeddings_dim (int): dimension of embedding to construct adjacency matrix
        gcn (int): location of graph convolution layers, if -1 add only at last, if 1 add only at first layer otherwise >2 add at all _layers
        readout (str): How to do graph readout (sum, mean/max)
        gcn_depth (int): Number of k hop propagations at each graph convolution layer.
        adjacency (numpy): Predefined adjacency matrix, default = None

    """

    def __init__(
        self,
        num_nodes: int,
        kernel_set: list,
        dropout: float,
        dilation_exponential: int,
        conv_channels: int,
        in_dim: int,
        num_classes: int,
        layers: int,
        embeddings_dim: int,
        gcn: int,
        readout: str,
        gcn_depth: int,
        adjacency: float,

    ):
        super(ASTGCN, self).__init__()

        self._num_nodes = num_nodes
        self._dropout = dropout
        self._layers = layers
        self._num_classes = num_classes

        if readout == 'conv':
             output_shape = num_nodes
        else:
             output_shape = conv_channels * 2

        self._stgnn_layers = nn.ModuleList()
        self._graph_constructor = GraphConstructor(num_nodes, dim=embeddings_dim) # k,dim are hyperparameters

        new_dilation = 1
        for j in range(1, layers + 1):
            # Check where to add gcn layers if any
            if (j == gcn or gcn >1):
                add_gcn = True
            elif (j==layers and gcn == -1 ):
                add_gcn = True
            else:
                add_gcn = False

            self._stgnn_layers.append(
                STGNNLayer(
                    dilation_exponential=dilation_exponential,
                    j=j,
                    conv_channels=conv_channels,
                    kernel_set=kernel_set,
                    new_dilation=new_dilation,
                    dropout=dropout,
                    num_nodes=num_nodes,
                    gcn_true=add_gcn,
                    gcn_depth=gcn_depth,
                    readout=readout,
                )
            )

            new_dilation *= dilation_exponential
        self._class_layer = nn.Linear(output_shape, num_classes)




    def forward(self, data):

            A = self._graph_constructor()
            X_s = []
            X = data.t
            for stgnn in self._stgnn_layers:
                    X, X_skip = stgnn(X, [A], True)
                    X_s.append(X_skip)

            X_final =  X_s.pop()
            X = self._class_layer(X_final)
            return X


class STGCN(nn.Module):
    """"
    Args:
        num_nodes (int): Number of nodes in the graph.
        kernel_set (list of int): List of kernel sizes.
        dropout (float): Droupout rate.
        dilation_exponential (int): Dilation exponential.
        conv_channels (int): Convolution channels. (Conv, Res)
        in_dim (int): Input dimension.
        num_classes (int): Number of classes.
        layers (int): Number of layers/blocks.
        embeddings_dim (int): dimension of embedding to construct adjacency matrix, Should be None in ST-GCN
        gcn (int): location of graph convolution layers, if -1 add only at last, if 1 add only at first layer otherwise >2 add at all _layers
        readout (str): How to do graph readout (sum, mean/max)
        gcn_depth (int): Number of k hop propagations at each graph convolution layer.
        adjacency (numpy): Predefined adjacency matrix

    """

    def __init__(
        self,
        num_nodes: int,
        kernel_set: list,
        dropout: float,
        dilation_exponential: int,
        conv_channels: int,
        in_dim: int,
        num_classes: int,
        layers: int,
        gcn: int,
        readout: str,
        gcn_depth: int,
        embeddings_dim: int,
        adjacency: float,
    ):
        super(STGCN, self).__init__()

        self._num_nodes = num_nodes
        self._dropout = dropout
        self._layers = layers
        self._num_classes = num_classes
        self.adjs = smax(torch.tensor(adjacency).float().to('cuda:1'))

        if readout == 'conv':
             output_shape = num_nodes
        else:
             output_shape = conv_channels * 2

        self._stgnn_layers = nn.ModuleList()

        new_dilation = 1
        for j in range(1, layers + 1):
            # Check where to add gcn layers if any
            if (j == gcn or gcn >1):
                add_gcn = True
            elif (j==layers and gcn == -1 ):
                add_gcn = True
            else:
                add_gcn = False

            self._stgnn_layers.append(
                STGNNLayer(
                    dilation_exponential=dilation_exponential,
                    j=j,
                    conv_channels=conv_channels,
                    kernel_set=kernel_set,
                    new_dilation=new_dilation,
                    dropout=dropout,
                    num_nodes=num_nodes,
                    gcn_true=add_gcn,
                    gcn_depth=gcn_depth,
                    readout=readout,
                )
            )

            new_dilation *= dilation_exponential
        self._class_layer = nn.Linear(output_shape, num_classes)




    def forward(self, data):
            A =  self.adjs

            X_s = []
            X = data.t
            for stgnn in self._stgnn_layers:
                    X, X_skip = stgnn(X, [A], True)
                    X_s.append(X_skip)

            X_final = X_s.pop()
            X = self._class_layer(X_final)
            return X
