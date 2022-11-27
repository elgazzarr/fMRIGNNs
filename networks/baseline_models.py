import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
from networks.utils import Temporal_Conv

MODELS = ['CNN_1D','MLP']

def get_model_class(model_name):
    """Return the dataset class with the given name."""
    if model_name not in globals():
        raise NotImplementedError("Model not found: {}".format(model_name))
    return globals()[model_name]

class CNN_1D(nn.Module):
    def __init__(self, nrois, f1, f2, dilation_exponential, k1, k2, dropout, readout):
        super(CNN_1D, self).__init__()
        self.readout = readout
        self.layer0 = Temporal_Conv(nrois, f1, k1, dilation_exponential, F.relu)
        self.layer1 = Temporal_Conv(f1, f2, k1, dilation_exponential, F.relu)
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.max = nn.AdaptiveMaxPool1d(1)
        dim = 2 if readout == 'meanmax' else 1
        self.drop = nn.Dropout(p=dropout)
        self.classify = nn.Linear(f2*dim, 2)

    def forward(self, data):
        x = torch.squeeze(data.t) # B, 1, T,C
        h0 = self.layer0(x)  # B,t,C
        h1 = self.layer1(h0)
        h_avg = torch.squeeze(self.avg(h1))
        h_max = torch.squeeze(self.max(h1))
        if self.readout == 'meanmax':
            h = torch.cat((h_avg, h_max),1)
        else:
            h = h_avg
        h = self.drop(h)
        hg = self.classify(h)

        return hg


class MLP(nn.Module):
    def __init__(self, nrois, f1, f2, **kwargs):
        super(MLP, self).__init__()
        in_dim = int(nrois * (nrois + 1) / 2)
        self.layer0 = Sequential(Linear(in_dim, f1), BatchNorm1d(f1), ReLU())
        self.layer1 = Sequential(Linear(f1, f2), BatchNorm1d(f2), ReLU())
        self.drop = nn.Dropout(p=0.5)
        self.classify = nn.Linear(f2, 2)

    def forward(self, data):
        x = data.x_flat
        h0 = self.layer0(x)
        h1 = self.layer1(h0)
        h = self.drop(h1)
        hg = self.classify(h)

        return hg
