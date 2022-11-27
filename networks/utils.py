import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, GINConv,  global_mean_pool,global_max_pool, global_add_pool
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
from torch_geometric.nn.inits import glorot, zeros
from collections import OrderedDict

class Sparsemax(nn.Module):
    """Sparsemax function."""

    def __init__(self, device='cuda:1', dim=None ):
        """Initialize sparsemax activation

        Args:
            dim (int, optional): The dimension over which to apply the sparsemax function.
        """
        super(Sparsemax, self).__init__()

        self.dim = -1 if dim is None else dim
        self.device = device

    def forward(self, input):
        """Forward function.

        Args:
            input (torch.Tensor): Input tensor. First dimension should be the batch size

        Returns:
            torch.Tensor: [batch_size x number_of_logits] Output tensor

        """
        # Sparsemax currently only handles 2-dim tensors,
        # so we reshape to a convenient shape and reshape back after sparsemax
        input = input.transpose(0, self.dim)
        original_size = input.size()
        input = input.reshape(input.size(0), -1)
        input = input.transpose(0, 1)
        dim = 1

        number_of_logits = input.size(dim)

        # Translate input by max for numerical stability
        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)

        # Sort input in descending order.
        # (NOTE: Can be replaced with linear time selection method described here:
        # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html)
        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range = torch.arange(start=1, end=number_of_logits + 1, step=1, device=self.device, dtype=input.dtype).view(1, -1)
        range = range.expand_as(zs)

        # Determine sparsity of projection
        bound = 1 + range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
        k = torch.max(is_gt * range, dim, keepdim=True)[0]

        # Compute threshold function
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)

        # Sparsemax
        self.output = torch.max(torch.zeros_like(input), input - taus)

        # Reshape back to original shape
        output = self.output
        output = output.transpose(0, 1)
        output = output.reshape(original_size)
        output = output.transpose(0, self.dim)

        return output

    def backward(self, grad_output):
        """Backward function."""
        dim = 1

        nonzeros = torch.ne(self.output, 0)
        sum = torch.sum(grad_output * nonzeros, dim=dim) / torch.sum(nonzeros, dim=dim)
        self.grad_input = nonzeros * (grad_output - sum.expand_as(grad_output))

        return self.grad_input




class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()
    def forward(self, x, target, smoothing=0.1):
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()


def graph_readout(x,batch,  method):

    if method == 'mean':
        return global_mean_pool(x,batch)

    elif method == 'meanmax':
        x_mean = global_mean_pool(x,batch)
        x_max = global_max_pool(x,batch)
        return torch.cat((x_mean, x_max),1)

    elif method == 'sum':
        return global_add_pool(x,batch)

    else:
        raise ValueError('Undefined readout opertaion')




class STGNNLayer(nn.Module):
    """ An implementation of the STGNN layer.

    Args:
        dilation_exponential (int): Dilation exponential.
        j (int): Iteration index.
        conv_channels (int): Convolution channels.
        kernel_set (list of int): List of kernel sizes.
        new_dilation (int): Dilation.
        gcn_true (bool): Whether to add graph convolution layer.
        dropout (float): Droupout rate.
        gcn_depth (int): Graph convolution depth.
        num_nodes (int): Number of nodes in the graph.
        readout (str): readout function to summarize graph

    """

    def __init__(
        self,
        dilation_exponential: int,
        j: int,
        conv_channels: int,
        kernel_set: list,
        new_dilation: int,
        gcn_true: bool,
        dropout: float,
        gcn_depth: int,
        num_nodes: int,
        readout: str,
    ):
        super(STGNNLayer, self).__init__()
        self._dropout = dropout
        self._readout = readout
        in_channels = 1 if j==1 else conv_channels
        self._gcn_true = gcn_true
        self._filter_conv = DilatedInception(
            in_channels,
            conv_channels,
            kernel_set=kernel_set,
            dilation_factor=new_dilation)

        self._gate_conv = DilatedInception(
            in_channels,
            conv_channels,
            kernel_set=kernel_set,
            dilation_factor=new_dilation)

        if self._gcn_true:
            self._graph_conv = Diff_GCN(conv_channels, conv_channels, gcn_depth, dropout, support_len=1)


        self._batchnorm = nn.BatchNorm2d(conv_channels)
        if self._readout == 'conv':
            self.end_conv = nn.Conv2d(in_channels=conv_channels,
                                              out_channels=1,
                                              kernel_size=(1,1),
                                              bias=True)
        else:
            self._graph_mean_pool = nn.AdaptiveAvgPool2d((conv_channels,1))
            self._graph_max_pool = nn.AdaptiveMaxPool2d((conv_channels, 1))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(
        self,
        X: torch.FloatTensor,
        A_list: torch.FloatTensor,
        training: bool,
        ) -> torch.FloatTensor:
        """
        Making a forward pass of STGCN layer.

        Arg types:
            * **X** (PyTorch FloatTensor) - Input feature tensor,
                with shape (batch_size, in_dim, num_nodes, seq_len).
            * **A_list**  - list of adjacency matricies to be used.
            * **training** (bool) - Whether in traning mode.

        Return types:
            * **X** (PyTorch FloatTensor) - Output sequence tensor,
                with shape (batch_size, in_dim, num_nodes, seq_len).
            * **X_skip** (PyTorch FloatTensor) - Output feature tensor for skip connection,
                with shape (batch_size, in_dim, num_nodes, seq_len).
        """

        X_filter = self._filter_conv(X)
        X_filter = torch.tanh(X_filter)
        X_gate = self._gate_conv(X)
        X_gate = torch.sigmoid(X_gate)
        X = X_filter * X_gate
        X = F.dropout(X, self._dropout, training=training)

        if self._gcn_true:
            X = self._graph_conv(X, A_list)

        X = self._batchnorm(X)

        if self._readout == 'conv':
            X_conv = F.relu(self.end_conv(X))
            X_pool = torch.mean(X_conv,-1)
            X_skip = torch.flatten(X_pool,start_dim=1)
        else:
            X_pool = torch.mean(X,-1) # Temporal pooling, output shape (B,C,N)  # Reconsider temporal pooling after mean and max
            X_mean = self._graph_mean_pool(X_pool) # Mean across Nodes
            X_max = self._graph_max_pool(X_pool)  # Max across Nodes
            X_skip = torch.cat((X_mean, X_max),1) # Concatenate Mean and Max
            X_skip = torch.squeeze(X_skip,-1)

        return X, X_skip


class DilatedInception(nn.Module):
    r"""An implementation of the dilated inception layer.
    For details see this paper: `"Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks."
    <https://arxiv.org/pdf/2005.11650.pdf>`_

    Args:
        c_in (int): Number of input channels.
        c_out (int): Number of output channels.
        kernel_set (list of int): List of kernel sizes.
        dilated_factor (int, optional): Dilation factor.
        pool (bol): wether to add pooling for downsampling.

    """

    def __init__(self, c_in: int, c_out: int, kernel_set: list, dilation_factor: int):
        super(DilatedInception, self).__init__()
        self._time_conv = nn.ModuleList()
        self._kernel_set = kernel_set
        c_out = int(c_out / len(self._kernel_set))
        for kern in self._kernel_set:
            self._time_conv.append(
                nn.Conv2d(c_in, c_out, (1, kern), dilation=(1, dilation_factor))
            )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, X_in: torch.FloatTensor) -> torch.FloatTensor:
        """
        Making a forward pass of dilated inception.

        Arg types:
            * **X_in** (Pytorch Float Tensor) - Input feature Tensor, with shape (batch_size, c_in, num_nodes, seq_len).

        Return types:
            * **X** (PyTorch Float Tensor) - Hidden representation for all nodes,
            with shape (batch_size, c_out, num_nodes, seq_len-6).
        """
        X = []
        for i in range(len(self._kernel_set)):
            X.append(self._time_conv[i](X_in))
        for i in range(len(self._kernel_set)):
            X[i] = X[i][..., -X[-1].size(3) :]
        X = torch.cat(X, dim=1)
        return X

class GraphConstructor(nn.Module):
    r"""An implementation of the graph learning layer to construct an adjacency matrix.
    For details see this paper: `"Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks."
    <https://arxiv.org/pdf/2005.11650.pdf>`_

    Args:
        nnodes (int): Number of nodes in the graph.
        k (int): Number of largest values to consider in constructing the neighbourhood of a node (pick the "nearest" k nodes).
        dim (int): Dimension of the node embedding.
    """

    def __init__(self, nnodes: int, dim: int):
        super(GraphConstructor, self).__init__()
        self.E = torch.nn.Linear(dim, nnodes)
        #self.E = nn.Parameter(torch.randn(nnodes, dim), requires_grad=True)
        self._reset_parameters()
        #print(self.E.shape)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self) -> torch.FloatTensor:
        """
        Making a forward pass to construct an adjacency matrix from node embeddings.

        Return types:
            * **A** (PyTorch Float Tensor) - Adjacency matrix constructed from node embeddings.
        """
        sparse =  Sparsemax(dim=1)
        A = sparse(F.relu(torch.mm(self.E.weight, self.E.weight.transpose(0, 1))))

        return A

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class Diff_GCN(nn.Module):

    """
        Implements Graph Convolution via diffusion process through K order steps.

        Args:
        c_in (int): Number of input channels in each node.
        c_out (int): Desired number of output channels in the next layer after aggregating K order neigbours and applying linear transformation
        dropout (float): droput probability.
        support_len (int): number of adjacency matricies, default = 1 (adaptive)
        order (int): K diffusion steps (k-order neigbourhood to propagate features from), default = 2
    """

    def __init__(self,c_in,c_out, order, dropout, support_len):
        super(Diff_GCN,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h



class Temporal_Conv(nn.Module):
    """Update the node feature tv with applying Temporal Convolutions to get tv+1."""
    'used for CNN_1D model not graphs'

    def __init__(self, in_channels, filters, k, d, activation, causal=False):
        super(Temporal_Conv, self).__init__()
        self.causal = causal

        if causal:
            p = (k - 1) * d
        else:
            p = 0

        self.conv1 = nn.Conv1d(in_channels, filters, k, dilation=d, padding=p)
        self.activation = activation
        self.bn = nn.BatchNorm1d(filters)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        h = self.conv1(x)
        if self.causal:
            h = h[:, :, :-self.conv1.padding[0]]
        h = self.bn(h)
        h = self.activation(h)
        h = self.pool(h)

        return h
