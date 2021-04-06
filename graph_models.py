import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from GraphLayers import GraphLayer
# Graph Neural Networks
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.ReLU = nn.LeakyReLU()

    def forward(self, x):
        x = self.ReLU(self.fc1(x))
        x = self.ReLU(self.fc2(x))
        x = self.ReLU(self.fc3(x))
        return x  # need revision


class GATConv(nn.Module):
    def __init__(self, in_features, out_features, num_head, slope_alpha=0.2, bias=True):
        super(GATConv, self).__init__()
        #self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.slope_alpha = slope_alpha
        self.num_head = num_head
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        self.LeakyReLU = nn.LeakyReLU(self.slope_alpha)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        if self.bias is not None:
            self.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, x, adj): # x refers to selected features
        ls = []
        for i in range(adj.shape[0]):
            index = np.ndarray.item(np.array(np.nonzero(adj[i, :])))
            ls.append(x[index, :])
        select_feature = torch.stack(ls, dim=0)  # select self and neighboring agents' features
        # suppose K neighbors and features size L, select_feature should have dimension of (K+1)*L

        out = []
        for i in range(self.num_head):
            neighbor_dim = select_feature.shape[0]
            select_feature = F.dropout(select_feature, self.dropout, training=self.training)  # Is it necessary?
            h = torch.matmul(select_feature, self.weight)  # is it necessary after MLP dimension shrink?
            coef_nomi = torch.zeros(neighbor_dim - 1)
            for j in range(1, neighbor_dim):
                hij = torch.cat((h[0, :], h[j, :]), dim=1)
                coef_nomi[j - 1] = torch.exp(self.LeakyReLU(torch.matmul(hij, self.a)))
            coef_deno = torch.sum(coef_nomi)
            att = torch.zeros(neighbor_dim - 1)
            for j in range(1, neighbor_dim):
                alpha = torch.div(coef_nomi[j - 1], coef_deno)
                att[j - 1] = torch.matmul(alpha, h[j, :])
            h_prime = torch.sum(att) # do we need nonlinear operator/activation function?
            # in the last layer we need average pooling operation and where to put it???
            if self.bias is not None:
                h_prime = h_prime + self.bias
            out.append(h_prime)
        out = torch.cat(out, dim=1)
        return out  # dimension 1 * (self.num_head * out_features)


#####################################
#Transition Block will be as same as the graph convolution layer#
#how to constrain the number of feature maps#
#####################################


class BottleneckLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BottleneckLayer, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.LeakyReLU(inplace=True)
        self.bn = nn.BatchNorm1d(in_channels)
        #Is bn and relu really necessary in this case?

    def forward(self, x):
        out = self.conv1(self.relu(self.bn(x)))
        return torch.cat([x, out], 1) # Not sure if this is necessary and needs revision

############
# bottleneck layer should come with avg pooling layer



class GDB(nn.Module):
    def __init__(self, num_layers, input_size, growth_rate, block, dropRate=0.0):
        super(GDB, self).__init__()
        # self.MLP = MLP()  # need revision
        #self.gc = GATConv()  # need revision
        self.layer = self._make_layer(block, input_size, growth_rate, num_layers, dropRate)
        #self.dropout = dropout
        #self.ReLU = nn.LeakyReLU()

    def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes + i * growth_rate, growth_rate, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x) # channel dimension?


class GraphDensenet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, depth, growth_rate, reduction,
                 bottleneck=True, dropRate=0.0, num_head=3, slope_alpha=0.2, bias=True):
        super(GraphDensenet, self).__init__()
        self.MLP = MLP(input_size, hidden_size, output_size)
        in_channels = 2 * growth_rate
        n = (depth - 4) / 3 # need revision
        if bottleneck == True:
            n = n/2
            block = BottleneckLayer
        else:
            block = GATConv
        n = int(n)

        # 1st block
        self.block1 = GDB(n, in_channels, growth_rate, block, dropRate)
        in_channels = int(in_channels + n*growth_rate)
        # 1st transition block
        out_channels = int(math.floor(in_channels*reduction))
        self.trans1 = GATConv(in_channels, out_channels, num_head, slope_alpha, bias)
        in_channels = int(math.floor(in_channels*reduction))

        # 2nd block
        self.block2 = GDB(n, in_channels, growth_rate, block, dropRate)
        in_channels = int(in_channels + n*growth_rate)
        # 2nd transition block
        out_channels = int(math.floor(in_channels*reduction))
        self.trans2 = GATConv(in_channels, out_channels, num_head, slope_alpha, bias)
        in_channels = int(math.floor(in_channels*reduction))

        # 3rd block
        self.block3 = GDB(n, in_channels, growth_rate, block, dropRate)
        in_channels = int(in_channels + n*growth_rate)

        #global normalization
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

        self.in_channels = in_channels


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def forward(self, x):
        out = self.MLP(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.block3(out)
        # out = self.relu(self.bn1(out))
        # is avg_pooling necessary?
        return out # out is the output latent input state fed into DDPG network

