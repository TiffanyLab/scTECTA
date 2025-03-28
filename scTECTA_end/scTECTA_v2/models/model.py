import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear
from .layers import GCNConv, GRL, DomainDiscriminator


class Model(torch.nn.Module):

    def __init__(self,in_size, hid_size, out_size, dropout_ratio):
        super(Model, self).__init__()
        self.conv1 = GCNConv(in_size, hid_size)
        self.conv2 = GCNConv(hid_size, hid_size)
        self.cls = GCNConv(hid_size,out_size)
        self.domain_discriminator = Linear(hid_size, 2)
        self.grl = GRL()
        self.dropout_ratio = dropout_ratio


    def forward(self, x, edge_index, conv_time=30):
        x = self.feat_bottleneck(x, edge_index, conv_time)
        x = self.feat_classifier(x, edge_index)

        return x

    def feat_bottleneck(self, x, edge_index, conv_time=30):
        x = self.conv1(x, edge_index, 0)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.conv2(x, edge_index, conv_time)
        x = F.relu(x)

        return x

    def feat_classifier(self, x, edge_index, conv_time=1):
        x = self.cls(x, edge_index, conv_time)

        return x

    def domain_classifier(self, x, edge_index=None, conv_time=1):
        h_grl = self.grl(x)
        d_logit = self.domain_discriminator(h_grl)

        return d_logit