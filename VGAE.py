import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import GraphConvolution


class fGCNModelVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(fGCNModelVAE, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1)  # GraphConvolution (1000->256)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2)  # GraphConvolution (256->256)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2)  # GraphConvolution (256->256)
        self.dropout = dropout  # 0
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x, adj):
        hidden1 = F.relu(self.gc1(x, adj))
        hidden1 = F.dropout(hidden1, self.dropout, training=self.training)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)  # (3611,256)
        z = self.reparameterize(mu, logvar)  # (3611,256)
        pred = self.dc(z)  # (3611,3611)
        return pred, mu, logvar


class sGCNModelVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(sGCNModelVAE, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1)  # GraphConvolution (1000->256)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2)  # GraphConvolution (256->256)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2)  ##GraphConvolution (256->256)
        self.dropout = dropout
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x, adj):
        hidden1 = F.relu(self.gc1(x, adj))
        hidden1 = F.dropout(hidden1, self.dropout, training=self.training)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        pred = self.dc(z)
        return pred, mu, logvar


class InnerProductDecoder(nn.Module):
    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj
