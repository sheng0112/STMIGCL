import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution


class GCN_Encoder(nn.Module):
    def __init__(self, nfeat, nhid, nemb, dropout):
        super(GCN_Encoder, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)  # GraphConvolution (1000->256)
        self.gc2 = GraphConvolution(nhid, nemb)  # GraphConvolution (256->256)
        self.dropout = dropout  # 0.5

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)  # (3611.256)
        return x


class Linear_Classifier(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(Linear_Classifier, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)  # (256, 7)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return F.log_softmax(ret, dim=1)


class Attention_Emb(nn.Module):
    def __init__(self, in_size):
        super(Attention_Emb, self).__init__()
        self.project = nn.Linear(in_size, 1, bias=False)

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1)
