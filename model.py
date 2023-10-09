import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution


class AEncoder(nn.Module):
    def __init__(self,
                 input_dim,
                 linear_encoder_hidden,
                 linear_decoder_hidden,
                 activate="relu",
                 p_drop=0.01):
        super(AEncoder, self).__init__()
        self.input_dim = input_dim
        self.linear_encoder_hidden = linear_encoder_hidden
        self.linear_decoder_hidden = linear_decoder_hidden
        self.activate = activate
        self.p_drop = p_drop

        current_encoder_dim = self.input_dim
        self.encoder = nn.Sequential()
        for le in range(len(self.linear_encoder_hidden)):
            self.encoder.add_module(f'encoder_L{le}',
                                    buildNetwork(current_encoder_dim, self.linear_encoder_hidden[le], self.activate,
                                                 self.p_drop))
            current_encoder_dim = self.linear_encoder_hidden[le]

        current_decoder_dim = linear_decoder_hidden[0]
        self.decoder = nn.Sequential()

        for ld in range(1, len(self.linear_decoder_hidden)):
            self.decoder.add_module(f'decoder_L{ld}',
                                    buildNetwork(current_decoder_dim, self.linear_decoder_hidden[ld], self.activate,
                                                 self.p_drop))
            current_decoder_dim = self.linear_decoder_hidden[ld]

        self.decoder.add_module(f'decoder_L{len(self.linear_decoder_hidden)}',
                                buildNetwork(self.linear_decoder_hidden[-1],
                                             self.input_dim, "sigmoid", self.p_drop))

    def forward(self, x):
        feat = self.encoder(x)
        return feat

    def dforward(self, x):
        feat = self.decoder(x)
        return feat


def buildNetwork(
        in_features,
        out_features,
        activate="relu",
        p_drop=0.0
):
    net = []
    net.append(nn.Linear(in_features, out_features))
    net.append(nn.BatchNorm1d(out_features, momentum=0.01, eps=0.001))
    if activate == "relu":
        net.append(nn.ELU())
    elif activate == "sigmoid":
        net.append(nn.Sigmoid())
    if p_drop > 0:
        net.append(nn.Dropout(p_drop))
    return nn.Sequential(*net)


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


class Attention_Emb(nn.Module):
    def __init__(self, in_size):
        super(Attention_Emb, self).__init__()
        self.project = nn.Linear(in_size, 1, bias=False)

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1)


class MIGCL(nn.Module):
    def __init__(self,
                 input_dim,
                 num_en,
                 num_de,
                 nhid,
                 nemb,
                 dropout):
        super(MIGCL, self).__init__()
        self.encoder = AEncoder(input_dim, num_en, num_de)
        self.GCL = GCN_Encoder(num_en[-1], nhid, nemb, dropout)
        self.attention = Attention_Emb(nemb)

    def forward(self, feat, fadj, sadj):
        z = self.encoder(feat)
        femb = self.GCL(z, fadj)
        semb = self.GCL(z, sadj)
        emb = self.attention(torch.stack([femb, semb], dim=1))
        de = self.encoder.dforward(emb)

        return femb, semb, de, emb
