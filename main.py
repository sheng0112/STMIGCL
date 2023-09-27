import torch
import random
import numpy as np
import scanpy as sc
import pandas as pd
from sklearn import metrics
import torch.nn.functional as F

from args import init_args
from VGAE import fGCNModelVAE, sGCNModelVAE
from loss import VGAE_Loss, Implicit_Contrastive_Loss
from model import GCN_Encoder, Linear_Classifier, Attention_Emb
from utils import preprocess, features_construct_graph, spatial_construct_graph, pred_result, Reconstruct_Ratio

"""
import matplotlib as plt
plt.use('TkAgg')
"""


def train():
    fadj, fadj_ori, fadj_label, fpos_weight, fnorm = features_construct_graph(feat, args.k)
    fadj = fadj.to(device)
    fadj_label = fadj_label.to(device)

    sadj, sadj_ori, sadj_label, spos_weight, snorm = spatial_construct_graph(adata, args.radius)
    sadj = sadj.to(device)
    sadj_label = sadj_label.to(device)

    fVGAE = fGCNModelVAE(feat.shape[1], args.hidden_size, args.emb_size, 0).to(device)
    sVGAE = sGCNModelVAE(feat.shape[1], args.hidden_size, args.emb_size, 0).to(device)

    Encoder = GCN_Encoder(nfeat=feat.shape[1], nhid=args.hidden_size, nemb=args.emb_size, dropout=args.dropout).to(
        device)

    Classifier = Linear_Classifier(args.emb_size, args.n_cluster).to(device)

    Attention = Attention_Emb(args.emb_size).to(device)

    # OPTIMIZER:
    optimizer_fvgae = torch.optim.Adam(fVGAE.parameters(), lr=0.01, weight_decay=0)
    optimizer_svgae = torch.optim.Adam(sVGAE.parameters(), lr=0.01, weight_decay=0)
    optimizer_encoder = torch.optim.Adam(Encoder.parameters(), lr=args.lr, weight_decay=args.l2)
    optimizer_classifier = torch.optim.Adam(Classifier.parameters(), lr=5e-3, weight_decay=5e-4)

    best_adj_acc = 0
    adj_acc = 0.1

    for epoch in range(args.epochs):
        fVGAE.train()
        sVGAE.train()
        Encoder.train()
        Classifier.train()

        fpred, fmu, flog_sigma = fVGAE(feat, fadj)
        femb = Encoder(feat, fadj)

        spred, smu, slog_sigma = sVGAE(feat, sadj)
        semb = Encoder(feat, sadj)

        if epoch < args.epochs * 0.5:
            fVGAE_loss = VGAE_Loss(preds=fpred, labels=fadj_label, mu=fmu, logvar=flog_sigma,
                                   n_nodes=feat.shape[0], norm=fnorm, pos_weight=fpos_weight)
            # 14.0678, 472.3860, 8.6785, 6.2431 6.1737
            optimizer_fvgae.zero_grad()
            fVGAE_loss.backward()
            optimizer_fvgae.step()

            sVGAE_loss = VGAE_Loss(preds=spred, labels=sadj_label, mu=smu, logvar=slog_sigma,
                                   n_nodes=feat.shape[0], norm=snorm, pos_weight=spos_weight)
            # 14.977, 716.3742, 11.4162, 6.1448 6.0926
            optimizer_svgae.zero_grad()
            sVGAE_loss.backward()
            optimizer_svgae.step()

        # Use latent distributions from the best VGAE so far.
        if adj_acc > best_adj_acc:
            fpred, fmu, flog_sigma = fVGAE(feat, fadj)
            fmu_best = fmu.detach()
            fsigma = torch.exp(flog_sigma.detach())

            spred, smu, slog_sigma = sVGAE(feat, sadj)
            smu_best = smu.detach()
            ssigma = torch.exp(slog_sigma.detach())

            best_adj_acc = adj_acc

        # Get the Contrastive Loss
        fContrastive_loss = Implicit_Contrastive_Loss(Z=femb, mu=fmu_best, sigma2=fsigma ** 2,
                                                      # 6.2915 6.3086 6.2732 11.1135
                                                      tau=args.tau, num_samples=args.num_samples,
                                                      device=device)
        sContrastive_loss = Implicit_Contrastive_Loss(Z=semb, mu=smu_best, sigma2=ssigma ** 2,
                                                      # 11.1917 9.1787 7.3612 10.7859
                                                      tau=args.tau, num_samples=args.num_samples,
                                                      device=device)
        Contrastive_loss = fContrastive_loss + sContrastive_loss
        # 17.4832,15.4873,13.6344, 21.8994 20.4468

        optimizer_encoder.zero_grad()
        Contrastive_loss.backward(retain_graph=True)
        optimizer_encoder.step()

        femb = Encoder(feat, fadj)
        semb = Encoder(feat, sadj)
        emb = Attention(torch.stack([femb, semb], dim=1))  # (3611,256)

        for iteration in range(100):
            optimizer_classifier.zero_grad()
            y_pred = Classifier(emb.detach())

            classifier_loss = F.nll_loss(y_pred, y)  # 负对数似然损失函数，只选取了一部分
            classifier_loss.backward()
            optimizer_classifier.step()

        Encoder.eval()  # 调用模型的 eval() 方法可以将模型设置为评估模式
        Classifier.eval()
        femb = Encoder(feat, fadj)
        semb = Encoder(feat, sadj)
        emb = Attention(torch.stack([femb, semb], dim=1))
        out = Classifier(emb.detach())
        _, y_pred = out.max(dim=1)
        y_pred = y_pred.cpu()

        # acc = pred_result(val_pred, y)
        # acc_list.append(acc)

        nmi = metrics.normalized_mutual_info_score(y.cpu(), y_pred)  # 0.62 0.687 0.7147 0.7145
        ari = metrics.adjusted_rand_score(y.cpu(), y_pred)  # 0.49 0.633 0.6836 0.67789
        print('Iter {}'.format(epoch), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari))

        fadj_acc = Reconstruct_Ratio(fpred.cpu(), fadj_ori)  # 0.0097
        sadj_acc = Reconstruct_Ratio(spred.cpu(), sadj_ori)  # 0.014
        adj_acc = fadj_acc + sadj_acc  # 0.0239 0.0239 0.114 0.98 0.979

    Encoder.eval()
    femb = Encoder(feat, fadj)
    semb = Encoder(feat, sadj)
    emb = Attention(torch.stack([femb, semb], dim=1))
    Classifier.eval()
    out = Classifier(emb.detach())
    _, y_pred = out.max(dim=1)
    y_pred = y_pred.cpu()

    nmi = metrics.normalized_mutual_info_score(y.cpu(), y_pred)
    ari = metrics.adjusted_rand_score(y.cpu(), y_pred)
    print('Final', ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari))

    adata.obs['y_pred'] = y_pred
    adata.obs['y_pred'] = adata.obs['y_pred'].astype('int')
    adata.obs['y_pred'] = adata.obs['y_pred'].astype('category')

    sc.pl.spatial(adata,
                  color=["ground_truth", "y_pred"],
                  title=["Ground truth", "ARI=%.4f" % ari],
                  show=True)


if __name__ == '__main__':
    args = init_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    adata = sc.read_visium('/opt/data/private/DataSet/Human_breast_cancer (10x)', count_file='filtered_feature_bc_matrix.h5')
    adata.var_names_make_unique()  # (3639,33538)

    label = pd.read_csv('/opt/data/private/DataSet/Human_breast_cancer (10x)' + '/metadata.tsv', sep='\t')
    adata.obs['label'] = torch.LongTensor(pd.factorize(label['fine_annot_type'].astype("category"))[0])
    adata.obs['ground_truth'] = label['fine_annot_type'].values
    adata = adata[~pd.isnull(adata.obs['ground_truth'])]
    y = torch.LongTensor(adata.obs['label'].values).to(device)

    preprocess(args, adata)
    feat = torch.FloatTensor(adata.obsm['feat'].copy()).to(device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    train()
