import os
import torch
import random
import numpy as np
import scanpy as sc
import torch.nn.functional as F
import pandas as pd
from sklearn import metrics

from args import init_args
from VGAE import fGCNModelVAE, sGCNModelVAE
from loss import VGAE_Loss, Implicit_Contrastive_Loss
from model import GCN_Encoder, Attention_Emb, MIGCL
from utils import preprocess, features_construct_graph, spatial_construct_graph, Reconstruct_Ratio  # , clustering


def train():
    fadj, fadj_ori, fadj_label, fpos_weight, fnorm = features_construct_graph(feat, args.k)
    fadj = fadj.to(device)
    fadj_label = fadj_label.to(device)

    sadj, sadj_ori, sadj_label, spos_weight, snorm = spatial_construct_graph(adata, args.radius)
    sadj = sadj.to(device)
    sadj_label = sadj_label.to(device)

    fVGAE = fGCNModelVAE(feat.shape[1], args.num_en[0], args.emb_size, 0).to(device)
    sVGAE = sGCNModelVAE(feat.shape[1], args.num_en[0], args.emb_size, 0).to(device)

    model = MIGCL(feat.shape[1], args.num_en, args.num_de, args.hidden_size, args.emb_size, dropout=args.dropout).to(
        device)

    # OPTIMIZER:
    optimizer_fvgae = torch.optim.Adam(fVGAE.parameters(), lr=0.01, weight_decay=0)
    optimizer_svgae = torch.optim.Adam(sVGAE.parameters(), lr=0.01, weight_decay=0)
    optimizer_encoder = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    best_adj_acc = 0
    adj_acc = 0.1

    for epoch in range(1, args.epochs + 1):
        fVGAE.train()
        sVGAE.train()
        model.train()

        femb, semb, de, _ = model(feat, fadj, sadj)

        fpred, fmu, flog_sigma = fVGAE(feat, fadj)
        spred, smu, slog_sigma = sVGAE(feat, sadj)

        # if epoch < args.epochs:
        fVGAE_loss = VGAE_Loss(preds=fpred, labels=fadj_label, mu=fmu, logvar=flog_sigma,
                               n_nodes=feat.shape[0], norm=fnorm, pos_weight=fpos_weight)
        optimizer_fvgae.zero_grad()
        fVGAE_loss.backward()
        optimizer_fvgae.step()

        sVGAE_loss = VGAE_Loss(preds=spred, labels=sadj_label, mu=smu, logvar=slog_sigma,
                               n_nodes=feat.shape[0], norm=snorm, pos_weight=spos_weight)
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
                                                      tau=args.tau, num_samples=args.num_samples,
                                                      device=device)
        sContrastive_loss = Implicit_Contrastive_Loss(Z=semb, mu=smu_best, sigma2=ssigma ** 2,
                                                      tau=args.tau, num_samples=args.num_samples,
                                                      device=device)
        Contrastive_loss = fContrastive_loss + sContrastive_loss
        recon_loss = F.mse_loss(feat, de)
        loss = Contrastive_loss + recon_loss

        optimizer_encoder.zero_grad()
        loss.backward(retain_graph=True)
        optimizer_encoder.step()

        emb = model(feat, fadj, sadj)[-1]

        fadj_acc = Reconstruct_Ratio(fpred.cpu(), fadj_ori)
        sadj_acc = Reconstruct_Ratio(spred.cpu(), sadj_ori)
        adj_acc = fadj_acc + sadj_acc

        if epoch % 10 == 0:
            np.savetxt('/opt/data/private/save/AE' + str(epoch) + 'data.csv', emb.detach().cpu().numpy())
            print(epoch)

    # adata.obsm['emb'] = emb.detach().cpu().numpy()


if __name__ == '__main__':

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    args = init_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    adata = sc.read_visium('/opt/data/private/DataSet/DLPFC/151507', count_file='filtered_feature_bc_matrix.h5')
    adata.var_names_make_unique()

    preprocess(args, adata)
    feat = torch.FloatTensor(adata.obsm['feat'].copy()).to(device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    train()

    # np.savetxt('/opt/data/private/save/' + str(epochs[i]) + 'data.csv', adata.obsm['emb'])

    """
    label = pd.read_csv('/opt/data/private/DataSet/DLPFC/151671' + '/metadata.tsv', sep='\t')
    adata.obs['ground_truth'] = label['layer_guess'].values
    adata = adata[~pd.isnull(adata.obs['ground_truth'])]

    radius = 50
    tool = 'louvain'

    if tool == 'mclust':
        clustering(adata, args.n_cluster, radius=radius, method=tool, refinement=True)
    elif tool in ['leiden', 'louvain']:
        clustering(adata, args.n_cluster, radius=radius, method=tool, start=0.1, end=1.5, increment=0.01, refinement=False)

    ARI = metrics.adjusted_rand_score(adata.obs['domain'], adata.obs['ground_truth'])
    NMI = metrics.normalized_mutual_info_score(adata.obs['domain'], adata.obs['ground_truth'])
    print(ARI)
    print(NMI)

    sc.pl.spatial(adata,
                  img_key="hires",
                  color=["domain"],
                  title=["ARI=%.2f" % ARI],
                  show=True)
    """
