import torch
import sklearn
import numpy as np
import scanpy as sc
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
from sklearn.neighbors import kneighbors_graph


def preprocess(args, adata):
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=args.n_top_genes)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    # sc.pp.scale(adata, zero_center=False, max_value=10)
    get_feature(adata)


def get_feature(adata):
    adata_Vars = adata[:, adata.var['highly_variable']]

    if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
        feat = adata_Vars.X.toarray()[:, ]
    else:
        feat = adata_Vars.X[:, ]

    adata.obsm['feat'] = feat


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def features_construct_graph(features, k=15, mode="connectivity", metric="cosine"):
    features = features.cpu().numpy()
    adj_ori = kneighbors_graph(features, k, mode=mode, metric=metric, include_self=True)
    adj_ori = adj_ori.toarray()
    row, col = np.diag_indices_from(adj_ori)
    adj_ori[row, col] = 0
    adj_ori = torch.FloatTensor(adj_ori)

    adj = sp.coo_matrix(adj_ori, dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    adj_label = adj_ori + torch.eye(adj_ori.shape[0])

    pos_weight = (adj_ori.shape[0] ** 2 - adj_ori.sum()) / adj_ori.sum()
    norm = adj_ori.shape[0] ** 2 / (2 * (adj_ori.shape[0] ** 2 - adj_ori.sum()))

    return adj, adj_ori, adj_label, pos_weight, norm


def spatial_construct_graph(adata, radius=150):
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']
    adj_ori = np.zeros((coor.shape[0], coor.shape[0]))
    nbrs = sklearn.neighbors.NearestNeighbors(radius=radius).fit(coor)
    distances, indices = nbrs.radius_neighbors(coor, return_distance=True)  # (4221, 1)
    for it in range(indices.shape[0]):
        adj_ori[[it] * indices[it].shape[0], indices[it]] = 1
    row, col = np.diag_indices_from(adj_ori)
    adj_ori[row, col] = 0
    adj_ori = torch.FloatTensor(adj_ori)

    adj = sp.coo_matrix(adj_ori, dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    adj_label = adj_ori + torch.eye(adj_ori.shape[0])

    pos_weight = (adj_ori.shape[0] ** 2 - adj_ori.sum()) / adj_ori.sum()
    norm = adj_ori.shape[0] ** 2 / (2 * (adj_ori.shape[0] ** 2 - adj_ori.sum()))

    return adj, adj_ori, adj_label, pos_weight, norm


def pred_result(y_pred, y_true):
    correct = float(y_pred.eq(y_true).sum().item())
    return correct


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


def Reconstruct_Ratio(pred, adj_ori):
    adj_pred = pred.reshape(-1)
    adj_pred = (sigmoid(adj_pred) > 0.5).float()
    adj_true = (adj_ori + torch.eye(adj_ori.shape[0])).reshape(-1)
    adj_acc = float(adj_pred.eq(adj_true).sum().item()) / adj_pred.shape[0]
    return adj_acc
