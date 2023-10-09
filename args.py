import argparse


def init_args():
    parser = argparse.ArgumentParser(description='Graph Contrastive Learning with Implicit Augmentations')

    # Experiment Settings
    parser.add_argument('--seed', type=int, default=3)  ###
    parser.add_argument('--n_top_genes', type=int, default=1000)  ###500,0.42 800,0.43

    # Model Design
    parser.add_argument('--hidden_size', type=int, default=32)  ###
    parser.add_argument('--emb_size', type=int, default=32)  ###
    parser.add_argument('--num_en', type=int, default=[128, 64])
    parser.add_argument('--num_de', type=int, default=[32, 128])
    parser.add_argument('--dropout', type=float, default=0.5)  ###
    parser.add_argument('--k', type=int, default=50)  ###
    parser.add_argument('--radius', type=int, default=150)  ###
    #parser.add_argument('--n_cluster', type=int, default=7)  ###

    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=200)  ###40,0.53
    parser.add_argument('--num_samples', type=int, default=500)  ###
    #k = 30, radius = 500, epochs = 40, mclust, 0.53
    parser.add_argument('--lr', type=float, default=0.001)  ###
    parser.add_argument('--l2', type=float, default=5e-3)  ###

    # Model Hyperparameters
    parser.add_argument('--tau', type=float, default=1)  ###

    return parser.parse_args()
