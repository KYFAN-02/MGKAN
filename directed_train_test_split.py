import math
import torch
from torch_geometric.utils import negative_sampling
from torch_geometric.data import Data
import numpy as np
import config


def data_split(data, fold, seed, val_ratio=config.val_ratio, test_ratio=config.test_ratio):

    torch.manual_seed(seed)
    num_nodes = data.num_nodes
    row_original, col_original = data.edge_index
    data.edge_index = None
    n_v = int(math.floor(val_ratio * row_original.size(0)))
    n_t = int(math.floor(test_ratio * row_original.size(0)))
    n_a = int(math.floor(row_original.size(0)))

    # Positive edges.
    perm = torch.randperm(row_original.size(0))
    start_step = int(fold / config.fold * perm.size().numel())
    perm_repeat = torch.cat([perm, perm], dim=0)
    row, col = row_original[perm_repeat], col_original[perm_repeat]

    r, c = row[start_step:start_step + n_v], col[start_step:start_step + n_v]
    data.val_pos_edge_index = torch.stack([r, c], dim=0)    # validation set
    r, c = row[start_step + n_v:start_step + n_v + n_t], col[start_step + n_v:start_step + n_v + n_t]
    data.test_pos_edge_index = torch.stack([r, c], dim=0)   # train set
    r, c = row[start_step + n_v + n_t:start_step + n_a], col[start_step + n_v + n_t:start_step + n_a]   # test set
    data.train_pos_edge_index = torch.stack([r, c], dim=0)

    # Negative edges.
    data.val_neg_edge_index = negative_sampling(
        edge_index=torch.cat([data.train_pos_edge_index, data.val_pos_edge_index], dim=1),
        num_nodes=num_nodes,
        num_neg_samples=n_v
    )

    data.test_neg_edge_index = negative_sampling(
        edge_index=torch.cat([data.train_pos_edge_index, data.val_pos_edge_index, data.test_pos_edge_index], dim=1),
        num_nodes=num_nodes,
        num_neg_samples=n_t
    )

    return data
