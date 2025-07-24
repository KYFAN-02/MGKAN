import torch
from torch_geometric.utils import to_dense_adj,dense_to_sparse

def first_order_adj(A):
    torch.diagonal(A).zero_()
    Ain = dense_to_sparse(norm(A.T))
    Aout = dense_to_sparse(norm(A))
    return Ain, Aout

def second_order_adj(edge_index):
    eps = 1e-10
    A = to_dense_adj(edge_index).squeeze(0)
    A.fill_diagonal_(0)
    row_sums = A.sum(dim=1, keepdim=True) + eps
    Ain = torch.mm((A / row_sums).T, A)
    col_sums = A.sum(dim=0, keepdim=True) + eps
    Aout = torch.mm(A, (A / col_sums).T)
    return dense_to_sparse(Ain), dense_to_sparse(Aout)

def norm(adj):
    D = adj.sum(dim=1) + 1
    D_inv_sqrt = D.pow(-0.5)
    D_inv_sqrt[D == 0] = 0
    norm_adj = D_inv_sqrt.unsqueeze(1) * adj * D_inv_sqrt.unsqueeze(0)
    return norm_adj