import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve,\
    accuracy_score, f1_score, recall_score, precision_score
from torch_geometric.utils import (negative_sampling, remove_self_loops,
                                   add_self_loops)
from torch_geometric.nn.inits import reset
import config

class RoleAlignDecoder(torch.nn.Module):
    def __init__(self):
        super(RoleAlignDecoder, self).__init__()
        self.out_channels = 2*config.out_channels
        self.bilinear=nn.Bilinear(self.out_channels, self.out_channels, 1, bias=False)

    def forward(self, x_in, x_out, edge_index, sigmoid=True):
        value=self.bilinear(x_out[edge_index[0]],x_in[edge_index[1]]).squeeze(1)
        return torch.sigmoid(value) if sigmoid else value

class Predictor(torch.nn.Module):

    def __init__(self, encoder, decoder=None):
        super(Predictor, self).__init__()
        self.encoder = encoder
        self.decoder = RoleAlignDecoder() if decoder is None else decoder
        Predictor.reset_parameters(self)

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.decoder)

    def encode(self, *args, **kwargs):

        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):

        return self.decoder(*args, **kwargs)

    def recon_loss(self, x_in, x_out, pos_edge_index, neg_edge_index=None):

        pos_loss = -torch.log(self.decoder(x_in, x_out, pos_edge_index, sigmoid=True) + config.EPS).mean()
        if neg_edge_index is None:
            #Each training epoch extracts negative training samples
            neg_edge_index = negative_sampling(pos_edge_index, num_nodes=x_in.size(0))
        neg_loss = -torch.log(1 - self.decoder(x_in, x_out, neg_edge_index, sigmoid=True) + config.EPS).mean()
        return pos_loss + neg_loss

    def test(self, x_in, x_out, pos_edge_index, neg_edge_index):
        if config.task1:
            pos_y = x_in.new_ones(pos_edge_index.size(1))
            neg_y = x_in.new_zeros(neg_edge_index.size(1))
            y = torch.cat([pos_y, neg_y], dim=0)
            pos_pred = self.decoder(x_in, x_out, pos_edge_index, sigmoid=True)
            neg_pred = self.decoder(x_in, x_out, neg_edge_index, sigmoid=True)
            pred = torch.cat([pos_pred, neg_pred], dim=0)
            y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

            return roc_auc_score(y, pred), average_precision_score(y, pred), accuracy_score(y, pred.round()), \
                   f1_score(y, pred.round()), precision_score(y, pred.round()), recall_score(y, pred.round())
        else:
            num_edges= pos_edge_index.size(1)//2

            pos_edges = pos_edge_index[:, :num_edges]
            reverse_edges = (pos_edge_index[: ,num_edges:2*num_edges]).flip(0)
            neg_edges = negative_sampling(torch.cat([pos_edges,reverse_edges],dim=1), num_nodes=x_in.size(0), num_neg_samples=pos_edge_index.size(1))

            all_edges = torch.cat([pos_edges, reverse_edges, neg_edges], dim=1)

            y = torch.cat([
                torch.zeros(num_edges, dtype=torch.long),
                torch.ones(num_edges, dtype=torch.long),
                torch.ones(pos_edge_index.size(1), dtype=torch.long)*2
            ], dim=0)

            perm = torch.randperm(all_edges.size(1))
            all_edges = all_edges[:, perm]
            y = y[perm]

            #Predicts the two-direction probabilities of all edges
            forward_proba = self.decoder(x_in, x_out, all_edges, sigmoid=True)
            reverse_proba = self.decoder(x_in, x_out, all_edges.flip(0), sigmoid=True)

            prob_matrix = torch.stack([
                forward_proba * (1 - reverse_proba),
                reverse_proba * (1 - forward_proba),
                (1 - forward_proba) * (1 - reverse_proba)
            ], dim=1)
            prob_matrix = torch.softmax(prob_matrix, dim=1)

            y_np = y.cpu().numpy()
            prob_matrix_np = prob_matrix.detach().cpu().numpy()

            average_type = 'macro'
            class_matrix=prob_matrix_np.argmax(axis=1)

            return (
                roc_auc_score(y_np, prob_matrix_np, multi_class='ovr', average=average_type),
                average_precision_score(y_np, prob_matrix_np),
                accuracy_score(y_np, class_matrix),
                f1_score(y_np, class_matrix, average=average_type),
                precision_score(y_np, class_matrix, average=average_type),
                recall_score(y_np, class_matrix, average=average_type)
            )