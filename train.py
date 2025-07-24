import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from Decoder import Predictor
from directed_train_test_split import data_split
from model.models import MGKAN
import copy
import config

print(torch.cuda.get_device_name())
#Dataset
fold=2024
#DDI_edge_list
edge_list = np.loadtxt(f'./newdataset/{fold}/new_edgelist.txt', dtype=np.int64)
#Node features
feature = pd.read_csv(f'./newdataset/{fold}/FP_feature_100.csv', header=0, index_col=0)
#sim_matrix
target_sim=pd.read_csv(f'./newdataset/{fold}/target_sim.csv', header=0, index_col=0)
enzyme_sim=pd.read_csv(f'./newdataset/{fold}/enzyme_sim.csv', header=0, index_col=0)
transporter_sim=pd.read_csv(f'./newdataset/{fold}/transporter_sim.csv', header=0, index_col=0)

edge_index = torch.tensor(edge_list).t().contiguous()
num_nodes = len(set(edge_index.flatten().tolist()))
features = torch.from_numpy(feature.values).to(torch.float32)
Sim_adj = torch.from_numpy((target_sim.values+enzyme_sim.values+transporter_sim.values)).float().contiguous().cuda()
sim_index, sim_weight = dense_to_sparse(Sim_adj)

def train():
    model.train()
    optimizer.zero_grad()
    x_in, x_out = model.encode(x, train_pos_edge_index,sim_index, sim_weight)
    loss = model.recon_loss(x_in, x_out, train_pos_edge_index)
    loss.backward()
    optimizer.step()
    return float(loss)

def test(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        x_in, x_out = model.encode(x, train_pos_edge_index,sim_index, sim_weight)
    return model.test(x_in, x_out, pos_edge_index, neg_edge_index)

def testfinal(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        x_in, x_out = model.encode(x, train_pos_edge_index,sim_index, sim_weight)
    return model.test(x_in, x_out, pos_edge_index, neg_edge_index)


def initialize_list():
    lists = [[] for _ in range(6)]
    return [lists[i] for i in range(6)]


target = ["auc", "ap", "acc", "f1", "pre", "re"]
auc_list, ap_list, f1_list, acc_list, pre_list, re_list = initialize_list()
target_list = [auc_list, ap_list,  f1_list, acc_list, pre_list, re_list]
for i in range(config.number):

    auc_l, ap_l, f1_l, acc_l, pre_l, re_l = initialize_list()
    target_l = [auc_l, ap_l, f1_l, acc_l, pre_l, re_l]
    for fold in range(config.fold):
        data = Data(edge_index=edge_index, num_nodes=num_nodes, x=features)
        data = data_split(data, fold, config.seed)
        print(data)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_pos_edge_index = data.train_pos_edge_index.to(device)

        x = data.x.to(device)
        model = Predictor(MGKAN(data.num_node_features, config.out_channels)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)


        min_loss_val = config.min_loss_val
        best_model = None
        min_epoch = config.min_epoch
        for epoch in range(1, config.epochs + 1):
            loss = train()
            if epoch % 50 == 0:
                auc, ap, acc, f1, pre, re = test(data.val_pos_edge_index, data.val_neg_edge_index)
                print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}, ACC: {:.4f}, F1: {:.4f}, PRE: {:.4f}, RE: {:.4f},'
                          .format(epoch, auc, ap, acc, f1, pre, re))
            if epoch > min_epoch and loss <= min_loss_val:
                min_loss_val = loss
                best_model = copy.deepcopy(model)
        model = best_model
        auc, ap, acc, f1, pre, re = testfinal(data.test_pos_edge_index, data.test_neg_edge_index)
        print('final. AUC: {:.4f}, AP: {:.4f}, ACC: {:.4f}, F1: {:.4f}, PRE: {:.4f}, RE: {:.4f},'
              .format(auc, ap, acc, f1, pre, re))
        for j in range(6):
            target_l[j].append(eval(target[j]))
    for j in range(6):
        target_list[j].append(np.mean(target_l[j]))
for j in range(6):
    # print(np.mean(target_list[j]), np.std(target_list[j]))
    print(str(round(np.mean(target_list[j]) * 100, 2))+"±"+str(round(np.std(target_list[j]) * 100, 2)))

