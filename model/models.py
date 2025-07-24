import torch
import torch.nn as nn
import torch.nn.functional as F
from model.kan_layer import KANLinear
from model.KANConv import GKAN
from model.utils import second_order_adj

class AttentionModule(torch.nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.linear = nn.Linear(embed_dim, embed_dim)
        self.q = nn.Parameter(torch.randn(embed_dim))
        self.norm = nn.LayerNorm(embed_dim)
        self.tanh = nn.Tanh()

    def forward(self, feature):

        feature = self.tanh(self.linear(self.norm(feature)))

        attention_logits = torch.matmul(feature, self.q)

        return attention_logits

class FeatureFusion(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim=input_dim
        self.num_feats=3
        self.attention_module = AttentionModule(input_dim)
        self.nonlinear = torch.nn.Sequential(
            nn.BatchNorm1d(self.num_feats * input_dim),
            KANLinear(self.num_feats * input_dim, input_dim),
            nn.BatchNorm1d(input_dim),


        )
    def forward(self, *features):
        attention_logits = [self.attention_module(feat) for feat in features]
        combined_attention = torch.stack(attention_logits, dim=0)
        alpha_combined = F.softmax(combined_attention, dim=0)
        alphas = torch.unbind(alpha_combined, dim=0)

        weight_feat=sum(alpha.unsqueeze(-1) * feat for alpha, feat in zip(alphas, features))

        nonlin_feat=self.nonlinear(torch.concat(features,dim=-1))

        z = torch.concat([weight_feat,nonlin_feat],dim=-1)

        return z

class MGKAN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MGKAN, self).__init__()

        self.out_channels = out_channels
        self.hidden_channels=2*out_channels

        self.GCN_DDI = GKAN(in_channels=in_channels,hidden_channels=self.hidden_channels,out_channels=self.out_channels)
        self.GCN_co = GKAN(in_channels=in_channels,hidden_channels=self.hidden_channels,out_channels=self.out_channels)
        self.GCN_sim = GKAN(in_channels=in_channels,hidden_channels=self.hidden_channels,out_channels=self.out_channels)

        self.in_fusion = FeatureFusion(self.out_channels)
        self.out_fusion = FeatureFusion(self.out_channels)

    def forward(self, x, edge_index ,sim_index ,sim_weight):

        (in2_index, in2_weight),(out2_index, out2_weight) = second_order_adj(edge_index)

        x_in1 = self.GCN_DDI(x,edge_index)
        x_out1 = self.GCN_DDI(x,edge_index.flip(0))

        x_in2 = self.GCN_co(x,in2_index,in2_weight)
        x_out2 = self.GCN_co(x,out2_index,out2_weight)

        x_sim=self.GCN_sim(x,sim_index,sim_weight)

        x_in=self.in_fusion(x_in1,x_in2,x_sim)
        x_out=self.out_fusion(x_out1,x_out2,x_sim)

        return x_in, x_out