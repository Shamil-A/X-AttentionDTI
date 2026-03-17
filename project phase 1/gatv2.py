import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool

class GATv2Encoder(nn.Module):
    def __init__(self, num_node_features=78, embed_dim=128, heads=4, dropout=0.2):
        super(GATv2Encoder, self).__init__()
        self.gat1 = GATv2Conv(num_node_features, embed_dim, heads=heads, dropout=dropout)
        self.gat2 = GATv2Conv(embed_dim * heads, embed_dim, heads=1, dropout=dropout)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        x = F.elu(x)
        x = global_mean_pool(x, batch)  # Pool to get [B, H] vector
        return x
