import torch
import torch.nn as nn
import torch.nn.functional as F
from models.gatv2 import GATv2Encoder
from models.cnn_protein import ProteinCNN
from models.cross_attention import CrossModalAttention

class DrugTargetFusionModel(nn.Module):
    def __init__(self, gat_embed_dim=128, prot_embed_dim=128, hidden_dim=256):
        super(DrugTargetFusionModel, self).__init__()
        self.gat_encoder = GATv2Encoder(embed_dim=gat_embed_dim)
        self.prot_encoder = ProteinCNN(embedding_dim=prot_embed_dim)
        self.cross_attn = CrossModalAttention(embed_dim=gat_embed_dim, num_heads=4)

        fusion_input_dim = gat_embed_dim + prot_embed_dim + gat_embed_dim + prot_embed_dim
        self.fc1 = nn.Linear(fusion_input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, drug_data, prot_seq):
        drug_vec = self.gat_encoder(drug_data)    # [B, H]
        prot_vec = self.prot_encoder(prot_seq)    # [B, H]
        drug_ctx, prot_ctx = self.cross_attn(drug_vec, prot_vec)
        fused = torch.cat([drug_vec, prot_vec, drug_ctx, prot_ctx], dim=1)
        x = F.relu(self.fc1(fused))
        return self.fc2(x)
