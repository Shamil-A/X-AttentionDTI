import torch
import torch.nn as nn

class CrossModalAttention(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4):
        super(CrossModalAttention, self).__init__()
        self.attn_drug_to_prot = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.attn_prot_to_drug = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, drug_vec, prot_vec):
        # Add sequence dimension for attention
        drug_vec = drug_vec.unsqueeze(1)  # [B, 1, H]
        prot_vec = prot_vec.unsqueeze(1)  # [B, 1, H]

        drug_ctx, _ = self.attn_drug_to_prot(drug_vec, prot_vec, prot_vec)
        prot_ctx, _ = self.attn_prot_to_drug(prot_vec, drug_vec, drug_vec)

        # Remove seq dimension
        drug_ctx = drug_ctx.squeeze(1)  # [B, H]
        prot_ctx = prot_ctx.squeeze(1)  # [B, H]

        return drug_ctx, prot_ctx
