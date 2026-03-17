import torch
import torch.nn as nn
import torch.nn.functional as F

class ProteinCNN(nn.Module):
    def __init__(self, vocab_size=26, embedding_dim=128):
        super(ProteinCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=32, kernel_size=8, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=8, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Projection layer to fixed embedding_dim (default 128)
        self.fc = nn.Linear(64, embedding_dim)

    def forward(self, x):
        # x: [B, seq_len]
        x = self.embedding(x.long())     # [B, seq_len, embed_dim]
        x = x.permute(0, 2, 1)          # [B, embed_dim, seq_len]

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        # Global pooling → fixed size [B, 64]
        x = F.adaptive_max_pool1d(x, 1).squeeze(-1)

        # Project to [B, embedding_dim] (128 by default)
        x = self.fc(x)
        return x
