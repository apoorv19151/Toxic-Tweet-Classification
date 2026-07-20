import torch
import torch.nn as nn

class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        *,
        vocab_size,
        embedding_dim=300,
        hidden_dim=128,
        num_layers=2,
        dropout=0.5,
        num_classes=6,
        padding_idx=0,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(
            in_features=hidden_dim * 2,
            out_features=num_classes,
        )

    def forward(self, x):
        embedded = self.embedding(x)

        _, (hidden, _) = self.lstm(embedded)

        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)

        hidden = self.dropout(hidden)

        logits = self.fc(hidden)

        return logits

__all__ = [
    "BiLSTMClassifier",
]
