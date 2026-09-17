"""
Lightweight Temporal Transformer for Isolated Indian Sign Language (ISL) Recognition.
Input shape: [Batch, 30, 258]
Output shape: [Batch, num_classes]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Standard Sinusoidal Positional Encoding for Temporal Sequences.
    """
    def __init__(self, d_model: int, max_len: int = 30, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: [Batch, Sequence_Len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TemporalTransformer(nn.Module):
    """
    Lightweight Temporal Transformer Architecture.
    
    1. Linear Projection: Maps 258 spatial landmark features to hidden dimension d_model.
    2. Positional Encoding: Adds temporal position context.
    3. Transformer Encoder: 3 to 4 Multi-Head Self-Attention layers.
    4. Global Temporal Pooling: Averages feature embeddings across the 30-frame sequence.
    5. Linear Classifier Head: Maps temporal representation to class probabilities/logits.
    """
    def __init__(
        self,
        input_dim: int = 258,
        num_classes: int = 100,
        seq_len: int = 30,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1
    ):
        super(TemporalTransformer, self).__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.d_model = d_model

        # 1. Feature Projection Layer [258 -> d_model]
        self.input_proj = nn.Linear(input_dim, d_model)
        self.layer_norm_in = nn.LayerNorm(d_model)

        # 2. Temporal Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=seq_len, dropout=dropout)

        # 3. Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        # 4. Classification Head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape [Batch, 30, 258]
        Returns:
            Logits tensor of shape [Batch, num_classes]
        """
        batch_size, seq_len, num_feats = x.shape
        assert num_feats == self.input_dim, f"Expected {self.input_dim} features per frame, got {num_feats}"

        # 1. Linear Projection
        out = self.input_proj(x)  # [Batch, 30, d_model]
        out = self.layer_norm_in(out)

        # 2. Positional Encoding
        out = self.pos_encoder(out)  # [Batch, 30, d_model]

        # 3. Transformer Encoder
        out = self.transformer_encoder(out)  # [Batch, 30, d_model]

        # 4. Global Temporal Average Pooling
        out = torch.mean(out, dim=1)  # [Batch, d_model]

        # 5. Classifier Head
        logits = self.classifier(out)  # [Batch, num_classes]

        return logits


if __name__ == "__main__":
    # Quick sanity test
    dummy_input = torch.randn(16, 30, 258)
    model = TemporalTransformer(num_classes=50)
    output = model(dummy_input)
    print(f"Sanity Check: Input shape {dummy_input.shape} -> Output shape {output.shape}")
