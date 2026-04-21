"""
Temporal probes for span-level hallucination detection.

Instead of aggregating the per-token lookback-ratio sequence down to a
single vector (Step 2), these models read the full (C, T) sequence and
learn where in the span the signal lives. The proposal calls this the
"Temporal Lookback Dynamics" component.

Two architectures, both small to avoid overfitting AggreFact's ~1k train
examples:

    LookbackCNN  — two 1D conv layers over channels=L*H, global max-pool,
                   MLP head. Captures short local attention patterns.

    LookbackLSTM — single-layer LSTM on (T, C), last hidden → MLP head.
                   Captures long-range temporal structure.

Both accept a padding mask so variable-length spans can live in one batch:
positions where mask==False are zeroed (CNN) or skipped (LSTM via
pack_padded_sequence).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LookbackCNN(nn.Module):
    """
    Input : x (B, C, T) — channels = L*H (lookback only) or wider if extra
                          feature streams are concatenated along channel dim.
            mask (B, T)  — True for valid positions, False for pad.
    Output: logits (B,)  — positive = faithful (matches step 2 convention).
    """

    def __init__(self, in_channels: int, hidden: int = 64,
                 kernel_sizes=(3, 5), dropout: float = 0.3):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, hidden, kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden * len(kernel_sizes), hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Zero padded positions so they don't affect conv activations or
        # (crucially) the global max-pool.
        x = x * mask.unsqueeze(1).float()

        pooled = []
        for conv in self.convs:
            h = F.relu(conv(x))                                 # (B, hidden, T)
            # Mask pad positions to -inf before global max-pool. Using a
            # very negative number rather than -inf avoids NaN if a whole
            # row happens to be all-pad (shouldn't happen but be safe).
            h = h.masked_fill(~mask.unsqueeze(1), -1e4)
            h = h.amax(dim=-1)                                  # (B, hidden)
            pooled.append(h)

        z = torch.cat(pooled, dim=-1)                           # (B, hidden * K)
        z = self.dropout(z)
        return self.head(z).squeeze(-1)                         # (B,)


class LookbackLSTM(nn.Module):
    """
    Input : x (B, C, T), mask (B, T).
    Output: logits (B,).

    Internally transposes to (B, T, C) and uses pack_padded_sequence so
    padded steps genuinely don't affect the hidden state.
    """

    def __init__(self, in_channels: int, hidden: int = 64,
                 num_layers: int = 1, dropout: float = 0.3,
                 bidirectional: bool = True):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        d = hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # (B, C, T) → (B, T, C)
        x = x.transpose(1, 2).contiguous()
        lengths = mask.sum(dim=-1).clamp(min=1).cpu()

        # pack_padded_sequence requires lengths on CPU and enforce_sorted=False
        # so we don't have to reorder the batch ourselves.
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        # h_n : (num_layers * num_directions, B, hidden)
        # Take the final layer's forward (and backward) hidden states.
        if self.lstm.bidirectional:
            fwd = h_n[-2]
            bwd = h_n[-1]
            h = torch.cat([fwd, bwd], dim=-1)                   # (B, 2*hidden)
        else:
            h = h_n[-1]                                         # (B, hidden)
        return self.head(h).squeeze(-1)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
