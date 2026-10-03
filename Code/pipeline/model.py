"""Models: the original CNN -> BiGRU -> attention, and the configurable SeqNet family."""

from __future__ import annotations

import math

import torch.nn as nn

from .config import NUM_CLASSES, ModelConfig, SeqModelConfig


def adjust_num_heads(embed_dim: int, requested_heads: int) -> int:
    """Largest head count <= requested that divides embed_dim."""
    for h in range(max(requested_heads, 1), 0, -1):
        if embed_dim % h == 0:
            return h
    return 1


def init_weights(module):
    if isinstance(module, (nn.Conv1d, nn.Linear)):
        nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
        if module.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
            bound = 1 / math.sqrt(max(1, fan_in))
            nn.init.uniform_(module.bias, -bound, bound)
    elif isinstance(module, (nn.GRU, nn.LSTM)):
        for name, param in module.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)


class CNNFrontEnd(nn.Module):
    """Input (B, T, C) -> output (B, T, F)."""

    def __init__(self, in_channels, out_channels=64, kernel_size=3, dropout=0.2):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=pad),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x.permute(0, 2, 1)).permute(0, 2, 1)


class CNNBiGRU_Attn(nn.Module):
    def __init__(
        self,
        input_channels: int,
        cnn_ch: int = 64,
        gru_hidden: int = 128,
        gru_layers: int = 2,
        attn_heads: int = 4,
        dropout: float = 0.3,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        self.cnn = CNNFrontEnd(input_channels, out_channels=cnn_ch, dropout=dropout)
        self.gru = nn.GRU(
            input_size=cnn_ch,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        embed_dim = gru_hidden * 2
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=adjust_num_heads(embed_dim, attn_heads),
            batch_first=True,
            dropout=0.1,
        )
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        self.apply(init_weights)

    def forward(self, x):
        x = self.cnn(x)
        x, _ = self.gru(x)
        x = self.norm(x)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        return self.classifier(attn_out.mean(dim=1))


class ConvBlock(nn.Module):
    """Conv (standard or depthwise-separable) -> BatchNorm -> ReLU -> optional max-pool."""

    def __init__(self, c_in: int, c_out: int, kernel_size: int, pool: int, separable: bool):
        super().__init__()
        pad = kernel_size // 2
        if separable:
            conv = [
                nn.Conv1d(c_in, c_in, kernel_size, padding=pad, groups=c_in, bias=False),
                nn.Conv1d(c_in, c_out, 1, bias=False),
            ]
        else:
            conv = [nn.Conv1d(c_in, c_out, kernel_size, padding=pad, bias=False)]
        layers = [*conv, nn.BatchNorm1d(c_out), nn.ReLU()]
        if pool > 1:
            layers.append(nn.MaxPool1d(pool))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SeqNet(nn.Module):
    """Configurable model family used for the M3 comparison (see SeqModelConfig).

    Input (B, T, C). The conv front-end shortens the sequence by the product of
    the downsample factors before any recurrent layer or attention, so their
    cost falls with it.
    """

    def __init__(self, input_channels: int, cfg: SeqModelConfig, num_classes: int):
        super().__init__()
        blocks, c_in = [], input_channels
        for c_out, pool in zip(cfg.conv_channels, cfg.downsample):
            blocks.append(ConvBlock(c_in, c_out, cfg.kernel_size, pool, cfg.separable))
            c_in = c_out
        self.front = nn.Sequential(*blocks)
        # Without conv blocks, a single downsample factor average-pools the raw input.
        self.input_pool = (
            nn.AvgPool1d(cfg.downsample[0]) if not cfg.conv_channels and cfg.downsample and cfg.downsample[0] > 1
            else None
        )
        self.front_dropout = nn.Dropout(cfg.dropout)

        self.rnn = None
        feat = c_in
        if cfg.rnn != "none":
            rnn_cls = nn.GRU if cfg.rnn == "gru" else nn.LSTM
            self.rnn = rnn_cls(
                input_size=c_in, hidden_size=cfg.rnn_hidden, num_layers=cfg.rnn_layers, batch_first=True,
                bidirectional=cfg.bidirectional, dropout=cfg.dropout if cfg.rnn_layers > 1 else 0.0,
            )
            feat = cfg.rnn_hidden * (2 if cfg.bidirectional else 1)
        self.norm = nn.LayerNorm(feat)

        self.attn = None
        if cfg.attention:
            self.attn = nn.MultiheadAttention(
                embed_dim=feat, num_heads=adjust_num_heads(feat, cfg.attn_heads), batch_first=True, dropout=0.1,
            )
        hidden = min(128, max(feat, num_classes))
        self.classifier = nn.Sequential(
            nn.Linear(feat, hidden), nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(hidden, num_classes),
        )
        self.apply(init_weights)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, C, T)
        if self.input_pool is not None:
            x = self.input_pool(x)
        x = self.front_dropout(self.front(x)).permute(0, 2, 1)  # (B, T', F)
        if self.rnn is not None:
            x, _ = self.rnn(x)
        x = self.norm(x)
        if self.attn is not None:
            x, _ = self.attn(x, x, x, need_weights=False)
        return self.classifier(x.mean(dim=1))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(input_channels: int, cfg: ModelConfig | SeqModelConfig, num_classes: int = NUM_CLASSES) -> nn.Module:
    if isinstance(cfg, SeqModelConfig):
        return SeqNet(input_channels, cfg, num_classes)
    return CNNBiGRU_Attn(
        input_channels=input_channels,
        cnn_ch=cfg.cnn_channels,
        gru_hidden=cfg.gru_hidden,
        gru_layers=cfg.gru_layers,
        attn_heads=cfg.attn_heads,
        dropout=cfg.dropout,
        num_classes=num_classes,
    )
