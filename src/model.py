"""A compact U-Net for binary building segmentation (Ronneberger et al., 2015).

Shape walk-through for the default config (6 bands, base 32, depth 4) and a
64 x 64 input:

    input              (B,   6, 64, 64)
    encoder level 1    (B,  32, 64, 64)  -> skip 1, then max-pool -> 32 x 32
    encoder level 2    (B,  64, 32, 32)  -> skip 2, then max-pool -> 16 x 16
    encoder level 3    (B, 128, 16, 16)  -> skip 3, then max-pool ->  8 x  8
    encoder level 4    (B, 256,  8,  8)  -> skip 4, then max-pool ->  4 x  4
    bottleneck         (B, 512,  4,  4)
    decoder 4..1       up-sample x2, concatenate the matching skip, conv block
    head (1x1 conv)    (B,   1, 64, 64)  raw logits

The model returns LOGITS (unbounded scores). Apply torch.sigmoid to get
building probabilities; the loss (BCEWithLogitsLoss) works on logits directly
because that is numerically more stable than sigmoid followed by BCE.
"""
from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Two 3x3 convolutions, each followed by BatchNorm and ReLU.

    3x3 kernels with padding=1 keep the spatial size unchanged. BatchNorm
    stabilizes training with small batches of differently-lit tiles.
    """

    def __init__(self, c_in: int, c_out: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 6, base_channels: int = 32, depth: int = 4, dropout: float = 0.0):
        super().__init__()
        chs = [base_channels * 2**i for i in range(depth)]  # e.g. [32, 64, 128, 256]

        # Encoder ("contracting path"): learns increasingly abstract features
        # while pooling halves the resolution at each level.
        self.encoders = nn.ModuleList()
        c_prev = in_channels
        for c in chs:
            self.encoders.append(ConvBlock(c_prev, c))
            c_prev = c
        self.pool = nn.MaxPool2d(2)

        # Bottleneck: widest, lowest-resolution representation (most context).
        self.bottleneck = ConvBlock(chs[-1], chs[-1] * 2, dropout)

        # Decoder ("expanding path"): transposed convolutions double the
        # resolution; skip connections re-inject the fine spatial detail from
        # the encoder that pooling discarded.
        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        c_prev = chs[-1] * 2
        for c in reversed(chs):
            self.upsamples.append(nn.ConvTranspose2d(c_prev, c, kernel_size=2, stride=2))
            self.decoders.append(ConvBlock(c * 2, c))  # *2 because of the concatenated skip
            c_prev = c

        # 1x1 convolution maps the features of each pixel to one logit.
        self.head = nn.Conv2d(chs[0], 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        factor = 2 ** len(self.encoders)
        assert h % factor == 0 and w % factor == 0, f"input size must be divisible by {factor}"
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.upsamples, self.decoders, reversed(skips)):
            x = up(x)
            x = dec(torch.cat([x, skip], dim=1))
        return self.head(x)


def build_model(cfg: dict) -> UNet:
    m = cfg["model"]
    return UNet(m["in_channels"], m["base_channels"], m["depth"], m["dropout"])


def save_checkpoint(path, model: nn.Module, cfg: dict, extra: dict) -> None:
    torch.save({"state_dict": model.state_dict(), "model_cfg": cfg["model"], **extra}, path)


def load_checkpoint(path, device="cpu") -> tuple[UNet, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    m = ckpt["model_cfg"]
    model = UNet(m["in_channels"], m["base_channels"], m["depth"], m["dropout"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
