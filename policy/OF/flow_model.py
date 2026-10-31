"""Small role-aware object-flow predictor used by the Bench2Dex adaptation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RoleFlowNet(nn.Module):
    """Predict future masks for box, cleaner, and soap from one RGB frame."""

    def __init__(
        self,
        n_objects: int = 3,
        horizon: int = 16,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        self.n_objects = int(n_objects)
        self.horizon = int(horizon)
        c = int(base_channels)
        self.enc1 = ConvBlock(3, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.enc4 = ConvBlock(c * 4, c * 8)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.dec3 = ConvBlock(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = ConvBlock(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = ConvBlock(c * 2, c)
        self.head = nn.Conv2d(c, self.n_objects * self.horizon, 1)
        self.output_pool = nn.AvgPool2d(2)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(image)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        logits = self.output_pool(self.head(d1))
        batch = logits.shape[0]
        return logits.reshape(
            batch, self.n_objects, self.horizon, logits.shape[-2], logits.shape[-1]
        )


def load_flow_model(checkpoint_path: str, device: torch.device) -> RoleFlowNet:
    payload = torch.load(checkpoint_path, map_location="cpu")
    args = payload.get("args", {})
    model = RoleFlowNet(
        n_objects=len(payload.get("objects", (None, None, None))),
        horizon=int(args.get("horizon", 16)),
        base_channels=int(args.get("base_channels", 32)),
    )
    model.load_state_dict(payload["model"])
    return model.to(device).eval()


@torch.no_grad()
def predict_flow(model: RoleFlowNet, image: torch.Tensor) -> torch.Tensor:
    """Predict a (B, objects*horizon, H, W) probability tensor."""
    logits = model(image)
    batch, n_objects, horizon, height, width = logits.shape
    return torch.sigmoid(logits).reshape(batch, n_objects * horizon, height, width)

