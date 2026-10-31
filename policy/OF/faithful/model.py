"""Transformer object-flow predictor matching the paper's interface."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ObjectFlowTransformer(nn.Module):
    """Predict h future binary masks from ho agent-view frames and language."""

    def __init__(
        self,
        image_height: int = 128,
        image_width: int = 160,
        mask_height: int = 64,
        mask_width: int = 80,
        obs_steps: int = 3,
        horizon: int = 16,
        lang_dim: int = 768,
        dim: int = 256,
        depth: int = 6,
        decoder_depth: int = 4,
        heads: int = 8,
        ff_dim: int = 1024,
        patch: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if image_height % patch or image_width % patch:
            raise ValueError("image size must be divisible by patch size")
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.mask_height = int(mask_height)
        self.mask_width = int(mask_width)
        self.obs_steps = int(obs_steps)
        self.horizon = int(horizon)
        self.patch = int(patch)
        self.grid_h = image_height // patch
        self.grid_w = image_width // patch
        self.n_tokens = self.grid_h * self.grid_w

        self.patch_embed = nn.Conv2d(3, dim, patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, self.n_tokens, dim))
        self.frame_emb = nn.Parameter(torch.zeros(1, obs_steps, dim))
        self.lang_proj = nn.Linear(int(lang_dim), dim)
        self.query = nn.Parameter(torch.zeros(1, horizon, dim))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=decoder_depth)
        self.mask_mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, self.n_tokens),
            nn.GELU(),
        )
        self.mask_decoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, images: torch.Tensor, lang: torch.Tensor) -> torch.Tensor:
        """images: (B, obs_steps*3, H, W), lang: (B, lang_dim)."""
        batch = images.shape[0]
        frames = images.reshape(batch, self.obs_steps, 3, self.image_height, self.image_width)
        tokens = self.patch_embed(
            frames.reshape(batch * self.obs_steps, 3, self.image_height, self.image_width)
        )
        tokens = tokens.flatten(2).transpose(1, 2)
        tokens = tokens.reshape(batch, self.obs_steps, self.n_tokens, -1)
        tokens = tokens + self.pos[:, None] + self.frame_emb[:, :, None]
        memory = tokens.reshape(batch, self.obs_steps * self.n_tokens, -1)

        lang_token = self.lang_proj(lang).unsqueeze(1)
        memory = self.encoder(torch.cat([lang_token, memory], dim=1))
        queries = self.query.expand(batch, -1, -1) + lang_token
        decoded = self.decoder(queries, memory)
        spatial = self.mask_mlp(decoded).reshape(batch * self.horizon, 1, self.grid_h, self.grid_w)
        masks = self.mask_decoder(spatial)
        return masks.reshape(batch, self.horizon, self.mask_height, self.mask_width)


def focal_dice_from_prob(p: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Focal and Dice terms from the paper (per image, batch-averaged)."""
    p = p.float().clamp(1e-6, 1.0 - 1e-6)
    y = y.float()
    bce = F.binary_cross_entropy_with_logits(torch.logit(p), y, reduction="none")
    weight = (p + y - p * y).pow(2)
    focal = weight.mul(bce).flatten(1).mean(dim=1)
    inter = (p * y).flatten(1).sum(dim=1)
    denom = p.flatten(1).sum(dim=1) + y.flatten(1).sum(dim=1)
    dice = 1.0 - (2.0 * inter + 1e-6) / (denom + 1e-6)
    return focal, dice


def flow_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float = 5.0):
    """Lflow = alpha * frame loss + frame-combination loss."""
    p = torch.sigmoid(logits.float())
    flat_p = p.reshape(-1, *p.shape[-2:])
    flat_y = target.reshape(-1, *target.shape[-2:])
    frame_focal, frame_dice = focal_dice_from_prob(flat_p, flat_y)
    frame_focal = frame_focal.reshape(*p.shape[:2])
    frame_dice = frame_dice.reshape(*p.shape[:2])
    lff = (1.0 + frame_focal + frame_dice).mean()

    p_comb = (1.0 - torch.prod(1.0 - p, dim=1)).clamp(1e-6, 1.0 - 1e-6)
    y_comb = target.amax(dim=1)
    comb_focal, comb_dice = focal_dice_from_prob(p_comb, y_comb)
    lfc = (1.0 + comb_focal + comb_dice).mean()
    total = alpha * lff + lfc
    return total, {
        "loss": float(total.detach()),
        "frame_loss": float(lff.detach()),
        "comb_loss": float(lfc.detach()),
    }
