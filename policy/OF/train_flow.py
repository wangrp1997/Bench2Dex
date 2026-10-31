#!/usr/bin/env python3
"""Train the role-aware object-flow predictor on Bench2Dex task 09."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from bimanual_flow import (
    BimanualFlowDataset,
    FLOW_HORIZON,
    FLOW_STRIDE,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    MASK_HEIGHT,
    MASK_WIDTH,
    OBJECTS,
    build_split,
    load_split,
)
from flow_model import RoleFlowNet


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=FLOW_HORIZON)
    parser.add_argument("--stride", type=int, default=FLOW_STRIDE)
    parser.add_argument("--image_height", type=int, default=IMAGE_HEIGHT)
    parser.add_argument("--image_width", type=int, default=IMAGE_WIDTH)
    parser.add_argument("--mask_height", type=int, default=MASK_HEIGHT)
    parser.add_argument("--mask_width", type=int, default=MASK_WIDTH)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    return parser


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def flow_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pos_weight = torch.tensor(5.0, device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    prob = torch.sigmoid(logits)
    dims = tuple(range(2, prob.ndim))
    dice = 1.0 - (2.0 * (prob * target).sum(dims) + 1.0) / (
        prob.sum(dims) + target.sum(dims) + 1.0
    )
    return bce + dice.mean()


@torch.no_grad()
def evaluate(model: RoleFlowNet, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    losses = []
    intersections = 0.0
    unions = 0.0
    for image, target in loader:
        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        logits = model(image)
        losses.append(float(flow_loss(logits, target)))
        prediction = torch.sigmoid(logits) >= 0.5
        truth = target >= 0.5
        intersections += float((prediction & truth).sum())
        unions += float((prediction | truth).sum())
    return {
        "val_loss": float(np.mean(losses)) if losses else float("nan"),
        "val_iou": intersections / max(unions, 1.0),
    }


def main() -> None:
    args = build_parser().parse_args()
    seed_everything(args.seed)
    if not Path(args.split).is_file():
        build_split(args.dataset_dir, args.split, seed=args.seed)
    split = load_split(args.split)

    train_set = BimanualFlowDataset(
        args.dataset_dir,
        split["train_files"],
        image_height=args.image_height,
        image_width=args.image_width,
        mask_height=args.mask_height,
        mask_width=args.mask_width,
        horizon=args.horizon,
        stride=args.stride,
    )
    val_set = BimanualFlowDataset(
        args.dataset_dir,
        split["val_files"],
        image_height=args.image_height,
        image_width=args.image_width,
        mask_height=args.mask_height,
        mask_width=args.mask_width,
        horizon=args.horizon,
        stride=args.stride,
    )
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    device = torch.device(args.device)
    model = RoleFlowNet(
        n_objects=len(OBJECTS),
        horizon=args.horizon,
        base_channels=args.base_channels,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_iou = -1.0
    history = []
    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for image, target in train_loader:
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            loss = flow_loss(model(image), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss))
        scheduler.step()
        metrics = evaluate(model, val_loader, device)
        metrics.update(
            epoch=epoch,
            train_loss=float(np.mean(train_losses)),
            lr=float(optimizer.param_groups[0]["lr"]),
        )
        history.append(metrics)
        print(json.dumps(metrics), flush=True)
        payload = {"model": model.state_dict(), "args": vars(args), "objects": list(OBJECTS)}
        torch.save(payload, output_dir / "flow_last.pt")
        if metrics["val_iou"] > best_iou:
            best_iou = metrics["val_iou"]
            torch.save(payload, output_dir / "flow_best.pt")
    (output_dir / "metrics.json").write_text(json.dumps(history, indent=2) + "\n")
    print(f"best val_iou={best_iou:.4f}; checkpoint={output_dir / 'flow_best.pt'}")


if __name__ == "__main__":
    main()

