#!/usr/bin/env python3
"""Train the faithful object-flow predictor on task 09."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from policy.OF.faithful.common import (
    AGENT_CAMERA,
    FLOW_HORIZON,
    FLOW_STRIDE,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    MASK_HEIGHT,
    MASK_WIDTH,
    dataset_files,
    encode_instruction,
    instruction_of,
    load_split,
)
from policy.OF.faithful.dataset import FaithfulFlowDataset
from policy.OF.faithful.model import ObjectFlowTransformer, flow_loss


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--masks_dir", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--horizon", type=int, default=FLOW_HORIZON)
    parser.add_argument("--stride", type=int, default=FLOW_STRIDE)
    parser.add_argument("--image_height", type=int, default=IMAGE_HEIGHT)
    parser.add_argument("--image_width", type=int, default=IMAGE_WIDTH)
    parser.add_argument("--mask_height", type=int, default=MASK_HEIGHT)
    parser.add_argument("--mask_width", type=int, default=MASK_WIDTH)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--decoder_depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ff_dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--no_amp", action="store_true")
    return parser


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(args, lang_dim: int) -> ObjectFlowTransformer:
    return ObjectFlowTransformer(
        image_height=args.image_height,
        image_width=args.image_width,
        mask_height=args.mask_height,
        mask_width=args.mask_width,
        horizon=args.horizon,
        lang_dim=lang_dim,
        dim=args.dim,
        depth=args.depth,
        decoder_depth=args.decoder_depth,
        heads=args.heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    )


@torch.no_grad()
def evaluate(model, loader, device, alpha: float, amp: bool) -> dict[str, float]:
    model.eval()
    losses = {"loss": [], "frame_loss": [], "comb_loss": []}
    intersections = 0.0
    unions = 0.0
    for image, target, lang in loader:
        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        lang = lang.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits = model(image, lang)
            _, terms = flow_loss(logits, target, alpha=alpha)
        for key in losses:
            losses[key].append(float(terms[key]))
        prediction = torch.sigmoid(logits.float()) >= 0.5
        truth = target.float() >= 0.5
        intersections += float((prediction & truth).sum())
        unions += float((prediction | truth).sum())
    result = {f"val_{key}": float(np.mean(value)) for key, value in losses.items()}
    result["val_iou"] = intersections / max(unions, 1.0)
    return result


def main() -> None:
    args = build_parser().parse_args()
    seed_everything(args.seed)
    split = load_split(args.split)
    train_files = split["train_files"]
    val_files = split["val_files"]
    if not train_files or not val_files:
        raise ValueError("split must contain both train_files and val_files")

    first_path = Path(args.dataset_dir) / train_files[0]
    import h5py

    with h5py.File(str(first_path), "r") as h5:
        instruction = instruction_of(h5)
    instruction_embedding = encode_instruction(instruction)
    print(f"instruction: {instruction}", flush=True)
    print(f"BERT embedding: {instruction_embedding.shape}", flush=True)

    train_set = FaithfulFlowDataset(
        args.dataset_dir,
        args.masks_dir,
        train_files,
        instruction_embedding,
        horizon=args.horizon,
        stride=args.stride,
        image_height=args.image_height,
        image_width=args.image_width,
        mask_height=args.mask_height,
        mask_width=args.mask_width,
        train=True,
    )
    val_set = FaithfulFlowDataset(
        args.dataset_dir,
        args.masks_dir,
        val_files,
        instruction_embedding,
        horizon=args.horizon,
        stride=args.stride,
        image_height=args.image_height,
        image_width=args.image_width,
        mask_height=args.mask_height,
        mask_width=args.mask_width,
        train=False,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    device = torch.device(args.device)
    model = build_model(args, instruction_embedding.shape[0]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    amp = not args.no_amp and device.type == "cuda"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_iou = -1.0
    history: list[dict] = []
    global_step = 0
    print(
        f"train={len(train_set)} val={len(val_set)} "
        f"steps/epoch={len(train_loader)} amp={amp}",
        flush=True,
    )

    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for image, target, lang in train_loader:
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            lang = lang.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logits = model(image, lang)
                loss, terms = flow_loss(logits, target, alpha=args.alpha)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(terms["loss"]))
            global_step += 1
            if global_step % 50 == 0:
                print(f"epoch {epoch} step {global_step} loss={train_losses[-1]:.4f}", flush=True)
            if args.max_steps and global_step >= args.max_steps:
                break
        scheduler.step()
        metrics = evaluate(model, val_loader, device, args.alpha, amp)
        metrics.update(epoch=epoch, train_loss=float(np.mean(train_losses)), step=global_step)
        history.append(metrics)
        print(json.dumps(metrics), flush=True)
        (output_dir / "metrics.json").write_text(json.dumps(history, indent=2) + "\n")
        payload = {
            "model": model.state_dict(),
            "args": vars(args),
            "instruction": instruction,
            "instruction_embedding": torch.from_numpy(instruction_embedding),
            "objects": ["task_relevant_merged"],
        }
        torch.save(payload, output_dir / "flow_last.pt")
        if metrics["val_iou"] > best_iou:
            best_iou = metrics["val_iou"]
            torch.save(payload, output_dir / "flow_best.pt")
        if args.max_steps and global_step >= args.max_steps:
            break
    print(f"best val_iou={best_iou:.4f}; checkpoint={output_dir / 'flow_best.pt'}")


if __name__ == "__main__":
    main()
