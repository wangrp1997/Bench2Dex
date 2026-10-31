#!/usr/bin/env python3
"""Append faithful 16-step object flow to an existing DP Zarr store."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F
import zarr

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from policy.OF.faithful.common import (
    AGENT_CAMERA,
    FLOW_HORIZON,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    OBS_STEPS,
    dataset_files,
    decode_rgb,
    resolve_cutoff,
)
from policy.OF.faithful.model import ObjectFlowTransformer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--zarr", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_height", type=int, default=16)
    parser.add_argument("--output_width", type=int, default=20)
    parser.add_argument("--key", default="object_flow_faithful")
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def load_model(checkpoint: str, device: torch.device) -> tuple[ObjectFlowTransformer, np.ndarray]:
    payload = torch.load(checkpoint, map_location="cpu")
    args = payload["args"]
    lang_embedding = payload["instruction_embedding"].numpy().astype(np.float32)
    model = ObjectFlowTransformer(
        image_height=args["image_height"],
        image_width=args["image_width"],
        mask_height=args["mask_height"],
        mask_width=args["mask_width"],
        obs_steps=OBS_STEPS,
        horizon=args["horizon"],
        lang_dim=lang_embedding.shape[0],
        dim=args["dim"],
        depth=args["depth"],
        decoder_depth=args["decoder_depth"],
        heads=args["heads"],
        ff_dim=args["ff_dim"],
        dropout=args["dropout"],
    )
    model.load_state_dict(payload["model"])
    return model.to(device).eval(), lang_embedding


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    model, lang_embedding = load_model(args.checkpoint, device)
    files = dataset_files(args.dataset_dir)
    if args.max_episodes:
        files = files[: args.max_episodes]

    root = zarr.open(str(args.zarr), mode="a")
    total_steps = int(root["data/action"].shape[0])
    channels = int(FLOW_HORIZON)
    shape = (total_steps, channels, args.output_height, args.output_width)
    data = root["data"]
    if args.key in data:
        if args.overwrite:
            del data[args.key]
        elif data[args.key].shape != shape:
            raise ValueError(f"existing {args.key} shape {data[args.key].shape} != {shape}")
    if args.key not in data:
        flow_array = data.create_dataset(
            args.key,
            shape=shape,
            chunks=(min(64, total_steps), channels, args.output_height, args.output_width),
            dtype="u1",
        )
    else:
        flow_array = data[args.key]

    lang = torch.from_numpy(lang_embedding).float().unsqueeze(0)
    offset = 0
    for episode_idx, path in enumerate(files):
        with h5py.File(str(path), "r") as h5:
            total = int(h5["robot/qpos"].shape[0])
            cutoff = resolve_cutoff(h5, total)
            images = []
            for frame_idx in range(cutoff):
                image = decode_rgb(h5[f"cameras/{AGENT_CAMERA}/rgb"][frame_idx])
                images.append(
                    cv2.resize(
                        image,
                        (IMAGE_WIDTH, IMAGE_HEIGHT),
                        interpolation=cv2.INTER_AREA,
                    )
                )
            images = np.stack(images, axis=0)
            for start in range(0, cutoff, args.batch_size):
                end = min(start + args.batch_size, cutoff)
                idx = np.arange(start, end)
                past = np.stack(
                    [
                        images[np.maximum(0, idx - 2)],
                        images[np.maximum(0, idx - 1)],
                        images[idx],
                    ],
                    axis=1,
                )
                batch = torch.from_numpy(past).float().to(device) / 255.0
                batch = batch.permute(0, 1, 4, 2, 3).reshape(
                    end - start, OBS_STEPS * 3, IMAGE_HEIGHT, IMAGE_WIDTH
                )
                batch_lang = lang.repeat(end - start, 1).to(device)
                with torch.inference_mode():
                    logits = model(batch, batch_lang)
                    flow = torch.sigmoid(logits.float())
                    flow = F.interpolate(
                        flow,
                        size=(args.output_height, args.output_width),
                        mode="bilinear",
                        align_corners=False,
                    )
                flow = (flow * 255.0).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
                flow_array[offset : offset + end - start] = flow
                offset += end - start
        print(
            f"episode {episode_idx + 1}/{len(files)} {path.name}: "
            f"{cutoff} frames, total={offset}",
            flush=True,
        )
    if offset != total_steps:
        raise RuntimeError(f"wrote {offset} frames but Zarr has {total_steps}")
    print(f"{args.key} ready: {args.zarr}/data/{args.key} {shape}")


if __name__ == "__main__":
    main()
