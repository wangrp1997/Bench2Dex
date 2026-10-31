#!/usr/bin/env python3
"""Append predicted role-aware flow to an existing DP Zarr dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F
import zarr

from bimanual_flow import (
    AGENT_CAMERA,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    dataset_files,
    decode_rgb,
    resolve_cutoff,
)
from flow_model import load_flow_model, predict_flow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--zarr", required=True)
    parser.add_argument("--flow_checkpoint", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_height", type=int, default=16)
    parser.add_argument("--output_width", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_episodes", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    model = load_flow_model(args.flow_checkpoint, device)
    files = dataset_files(args.dataset_dir)
    if args.max_episodes:
        files = files[: args.max_episodes]

    root = zarr.open(str(args.zarr), mode="a")
    total_steps = int(root["data/action"].shape[0])
    channels = int(model.n_objects * model.horizon)
    shape = (total_steps, channels, args.output_height, args.output_width)
    data = root["data"]
    if "object_flow" in data:
        flow_array = data["object_flow"]
        if flow_array.shape != shape:
            raise ValueError(f"existing object_flow shape {flow_array.shape} != {shape}")
    else:
        flow_array = data.create_dataset(
            "object_flow",
            shape=shape,
            chunks=(min(64, total_steps), channels, args.output_height, args.output_width),
            dtype="u1",
        )

    offset = 0
    for episode_idx, path in enumerate(files):
        with h5py.File(str(path), "r") as h5:
            total = int(h5["robot/qpos"].shape[0])
            cutoff = resolve_cutoff(h5, total)
            for start in range(0, cutoff, args.batch_size):
                end = min(start + args.batch_size, cutoff)
                images = []
                for t in range(start, end):
                    image = decode_rgb(h5[f"cameras/{AGENT_CAMERA}/rgb"][t])
                    image = cv2.resize(
                        image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA
                    )
                    images.append(image.transpose(2, 0, 1))
                batch = torch.from_numpy(np.stack(images)).float().to(device) / 255.0
                flow = predict_flow(model, batch)
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
    print(f"object_flow ready: {args.zarr}/data/object_flow {shape}")


if __name__ == "__main__":
    main()

