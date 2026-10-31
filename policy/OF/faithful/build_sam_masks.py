#!/usr/bin/env python3
"""Build per-frame SAM masks for task-relevant objects (Cutie substitute)."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from transformers import SamModel, SamProcessor

from .common import (
    AGENT_CAMERA,
    MASK_HEIGHT,
    MASK_WIDTH,
    OBJECTS,
    dataset_files,
    decode_rgb,
    resolve_cutoff,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--model", default="facebook/sam-vit-base")
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _load_model(model_name: str, device: torch.device):
    processor = SamProcessor.from_pretrained(model_name)
    model = SamModel.from_pretrained(model_name).eval().to(device)
    return processor, model


def _boxes_and_visibility(h5: h5py.File, frame_idx: int) -> tuple[list[list[float]], list[bool]]:
    boxes: list[list[float]] = []
    visible: list[bool] = []
    for obj in OBJECTS:
        xyxy = h5[f"labels/box2d/{AGENT_CAMERA}/{obj}/xyxy"][frame_idx]
        is_visible = bool(h5[f"labels/box2d/{AGENT_CAMERA}/{obj}/visible"][frame_idx])
        box = np.asarray(xyxy, dtype=np.float32).tolist()
        if not is_visible or box[2] <= box[0] or box[3] <= box[1]:
            box = [0.0, 0.0, 1.0, 1.0]
            is_visible = False
        boxes.append(box)
        visible.append(is_visible)
    return boxes, visible


def _to_small(mask: np.ndarray) -> np.ndarray:
    small = cv2.resize(
        mask.astype(np.uint8),
        (MASK_WIDTH, MASK_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )
    return (small > 0).astype(np.uint8) * 255


def process_episode(
    path: Path,
    output_dir: Path,
    processor: SamProcessor,
    model: SamModel,
    device: torch.device,
    batch_size: int,
) -> None:
    out_path = output_dir / path.name
    with h5py.File(str(path), "r") as h5:
        total = int(h5["robot/qpos"].shape[0])
        cutoff = resolve_cutoff(h5, total)
        with h5py.File(str(out_path), "w") as out:
            objects = out.create_dataset(
                "objects",
                shape=(cutoff, len(OBJECTS), MASK_HEIGHT, MASK_WIDTH),
                dtype="u1",
                chunks=(min(64, cutoff), len(OBJECTS), MASK_HEIGHT, MASK_WIDTH),
            )
            merged = out.create_dataset(
                "merged",
                shape=(cutoff, MASK_HEIGHT, MASK_WIDTH),
                dtype="u1",
                chunks=(min(64, cutoff), MASK_HEIGHT, MASK_WIDTH),
            )
            out.attrs["source"] = str(path)
            out.attrs["camera"] = AGENT_CAMERA
            out.attrs["objects"] = ",".join(OBJECTS)
            out.attrs["segmentation"] = "SAM box-prompt, Cutie substitute"
            for start in range(0, cutoff, batch_size):
                end = min(start + batch_size, cutoff)
                images = []
                all_boxes = []
                all_visible = []
                for frame_idx in range(start, end):
                    images.append(decode_rgb(h5[f"cameras/{AGENT_CAMERA}/rgb"][frame_idx]))
                    boxes, visible = _boxes_and_visibility(h5, frame_idx)
                    all_boxes.append(boxes)
                    all_visible.append(visible)
                inputs = processor(images=images, input_boxes=all_boxes, return_tensors="pt")
                inputs = {
                    key: value.to(device) if isinstance(value, torch.Tensor) else value
                    for key, value in inputs.items()
                }
                with torch.inference_mode():
                    outputs = model(**inputs, multimask_output=True)
                masks = processor.image_processor.post_process_masks(
                    outputs.pred_masks.cpu(),
                    inputs["original_sizes"].cpu(),
                    inputs["reshaped_input_sizes"].cpu(),
                )
                scores = outputs.iou_scores.cpu()
                for local_idx, frame_idx in enumerate(range(start, end)):
                    per_object = np.zeros(
                        (len(OBJECTS), MASK_HEIGHT, MASK_WIDTH), dtype=np.uint8
                    )
                    for obj_idx, is_visible in enumerate(all_visible[local_idx]):
                        if not is_visible:
                            continue
                        best = int(scores[local_idx, obj_idx].argmax())
                        mask = masks[local_idx][obj_idx, best].numpy() > 0
                        per_object[obj_idx] = _to_small(mask)
                    objects[frame_idx] = per_object
                    merged[frame_idx] = np.any(per_object > 0, axis=0).astype(np.uint8) * 255
                print(
                    f"{path.name}: {end}/{cutoff} frames "
                    f"({100.0 * end / cutoff:.1f}%)",
                    flush=True,
                )


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    processor, model = _load_model(args.model, device)
    files = dataset_files(args.dataset_dir)
    if args.max_episodes:
        files = files[: args.max_episodes]
    started = time.time()
    for episode_idx, path in enumerate(files):
        out_path = output_dir / path.name
        if out_path.exists() and not args.overwrite:
            print(f"skip {path.name}: {out_path} exists", flush=True)
            continue
        if out_path.exists():
            out_path.unlink()
        print(f"[{episode_idx + 1}/{len(files)}] {path.name}", flush=True)
        process_episode(path, output_dir, processor, model, device, args.batch_size)
    elapsed = time.time() - started
    print(f"done: {len(files)} episodes in {elapsed / 60:.1f} min -> {output_dir}")


if __name__ == "__main__":
    main()
