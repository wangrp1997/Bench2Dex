"""Dataset for future SAM masks, past RGB frames, and BERT instructions."""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .common import (
    AGENT_CAMERA,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    MASK_HEIGHT,
    MASK_WIDTH,
    OBS_STEPS,
    decode_rgb,
    resolve_cutoff,
)


def _color_jitter(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32) / 255.0
    mean = image.mean(axis=(0, 1), keepdims=True)
    image = (image - mean) * random.uniform(0.8, 1.2) + mean
    gray = image.mean(axis=2, keepdims=True)
    image = gray + (image - gray) * random.uniform(0.8, 1.2)
    image = image * random.uniform(0.8, 1.2)
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


class FaithfulFlowDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str | Path,
        masks_dir: str | Path,
        files: list[str],
        instruction_embedding: np.ndarray,
        *,
        horizon: int = 16,
        stride: int = 1,
        obs_steps: int = OBS_STEPS,
        image_height: int = IMAGE_HEIGHT,
        image_width: int = IMAGE_WIDTH,
        mask_height: int = MASK_HEIGHT,
        mask_width: int = MASK_WIDTH,
        train: bool = True,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.masks_dir = Path(masks_dir)
        self.files = [str(name) for name in files]
        self.instruction_embedding = np.asarray(instruction_embedding, dtype=np.float32)
        self.horizon = int(horizon)
        self.stride = int(stride)
        self.obs_steps = int(obs_steps)
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.mask_height = int(mask_height)
        self.mask_width = int(mask_width)
        self.train = bool(train)
        self._h5 = None
        self._mask_h5 = None
        self._h5_path = None
        self._mask_path = None

        self.indices: list[tuple[str, int]] = []
        for name in self.files:
            with h5py.File(str(self.dataset_dir / name), "r") as h5:
                cutoff = resolve_cutoff(h5, int(h5["robot/qpos"].shape[0]))
            self.indices.extend((name, t) for t in range(cutoff))

    def _open(self, name: str) -> tuple[h5py.File, h5py.File]:
        image_path = str(self.dataset_dir / name)
        mask_path = str(self.masks_dir / name)
        if self._h5_path != image_path:
            if self._h5 is not None:
                self._h5.close()
            self._h5 = h5py.File(image_path, "r")
            self._h5_path = image_path
        if self._mask_path != mask_path:
            if self._mask_h5 is not None:
                self._mask_h5.close()
            self._mask_h5 = h5py.File(mask_path, "r")
            self._mask_path = mask_path
        return self._h5, self._mask_h5

    def __len__(self) -> int:
        return len(self.indices)

    def _augment(self, frames: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.train:
            return frames, masks
        frames = np.stack([_color_jitter(frame) for frame in frames], axis=0)
        dx = random.randint(-8, 8)
        dy = random.randint(-8, 8)
        transform = np.float32([[1, 0, dx], [0, 1, dy]])
        frames = np.stack(
            [
                cv2.warpAffine(
                    frame,
                    transform,
                    (self.image_width, self.image_height),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101,
                )
                for frame in frames
            ],
            axis=0,
        )
        mask_transform = np.float32(
            [
                [1, 0, dx * self.mask_width / self.image_width],
                [0, 1, dy * self.mask_height / self.image_height],
            ]
        )
        masks = np.stack(
            [
                cv2.warpAffine(
                    mask,
                    mask_transform,
                    (self.mask_width, self.mask_height),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                )
                for mask in masks
            ],
            axis=0,
        )
        return frames, masks

    def __getitem__(self, index: int):
        name, t = self.indices[index]
        h5, mask_h5 = self._open(name)
        cutoff = resolve_cutoff(h5, int(h5["robot/qpos"].shape[0]))
        frame_ids = [max(0, t - self.obs_steps + 1 + i) for i in range(self.obs_steps)]
        frames = []
        for frame_idx in frame_ids:
            image = decode_rgb(h5[f"cameras/{AGENT_CAMERA}/rgb"][frame_idx])
            frames.append(
                cv2.resize(
                    image,
                    (self.image_width, self.image_height),
                    interpolation=cv2.INTER_AREA,
                )
            )
        times = np.minimum(t + self.stride * np.arange(self.horizon), cutoff - 1)
        unique_times, inverse = np.unique(times, return_inverse=True)
        masks = mask_h5["merged"][unique_times][inverse].astype(np.float32) / 255.0
        frames, masks = self._augment(np.stack(frames, axis=0), masks)
        frames = frames.transpose(0, 3, 1, 2).reshape(
            self.obs_steps * 3, self.image_height, self.image_width
        )
        return (
            torch.from_numpy(np.ascontiguousarray(frames)).float() / 255.0,
            torch.from_numpy(np.ascontiguousarray(masks)).float(),
            torch.from_numpy(self.instruction_embedding.copy()).float(),
        )

    def __del__(self):
        if self._h5 is not None:
            self._h5.close()
        if self._mask_h5 is not None:
            self._mask_h5.close()
