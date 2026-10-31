"""Role-aware object-flow data utilities for Bench2Dex task 09.

The released replay data has one behaviour per demonstration, so this module
uses all task-relevant objects instead of collapsing them into one binary mask.
That preserves object roles, which matter for bimanual hand/object allocation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch


OBJECTS = (
    "obj_154_wooden_box_1",
    "obj_200_cleaner_2",
    "obj_209_soap_3",
)
AGENT_CAMERA = "cam_stereo_left"
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 160
MASK_HEIGHT = 64
MASK_WIDTH = 80
FLOW_HORIZON = 16
FLOW_STRIDE = 3


def resolve_cutoff(h5: h5py.File, total: int) -> int:
    """Return the frame count before homing starts."""
    marker = h5.get("meta/homing_start_sim_step")
    if marker is None or "time/sim_step" not in h5:
        return total
    homing = int(marker[()])
    if homing < 0:
        return total
    cutoff = int(np.searchsorted(h5["time/sim_step"][:], homing, side="left"))
    return cutoff if 0 < cutoff < total else total


def episode_group_key(path: str | Path) -> str:
    """Group visual/scene variants that share exactly the same robot trajectory."""
    with h5py.File(str(path), "r") as h5:
        qpos = np.ascontiguousarray(h5["robot/qpos"][:])
    return hashlib.sha256(qpos.tobytes()).hexdigest()


def dataset_files(dataset_dir: str | Path) -> list[Path]:
    files = sorted(Path(dataset_dir).glob("episode_*.hdf5"))
    if not files:
        raise FileNotFoundError(f"no episode_*.hdf5 under {dataset_dir}")
    return files


def build_split(
    dataset_dir: str | Path,
    output: str | Path,
    *,
    train_groups: int = 40,
    val_groups: int = 10,
    seed: int = 0,
) -> dict:
    """Split by unique qpos hash so `_1` scene variants cannot leak across splits."""
    files = dataset_files(dataset_dir)
    groups: dict[str, list[str]] = {}
    for path in files:
        groups.setdefault(episode_group_key(path), []).append(path.name)
    if len(groups) < train_groups + val_groups:
        raise ValueError(
            f"need {train_groups + val_groups} unique trajectories, found {len(groups)}"
        )

    keys = sorted(groups)
    np.random.default_rng(seed).shuffle(keys)
    val_keys = set(keys[:val_groups])
    train_keys = set(keys[val_groups : val_groups + train_groups])
    episode_indices = {path.name: i for i, path in enumerate(files)}
    train_files = sorted(name for key in train_keys for name in groups[key])
    val_files = sorted(name for key in val_keys for name in groups[key])

    split = {
        "dataset_dir": str(Path(dataset_dir).resolve()),
        "seed": seed,
        "group_key": "sha256(robot/qpos) over the full episode",
        "objects": list(OBJECTS),
        "agent_camera": AGENT_CAMERA,
        "train_files": train_files,
        "val_files": val_files,
        "episode_indices": episode_indices,
        "train_episode_indices": sorted(episode_indices[name] for name in train_files),
        "val_episode_indices": sorted(episode_indices[name] for name in val_files),
        "groups": {key: sorted(groups[key]) for key in sorted(groups)},
        "train_groups": sorted(train_keys),
        "val_groups": sorted(val_keys),
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    return split


def load_split(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def decode_rgb(raw) -> np.ndarray:
    data = raw if isinstance(raw, bytes) else bytes(raw)
    bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("failed to decode RGB JPEG")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _box_mask(xyxy: np.ndarray, visible: bool, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.float32)
    if not visible:
        return mask
    x1, y1, x2, y2 = np.asarray(xyxy, dtype=np.float32)
    if x2 <= x1 or y2 <= y1:
        return mask
    x1 = int(np.clip(np.floor(x1 * width / 640.0), 0, width - 1))
    x2 = int(np.clip(np.ceil(x2 * width / 640.0), x1 + 1, width))
    y1 = int(np.clip(np.floor(y1 * height / 480.0), 0, height - 1))
    y2 = int(np.clip(np.ceil(y2 * height / 480.0), y1 + 1, height))
    mask[y1:y2, x1:x2] = 1.0
    return mask


def future_object_masks(
    h5: h5py.File,
    t: int,
    cutoff: int,
    *,
    horizon: int = FLOW_HORIZON,
    stride: int = FLOW_STRIDE,
    height: int = MASK_HEIGHT,
    width: int = MASK_WIDTH,
    camera: str = AGENT_CAMERA,
) -> np.ndarray:
    """Return object-role-aware future masks with shape (objects, horizon, H, W)."""
    times = np.minimum(t + stride * np.arange(horizon), cutoff - 1)
    masks = np.zeros((len(OBJECTS), horizon, height, width), dtype=np.float32)
    for obj_idx, obj in enumerate(OBJECTS):
        boxes = h5[f"labels/box2d/{camera}/{obj}/xyxy"]
        visible = h5[f"labels/box2d/{camera}/{obj}/visible"]
        for horizon_idx, frame_idx in enumerate(times):
            masks[obj_idx, horizon_idx] = _box_mask(
                boxes[int(frame_idx)], bool(visible[int(frame_idx)]), height, width
            )
    return masks


class BimanualFlowDataset(torch.utils.data.Dataset):
    """Sample (RGB, role-aware future object masks) pairs from replay HDF5."""

    def __init__(
        self,
        dataset_dir: str | Path,
        files: list[str],
        *,
        image_height: int = IMAGE_HEIGHT,
        image_width: int = IMAGE_WIDTH,
        mask_height: int = MASK_HEIGHT,
        mask_width: int = MASK_WIDTH,
        horizon: int = FLOW_HORIZON,
        stride: int = FLOW_STRIDE,
        camera: str = AGENT_CAMERA,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.files = [str(name) for name in files]
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.mask_height = int(mask_height)
        self.mask_width = int(mask_width)
        self.horizon = int(horizon)
        self.stride = int(stride)
        self.camera = camera
        self._h5 = None
        self._h5_path = None

        self.indices: list[tuple[str, int]] = []
        for name in self.files:
            with h5py.File(str(self.dataset_dir / name), "r") as h5:
                cutoff = resolve_cutoff(h5, int(h5["robot/qpos"].shape[0]))
            self.indices.extend((name, t) for t in range(cutoff))

    def _open(self, name: str) -> h5py.File:
        path = str(self.dataset_dir / name)
        if self._h5_path != path:
            if self._h5 is not None:
                self._h5.close()
            self._h5 = h5py.File(path, "r")
            self._h5_path = path
        return self._h5

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        name, t = self.indices[index]
        h5 = self._open(name)
        cutoff = resolve_cutoff(h5, int(h5["robot/qpos"].shape[0]))
        image = decode_rgb(h5[f"cameras/{self.camera}/rgb"][t])
        image = cv2.resize(
            image,
            (self.image_width, self.image_height),
            interpolation=cv2.INTER_AREA,
        )
        masks = future_object_masks(
            h5,
            t,
            cutoff,
            horizon=self.horizon,
            stride=self.stride,
            height=self.mask_height,
            width=self.mask_width,
            camera=self.camera,
        )
        return (
            torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float() / 255.0,
            torch.from_numpy(masks),
        )

    def __del__(self):
        if self._h5 is not None:
            self._h5.close()

