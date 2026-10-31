"""Shared constants and small data helpers for the faithful flow version."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import h5py
import numpy as np


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
FLOW_STRIDE = 1
OBS_STEPS = 3
LANG_DIM = 768


def resolve_cutoff(h5: h5py.File, total: int) -> int:
    marker = h5.get("meta/homing_start_sim_step")
    if marker is None or "time/sim_step" not in h5:
        return total
    homing = int(marker[()])
    if homing < 0:
        return total
    cutoff = int(np.searchsorted(h5["time/sim_step"][:], homing, side="left"))
    return cutoff if 0 < cutoff < total else total


def dataset_files(dataset_dir: str | Path) -> list[Path]:
    files = sorted(Path(dataset_dir).glob("episode_*.hdf5"))
    if not files:
        raise FileNotFoundError(f"no episode_*.hdf5 under {dataset_dir}")
    return files


def decode_rgb(raw) -> np.ndarray:
    data = raw if isinstance(raw, bytes) else bytes(raw)
    bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("failed to decode RGB JPEG")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_split(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def instruction_of(h5: h5py.File) -> str:
    value = h5["meta/instruction"][()]
    return value.decode() if isinstance(value, bytes) else str(value)


def episode_group_key(path: str | Path) -> str:
    """Group scene variants that share exactly the same robot trajectory."""
    with h5py.File(str(path), "r") as h5:
        qpos = np.ascontiguousarray(h5["robot/qpos"][:])
    return hashlib.sha256(qpos.tobytes()).hexdigest()


def encode_instruction(text: str, model_name: str = "bert-base-uncased") -> np.ndarray:
    """Return the frozen BERT [CLS] embedding for a task instruction."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModel.from_pretrained(model_name, local_files_only=True).eval()
    batch = tokenizer([text], padding=True, truncation=True, max_length=64, return_tensors="pt")
    with torch.no_grad():
        output = model(**batch).last_hidden_state[:, 0]
    return output[0].float().cpu().numpy()
