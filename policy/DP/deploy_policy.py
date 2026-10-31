import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from .dp_model import DP


_ACTIVE_DOF_INFO = None
_FLOW_MODEL = None
_FLOW_DEVICE = None
_FLOW_KIND = None
_FLOW_KEY = "object_flow"
_FLOW_LANG = None
_FLOW_HISTORY = []


def _maybe_select_active(qpos):
    global _ACTIVE_DOF_INFO
    if _ACTIVE_DOF_INFO is None:
        return qpos
    arr = np.asarray(qpos, dtype=np.float32)
    if arr.shape[-1] != _ACTIVE_DOF_INFO.full_dof:
        return qpos
    from robots.active_dof_utils import select_active
    return select_active(arr, _ACTIVE_DOF_INFO)


def _observation_root(observation):
    return observation.get("observation", observation)


def _encode_camera(observation, candidates):
    root = _observation_root(observation)
    for name in candidates:
        if name in root and "rgb" in root[name]:
            return np.moveaxis(root[name]["rgb"], -1, 0) / 255
    raise KeyError(f"None of the camera observations are available: {candidates}")


def _flow_image(obs):
    image = np.rint(np.clip(obs["stereo_left_cam"].transpose(1, 2, 0), 0.0, 1.0) * 255.0)
    return cv2.resize(image.astype(np.uint8), (160, 128), interpolation=cv2.INTER_AREA)


def _load_flow_checkpoint(path, device):
    global _FLOW_KIND, _FLOW_LANG
    payload = torch.load(path, map_location="cpu")
    if "instruction_embedding" in payload:
        from policy.OF.faithful.model import ObjectFlowTransformer

        args = payload["args"]
        lang = payload["instruction_embedding"].float().cpu()
        model = ObjectFlowTransformer(
            image_height=args["image_height"],
            image_width=args["image_width"],
            mask_height=args["mask_height"],
            mask_width=args["mask_width"],
            obs_steps=3,
            horizon=args["horizon"],
            lang_dim=lang.shape[0],
            dim=args["dim"],
            depth=args["depth"],
            decoder_depth=args["decoder_depth"],
            heads=args["heads"],
            ff_dim=args["ff_dim"],
            dropout=args["dropout"],
        )
        model.load_state_dict(payload["model"])
        _FLOW_KIND = "faithful"
        _FLOW_LANG = lang
        return model.to(device).eval()
    from policy.OF.flow_model import load_flow_model

    _FLOW_KIND = "role"
    _FLOW_LANG = None
    return load_flow_model(str(path), device)


def encode_obs(observation):
    global _FLOW_HISTORY
    obs = {}
    camera_candidates = {
        "head_cam":         ("head_camera", "cam_overhead"),
        "right_cam":        ("cam_wrist_right", "right_camera"),
        "left_cam":         ("cam_wrist_left", "left_camera"),
        "stereo_left_cam":  ("cam_stereo_left",),
        "stereo_right_cam": ("cam_stereo_right",),
    }
    for obs_key, candidates in camera_candidates.items():
        try:
            obs[obs_key] = _encode_camera(observation, candidates)
        except KeyError:
            pass

    joint_action = observation.get("joint_action", {})
    if "vector" in joint_action:
        obs["agent_pos"] = _maybe_select_active(joint_action["vector"])
    elif "qpos" in joint_action:
        obs["agent_pos"] = _maybe_select_active(joint_action["qpos"])
    else:
        raise KeyError("joint_action must contain 'vector' or 'qpos'")
    if _FLOW_MODEL is not None:
        if "stereo_left_cam" not in obs:
            raise KeyError("object flow requires the cam_stereo_left observation")
        image = _flow_image(obs)
        if _FLOW_KIND == "faithful":
            if not _FLOW_HISTORY:
                _FLOW_HISTORY.append(image)
            _FLOW_HISTORY.append(image)
            _FLOW_HISTORY = _FLOW_HISTORY[-3:]
            while len(_FLOW_HISTORY) < 3:
                _FLOW_HISTORY.insert(0, _FLOW_HISTORY[0])
            frames = np.stack(_FLOW_HISTORY, axis=0).transpose(0, 3, 1, 2)
            frames = frames.reshape(9, 128, 160)
            batch = torch.from_numpy(np.ascontiguousarray(frames)).unsqueeze(0).float()
            batch = batch.to(_FLOW_DEVICE) / 255.0
            lang = _FLOW_LANG.unsqueeze(0).to(_FLOW_DEVICE)
            with torch.inference_mode():
                logits = _FLOW_MODEL(batch, lang)
                flow = torch.sigmoid(logits.float())
        else:
            from policy.OF.flow_model import predict_flow

            image = torch.from_numpy(image.transpose(2, 0, 1).copy()).unsqueeze(0).float()
            image = image.to(_FLOW_DEVICE) / 255.0
            flow = predict_flow(_FLOW_MODEL, image)
        flow = F.interpolate(flow, size=(16, 20), mode="bilinear", align_corners=False)
        obs[_FLOW_KEY] = flow[0].cpu().numpy()
    return obs


def get_model(usr_args):
    global _ACTIVE_DOF_INFO, _FLOW_MODEL, _FLOW_DEVICE, _FLOW_KEY, _FLOW_HISTORY
    _ACTIVE_DOF_INFO = None
    _FLOW_MODEL = None
    _FLOW_DEVICE = None
    _FLOW_KEY = "object_flow"
    _FLOW_HISTORY = []
    if usr_args.get("use_active_dof", True):
        robot_key = usr_args.get("robot_key") or None
        if robot_key:
            from robots.active_dof_utils import get_active_dof_info
            _ACTIVE_DOF_INFO = get_active_dof_info(robot_key)
            print(
                f"[DP] Active DOF: robot={robot_key} "
                f"dims={_ACTIVE_DOF_INFO.active_dof} (full={_ACTIVE_DOF_INFO.full_dof})"
            )

    ckpt_file = usr_args.get("checkpoint_path")
    if not ckpt_file:
        raise ValueError("DP deploy requires 'checkpoint_path' (path to a .ckpt file).")

    training_config_path = usr_args.get("training_config_path")
    if not training_config_path:
        raise ValueError(
            "DP deploy requires 'training_config_path' (e.g. "
            "policy/DP/diffusion_policy/config/robot_dp_36_dex2scene_pretrained.yaml)."
        )

    with open(training_config_path, "r", encoding="utf-8") as f:
        model_training_config = yaml.safe_load(f)

    n_obs_steps = model_training_config["n_obs_steps"]
    n_action_steps = model_training_config["n_action_steps"]
    device = usr_args.get("device", "cuda:0")

    model = DP(ckpt_file, n_obs_steps=n_obs_steps, n_action_steps=n_action_steps, device=device)
    _FLOW_KEY = getattr(model.policy, "flow_key", "object_flow")

    flow_checkpoint = usr_args.get("object_flow_checkpoint") or os.environ.get(
        "OBJECT_FLOW_CHECKPOINT"
    )
    if flow_checkpoint and str(flow_checkpoint).lower() not in ("", "null", "none"):
        _FLOW_DEVICE = torch.device(device)
        _FLOW_MODEL = _load_flow_checkpoint(str(flow_checkpoint), _FLOW_DEVICE)
        print(f"[DP] Loaded flow predictor: kind={_FLOW_KIND} key={_FLOW_KEY}")
    return model


def _model_call(model, func_name, obs=None):
    if hasattr(model, "call"):
        return model.call(func_name=func_name, obs=obs)
    method = getattr(model, func_name)
    return method(obs) if obs is not None else method()


def eval(TASK_ENV, model, observation):
    """
    TASK_ENV: Task Environment Class, you can use this class to interact with the environment
    model: The model from 'get_model()' function
    observation: The observation about the environment
    """
    obs = encode_obs(observation)
    TASK_ENV.get_instruction()

    actions = _model_call(model, "get_action", obs)

    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        _model_call(model, "update_obs", obs)


def reset_model(model):
    global _FLOW_HISTORY
    _FLOW_HISTORY = []
    _model_call(model, "reset_model")
