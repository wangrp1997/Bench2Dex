# Role-Aware Bimanual Object Flow

This is an independent adaptation of the object-flow idea in Chen et al.,
"Cross-Behavior Learning With Object Flow Prediction for Robotic Manipulation"
(IEEE TRO 2026), to Bench2Dex task 09. The paper's public code was not found,
so this is not an official implementation.

## Adaptation

- Agent view: `cam_stereo_left`, which is also an input to ACT and DP.
- Object roles are preserved as three channels: wooden box, cleaner, soap.
- The predictor forecasts 16 future binary masks per object from one RGB frame.
- DP receives the predicted flow through a small learned flow encoder in addition
  to the same four cameras and 40-D active joint state used by the baselines.
- Training and validation are split by `sha256(robot/qpos)`, so the `_1`
  scene variants cannot leak across splits.
- Masks are derived from per-frame `labels/box2d` boxes. This is coarser than the
  paper's Cutie segmentation and must be reported as such.

## Pipeline

```bash
python policy/OF/build_split.py \
  --dataset_dir "$DATA" --output "$DATA/object_flow_split.json"

python policy/OF/train_flow.py \
  --dataset_dir "$DATA" --split "$DATA/object_flow_split.json" \
  --output_dir output/object_flow/09_flow_seed0 --device cuda:0

python policy/DP/process_hdf5_to_zarr.py \
  --dataset_dir "$DATA" --output "$ZARR" --truncate_at_homing

python policy/OF/precompute_flow.py \
  --dataset_dir "$DATA" --zarr "$ZARR" \
  --flow_checkpoint output/object_flow/09_flow_seed0/flow_best.pt --device cuda:0
```

The action-policy comparison is `ACT`, `BC-DP`, and `RA-OF-DP` on the same
trajectory split and evaluation seeds. The released 09 data contains one
behaviour family only, so this experiment tests role-aware prediction and
bimanual conditioning, not cross-behaviour transfer.
