"""P3b: diagnose why the fingertip taxel reads zero."""
import os
import numpy as np
import genesis as gs

gs.init(backend=gs.cpu, logging_level='error')
sc = gs.Scene(show_viewer=False)
sc.add_entity(gs.morphs.Plane())
urdf = os.path.join(os.path.dirname(gs.__file__), "assets", "urdf", "shadow_hand", "shadow_hand.urdf")
hand = sc.add_entity(gs.morphs.URDF(file=urdf, pos=(0, 0, 0.3), fixed=True))
tip = "index_finger_distal"
link = hand.get_link(tip)
print(f"   link {tip}: idx_local={link.idx_local}")

# 用 link 的几何包围盒推断表面位置（局部系）
try:
    AABB = link.get_AABB()
    print(f"   link AABB = {AABB}")
except Exception as e:
    print(f"   AABB 取不到: {type(e).__name__}")

# 在多个候选位置放探针，看哪个能读到东西（用一个明显包住指尖的盒子）
grid = gs.utils.geom.generate_grid_points_on_plane(
    (-0.006, -0.006, 0.006), (0.006, 0.006, 0.006), normal=(0, 0, 1), nx=4, ny=4)
print(f"   探针网格 (link-local, z=+6mm): shape={grid.shape}")
tax = sc.add_sensor(gs.sensors.KinematicTaxel(
    entity_idx=hand.idx, link_idx_local=link.idx_local,
    probe_local_pos=grid.reshape(-1, 3), probe_radius=0.004,
    normal_stiffness=5000.0, normal_exponent=1.5))
deep = sc.add_sensor(gs.sensors.ContactDepthProbe(
    entity_idx=hand.idx, link_idx_local=link.idx_local,
    probe_local_pos=grid.reshape(-1, 3), probe_radius=0.004))
box = sc.add_entity(gs.morphs.Box(size=(0.03, 0.03, 0.03), pos=(0, 0, 0.30)))
sc.build()
for _ in range(5):
    sc.step()

p = link.get_pos(); q = link.get_quat()
p = p.cpu().numpy() if hasattr(p,'cpu') else np.asarray(p)
print(f"   指尖世界位置 = {np.round(p,4)}")
for tag, zoff in (("指尖中心", 0.0), ("下方1cm", -0.01), ("下方2cm", -0.02)):
    box.set_pos((p[0], p[1], p[2] + zoff))
    for _ in range(6):
        sc.step()
    f = tax.read().force
    f = f.cpu().numpy() if hasattr(f,'cpu') else np.asarray(f)
    d = deep.read()
    d = d.cpu().numpy() if hasattr(d,'cpu') else np.asarray(d)
    print(f"   box@{tag:8s} |F|max={np.abs(f).max():.4f} N  depth_max={np.max(d):.5f} m")
