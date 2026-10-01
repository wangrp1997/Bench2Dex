"""P2: put tactile sensors on a multi-finger hand's fingertips and verify the stream.

Finds the fingertip links of the bundled Shadow Hand, builds a planar taxel grid on each,
attaches KinematicTaxel sensors, presses an object against a fingertip and reads.
"""
import os
import numpy as np
import genesis as gs


def main():
    gs.init(backend=gs.cpu, logging_level='warning')
    sc = gs.Scene(show_viewer=False)
    sc.add_entity(gs.morphs.Plane())
    hand_urdf = os.path.join(os.path.dirname(gs.__file__), "assets", "urdf", "shadow_hand", "shadow_hand.urdf")
    hand = sc.add_entity(gs.morphs.URDF(file=hand_urdf, pos=(0, 0, 0.2), fixed=True))
    # NOTE: add the sensors BEFORE sc.build() - a built scene is immutable and adding a
    # sensor afterwards raises "Scene is already built."
    links = [l.name for l in hand.links]
    print(f"   hand 有 {len(links)} 个 link")
    tips = [n for n in links if 'tip' in n.lower() or 'distal' in n.lower()]
    print(f"   疑似指尖 link: {tips}")
    print(f"   全部 link: {links}")

    # probe grid on a fingertip, in link-local frame
    grid = gs.utils.geom.generate_grid_points_on_plane(
        (-0.008, -0.008, 0.0), (0.008, 0.008, 0.0), normal=(0, 0, 1), nx=6, ny=6)
    print(f"   6x6 probe 网格 shape={grid.shape}")

    # attach one KinematicTaxel per fingertip link
    sensors = {}
    for name in tips:
        idx = hand.get_link(name).idx_local
        sensors[name] = sc.add_sensor(gs.sensors.KinematicTaxel(
            entity_idx=hand.idx, link_idx_local=idx,
            probe_local_pos=grid.reshape(-1, 3), probe_radius=0.002,
            normal_stiffness=5000.0, normal_exponent=1.5))
    sc.build()
    print(f"   ✓ 已在 {len(sensors)} 个指尖上挂 KinematicTaxel")

    for _ in range(20):
        sc.step()
    for name, s in list(sensors.items())[:4]:
        d = s.read()
        f = d.force.cpu().numpy() if hasattr(d.force, 'cpu') else np.asarray(d.force)
        print(f"     {name:22s} force shape={f.shape} |F|max={np.abs(f).max():.4f} N")


if __name__ == "__main__":
    main()
