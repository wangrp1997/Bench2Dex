"""P3: verify the fingertip tactile stream actually responds to contact.

Attaches a KinematicTaxel grid to one fingertip, pushes a box into it at increasing
penetration, and checks that the sensed force grows monotonically. If this fails, nothing
built on top of the tactile stream can be trusted.
"""
import os
import numpy as np
import genesis as gs


def main():
    gs.init(backend=gs.cpu, logging_level='error')
    sc = gs.Scene(show_viewer=False)
    sc.add_entity(gs.morphs.Plane())
    urdf = os.path.join(os.path.dirname(gs.__file__), "assets", "urdf", "shadow_hand", "shadow_hand.urdf")
    hand = sc.add_entity(gs.morphs.URDF(file=urdf, pos=(0, 0, 0.3), fixed=True))
    tip = "index_finger_distal"
    tip_idx = hand.get_link(tip).idx_local

    grid = gs.utils.geom.generate_grid_points_on_plane(
        (-0.006, -0.006, 0.0), (0.006, 0.006, 0.0), normal=(0, 0, 1), nx=6, ny=6)
    tax = sc.add_sensor(gs.sensors.KinematicTaxel(
        entity_idx=hand.idx, link_idx_local=tip_idx,
        probe_local_pos=grid.reshape(-1, 3), probe_radius=0.003,
        normal_stiffness=5000.0, normal_exponent=1.5))
    # a small box we will push into the fingertip
    box = sc.add_entity(gs.morphs.Box(size=(0.02, 0.02, 0.02), pos=(0, 0, 0.34)))
    sc.build()

    # first find the fingertip world position so we can aim the box at it
    for _ in range(5):
        sc.step()
    tip_pos = hand.get_link(tip).get_pos()
    tip_pos = tip_pos.cpu().numpy() if hasattr(tip_pos, 'cpu') else np.asarray(tip_pos)
    print(f"   指尖 {tip} 世界位置 = {np.round(tip_pos, 4)}")

    print(f"   {'penetration':>12s} {'|F|max (N)':>12s} {'|F|sum (N)':>12s}")
    prev = -1.0
    ok = True
    for pen in (0.0, 0.001, 0.002, 0.004, 0.006):
        # place the box just touching / overlapping the fingertip from below
        box.set_pos((tip_pos[0], tip_pos[1], tip_pos[2] - 0.010 - 0.010 + pen))
        for _ in range(8):
            sc.step()
        d = tax.read()
        f = d.force.cpu().numpy() if hasattr(d.force, 'cpu') else np.asarray(d.force)
        fmax, fsum = float(np.abs(f).max()), float(np.abs(f).sum())
        print(f"   {pen:12.4f} {fmax:12.4f} {fsum:12.4f}")
        if pen > 0 and fmax < prev - 1e-6:
            ok = False
        prev = fmax
    print(f"\n   单调性: {'✓ 力随穿透增加' if ok else '✗ 非单调，需检查'}")


if __name__ == "__main__":
    main()
