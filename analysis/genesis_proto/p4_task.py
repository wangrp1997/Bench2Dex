"""P4: minimal contact-rich task where TOUCH IS NECESSARY - fingertip force regulation.

A fixed object sits above the index fingertip; the finger must press it with a target force
that is observable ONLY through the tactile stream. Vision is omitted on purpose: the routing
question is about tactile prediction, not vision, and this keeps it runnable on CPU.

The task is deliberately small - it exists to exercise the full loop (obs -> action -> reward
-> tactile prediction -> routing). A richer grasp/lift task replaces it once GPU training is
available.
"""
import os
import numpy as np
import genesis as gs

N_TAX = 4
F_TARGET = 2.0          # N
OBJ_GAP = 0.004         # object bottom sits this far above the fingertip at rest


class PressForceEnv:
    def __init__(self, seed=0):
        self.rng = np.random.RandomState(seed)
        gs.init(backend=gs.cpu, logging_level='error')
        sc = gs.Scene(show_viewer=False, sim_options=gs.options.SimOptions(dt=0.01))
        sc.add_entity(gs.morphs.Plane())
        urdf = os.path.join(os.path.dirname(gs.__file__), "assets", "urdf", "shadow_hand", "shadow_hand.urdf")
        self.hand = sc.add_entity(gs.morphs.URDF(file=urdf, pos=(0, 0, 0.3), fixed=True))
        grid = gs.utils.geom.generate_grid_points_on_plane(
            (-0.005, -0.005, 0.006), (0.005, 0.005, 0.006), normal=(0, 0, 1), nx=N_TAX, ny=N_TAX)
        tip = "index_finger_distal"
        self.tip = tip
        self.tax = sc.add_sensor(gs.sensors.KinematicTaxel(
            entity_idx=self.hand.idx, link_idx_local=self.hand.get_link(tip).idx_local,
            probe_local_pos=grid.reshape(-1, 3), probe_radius=0.004,
            normal_stiffness=5000.0, normal_exponent=1.5))
        self.obj = sc.add_entity(gs.morphs.Box(size=(0.03, 0.03, 0.03), pos=(0, 0, 1.0), fixed=True))
        self.sc = sc
        sc.build()
        for _ in range(5):
            sc.step()
        p = self.hand.get_link(tip).get_pos()
        p = p.cpu().numpy() if hasattr(p, 'cpu') else np.asarray(p)
        self.rest = p.copy()
        # park the fixed object just above the fingertip
        self.obj.set_pos((float(p[0]), float(p[1]), float(p[2]) + 0.015 + OBJ_GAP))
        for _ in range(5):
            sc.step()
        self.tip_dof = self.hand.get_link(tip).idx_local     # dof index of the fingertip joint
        self.n_dof = self.hand.n_dofs
        self.obs_dim = self.n_dof * 2 + N_TAX * N_TAX * 3 + 1

    def reset(self):
        """Reset the scene and re-park the fixed object above the fingertip."""
        self.sc.reset()
        for _ in range(5):
            self.sc.step()
        p = self.hand.get_link(self.tip).get_pos()
        p = p.cpu().numpy() if hasattr(p, 'cpu') else np.asarray(p)
        self.obj.set_pos((float(p[0]), float(p[1]), float(p[2]) + 0.015 + OBJ_GAP))
        for _ in range(5):
            self.sc.step()
        return self._obs()

    def _tactile(self):
        d = self.tax.read().force
        a = d.cpu().numpy() if hasattr(d, 'cpu') else np.asarray(d)
        return a.reshape(-1), float(np.abs(a).sum())

    def _obs(self):
        q = self.hand.get_dofs_position(); v = self.hand.get_dofs_velocity()
        q = (q.cpu().numpy() if hasattr(q,'cpu') else np.asarray(q)).reshape(-1)
        v = (v.cpu().numpy() if hasattr(v,'cpu') else np.asarray(v)).reshape(-1)
        t, ftot = self._tactile()
        return np.concatenate([q, v, t, [ftot]]).astype(np.float32)

    def step(self, action):
        """action: per-dof position delta (scaled)."""
        q = self.hand.get_dofs_position()
        q = (q.cpu().numpy() if hasattr(q,'cpu') else np.asarray(q)).reshape(-1)
        tgt = np.clip(q + np.asarray(action, np.float32).reshape(-1), -1.2, 1.2)
        self.hand.control_dofs_position(tgt)
        self.sc.step()
        t, ftot = self._tactile()
        rew = -abs(ftot - F_TARGET)
        return self._obs(), float(rew), ftot


if __name__ == "__main__":
    env = PressForceEnv()
    print(f"   ✓ build OK: dof={env.n_dof}, obs_dim={env.obs_dim}, 指尖静止位置={np.round(env.rest,4)}")
    print(f"   物体位置 = {np.round(env.obj.get_pos().cpu().numpy() if hasattr(env.obj.get_pos(),'cpu') else np.asarray(env.obj.get_pos()),4)}")
    print(f"\n   {'step':>4s} {'|F|sum(N)':>12s} {'reward':>10s}")
    for i in range(10):
        a = np.zeros(env.n_dof, np.float32)
        a[env.tip_dof] = 0.08      # drive the fingertip joint upward toward the object
        o, r, f = env.step(a)
        print(f"   {i:4d} {f:12.4f} {r:10.4f}")
