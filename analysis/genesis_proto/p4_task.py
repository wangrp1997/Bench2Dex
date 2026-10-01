"""P4 (v4, official geometry): force regulation on a fixed tactile pad.

The geometry follows the shipped `examples/sensors/tactile_sandbox.py`: probes sit ON the top
surface of a fixed platform with an upward normal, and an object is pressed DOWN onto it. That
makes the summed tactile response a direct, monotone function of the object's height, so the
task is well posed by construction.

Why the earlier designs failed:
  v1 fixed object, finger presses   -> unreachable steady-state forces around the target
  v2 dynamic box on one fingertip   -> topples off
  v3 finger near a scripted object  -> curling the finger moves the pad sideways, not into the
                                       object, so the action had almost no effect on the force
                                       (every controller setting returned the same number)
"""
import numpy as np
import genesis as gs

GRID = 6                 # 6x6 taxels
PAD = 0.08               # platform half-size in metres (full size 8 cm)
PAD_Z = 0.02             # platform thickness
OBJ = 0.03               # object size
STIFF = 300.0
PROBE_R = 0.004
# Calibrated against the FORCE (not the depth): with a dynamic object the summed tactile force
# spans roughly 3.5 (object 5 mm up) to 15.4 (object 5 mm down). An earlier value of 0.175 came
# from a depth reading and was off by ~40x, which is why every action scored ~-1.
F_TARGET = 8.0


class PadForceEnv:
    """Agent sets the object's height; the reward is the tactile force error."""

    def __init__(self, seed=0, randomise_target=True):
        self.rng = np.random.RandomState(seed)
        self.randomise_target = randomise_target
        gs.init(backend=gs.cpu, logging_level='error')
        sc = gs.Scene(show_viewer=False, sim_options=gs.options.SimOptions(dt=0.01))
        sc.add_entity(gs.morphs.Plane())
        # fixed platform; probes on its TOP face, normal up (the official sandbox layout)
        self.pad = sc.add_entity(gs.morphs.Box(size=(PAD, PAD, PAD_Z), pos=(0, 0, PAD_Z / 2), fixed=True))
        grid = gs.utils.geom.generate_grid_points_on_plane(
            lo=(-0.006, -0.006, PAD_Z / 2), hi=(0.006, 0.006, PAD_Z / 2),
            normal=(0, 0, 1), nx=GRID, ny=GRID)
        self.tax = sc.add_sensor(gs.sensors.KinematicTaxel(
            entity_idx=self.pad.idx, link_idx_local=0,
            probe_local_pos=grid.reshape(-1, 3), probe_radius=PROBE_R,
            normal_stiffness=STIFF, normal_exponent=1.5))
        z0 = PAD_Z / 2 + OBJ / 2 + 0.004          # resting just above the pad
        self.z0 = z0
        # The object MUST be dynamic. A fixed=True (kinematic) body is invisible to the SDF /
        # raycast probe query, which is why every earlier version read exactly zero no matter
        # how far the object was pushed into the pad. We still command its height explicitly
        # every step, so it stays controllable while remaining visible to the sensors.
        self.obj = sc.add_entity(gs.morphs.Box(size=(OBJ, OBJ, OBJ), pos=(0, 0, z0)))
        self.sc = sc
        sc.build()
        st = sc.step
        for _ in range(3):
            st()
        self.act_dim = 1
        self.obs_dim = GRID * GRID * 3 + 2        # tactile + [z_rel, f_target]

    def reset(self):
        self.sc.reset()
        # WIDE randomisation across nearly the whole reachable band (3.5 - 15.4). With only
        # +-40% the optimal action never strayed far from 0, a constant action scored almost as
        # well as the optimum, and PPO had nothing to learn. Spanning the band forces the policy
        # to read the force and adapt every episode.
        self._target = (self.rng.uniform(4.5, 14.5)) if self.randomise_target else F_TARGET
        self._z = self.z0
        self._place()
        for _ in range(3):
            self.sc.step()
        return self._obs()

    def _place(self):
        self.obj.set_pos((0.0, 0.0, float(self._z)))

    def _tactile(self):
        d = self.tax.read().force
        a = d.cpu().numpy() if hasattr(d, 'cpu') else np.asarray(d)
        a = a.reshape(-1)
        return a, float(np.abs(a).sum())

    def _obs(self):
        t, f = self._tactile()
        return np.concatenate([t, [self._z - self.z0, self._target]]).astype(np.float32)

    def step(self, action):
        """action[0] in [-1,1] sets the object's downward displacement (0.5 mm per unit)."""
        a = float(np.asarray(action).reshape(-1)[0])
        self._z = self.z0 - np.clip(a, -1.0, 1.0) * 0.005
        self._place()
        self.sc.step()
        t, f = self._tactile()
        rew = -(1.0 - np.exp(-abs(f - self._target)))
        return self._obs(), float(rew), f


if __name__ == "__main__":
    e = PadForceEnv()
    e.reset()
    print(f"   obs_dim={e.obs_dim} act_dim={e.act_dim}  目标力={e._target:.3f}")
    print(f"   {'action':>8s} {'z_rel(mm)':>10s} {'|F|sum':>9s} {'reward':>9s}")
    for a in (-1.0, -0.5, 0.0, 0.3, 0.6, 1.0):
        e.reset()
        r = None
        for _ in range(10):
            o, r, f = e.step(np.array([a], np.float32))
        print(f"   {a:8.2f} {(e._z-e.z0)*1000:10.3f} {f:9.3f} {r:9.4f}")
