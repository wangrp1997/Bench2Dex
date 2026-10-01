"""P5b: ceiling probe with a correctly-signed proportional controller.

Sign convention, established by direct diagnostic: a NEGATIVE command OPENS the finger and
REDUCES the force, so to raise the force the command must be POSITIVE. An earlier version had
this inverted, which drove the finger away from the object and pinned the force at ~1.0 for
every controller setting - the identical-results-across-all-cells signature of a dead actuator.
"""
import numpy as np
from p4_task import TrackForceEnv, F_TARGET

env = TrackForceEnv()
print(f"   F_TARGET={F_TARGET}  act_dim={env.act_dim}")
print(f"   {'kp':>6s} {'maxstep':>8s} {'meanF':>9s} {'stdF':>8s} {'meanRew':>10s}")
best = (-9.0, None)
for kp in (2.0, 5.0, 20.0):
    for ms in (0.005, 0.02):
        env.reset()
        fs, rs = [], []
        for _ in range(150):
            f = env._tactile()[1]
            err = F_TARGET - f                 # >0 means too little force -> need POSITIVE cmd
            cmd = float(np.clip(kp * err, -ms, ms))
            o, r, f2 = env.step(np.full(env.act_dim, cmd, np.float32))
            fs.append(f2); rs.append(r)
        mf, sf, mr = float(np.mean(fs)), float(np.std(fs)), float(np.mean(rs))
        print(f"   {kp:6.1f} {ms:8.3f} {mf:9.3f} {sf:8.3f} {mr:10.4f}")
        if mr > best[0]:
            best = (mr, (kp, ms))
print(f"\n   最佳: kp/maxstep={best[1]} 奖励={best[0]:.4f}   （0 = 完美）")
