"""P5: compact PPO with a SWITCHABLE tactile-prediction route.

This is the prototype of the experiment that resolves the TacWAM / Compact VT-WM / HapticWAM
contradiction. One policy, one tactile predictor, three routes and two prediction targets:

  route = none : action head sees [proprio, tactile]
  route = aux  : same, PLUS an auxiliary loss on the predictor (predictor never feeds action)
  route = test : the predictor's output is appended to the action head's input
  target= raw  : predict the future tactile VECTOR
  target= task : predict the future CONTACT FORCE (the quantity the reward uses; 1 number)

Hypothesis (from the surveys): aux/raw and aux/task train fine; test/raw is what produced the
reported collapses (TacWAM 55->37.5->7.5); predicting the task functional should be easier and
at least not worse.
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from p4_task import PadForceEnv, GRID as N_TAX, F_TARGET

TAC_DIM = N_TAX * N_TAX * 3


class Policy(nn.Module):
    def __init__(self, obs_dim, act_dim, pred_dim, route, hidden=128):
        super().__init__()
        self.route = route
        self.enc = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh())
        # predictor: future tactile from current obs (always trained on the aux loss)
        self.pred = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, pred_dim))
        extra = pred_dim if route == "test" else 0
        self.act = nn.Linear(hidden + extra, act_dim)
        self.logstd = nn.Parameter(torch.zeros(act_dim) - 1.0)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.enc(x)
        if self.route == "test":
            p = self.pred(x)
            h = torch.cat([h, p], -1)
        mu = self.act(h)
        return mu, self.critic(h).squeeze(-1), self.pred(x)


# act_scale matched to the calibrated control authority (see p5b_ceiling.py): the
# effective step must be ~0.005-0.02 rad/step, not 0.08 which saturated the finger.
def rollout(env, net, horizon, act_scale=1.0):
    obs, acts, logps, rews, vals, preds, tacs, forces = [], [], [], [], [], [], [], []
    o = env.reset()
    for _ in range(horizon):
        x = torch.as_tensor(o, dtype=torch.float32)[None]
        with torch.no_grad():
            mu, v, p = net(x)
            std = net.logstd.exp()
            a = torch.distributions.Normal(mu, std).sample()
            lp = torch.distributions.Normal(mu, std).log_prob(a).sum(-1)
        a_np = (a[0].numpy() * act_scale).astype(np.float32)   # act_dim = finger dofs only
        o2, r, ftot = env.step(a_np)
        obs.append(o); acts.append(a[0].numpy()); logps.append(float(lp)); rews.append(r)
        vals.append(float(v)); preds.append(p[0].numpy())
        # SUPERVISION TARGET = the tactile state AFTER executing the action, i.e. the FUTURE
        # contact. An earlier version used the current force for the 'task' target, which is
        # not a prediction at all.
        tacs.append(o2[-TAC_DIM-1:-1]); forces.append(o2[-1])
        o = o2
    return (np.array(obs, np.float32), np.array(acts, np.float32), np.array(logps, np.float32),
            np.array(rews, np.float32), np.array(vals, np.float32),
            np.array(preds, np.float32), np.array(tacs, np.float32),
            np.array(forces, np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", default="none", choices=["none", "aux", "test"])
    ap.add_argument("--target", default="task", choices=["raw", "task"])
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--horizon", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    env = PadForceEnv(seed=a.seed)
    obs_dim, act_dim = env.obs_dim, env.act_dim
    pred_dim = TAC_DIM if a.target == "raw" else 1
    net = Policy(obs_dim, act_dim, pred_dim, a.route)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    K = 1 if a.target == "raw" else 0     # for 'task', the predicted scalar is the force itself

    hist = []
    for it in range(a.iters):
        O, A, LP, R, V, P, T, Fn = rollout(env, net, a.horizon)
        # returns
        adv = R - V; adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        # LOSSES
        mu, val, pred = net(torch.as_tensor(O))
        dist = torch.distributions.Normal(mu, net.logstd.exp())
        lp = dist.log_prob(torch.as_tensor(A)).sum(-1)
        ratio = (lp - torch.as_tensor(LP)).exp()
        pg = -torch.min(ratio * torch.as_tensor(adv),
                        ratio.clamp(0.8, 1.2) * torch.as_tensor(adv)).mean()
        vloss = F.mse_loss(val, torch.as_tensor(R))
        if a.target == "raw":
            # predict the FULL future tactile vector
            ploss = F.mse_loss(pred, torch.as_tensor(T))
        else:
            # predict only the FUTURE contact force - the quantity the reward is about
            ploss = F.mse_loss(pred.squeeze(-1), torch.as_tensor(Fn))
        loss = pg + 0.5 * vloss + (ploss if a.route != "none" else 0.0)
        opt.zero_grad(); loss.backward(); opt.step()
        meanF = float(np.abs(T).sum(1).mean())
        hist.append({"iter": it, "mean_rew": float(R.mean()), "mean_F": meanF,
                     "pred_loss": float(ploss.detach())})
        if it % 5 == 0:
            print(f"IT {it:4d} rew={R.mean():8.4f} meanF={meanF:7.3f} "
                  f"predloss={float(ploss.detach()):.4f}", flush=True)
    best = max(h["mean_rew"] for h in hist[-5:])
    res = {"route": a.route, "target": a.target, "seed": a.seed,
           "final_rew": float(np.mean([h["mean_rew"] for h in hist[-5:]])),
           "best_rew": float(best), "final_predloss": hist[-1]["pred_loss"], "hist": hist}
    print(f"   => route={a.route} target={a.target} 最终奖励={res['final_rew']:.3f} "
          f"predloss={res['final_predloss']:.4f}")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
