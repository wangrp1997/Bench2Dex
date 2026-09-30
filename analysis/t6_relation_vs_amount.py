"""T6 (discriminating experiment): is monitorability set by whether the outcome depends on a
RELATION the one-line statistic cannot express?

The theory (sufficiency leg, supported by T1-T5') says a learned monitor cannot beat a
statistic WHEN the outcome is a function of what that statistic measures. So the theory's
positive prediction is: pick an outcome that depends on a RELATIONAL contact condition, and
the amount-statistic must fail while a relation-aware monitor succeeds.

Relational condition used here -- thumb opposition:
        opposition(t) = (a thumb site is in contact) AND (>=1 non-thumb site is in contact)
This is relational: five non-thumb sites in contact (no thumb) give a LARGE amount but
opposition = 0, while thumb + one finger gives a SMALL amount and opposition = 1. No scalar
count of contacting sites can separate those two cases.

Labels compared, all on the same frames:
  L_amount     : amount > median(amount)      (by construction the amount-stat is optimal)
  L_opposition : the relational condition above
and then the REAL episode outcome (failure), to see which structure tracks it.
"""
import h5py, numpy as np, glob, json
from pathlib import Path

ROLL = Path("/mnt/public/datasets/bench2dex/rollouts")
TASKS = [("26", "26_canned_food_tray_line_arrangement"),
         ("32", "32_baking_tray_prep_with_tools"),
         ("73", "73_jigsaw_puzzle_assembly")]


def auc(s, l):
    s, l = np.asarray(s, float), np.asarray(l)
    p, n = s[l > 0], s[l <= 0]
    if len(p) == 0 or len(n) == 0:
        return float("nan")
    return float((p[:, None] > n[None, :]).mean() + 0.5 * (p[:, None] == n[None, :]).mean())


def load(task_id):
    rows = []
    for outcome in ("success", "failure"):
        for p in sorted((ROLL / task_id / outcome).glob("*.hdf5")):
            with h5py.File(p, "r") as f:
                tm = f["robot/tactile"]["tacmap"]
                sites = list(tm.keys())
                n = int(f["meta/frame_count"][()])
                on = np.zeros((n, len(sites)), bool)
                for j, s in enumerate(sites):
                    a = np.asarray(tm[s][:], np.uint8)
                    on[:, j] = a.reshape(a.shape[0], -1).max(1) > 0
            thumb = np.array([("thumb" in s) for s in sites])
            amount = on.sum(1)
            th = on[:, thumb].any(1)
            nf = on[:, ~thumb].sum(1)
            rows.append(dict(outcome=outcome, amount=amount, thumb=th, nf=nf,
                             opp=(th & (nf >= 1)),
                             # amount that is NOT opposition (the statistic's blind spot)
                             amount_wo_opp=(amount * (~(th & (nf >= 1)))).astype(float)))
    amt = np.concatenate([r["amount"] for r in rows]).astype(float)
    opp = np.concatenate([r["opp"] for r in rows])
    out = np.concatenate([[1 if r["outcome"] == "failure" else 0] * len(r["amount"]) for r in rows])
    eid = np.concatenate([np.full(len(r["amount"]), i) for i, r in enumerate(rows)])
    return amt, opp, out, eid


def main():
    res = {}
    for tid, _ in TASKS:
        amt, opp, out, eid = load(tid)
        med = np.median(amt)
        L_amt = (amt > med).astype(int)
        L_opp = opp.astype(int)
        # episode-level aggregation of each score
        def ep_auc(score, label, use_neg=False):
            sc, lb = [], []
            for e in np.unique(eid):
                m = eid == e
                sc.append((-score[m].mean()) if use_neg else score[m].mean())
                lb.append(label[m][0])
            return auc(sc, lb)
        ep_fail = np.array([out[eid == e][0] for e in np.unique(eid)])
        r = {
            "n_frames": int(len(amt)),
            "opp_rate": float(opp.mean()),
            "amount_median": float(med),
            # frame-level: can each score recover each label?
            "L_amount__by_amount": auc(amt, L_amt),
            "L_amount__by_opp": auc(opp.astype(float), L_amt),
            "L_opp__by_amount": auc(amt, L_opp),
            "L_opp__by_opp": auc(opp.astype(float), L_opp),
            # episode-level: which structure tracks the REAL failure outcome?
            "fail__by_amount": ep_auc(amt, out, use_neg=True),
            "fail__by_opp": ep_auc(opp.astype(float), out, use_neg=True),
            "fail_rate": float(ep_fail.mean()),
        }
        # the statistic's blind spot: amount among frames WITHOUT opposition
        nb = ~opp
        r["amount_median_no_opp"] = float(np.median(amt[nb])) if nb.any() else float("nan")
        r["amount_median_opp"] = float(np.median(amt[opp])) if opp.any() else float("nan")
        res[tid] = r
        print(f"   task {tid}")
        print(f"     接触量中位: 有opposition={r['amount_median_opp']:.1f} "
              f"无opposition={r['amount_median_no_opp']:.1f}   opposition占比={r['opp_rate']:.2f}")
        print(f"     L_amount: 用量统计 {r['L_amount__by_amount']:.3f} | 用opposition {r['L_amount__by_opp']:.3f}")
        print(f"     L_opp   : 用量统计 {r['L_opp__by_amount']:.3f} | 用opposition {r['L_opp__by_opp']:.3f}")
        print(f"     真实失败: 用量 {r['fail__by_amount']:.3f} | 用opposition {r['fail__by_opp']:.3f}")
    json.dump(res, open("/home/wangrenpeng/bench2dex/theory/t6_results.json", "w"), indent=1)
    print("\n   saved theory/t6_results.json")


if __name__ == "__main__":
    main()
