"""T16: the paper's main figure -- two panels from the controlled sweep (T15)."""
import json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = json.load(open("/home/wangrenpeng/bench2dex/theory/t15_results.json"))
t1, t2 = R["test1"], R["test2"]

fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.5))

# ---- left: matched RELATION, varying AMOUNT (the control) ----
x = np.arange(len(t1))
a_amt = [r["auc_amount"] for r in t1]
a_lrn = [r["auc_learned"] for r in t1]
ax[0].bar(x - 0.19, a_amt, 0.38, label="contact-amount statistic", color="#3b6ea5")
ax[0].bar(x + 0.19, a_lrn, 0.38, label="learned monitor", color="#c4622d")
ax[0].axhline(0.5, ls=":", c="grey", lw=1)
ax[0].set_xticks(x); ax[0].set_xticklabels(["$B_{k1}$ vs $B_{k4}$", "$B_{k2}$ vs $B_{k4}$",
                                            "$B_{k3}$ vs $B_{k4}$"], fontsize=8)
ax[0].set_ylim(0, 1.08); ax[0].set_ylabel("AUC")
ax[0].set_title("(a) matched RELATION, varying AMOUNT\n(control: the statistic is not broken)",
                fontsize=9)
ax[0].legend(fontsize=7.5, loc="lower right", framealpha=0.9)

# ---- right: matched AMOUNT, varying RELATION ----
lab = [f"amount {r['amount']}\n{r['famA']} vs {r['famB']}" for r in t2]
x2 = np.arange(len(t2))
b_amt = [r["auc_amount"] for r in t2]
b_lrn = [r["auc_learned"] for r in t2]
ax[1].bar(x2 - 0.19, b_amt, 0.38, label="contact-amount statistic", color="#3b6ea5")
ax[1].bar(x2 + 0.19, b_lrn, 0.38, label="learned monitor", color="#c4622d")
ax[1].axhline(0.5, ls=":", c="grey", lw=1)
for xi, v in zip(x2, b_amt):
    ax[1].annotate("0.500", (xi - 0.19, v), textcoords="offset points", xytext=(0, 4),
                   ha="center", fontsize=7, color="#1f3f60")
ax[1].set_xticks(x2); ax[1].set_xticklabels(lab, fontsize=7.5)
ax[1].set_ylim(0, 1.08); ax[1].set_ylabel("AUC")
ax[1].set_title("(b) matched AMOUNT, varying RELATION\n(the statistic is exactly at chance)",
                fontsize=9)
ax[1].legend(fontsize=7.5, loc="lower right", framealpha=0.9)

for a in ax:
    a.tick_params(axis="y", labelsize=8); a.grid(axis="y", alpha=0.25)
fig.tight_layout()
out = "/home/wangrenpeng/bench2dex/theory/fig_main.png"
fig.savefig(out, dpi=200)
print(f"   saved {out}")
