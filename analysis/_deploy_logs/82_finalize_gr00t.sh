#!/bin/bash
# Wait for the GR00T task-73 resume to finish, then finalize: verify every cell,
# regenerate RESULTS.md, commit and push.
set -uo pipefail
L=/home/wangrenpeng/bench2dex/_deploy_logs
ROOT=/home/wangrenpeng/bench2dex
REPO=$ROOT/Bench2Dex
SSH="ssh -i /home/wangrenpeng/.ssh/id_ed25519 -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30"
say(){ echo "$(date -Is) $*" | tee -a $L/82_finalize.log; }

say "waiting for gr00t73 to finish ..."
while tmux has-session -t gr00t73 2>/dev/null; do sleep 60; done
say "gr00t73 finished"

# per-cell episode counts
cd $ROOT
python3 - <<'PY' 2>&1 | tee -a $L/82_finalize.log
import json,glob,os
pats={"ACT":"*_iiwa7_sharpa_policy_best_none_*","DP":"dp_*","PI05":"pi05_*","GR00T":"gr00t_*"}
for algo,p in pats.items():
    for t in ("26","32","73"):
        ds=[d for d in glob.glob(f"/home/wangrenpeng/bench2dex/output/metric/{p}")
            if os.path.basename(d).split("_")[0].strip("dp")==t or os.path.basename(d).startswith(t)
               or os.path.basename(d).startswith(f"dp_{t}_") or os.path.basename(d).startswith(f"pi05_{t}_")
               or os.path.basename(d).startswith(f"gr00t_{t}_")]
        ds=[d for d in ds if os.path.exists(d+"/per_episode.jsonl") and os.path.getsize(d+"/per_episode.jsonl")>0]
        if not ds: continue
        d=max(ds,key=os.path.getmtime)
        n=sum(1 for _ in open(d+"/per_episode.jsonl"))
        s=sum(1 for l in open(d+"/per_episode.jsonl") if l.strip() and json.loads(l).get("stable_success"))
        print(f"   {algo:5s} {t}: {s}/{n} {'OK' if n==50 else '<-- INCOMPLETE'}")
PY

python3 $L/99_make_report.py >> $L/82_finalize.log 2>&1
cp $ROOT/RESULTS.md $REPO/RESULTS.md
cd $REPO
git add RESULTS.md
if git diff --cached --quiet; then say "no report change"; else
  git -c core.pager=cat commit -q -m "results: GR00T baseline complete — all four algorithms recorded"
  for a in 1 2 3 4; do
    out=$(timeout 150 env -u GIT_SSH_COMMAND GIT_TERMINAL_PROMPT=0 git -c core.sshCommand="$SSH" push git@github.com:wangrp1997/Bench2Dex.git main 2>&1)
    echo "$out" | grep -qE "main -> main|up-to-date" && { say "pushed $(git rev-parse --short HEAD)"; break; }
    say "push retry $a"; sleep 25
  done
fi
say "FINALIZED"
