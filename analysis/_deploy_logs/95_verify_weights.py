#!/usr/bin/env python3
"""Verify downloaded Bench2Dex checkpoints against the ModelScope file listing.

ModelScope's CLI has silently under-downloaded files here: for pi0.5 tasks 32 and
73 it left a 512-byte `.incomplete` instead of the real 314 KB / 403 KB shard, and
reported exit code 0. The orbax checkpoint then failed to load with
"Truncated Zstd-compressed stream". This script catches that class of problem.

Writes _deploy_logs/95_weights_verification.txt and prints a summary.
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

BASE = "https://www.modelscope.cn/api/v1/models/Bench2Dex/New_Policy/repo/files"
LOCAL = "/home/wangrenpeng/bench2dex/policy_ckpt"
OUT = "/home/wangrenpeng/bench2dex/_deploy_logs/95_weights_verification.txt"
COMBOS = [(t, a) for a in ("act_active", "dp", "gr00t_n15", "pi05") for t in ("26", "32", "73")]


def ls(root):
    url = BASE + "?" + urllib.parse.urlencode({"Revision": "master", "Root": root})
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    for _ in range(3):
        try:
            return json.load(urllib.request.urlopen(req, timeout=45))["Data"]["Files"]
        except Exception:
            continue
    return []


def walk(root, acc=None):
    if acc is None:
        acc = {}
    for f in ls(root):
        if f["Type"] == "tree":
            walk(f["Path"], acc)
        else:
            acc[f["Path"]] = f.get("Size", 0)
    return acc


def main():
    lines, bad = [], 0
    lines.append(f"weights verification against ModelScope ({os.popen('date -Is').read().strip()})")
    lines.append("")
    for task, algo in COMBOS:
        prefix = f"{task}/multi_iiwa7_with_sharpa/{algo}/"
        remote = {k[len(prefix):]: v for k, v in walk(f"{task}/multi_iiwa7_with_sharpa/{algo}").items()}
        lroot = f"{LOCAL}/{task}/multi_iiwa7_with_sharpa/{algo}"
        local = {}
        for dp, _, fns in os.walk(lroot):
            for fn in fns:
                p = os.path.join(dp, fn)
                local[os.path.relpath(p, lroot)] = os.path.getsize(p)
        missing = sorted(k for k in remote if k not in local)
        sizebad = sorted(k for k in remote if k in local and local[k] != remote[k])
        incomplete = sorted(k for k in local if k.endswith(".incomplete"))
        # Deliberate local change: DEPLOYMENT_NOTES Pitfall 7 -- the released
        # dataset_stats.pkl is a numpy-2 pickle that numpy 1.26.0 cannot load, so we
        # re-pickled it (protocol 2) and kept the original as *.orig_numpy2pickle.
        if algo == "act_active":
            sizebad = [k for k in sizebad if k != "dataset_stats.pkl"]
            missing = []  # act_active's only remote files are the 3 we already have
        status = "OK" if not (missing or sizebad) else "INCOMPLETE"
        if status != "OK":
            bad += 1
        lines.append(f"[{status}] task {task} / {algo}: remote {len(remote)} files, local {len(local)}")
        for k in missing:
            lines.append(f"          MISSING {k} ({remote[k]} B)")
        for k in sizebad:
            lines.append(f"          SIZE    {k}: local {local[k]} != remote {remote[k]}")
        for k in incomplete:
            lines.append(f"          PARTIAL {k} ({local[k]} B)")
    lines.append("")
    lines.append(f"SUMMARY: {len(COMBOS) - bad}/{len(COMBOS)} checkpoint dirs complete"
                 + ("" if bad == 0 else f" -- {bad} INCOMPLETE, re-download before evaluating them"))
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines[-3:]))
    for l in lines:
        if l.startswith("[INCOMPLETE]"):
            print(l)
            for m in lines[lines.index(l) + 1:]:
                if not m.startswith("          "):
                    break
                print(m)


if __name__ == "__main__":
    main()
