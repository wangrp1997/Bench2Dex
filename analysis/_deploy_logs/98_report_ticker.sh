#!/bin/bash
# Regenerate RESULTS.md every 5 minutes so it is always current for review.
while true; do
  python3 /home/wangrenpeng/bench2dex/_deploy_logs/99_make_report.py >> /home/wangrenpeng/bench2dex/_deploy_logs/98_ticker.log 2>&1
  sleep 300
done
