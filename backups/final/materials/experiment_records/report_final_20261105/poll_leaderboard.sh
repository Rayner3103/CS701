#!/bin/bash
# Retries the public Phase 2 leaderboard every 30 minutes until it answers; saves the JSON next to the results.
E=/mnt/hdd/junkaichen/CS701/CS701/experiments/report_final_20261105
until curl -sS -m 25 -f 'https://web-production-c241e.up.railway.app/api/leaderboard?phase=2' -o $E/results/leaderboard_phase2.json 2>/dev/null && [ -s $E/results/leaderboard_phase2.json ]; do sleep 1800; done
date
cat $E/results/leaderboard_phase2.json | head -c 3000
