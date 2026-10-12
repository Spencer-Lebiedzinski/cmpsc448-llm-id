#!/bin/bash
# waits for the rq1/rq2 job then runs the rest one after another (they all share the gpu)
cd "$(dirname "$0")"
while pgrep -f "run_rq1_rq2.py cnn,lstm" > /dev/null; do sleep 15; done
.venv/bin/python src/run_rq3.py cnn,lstm > logs/rq3.log 2>&1
.venv/bin/python src/run_rq4.py ablate cnn,lstm > logs/rq4_ablate.log 2>&1
.venv/bin/python src/run_seeds.py > logs/seeds.log 2>&1
echo done > logs/queue_done
