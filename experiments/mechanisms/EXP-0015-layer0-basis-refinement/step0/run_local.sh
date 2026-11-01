#!/bin/bash
export AF_ROOT="$HOME/mnt/AF_PIKANs"
cd "$AF_ROOT/experiments/mechanisms/EXP-0015-layer0-basis-refinement/step0"
mkdir -p out
for p in piezo heat dfg; do python3 fit_step0.py --problem $p --arms C,O11,U41 --seeds 1,2,3 --adam 3000 --lbfgs 500 --ntrain 8000 --out out >> out/local.log 2>&1; done
python3 fit_step0.py --problem allen_cahn --arms C,O11,U41,X11 --seeds 1,2,3 --adam 3000 --lbfgs 500 --ntrain 8000 --out out >> out/local.log 2>&1
echo DONE >> out/local.log
