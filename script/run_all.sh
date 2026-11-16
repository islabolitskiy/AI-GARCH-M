#!/bin/sh
set -e
cd "$(dirname "$0")"
gcc -O3 -shared -fPIC -o libcore.so core.c -lm

MARKETS="SPX IXIC DJI RUT GSPTSE BVSP MXX FTSE GDAXI FCHI AEX SSMI IBEX N225 HSI AXJO KS11 BSESN TWII STI"

python3 01_data.py
for m in $MARKETS; do python3 02_estimate.py $m; done
for v in garch gjr egarch; do python3 03_bootstrap.py SPX $v 999; done
for i in IXIC DJI; do for v in garch gjr egarch; do python3 03_bootstrap.py $i $v 499; done; done
python3 04_subsamples.py
for m in $MARKETS; do python3 05_out_of_sample.py $m; done
python3 06_mc_calibrated.py null05 500
python3 06_mc_calibrated.py alt05 500
python3 06_mc_calibrated.py alt025 500
python3 06_mc_calibrated.py orig 300
for d in normal t5; do for T in 1000 2500 5000 20000; do python3 07_mc_thresholds.py $d $T 300; done; done
python3 08_replication.py
python3 10_panel_bootstrap.py garch 499
python3 10_panel_bootstrap.py gjr 499
python3 10_panel_bootstrap.py egarch 299
python3 11_panel_robust.py
python3 09_tables.py
