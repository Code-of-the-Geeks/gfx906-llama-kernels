#!/bin/bash
# IQ2_XS mmvq ablation sweep: rebuild with IQ2XS_PROBE=P, time n=1 shapes 3x. Appends to probe-sweep.log
S=~/scratch/iq2xs-spike-2026-11-23; T=~/work/llama.cpp-iq2xs-spike; H=$T/ggml/src/ggml-cuda/iq2xs-probe.cuh
G="systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=1G"
for P in ${@:-0 1 2 3 4 5}; do
  sed -i "s/^#define IQ2XS_PROBE [0-9]*/#define IQ2XS_PROBE $P/" $H
  $S/build.sh test-backend-ops > $S/probe-build-$P.log 2>&1 || { echo "P=$P BUILD-FAIL" >> $S/probe-sweep.log; continue; }
  for r in 1 2 3; do
    $G $T/build-spike/bin/test-backend-ops perf -o MUL_MAT -p "type_a=iq2_xs,type_b=f32,m=(4096|17408|5120),n=1,k=(14336|5120|17408)," 2>&1 | grep MUL_MAT | sed "s/^/P=$P r=$r /" >> $S/probe-sweep.log
  done
done
sed -i "s/^#define IQ2XS_PROBE [0-9]*/#define IQ2XS_PROBE 0/" $H
echo SWEEP-DONE >> $S/probe-sweep.log
