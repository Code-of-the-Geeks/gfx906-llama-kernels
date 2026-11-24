#!/bin/bash
# build iq2xs_l4_bench (ROCm 6.2 clang, gfx906); output next to this script or $1; resource usage in $OUT.resusage.txt
H=$(dirname "$(readlink -f "$0")"); OUT=${1:-$H/iq2xs_l4_bench}
/opt/rocm-6.2.0/lib/llvm/bin/clang++ -x hip --offload-arch=gfx906 --hip-link -O3 -std=c++17 "$H/iq2xs_l4_bench.cpp" -o "$OUT" \
  -nostdinc -isystem /opt/rocm-6.2.0/include -isystem /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include \
  -isystem /usr/include/c++/15 -isystem /usr/include/x86_64-linux-gnu/c++/15 -isystem /usr/include/x86_64-linux-gnu -isystem /usr/include \
  -L/opt/rocm-6.2.0/lib -Wl,-rpath,/opt/rocm-6.2.0/lib -Rpass-analysis=kernel-resource-usage 2> "$OUT.resusage.txt" || { grep -E "error" "$OUT.resusage.txt"; exit 1; }
echo built "$OUT"
