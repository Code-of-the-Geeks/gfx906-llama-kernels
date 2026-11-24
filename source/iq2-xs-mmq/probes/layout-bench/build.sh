#!/bin/bash
# build with ROCm 6.2 clang for gfx906 (system hipcc/LLVM dropped gfx906); output next to this script or $1
H=$(dirname "$(readlink -f "$0")"); OUT=${1:-$H/iq2xs_layout_bench}
/opt/rocm-6.2.0/lib/llvm/bin/clang++ -x hip --offload-arch=gfx906 --hip-link -O3 -std=c++17 "$H/iq2xs_layout_bench.cpp" -o "$OUT" \
  -nostdinc -isystem /opt/rocm-6.2.0/include -isystem /opt/rocm-6.2.0/lib/llvm/lib/clang/18/include \
  -isystem /usr/include/c++/15 -isystem /usr/include/x86_64-linux-gnu/c++/15 -isystem /usr/include/x86_64-linux-gnu -isystem /usr/include \
  -L/opt/rocm-6.2.0/lib -Wl,-rpath,/opt/rocm-6.2.0/lib -Rpass-analysis=kernel-resource-usage 2> "$OUT.resusage.txt" || { cat "$OUT.resusage.txt" | grep -E "error" ; exit 1; }
echo built "$OUT"
