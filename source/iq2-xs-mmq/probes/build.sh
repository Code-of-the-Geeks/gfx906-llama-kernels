#!/bin/bash
# two-step build of build-spike (libggml-hip DT_NEEDED .so.7 -> .so.6 patch, same as ternary-gemm/build-rocm634.sh)
set -e; B=~/work/llama.cpp-iq2xs-spike/build-spike
patch_so() { for f in $B/bin/libggml-hip.so*; do [ -f "$f" ] && [ ! -L "$f" ] && python3 -I -c "
import sys
p=sys.argv[1];b=open(p,'rb').read();open(p,'wb').write(b.replace(b'libamdhip64.so.7',b'libamdhip64.so.6'))" "$f"; done; }
cmake --build $B -j8 --target ggml-hip > /dev/null; patch_so
cmake --build $B -j8 --target ${@:-test-backend-ops} > /dev/null; patch_so
echo BUILD-OK
