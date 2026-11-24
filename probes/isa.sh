#!/bin/bash
# usage: isa.sh <probe> -> resusage-probe<P>.txt + isa-probe<P>.s for mmvq.cu (gfx906)
T=~/work/llama.cpp-iq2xs-spike; F=$T/build-spike/ggml/src/ggml-hip/CMakeFiles/ggml-hip.dir/flags.make; S=~/scratch/iq2xs-spike-2026-11-23
D=$(grep '^HIP_DEFINES' $F | cut -d= -f2-); I=$(grep '^HIP_INCLUDES' $F | cut -d= -f2-); FL=$(grep '^HIP_FLAGS' $F | cut -d= -f2-)
eval /opt/rocm-6.2.0/lib/llvm/bin/clang++ -x hip $D $I $FL -DIQ2XS_PROBE=$1 --cuda-device-only -S -o $S/isa-probe$1.s -Rpass-analysis=kernel-resource-usage $T/ggml/src/ggml-cuda/mmvq.cu 2> $S/resusage-probe$1.txt
