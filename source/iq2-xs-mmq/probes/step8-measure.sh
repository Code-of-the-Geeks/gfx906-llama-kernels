#!/bin/bash
# step 8: three-way real-model A/B: build-spike (stock) / build-spike-flagon (fast sign all, FS=1) / build-spike-fused (FS=2)
S=~/scratch/iq2xs-spike-2026-11-23; O=$S/step8; mkdir -p $O; T=~/work/llama.cpp-iq2xs-spike
M=~/work/models/Qwen3.8-27B-IQ2_XS.gguf; W=~/work/llama.cpp/wikitext-2-raw/wiki.test.raw
G="systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=1G"
echo "== start $(date -Is)" >> $O/progress.log
for op in MUL_MAT MUL_MAT_ID; do
  $G $T/build-spike-fused/bin/test-backend-ops -o $op -p type_a=iq2_xs > $O/gate-fused-$op.log 2>&1
  echo "GATE build-spike-fused $op exit=$? $(grep -c FAIL $O/gate-fused-$op.log) FAIL-lines, $(grep 'backends passed' $O/gate-fused-$op.log)" >> $O/progress.log
done
for r in 1 2 3; do for b in build-spike build-spike-flagon build-spike-fused; do
  $G $T/$b/bin/llama-bench -m $M -ngl 99 -fa 1 -lm none -p 512 -n 128 -r 3 -o jsonl 2>/dev/null | sed "s/^/{\"build\":\"$b\",\"inv\":$r,\"row\":/; s/\$/}/" >> $O/bench.jsonl
done; done
echo "bench done $(date -Is)" >> $O/progress.log
$G $T/build-spike-fused/bin/llama-perplexity -m $M -f $W -ngl 99 -fa on -lm none -b 1 -ub 1 --chunks 2 > $O/ppl-b1-fused.log 2>&1
$G $T/build-spike-fused/bin/llama-perplexity -m $M -f $W -ngl 99 -fa on -lm none -b 8 -ub 8 --chunks 8 > $O/ppl-b8-fused.log 2>&1
echo "ppl done $(date -Is)" >> $O/progress.log
export ROCM_PATH=/opt/rocm-6.2.0 PATH=/opt/rocm-6.2.0/bin:$PATH LD_LIBRARY_PATH=/opt/rocm-6.2.0/lib
cd $O && timeout 300 $G /opt/rocm-6.2.0/bin/rocprofv3 --kernel-trace --stats -d $O/prof-fused -o run -- $T/build-spike-fused/bin/llama-bench -m $M -ngl 99 -fa 1 -lm none -p 0 -n 64 -r 1 > $O/prof-fused.out 2>&1
echo "prof done rc=$? $(date -Is)" >> $O/progress.log
echo MEASURE-DONE >> $O/progress.log
