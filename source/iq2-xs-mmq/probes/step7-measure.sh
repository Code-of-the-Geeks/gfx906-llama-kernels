#!/bin/bash
# step 7: real-model A/B, build-spike (stock) vs build-spike-flagon (GGML_HIP_IQ2_XS_FAST_SIGN). Results appended to step7/*.log
S=~/scratch/iq2xs-spike-2026-11-23; O=$S/step7; mkdir -p $O; T=~/work/llama.cpp-iq2xs-spike
M=~/work/models/Qwen3.8-27B-IQ2_XS.gguf; W=~/work/llama.cpp/wikitext-2-raw/wiki.test.raw
G="systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=1G"
B0=$T/build-spike/bin; B1=$T/build-spike-flagon/bin
echo "== start $(date -Is)" >> $O/progress.log
# 1. gates
for b in build-spike build-spike-flagon; do for op in MUL_MAT MUL_MAT_ID; do
  $G $T/$b/bin/test-backend-ops -o $op -p type_a=iq2_xs > $O/gate-$b-$op.log 2>&1
  echo "GATE $b $op exit=$? $(grep -c FAIL $O/gate-$b-$op.log) FAIL-lines, $(grep 'backends passed' $O/gate-$b-$op.log)" >> $O/progress.log
done; done
# 2. op perf, interleaved
for r in 1 2 3; do for b in build-spike build-spike-flagon; do
  $G $T/$b/bin/test-backend-ops perf -o MUL_MAT -p "type_a=iq2_xs,type_b=f32,m=(17408|5120|6144|4096),n=(1|4|8),k=(5120|17408|6144|14336)," 2>&1 | grep MUL_MAT | sed "s/^/$b r=$r /" >> $O/opperf.log
done; done
echo "opperf done $(date -Is)" >> $O/progress.log
# 3. llama-bench, 3 interleaved invocations x 3 reps
for r in 1 2 3; do for b in build-spike build-spike-flagon; do
  $G $T/$b/bin/llama-bench -m $M -ngl 99 -fa 1 -lm none -p 512 -n 128 -r 3 -o jsonl 2>/dev/null | sed "s/^/{\"build\":\"$b\",\"inv\":$r,\"row\":/; s/\$/}/" >> $O/bench.jsonl
done; done
echo "bench done $(date -Is)" >> $O/progress.log
# 4. perplexity
$G $B0/llama-perplexity -m $M -f $W -ngl 99 -fa on -lm none --chunks 40 > $O/ppl-std-stock.log 2>&1
$G $B1/llama-perplexity -m $M -f $W -ngl 99 -fa on -lm none --chunks 40 > $O/ppl-std-flagon.log 2>&1
echo "ppl std done $(date -Is)" >> $O/progress.log
$G $B0/llama-perplexity -m $M -f $W -ngl 99 -fa on -lm none -b 8 -ub 8 --chunks 8 --kl-divergence-base $O/kld-base-b8.bin > $O/ppl-b8-stock.log 2>&1
$G $B1/llama-perplexity -m $M -f $W -ngl 99 -fa on -lm none -b 8 -ub 8 --chunks 8 > $O/ppl-b8-flagon.log 2>&1
$G $B1/llama-perplexity -m $M -ngl 99 -fa on -lm none -b 8 -ub 8 --chunks 8 --kl-divergence-base $O/kld-base-b8.bin --kl-divergence > $O/kld-b8-flagon-vs-stock.log 2>&1
echo "ppl b8 + kld done $(date -Is)" >> $O/progress.log
$G $B0/llama-perplexity -m $M -f $W -ngl 99 -fa on -lm none -b 1 -ub 1 --chunks 2 > $O/ppl-b1-stock.log 2>&1
$G $B1/llama-perplexity -m $M -f $W -ngl 99 -fa on -lm none -b 1 -ub 1 --chunks 2 > $O/ppl-b1-flagon.log 2>&1
echo "ppl b1 done $(date -Is)" >> $O/progress.log
echo MEASURE-DONE >> $O/progress.log
