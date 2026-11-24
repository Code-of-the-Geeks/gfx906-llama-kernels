#!/bin/bash
# usage: run-prompts.sh  -> runs 5 prompts x {tq2,q4km} x {think,nothink}, temp 0, n_predict 256
S=~/scratch/maple-quality-2026-11-24; B=~/work/llama.cpp-iq2xs-spike/build-spike/bin
declare -A M=([tq2]=~/models/maple-preview-TQ2_0-head-Q4_K.gguf [q4km]=~/models/maple-q4_k_m-metafix.gguf)
for mode in think nothink; do for p in $S/prompts/*.txt; do for m in tq2 q4km; do
  n=$(basename $p .txt); f=$S/out/$mode-$n.prompt
  { printf '<|im_start|>user\n'; cat $p; printf '<|im_end|>\n<|im_start|>assistant\n<think>\n'; [ $mode = nothink ] && printf '\n</think>\n\n'; } > $f
  systemd-run --user --scope -q -p MemoryMax=13G -p MemorySwapMax=1G $B/llama-completion -m ${M[$m]} -f $f -n 256 --temp 0 --seed 0 -ngl 99 -fa on -lm none -no-cnv --no-display-prompt -c 2048 > $S/out/$mode-$n-$m.txt 2> $S/out/$mode-$n-$m.log
  echo "$mode $n $m rc=$? $(wc -w < $S/out/$mode-$n-$m.txt)w"
done; done; done
