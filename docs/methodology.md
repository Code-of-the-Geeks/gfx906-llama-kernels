# Methodology: how every number in this repo was produced

The measurement loop that found the fast-sign win and killed six bad ideas.
Scripts are in `probes/`.

## 1. Never trust a number without its build

Every benchmark line states the binary that produced it. Our own September
baselines were wrong because they came from an older tree (n=5 "pathological
step" of 712 us simply did not exist on the newer build). Same shapes, two
builds: 126 us vs 89 us at n=1. State the tree, state the flags.

## 2. ISA dumps before and after every change

`probes/isa.sh` + `probes/isa_stats.py`: locate the kernel symbol in the built
`.hsaco`/ELF with `llvm-objdump -d`, extract the inner loop, histogram
instructions. This is what caught the quarter-rate multiplies that no source
review would find — the C++ looks identical, the ISA does not.

Key columns we track per loop: total instructions, `v_dot4_i32_i8` count,
`v_mul_lo_u32` count, vector load count, `s_waitcnt` count, VGPR, occupancy.

## 3. Loads-only floor

Ablation kernel: same loads, same shapes, dot replaced by a trivial accumulate.
This separates memory cost from instruction cost. IQ2_XS decode ran at
160-189 GB/s while the loads-only floor was 347 GB/s — proof the bottleneck was
compute, which redirected the whole project away from layout repacking.
Ablations in the other direction (VDR-4) *raised* the floor (347 to 233 GB/s)
and killed that lever immediately.

## 4. Microbench harness against real tensors, judged against llama.cpp

`probes/layout-bench/`: standalone HIP kernels reading real GGUF tensors
(metadata via gguf-py), double-precision CPU oracle for correctness, layouts
side by side (stock / planar / aligned / fused-scale / int8-plane). Rule: judge
every variant against **llama.cpp stock numbers**, never against the harness's
own copy of the kernel — the harness S-kernel ran 19-58% slower than llama.cpp
at n>=2 on long-k rows, which would have manufactured fake wins.

## 5. End-to-end gates for anything integrated

- `test-backend-ops test -o MUL_MAT_ID/-o GET_ROWS` vs CPU reference (bit-exact
  or bounded error), including regression shapes at the batch boundaries where
  dispatch changes (n=8 vs n=9).
- exhaustive self-test of any bit trick (65536 input values, 0 mismatches).
- llama-bench A/B, multiple repeats with variance.
- perplexity bit-identical in three chunk configs + KL divergence against a
  stock-vs-stock control (a nonzero KLD equal to the control means the residual
  comes from compressed logit storage, not the kernel).
- rocprofv3 per-kernel traces to localize regressions — that is how we found the
  fast sign helping fused calls -7.5% while hurting plain calls +17%, leading to
  the fused-only variant (the +4.3% winner).

## 6. Negative results are recorded, not deleted

Six rejected levers with full data (phased loads, LDS grid as a general lever,
VDR-4, three repacked layouts, sign table, int8 spill planes) live in
`source/`. The next person on this GPU should not spend an evening rediscovering
them.
