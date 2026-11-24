# IQ2_XS mmq/vec_dot on gfx906 (MI50) — spike started 2026-11-23

Goal: give IQ2_XS a dp4a-friendly fast path on MI50 (today it runs the generic
codebook-dequant path; nobody in the gfx906 ecosystem has an IQ2_XS MMQ).
Target model: Qwen3.8-27B IQ2_XS (6block), 10,591,877,792 B, fits 16 GB VRAM.

## Methodology (same as ternary-gemm, see ../ternary-gemm/README.md)
1. Baseline first: stock build-rocm (llama.cpp tree at ~/work/llama.cpp), tg128 + pp512 on the model,
   `test-backend-ops perf` per-shape timings for iq2_xs mul_mat (n=1 and n=512).
2. Correctness oracle: CPU vs GPU via test-backend-ops; PPL parity wikitext-2 before/after any kernel change.
3. Diagnose by ablation (strip dot / strip activations / widen loads), never guess.
4. Check register pressure for every MMQ variant: -Rpass-analysis=kernel-resource-usage, spills must be 0.

## IQ2_XS format notes (block_iq2_xs, QK_K=256)
- 256 weights per block: scales fp16 x16 sub-groups (16B), cc[8] codebook indices (8B),
  8x uint32 packed signs/qbits (32B) => bs=56, 2.31 bpw incl. scales.
- vec_dot unpacks via LUT: per 2-byte pair -> two symbol pairs from the 256-entry table (per sub-block),
  then scale. No dp4a: byte-per-symbol lookup + fp16 accumulate. That is the slow path to replace.
- Idea A (LUT->dp4a): precompute per-codebook-index a 4-byte symbol vector; activations q8_1;
  dot becomes indexed gathers + int8 dots. Idea B (bit-plane split like Q2_0 planar): not directly possible
  (codebook symbols are not +/-1 only), but the 256-entry table decomposes into low/high nibbles -> two LUTs.

## Artifacts
- modelstore: /mnt/raid5/modelstore/llm/qwen38-27b-iq2_xs/Qwen3.8-27B-IQ2_XS.gguf (sha256 recorded here after download)
- copy on vega20-box: ~/work/models/ (after modelstore download completes)

## Log
- 2026-11-23 download: 6block/Qwen3.8-27B-IQ2_XS.gguf, 10591877792 B, sha256 baa00667be0fc1aaffcd519565478f9ecc05f51eaabb9848f8723459f928c90c -> modelstore llm/qwen38-27b-iq2_xs/
- 2026-11-23 stock baseline (build16 ecdd29e8b, -ngl 99 -fa on): pp512 216.21 / tg128 19.42 t/s.
  (The op-perf numbers first recorded here — n=1 126us, n=5 712us "pathological step" — came from the OLD September build-rocm tree
  and do NOT apply to build16: corrected table in RESULTS.md step 0, n=1 89.5us monotone.)

## Outcome (2026-11-24, steps 1-8 in RESULTS.md)

**Shipped candidate: bit-exact fast sign application in fused mmvq instances only** — `-DGGML_HIP_IQ2_XS_FAST_SIGN=2`
(spike commits `214d49d49` + `820ac2476`, patches in `probes/`): replaces llama.cpp's HIP `unpack_ksigns` + `__vcmpne4`/`__vsub4`
emulation with a computed byte mask (24-bit spread multiply, `(g ^ m) + b`, carry-free because grid magnitudes are 8/25/43 < 128).
tg128 **+4.3%** (19.47 -> 20.31; independent rerun by the other agent 19.72 -> 20.46), pp512 unchanged, PPL bit-identical in 3 modes,
exhaustive 65536-value self-test 0 mismatches. Restriction matters: the same vec_dot in *plain* mmvq instances is +17% per call (rocprof trace).

Rejected levers (all measured, all gate-green, honest negatives): phased loads (latency hypothesis refuted — waves/SIMD hide it),
codebook in LDS (gathers worth only 3-6%; net negative at n=1 but -2..-11% at n=4/8), VDR 4 (floor worse: 347 -> 233 GB/s),
repacked layouts L1/L2/L3 (floor unmoved by alignment/plane-split; best -6.2%, L3 not bit-exact), ksigns64 table (+21..+37% —
trading ALU for vector-path gathers is the wrong direction on gfx906, third confirmation of that rule).

Core lesson: IQ2_XS mmvq sits at ~170-200 GB/s useful against a loads-only floor (~350-400 GB/s) that nothing load-side moved.
The remaining gap to Q4_K/Q8_0 (400-480 GB/s) is per-step compute on an activation-heavy floor (36 B activations per 9.25 B weights).
Premise corrections: stock already has dp4a vec_dot + MMQ for iq2_xs; block_iq2_xs is 74 B/256 weights (not 56); vega20-box GPU is a
Vega II with 32 GB, not a 16 GB MI50.

**Step 9 closed the RAM-for-compute question (negative):** pre-dequantized signed int8 planes (L4, 288 B/block, bit-exact) stream at
466-614 GB/s — proving the stock kernel is compute/latency-bound, not bandwidth-bound — but lose anyway: n=1 +8..+58% (3.89x bytes),
n=512 +24..+29% vs llama.cpp MMQ because tiled kernels already amortize tile decode over 128 columns (pre-dequant saves only 0.1-1.2%
inside the harness), and n=4 wins -11..-20% on 3 of 4 shapes but needs +13..+20 GiB VRAM this card cannot spare next to a usable KV cache.
Verdict: fast-sign-fused-only (step 8) stands as THE integration candidate; spill-RAM cannot beat the byte math at any n here.

