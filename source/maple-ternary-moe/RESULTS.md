# Maple ternary-MoE — rung 1, subtask 1: q2_0/tq2_0 MUL_MAT_ID gap analysis (2026-11-24)

Analysis only — no code changes in any tree. Target tree: `~/work/llama.cpp-iq2xs-spike`
(branch `spike/iq2-xs-mmq` @ 820ac2476, binaries `build-spike`, built at a5d139cb5 per llama-bench).
GPU: Vega 20 / gfx906, 32 GiB, HIP. Logs: `logs/gap-2026-11-24/`.

## TL;DR
- **q2_0 never fails** in MUL_MAT_ID on any of build-spike / build16 / furnace (77/77 OK each).
  Nothing in this subtask reproduced a q2_0 gap. If one exists, it is outside the test-backend-ops
  MUL_MAT_ID coverage (e.g. real-model shapes / planar-MMQ build flag) and needs a separate repro.
- **tq2_0 aborts on build16 and furnace** with `GGML_ASSERT(convert_func != nullptr)`: those binaries
  have TQ2_0 **mmvq** kernels but **no TQ2_0 dequant/convert**, so any MUL_MAT(_ID) the mmvq path
  doesn't take (batch > mmvq limit) falls into the dequant+GEMM path and aborts.
  supports_op is not the culprit: it claims TQ2_0, and the dispatcher can't actually run it.
- **build-spike already has the fix** (base commit 3a5e00d40 carries the build17 convert graft:
  TQ2_0 in to_fp16/to_fp32/to_bf16). tq2_0 MUL_MAT_ID 5/5 OK, and the real Maple GGUF runs full-GPU:
  **tg128 202.3 t/s, pp512 840.7 t/s**.
- **What's left is a performance cliff, not a crash:** for batch > 8, TQ2_0 MUL_MAT_ID drops into the
  host-synchronising sorted fallback (per-expert dequant→fp16 + hipBLAS over 256 experts):
  **pp8 546 → pp9 170 t/s**. That cliff hits every prefill and every multi-slot batch.
  There is also latent missing coverage (get_rows, `_nc` converts) that Maple doesn't trigger.

## Reproduction
`test-backend-ops -o MUL_MAT_ID -b ROCm0 -p type_a=<t>` in a `systemd-run --scope -p MemoryMax=8G` scope.

| binary | q2_0 | tq2_0 |
|---|---|---|
| build-spike (iq2xs-spike tree) | 77/77 OK | 5/5 OK |
| llama.cpp-master/build16 (binary from Nov 1, before the graft) | 77/77 OK | **abort** `ggml-cuda.cu:1498 GGML_ASSERT(convert_func != nullptr)` in `ggml_cuda_mul_mat_cublas` |
| furnace/build (src 3a683a124) | 77/77 OK | **abort** `ggml-cuda.cu:2050 GGML_ASSERT(convert_func != nullptr)` in `ggml_cuda_mul_mat_id` |

Note: the llama.cpp-master **source** has the convert graft as an uncommitted change to `convert.cu`
(that is build17). The build16 **binary** predates it, so build16 still aborts. Production stays untouched.

Real model on build-spike (`maple-preview-TQ2_0-head-Q4_K.gguf`, `-ngl 99 -fa 1 -r 2`):

| test | pp1 | pp4 | pp8 | **pp9** | pp16 | pp64 | pp512 | tg128 |
|---|---|---|---|---|---|---|---|---|
| t/s | 121 ±88 | 419 ±132 | 546 ±100 | **170 ±3** | 207 ±7 | 361 ±4 | 841 ±1 | 202.3 ±1.0 |

(pp1–pp8 have high variance with only 2 reps. The pp8→pp9 step is far larger than that noise.)

## Trace: where TQ2_0 goes in `ggml_cuda_mul_mat_id` (ggml-cuda.cu:2036)
Maple expert shapes: 256 experts, top-8, gate/up `[2048 → 512]`, down `[512 → 2048]`; all 168 matrix tensors TQ2_0.

1. **supports_op** (ggml-cuda.cu ~5512): TQ2_0 is listed for MUL_MAT/MUL_MAT_ID, so it returns true.
   That is correct only if the convert fallback exists, which is the build16/furnace bug.
   (Furnace's 3a683a124 tried to gate supports_op on kernel availability, but the test still aborts
   in the `mul_mat_id` sorted path, so that gate is incomplete.)
2. **mmvq-moe, ne2 ≤ 8**: `get_mmvq_mmid_max_batch(TQ2_0, gfx906)` → gcn table has no TQ2_0 entry, so it
   uses the default `MMVQ_MAX_BATCH_SIZE` = 8, and the fused `mul_mat_vec_q` with ids runs. Our
   `vec_dot_tq2_0_q8_1` (vecdotq.cuh:733) is dp4a + `perm` LUT, one 32-quant chunk per call, VDR 1.
   Covered and fast: this is the decode path (202 t/s).
3. **MMQ**: `ggml_cuda_should_use_mmq` (mmq.cu:355) has no TQ2_0 case, so it returns false.
4. **MMF**: `ggml_cuda_should_use_mmf` returns false for any quantized type.
5. **Sorted fallback**: `GGML_ASSERT(ggml_cuda_mul_mat_id_needs_sync)`. It does a D2H copy of ids plus a stream
   sync, a host-side sort, then for each expert with tokens: `ggml_cuda_mul_mat` → dequant via
   `ggml_get_to_fp16_cuda(TQ2_0)` (convert.cu:557, present in spike) → hipBLAS GEMM.
   **build16/furnace abort here.** In build-spike it works but is slow: it also breaks HIP graph capture for these
   batches, runs ≤ 256 small GEMMs per matrix, and does 3 sorted MUL_MAT_IDs per layer × 24 layers.

### Latent gaps in build-spike (not hit by the Maple GGUF)
- `getrows.cu`: no TQ2_0 case, and the GET_ROWS supports_op has no TQ2_0. Maple's `token_embd` is F16, so it
  isn't hit. Any TQ2_0-embedding GGUF would fall back to CPU for get_rows.
- `ggml_get_to_{fp16,bf16,fp32}_nc_cuda`: no TQ2_0. They're only reached for non-contiguous src0, and per-expert
  slices are contiguous, so this isn't hit.
- test-backend-ops: TQ2_0 MUL_MAT_ID coverage is 3 shapes (n_mats=4, n ∈ {1,16,32}). There's no
  256-expert/top-8 case, nothing at the n=8/9 boundary, and no Maple shapes.

## stamsam/llama.cpp approach (commits 60b68f9, 9ee03ee; read-only, not merged)
- `dequantize_tq2_0` with the correct ggml-quants packing (byte = 32·(e/128) + e%32, lane = (e/32)%4). It's
  registered in to_fp16/to_fp32 (after a QR 4→1 fix), get_rows, and the supports_op MUL_MAT + GET_ROWS lists.
- `vec_dot_tq2_0_q8_1`: scalar 32-iteration byte loop, VDR 1, qi 8. Ours is a strictly faster dp4a/perm
  version with the same chunking.
- **MoE policy:** `get_mmvq_mmid_max_batch(TQ2_0) = -1`, so *every* TQ2_0 MUL_MAT_ID (even batch 1)
  goes to dequant + GEMM. That's correct but slow (94 t/s decode on an RTX 4000 Ada). Our tree is already
  ahead on MoE: it keeps mmvq for ≤ 8.
- Worth taking from it: get_rows registration + GET_ROWS supports_op entry (the `_nc` tables aren't in stamsam either).
  Nothing in it addresses the batch > 8 cliff (no MMQ).

## Minimal patch plan (spike tree only; build16 / llama.cpp-master untouched)
**P0: correctness parity (small, ~20 lines)**
1. getrows.cu: add `case GGML_TYPE_TQ2_0: get_rows_cuda_q<QK_K, 1, dequantize_tq2_0>` and add TQ2_0 to the
   GET_ROWS supports_op list (stamsam pattern, already QR=1-correct).
2. convert.cu `_nc` tables: add TQ2_0 to fp16/bf16/fp32 `_nc` via the generic non-contiguous dequant
   template used by Q2_0 (QK_K, QR 1).
3. test-backend-ops: add Maple-shaped MUL_MAT_ID cases for TQ2_0 (and Q2_0 for parity):
   `n_mats=256, n_used=8, (m,k) ∈ {(512,2048),(2048,512)}, n ∈ {1,8,9,16,64,512}`, plus GET_ROWS TQ2_0.
   This turns the n=8/9 boundary into a regression test.
   Gate: all OK vs CPU on build-spike. Maple tg128 unchanged (~202).

**P1: remove the batch > 8 cliff, via TQ2_0 MMQ (the real rung-1 work)**
4. Add TQ2_0 to `ggml_cuda_should_use_mmq` + a `mmq-instance-tq2_0.cu` template instance.
5. `load_tiles_tq2_0` (mmq-load-tiles.cuh): unpack the 2-bit codes to int8 {-1,0,+1} with the same
   `perm 0x020100FF` trick the vec_dot uses. Write a Q8_0-style tile (MMQ_DP4A_TXS_Q8_0 /
   `MMQ_Q8_1_DS_LAYOUT_D4`), broadcasting the one fp16 `d` to all 8 sub-blocks of 32. Then reuse the existing
   `vec_dot_q8_0_q8_1_dp4a` unchanged. That keeps the new code to the tile loader only.
   (Q2_0, QK 64, already does this on the non-planar path. TQ2_0 differs only in block size 256, a single
   scale per 256 and the e/128/e%32 packing.)
6. Optional later: a planar variant like `GGML_HIP_Q2_0_PLANAR_MMQ`, only if P1 MMQ is LDS/VALU-bound.
   Gate for P1: pp9/pp16/pp64 stop falling off a cliff (target ≥ pp8 level, monotonic toward pp512), pp512 > 841,
   PPL bit-identical to the fallback (or within fp16-GEMM noise), tg128 unchanged.

**Not recommended:** stamsam's `mmid_max = -1` (costs decode); gating supports_op off (pushes MoE to CPU).

---

# Subtask 2: P0 correctness parity (2026-11-24)

Tree `~/work/llama.cpp-iq2xs-spike`, branch `spike/iq2-xs-mmq`, commit **27ca6cb51** (on top of 820ac2476).
Built into `build-spike` with `spikes/iq2-xs-mmq/probes/build.sh` (the two-step build that patches the
libggml-hip `libamdhip64.so.7 → .so.6` NEEDED entry; a plain `cmake --build` fails at the exe link).
Logs: `logs/p0-2026-11-24/`. P0 only, no MMQ work.

## Changes (+25 lines)
- `getrows.cu`: `case GGML_TYPE_TQ2_0: get_rows_cuda_q<QK_K, 1, dequantize_tq2_0>`; `ggml-cuda.cu`: TQ2_0 added to the GET_ROWS supports_op list.
- `convert.cu`: TQ2_0 in `ggml_get_to_{fp16,bf16,fp32}_nc_cuda` via `dequantize_block_cuda<QK_K, 1, dequantize_tq2_0>`.
- `test-backend-ops.cpp`: Maple-shaped MUL_MAT_ID for tq2_0 and q2_0 (`n_mats=256, n_used=8`, (m,k) ∈ {(512,2048),(2048,512)},
  n ∈ {1,8,9,16,64,512}, 12 cases per type), plus 2 multi-block GET_ROWS tq2_0 cases (n=1024 and n=2048 with a view).

## Gates
| gate | result |
|---|---|
| MUL_MAT_ID tq2_0 (ROCm0 vs CPU) | **15/15 OK** (3 old + 12 Maple-shaped, incl. n=8/9 on both shapes) |
| MUL_MAT_ID q2_0 | **87/87 OK** (75 old + 12 new) |
| GET_ROWS tq2_0 | **10/10 OK**, 0 not-supported (it used to be unsupported, so it fell back to CPU) |
| regression: MUL_MAT iq2_xs / MUL_MAT_ID iq2_xs / GET_ROWS iq2_xs | 49/49, 15/15, 8/8 OK |
| Maple llama-bench (`-ngl 99 -fa 1 -r 3`) | **tg128 203.6 ± 1.0** (was 202.3), pp512 837.8 ± 25.6, pp8 521 ± 184, pp9 167.7 ± 5.0 |

Correction to subtask 1: the old q2_0 MUL_MAT_ID log shows 75/75, not 77/77 (the tq2_0 "5/5" was 3 cases).

As expected, the pp8 → pp9 cliff (521 → 168) is unchanged: P0 is correctness only. Next: P1 (TQ2_0 MMQ).

---

# Subtask 3: P1 TQ2_0 MMQ (2026-11-24)

Tree `~/work/llama.cpp-iq2xs-spike`, branch `spike/iq2-xs-mmq`, commit **197884098** (on top of 27ca6cb51).
Built with `spikes/iq2-xs-mmq/probes/build.sh` (after a `cmake build-spike` reconfigure to pick up the new instance file).
Logs: `logs/p1-2026-11-24/`. The full rocprofv3 kernel traces are in `~/scratch/maple-p1-2026-11-24/prof/` (stats and summary are in logs).

## Result: the pp8→pp9 cliff is gone, target met (pp9 537 ≥ 450)
| Maple `-ngl 99 -fa 1` | pp8 | **pp9** | pp10 | pp12 | pp16 | pp24 | pp64 | pp512 | tg128 |
|---|---|---|---|---|---|---|---|---|---|
| before (P0, 27ca6cb51, -r 3) | 521 ± 184 | **167.7 ± 5.0** | — | — | 207* | — | 361* | 837.8 ± 25.6 | 203.6 ± 1.0 |
| P1 MMQ only (v1, -r 10) | 577 ± 94 | 402 ± 29 | 441 | 503 | 607 | 664 | 502 | 1700† | 203.5 ± 0.8† |
| **P1 final (v2, -r 10)** | 577 ± 93 | **537 ± 46** | 589 | 687 | **842** | 1022 | **1291** | **2407 ± 24** | — |
| P1 final (v2, -r 3, full run) | 569 ± 88 | 500 ± 86 | — | — | 791 | — | 1271 | 2396 ± 25 | **199.6 ± 0.6** |

\* subtask 1 numbers (same fallback path). † v1 -r 3 run. pp1–pp8 are the unchanged mmvq path; their high variance is the same as before.
Throughput now rises monotonically from pp9 upward. pp512 is 2.9× (the old fallback was dequant + hipBLAS per expert).

## Changes (+211/−2, one loader + registration)
- `mmq-load-tiles.cuh`: `ggml_cuda_mmq_load_tiles_tq2_0`. One TQ2_0 block (256 values) is exactly one MMQ_ITER_K step.
  16 threads/row, and each thread does 1 `get_int_b2` of packed int m and produces 4 tile ints
  (`(q >> 2l) & 0x03030303` → `v_perm` LUT `0x020100FF` = {−1,0,1,2}) at Q8_0 tile position `32·(m/8) + 8l + m%8`
  (follows the ggml packing: byte 32·(e/128)+e%32, crumb (e/32)%4). fp16 `d` is broadcast to all 8 Q8_0 scale slots.
  The dot is the existing `vec_dot_q8_0_q8_1_dp4a` (dp4a) / `_mma` (MMA layout path), reused unchanged. The y layout is D4.
- `mmq.cuh` / `mmq.cu`: TQ2_0 is added to ds-layout (D4), dp4a TXS (Q8_0), util funcs (dp4a + mma), `should_use_mmq`, switch_type,
  and DECL. Every `mmq-config-*.cuh` gets a TQ2_0 copy of the Q1_0 rows. New `template-instances/mmq-instance-tq2_0.cu`
  (+ generate_cu_files.py).
- **Tile selection (v2, TQ2_0+GCN only):** (a) MoE picks J against tokens/expert `ceil(ne12·n_used/n_expert)`
  (the existing RDNA3/4 rule) rather than ne12. (b) The TQ2_0 GCN J=16 config uses I=64 (was I=128).

## Why v1 (plain MMQ) only reached pp9 402: rocprofv3, per forward
| | pp8 (mmvq) | pp9 v1 (MMQ, J=16, I=128) |
|---|---|---|
| MoE matmuls | 7.9 ms | 9.4 ms (gate/up 129 µs, down 151 µs per call) |
| dense TQ2_0 attn proj | 1.7 ms (17 µs/call) | 5.7 ms (58 µs/call, only 4–16 WGs) |

- MoE MMQ time scaled with the number of experts touched, at about 130 GB/s effective weight traffic (pp9 129 µs → pp16 212 µs).
  At 72 picks / 256 experts ≈ 0.3 tokens per expert-slot, the J=16 tile is ~94% padding, so the dp4a + LDS work on dead columns dominates.
  It is **not** the ternary unpack: the loader is 1 load + 4×(shift, and, perm) + 4 LDS stores per 16 values, per 256-k step.
- With ne11 = 9, dense used one J=16 column tile with I=128 rows, i.e. 16 WGs (m = 2048) or 4 WGs (m = 512) on 60 CUs.
- The v2 tile-selection change fixes both: MoE uses J=8, and dense J=16 gets twice the WGs. Nothing in the kernel changed.

## Gates
| gate | result |
|---|---|
| 1. MUL_MAT_ID tq2_0 vs CPU | **15/15 OK** (incl. all 12 Maple-shaped: n=1/8/9/16/64/512 × both shapes). MUL_MAT tq2_0 48/48 (dense now MMQ), GET_ROWS tq2_0 10/10, MUL_MAT_ID q2_0 87/87 |
| 2. llama-bench pp8/pp9/pp512 | 577 / **537** / **2407** (before 521 / 167.7 / 838) |
| 3. tg128 | **199.6 ± 0.6** (P0 203.6; decode path untouched, mmvq) |
| 4. iq2_xs suites | MUL_MAT 49/49, MUL_MAT_ID 15/15, GET_ROWS 8/8 OK |
| PPL wikitext-2, 20×512 | GPU MMQ **103.969**, GPU `-ub 8` (mmvq only) 103.972, CPU 102.465. MMQ ≡ mmvq. The ~1.5% gap to CPU is the GPU q8_1 path in general, not MMQ |

## Notes / next
- Dense TQ2_0 at ne11 9–16 still has few WGs on GCN (no stream-k in the gcn configs). If small-batch prefill matters more,
  try a stream-k or I=32 config for dense TQ2_0.
- pp512 2407 makes Maple prefill on the MI50 ~2.9× the old fallback and ~7.8× CPU (307).
- The planar variant (P1 step 6) isn't needed: the kernel is not loader/VALU-bound on unpack.

---

# Subtask 4: quality sanity, TQ2_0 vs Q4_K_M (2026-11-24)

Both models ran on `build-spike` (spike commit 197884098), `-ngl 99 -fa on -lm none`, inside the usual `systemd-run` MemoryMax scope.
Logs, prompts, raw outputs, `side-by-side.txt` and the scripts are in `logs/quality-2026-11-24/`.

- **A:** `maple-preview-TQ2_0-head-Q4_K.gguf` is the official deepgrove GGUF: 5.5 GiB, ternary experts and attention, Q4_K head.
- **B:** `maple-q4_k_m.gguf` is 11.5 GiB (Q4_K + Q6_K).

**The Q4_K_M file had to be fixed before it would load.** Its converter wrote a different maple metadata schema:
- `maple.attention.sliding_window_pattern`, `swiglu_clamp_exp`, `expert_weights_norm`, `expert_gating_func`, `expert_weights_scale` and `vocab_size` are missing.
- `rope.dimension_count` is 0. The file adds `rope.*_swa` instead.

On the spike loader it fails with `array key not found: maple.attention.sliding_window_pattern`.
`logs/quality-2026-11-24/ggufpatch.py` wrote `~/work/models/maple-q4_k_m-metafix.gguf`. It copies those 7 KVs from the official GGUF, and the tensor infos and data byte-for-byte; all 291 names, shapes and types are identical.
The original file is untouched. `~/work/models/maple-tq2_0.gguf` (made at the same time) has the same metadata gap.

## 1. Perplexity: wikitext-2 test, 40 × 512
| model | size | PPL | Δ vs Q4_K_M |
|---|---|---|---|
| Q4_K_M (metafix) | 11.5 GiB | **84.206 ± 3.257** | — |
| TQ2_0 (official) | 5.5 GiB | **85.409 ± 3.309** | **+1.4 %** (well inside ±3.3) |

Parent independent rerun (same build-spike, same file): TQ2_0 **84.3205 +/- 3.253**, Q4_K_M metafix **83.8438 +/- 3.236** -> delta +0.57%, parity confirmed.
The +1.4% matches the ~1.5% GPU q8_1-activation gap seen in subtask 3 (GPU vs CPU on TQ2_0), so most of it is probably the activation path rather than the weight format.
The absolute PPL (~85) is high, but it is the same in both quants. Maple is a reasoning/chat model, and raw wikitext is far from its training distribution.

## 2. Five fixed prompts: temp 0, seed 0, n_predict 256
The chat template always opens `<think>`. Two modes were run:
- **think:** the template as written.
- **nothink:** `<think>\n\n</think>\n\n` pre-filled. The model sometimes re-opens `<think>` anyway.

Pass = the final answer after `</think>` is correct and follows the instructions.

| prompt | TQ2_0 think | Q4_K_M think | TQ2_0 nothink | Q4_K_M nothink |
|---|---|---|---|---|
| 1 code fix (avg, `//` + empty list) | trunc* | trunc* | trunc* | trunc* |
| 2 arithmetic (change from €50 → 30.50) | PASS | PASS | PASS | PASS |
| 3 JSON extraction (3 people) | PASS† | **FAIL** (Maria → Utrecht, she moved to Rotterdam) | PASS | PASS (identical to TQ2_0) |
| 4 3 bullets, no "health" | PASS | PASS | PASS | PASS |
| 5 Dutch one-sentence answer | PASS | PASS | PASS | PASS |

\* All four found both bugs and wrote the right fix (`if not xs: return 0` + `/`) in the reasoning, but ran out of the 256-token budget before the final code block.
This is a budget issue, the same in both quants, not a quality difference.
† It added the surname "Jansen" to Pieter (inferred from "her brother"); city and ages are right.

Outputs are near-identical across quants (same reasoning skeleton, often the same wording). The one real divergence (prompt 3, think mode) went in TQ2_0's favour.
Decode at n=256 was 193–196 t/s for TQ2_0.

## Verdict
**TQ2_0 degrades acceptably vs Q4_K_M. Effectively, it does not measurably degrade.** PPL +1.4% is inside the error bar, and the prompt results are 4 pass + 1 budget-truncated in both modes for TQ2_0, vs 3/5 (think) and 4/5 (nothink) for Q4_K_M.
That expectation holds for a ternary-trained model: the TQ2_0 file is the native weight format, and Q4_K_M only spends 2.1× the bytes to re-encode it.
On the MI50 TQ2_0 is also the faster file (P1: pp512 2407, tg ~200). There is no reason to ship a Q4_K_M of Maple.

**Is Maple-class ternary MoE usable as a real engine model?** Conditionally yes: as a fast, small-footprint worker for short, well-specified tasks. It is not yet a general engine model.
On the positives, it got the arithmetic, the extraction, the negative constraint and the Dutch prompt right at temp 0. At 5.5 GiB it decodes at ~200 t/s and prefills at ~2400 t/s on a single MI50, leaving most of the 32 GB for KV or a second model.
Three things stand in the way of an engine role:
- **Reasoning verbosity is the biggest gap.** The template forces `<think>`, a pre-closed think block is sometimes re-opened, and a trivial code fix used >256 tokens. Step 1b saw 14k-token reasoning on list prompts with no answer. Any deployment needs a reasoning budget (or a hard pre-closed think plus a stop-and-answer policy) and a larger n_predict.
- **The evidence is thin.** Five prompts and a wikitext PPL show the quant is sound. They say nothing about Maple vs our current engine models on coding, long context (SWA-512 with GA every 4th layer) or tool use. Run the fact set / harness vs Bonsai and Qwen-class before any deployment decision.
- **The tooling isn't standard.** Mainline-style converters emit incompatible maple metadata, and the GPU path relies on our spike kernels.

So the ladder's premise holds: a ternary MoE delivers model-grade answers, not just a kernel target. Whether Maple itself earns a slot depends on that head-to-head eval and on taming the thinking budget, not on the TQ2_0 format.
