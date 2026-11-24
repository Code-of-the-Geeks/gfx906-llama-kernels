# IQ2_XS on MI50 (gfx906): step 1 diagnosis (2026-11-23)

Scope: diagnosis only, no kernel changes kept. All probes were temporary and reverted (sources in `probes/`).

## Setup
- Engine: branch `spike/iq2-xs-mmq` in a **separate git worktree** `~/work/llama.cpp-iq2xs-spike` (of the same repo as
  `llama.cpp-master`). Commits: `3a5e00d40` = upstream `ecdd29e8b` + build16's uncommitted working-tree diff (Q2_0/TQ2_0 patch),
  `d4bc3920b` = test-only perf cases (model decode shapes per tensor type, iq2_xs/q4_0 bs 6,7). Worktree instead of editing
  `llama.cpp-master` in place because build16 compiles from that working tree; spike edits there would leak into any build16 rebuild.
  `llama.cpp-master`, build16 and all services untouched.
- Build `build-spike`: CMakeCache identical to build16's except paths (verified by diff), incl. `CMAKE_PREFIX_PATH=~/rocm-fake`
  and `-DGGML_HIP_Q2_0_PLANAR_MMQ=1`. Gotcha: libggml-hip links `libamdhip64.so.7` (system, from libamdhip64-dev) and the exe link fails;
  same two-step build + ELF string patch `.so.7 -> .so.6` as `ternary-gemm/build-rocm634.sh` (`probes/build.sh`).
- Correctness gate on build-spike: `test-backend-ops -o MUL_MAT -p type_a=iq2_xs` 2/2 backends OK (also OK under every rows probe).
- GPU idle otherwise (no llama services running), power cap 160 W. Op timings: 3 runs each, spread < 0.5% unless noted.

## 0. Corrections to README.md
- **Stock already has a dp4a IQ2_XS path.** `vec_dot_iq2_xs_q8_1` (vecdotq.cuh) uses `v_dot4_i32_i8`; IQ2_XS also has an MMQ path
  (`ggml_cuda_mmq_load_tiles_iq2_xs`, `mmq-instance-iq2_xs.cu`, DP4A tile `MMQ_DP4A_TXS_Q8_0_16`). The "no dp4a, fp16 accumulate,
  nobody has an IQ2_XS MMQ" premise is wrong; the question is why the existing dp4a path is slow.
- **Block format:** `block_iq2_xs` = `d` fp16 (2 B) + `qs[32]` uint16 (64 B; each = 9-bit index into the 512-entry `iq2xs_grid`
  of 8-byte rows + 7-bit index into the sign table, 8th sign by parity) + `scales[8]` (two 4-bit scales per byte) = **74 B / 256 weights
  = 2.3125 bpw**. Not 56 B.
- **The baseline op-perf numbers in README do not reproduce, and the n=5..8 step does not exist today.** build16's own
  `test-backend-ops` binary (run only) and build-spike agree to < 1% (`logs/perf-ab-4096.log`):

| MUL_MAT iq2_xs 4096x14336 | README baseline | build16 today | build-spike |
|---|---|---|---|
| n=1 | 126.02 us | 89.5 | 89.2 |
| n=2 | - | 140.6 | 140.5 |
| n=3 | - | 180.9 | 180.5 |
| n=4 | 201 | 152.6 | 152.4 |
| n=5 | **712** | **162.2** | 161.7 |
| n=6 / n=7 | - | - | 180.6 / 205.0 |
| n=8 | **1083** | **222.0** | 222.0 |
| n=512 | 6.63 ms (9.07 TFLOPS) | 5.445 ms | 5.447 ms (11.0 TFLOPS; Q4_0 3.78 ms = 15.9) |

  No raw logs of the README baseline exist (commit 6b25b09 has only the README text), so the earlier conditions cannot be checked.
  The model-level baseline **does** reproduce on build-spike (`logs/llama-bench-spike.log`, `-ngl 99 -fa 1 -lm none`, r=3):
  **pp512 216.13 ± 0.75, tg128 19.68 ± 0.01** (README: 216.21 / 19.42).
- What the n<=8 config actually does on `MMVQ_PARAMETERS_GCN` (dense MUL_MAT, mmvq used for all n<=8): nwarps 2 for n=1..4, 1 for
  n=5..8; rows_per_block 1 for n=1, 2 for n=2..8; VDR 2. No spills in any iq2_xs mmvq instance (n=1: 38 VGPR, occ 6; n=2..8: 58-64
  VGPR, occ 4 for both n=4 and n=5; `logs/resusage-iq2xs-q40-summary.txt`). The nwarps 2->1 switch at n=5 costs +6% (152 -> 162 us),
  not 3.5x. A smaller real oddity: n=3 (180.5) is slower than n=4 (152.4), and n=2 (140.5) is close to n=4; Q4_0 is monotonic
  (88/119/145). Not diagnosed. It matters little for decode, which is n=1 (n=2..4 only shows up in batched/spec decode).

## 1. Where decode time goes (n=1, per token, model shapes)
`logs/perf-stock-shapes.log`, counts from the GGUF (64 layers):

| type | m x k | per token | us (n=1) | GB/s weights | ms/token |
|---|---|---|---|---|---|
| iq2_xs | 17408x5120 (ffn gate/up) | 128 | 149.08 | 173 | 19.08 |
| iq2_xs | 5120x17408 (ffn down) | 56 | 136.12 | 189 | 7.62 |
| iq2_xs | 6144x5120 | 48 | 56.98 | 160 | 2.74 |
| iq2_xs | 5120x6144 | 48 | 50.68 | 179 | 2.43 |
| q4_K | 10240x5120 | 48 | 73.35 | 402 | 3.52 |
| q4_K | 248320x5120 (output) | 1 | 1638.78 | 436 | 1.64 |
| q8_0 | 12288x5120 | 16 | 139.05 | 481 | 2.22 |
| q8_0 | 5120x6144 | 16 | 72.81 | 459 | 1.16 |
| q8_0 | 1024x5120 | 32 | 17.91 | 311 | 0.57 |
| q2_K | 5120x17408 | 8 | 111.71 | 262 | 0.89 |

- Matmul total 41.9 ms of a 50.8 ms token (tg128 19.68). **IQ2_XS = 31.9 ms = 76% of matmul**, at 160-189 GB/s of weights,
  while Q4_K/Q8_0 run 400-480 GB/s on the same box. IQ2_XS is only 5.23 of the 9.85 GiB but dominates time.
- Upper bound if IQ2_XS mmvq ran at ~400 GB/s: 31.9 -> ~14 ms, token ~33 ms, **tg ~30 t/s (+50%)**. That is the prize for step 2.

## 2. Decode path (stock mmvq, n=1)
Per 256-weight block 8 threads (qi 16 / VDR 2); each thread per call: 32 weights = 2x `get_int_b2` of `qs` (8 B), 1 scale byte,
`d` (2 B), 32 B of q8_1 activations + `ds`. For each of 4 qs entries: gather one 8-byte row of `iq2xs_grid` (512x8 B = 4 KB,
`__device__` global table, dependent load on the qs value), sign byte via popcount parity (`unpack_ksigns`, multiply broadcast),
`__vcmpne4`/xor/`__vsub4` to apply signs (emulated byte-wise on gfx906), 2 dp4a. Then the 4-bit scale fixup and fp32 scale.

ISA of the n=1 kernel main loop (`logs/isa-loop-stats.txt`): **185 instructions per iteration, 8 `v_dot4`**, 151 other VALU
(41 and, 24 xor, 15 shr, **14 `v_mul_lo_u32` (quarter rate)**, 13 add, 8 sub, 8 or, 4 bcnt...), 10 vector loads
(5 dwordx2 = 4 grid gathers + qs, 2 dwordx4 + 1 dword activations, ushort d, ubyte scale), 9 waitcnt.

## 3. Ablations (temporary `IQ2XS_PROBE` variants of the vec_dot in a header only mmvq.cu included; reverted)
`logs/probe-sweep.log`; probe 0 = stock math through the same hook (matches stock to 0.2%).

| probe | 4096x14336 | 5120x17408 | 17408x5120 | loop instr / VGPR / occ |
|---|---|---|---|---|
| 0 stock | 88.9 | 136.1 | 149.0 | 185 / 38 / 6 |
| 1 loads only (qs, scales, d, activations; no gather, no sign math, no dot) | 48.9 (-45%) | 74.1 (-46%) | 76.6 (-49%) | - |
| 2 no activation loads (q8_1 qs/ds replaced by weight-derived values) | 75.2 (-15%) | 112.8 (-17%) | 136.2 (-9%) | 178 / 38 / 6 |
| 3 no grid gather (grid value computed from index) | 81.2 (-9%) | 122.6 (-10%) | 143.9 (-3%) | 181 / 32 / 8 |
| 4 wide weight loads (qs as one 8-byte load) | 89.8 (+1%) | 138.2 (+2%) | 149.1 (0%) | - |
| 5 no sign application (ksigns/vcmpne4/xor/vsub4 skipped) | 100.0 (**+12%**) | 163.8 (**+20%**) | 136.3 (-9%) | 69 / 24 / 8 |

Rows-per-block probe at n=1 (the lever that fixed Q2_0; `logs/rows-sweep.log`, gate OK for each), stock R=1 in brackets:

| rows | 4096x14336 [88.9] | 5120x6144 [50.7] | 5120x17408 [136.1] | 6144x5120 [57.0] | 17408x5120 [149.0] |
|---|---|---|---|---|---|
| 2 | 94.5 | 54.8 | 143.1 | 61.8 | 159.4 |
| 4 | 93.3 | 53.4 | 128.9 | 60.3 | 143.4 |
| 8 | 100.9 | 58.9 | 139.1 | 69.6 | 156.1 |

## 4. Findings
1. **The kernel is not bound by any single resource we can strip.** Loads-only is 52-55% of the stock runtime, so ~45% is the
   decode/dot body. But removing its pieces one at a time does not add up to that: activations -9..-17%, grid gather -3..-10%,
   and removing the sign math (63% of the loop's instructions, VGPR 38 -> 24, occupancy 6 -> 8) makes two of three shapes
   **slower** (+12%, +20%). So VALU issue count is not the limiter, and neither is occupancy.
2. **Weight load width is irrelevant** (probe 4: 0..+2%), same as for Q2_0.
3. **Activation reuse across rows is not the lever here** (rows 4: -4..-5% on the two k-long or m-long ffn shapes, +5..6% on the
   others; rows 8 worse everywhere), unlike Q2_0 (-32%). Activation traffic costs 9-17% (probe 2), less than Q2_0's ~27%.
4. **Even the loads-only floor is low:** 48.9 us on 4096x14336 = 347 GB/s of weights (336 GB/s on 17408x5120), below stock
   Q4_0's full kernel (~410 GB/s). Per thread per 32 weights the kernel moves ~9.25 B of weights but 36 B of activations plus
   small scattered loads (ubyte scale, ushort d, 2-byte-aligned qs), i.e. many small memory ops per useful byte.
5. Working interpretation for step 2 (supported by 1-4, not yet proven): the stock loop is **latency-bound on a dependent chain
   per thread** (qs load -> index -> dependent grid gather -> sign math -> dp4a, 4 times) with little independent work to
   overlap. Shortening one link does not help when the scheduler cannot overlap the others; probe 5's slowdown (fewer
   instructions between issue points, different load placement) fits that but needs an ISA diff to confirm. Directions to
   test (step 2, not done): more independent weights per thread (VDR 4 / two blocks in flight) so gathers overlap; the grid in
   LDS or a register-friendly decomposition instead of global gathers; cheaper sign application (one v_perm/bfi path instead of
   emulated vcmpne4/vsub4 and the quarter-rate multiply). The prefill MMQ path (n=512: 11.0 vs Q4_0 15.9 TFLOPS) is a separate,
   smaller item: pp is not the Vane bottleneck.

## Files
- `logs/`: raw op timings (A/B, model shapes, probe sweep, rows sweep), llama-bench, resource usage, ISA loop stats.
- `probes/`: probe header (`iq2xs-probe.cuh`), the mmvq.cu hook diff, sweep/ISA/build scripts (they expect
  `~/scratch/iq2xs-spike-2026-11-23` as work dir).

# Step 2: phased-load vec_dot (latency hypothesis test), 2026-11-23

**Result: negative.** No gain at n=1 (target was >= 15% on the FFN shapes); large regression at n=2..8. The simple
per-iteration latency hypothesis from step 1 finding 5 is refuted.

## Change
Spike commit `fc4272e81` (on `spike/iq2-xs-mmq`, patch also in `probes/step2-pipe-mmvq.patch`): opt-in
`-DGGML_HIP_IQ2_XS_PIPE_MMVQ=1` (HIP only, off by default) replaces `vec_dot_iq2_xs_q8_1`, whose only consumer is mmvq (n <= 8).
Same math, loads issued in phases pinned with `__builtin_amdgcn_sched_barrier(0)`:
1. all loads that do not depend on another load: qs (2 ints), scale byte, d, the 8 q8_1 activation ints, ds
2. all 4 `iq2xs_grid` gathers
3. sign application + dp4a + scales

Why this shape: the stock n=1 loop ISA already issues the 4 gathers back to back, but issues the activation loads only after
the gathers **and** 52 ALU of sign math have completed, i.e. three serialized memory round trips per iteration
(qs -> gathers -> activations). The variant's ISA has two: [ubyte, ushort, 2x dwordx4, dword, qs dwordx2] -> wait ->
[4 gathers] -> wait -> ALU/dp4a. n=1 kernel: 178 loop instructions, 36 VGPR, occupancy 7, 0 spills (stock: 185, 38, 6, 0).
LDS staging of the grid was not tried (it needs kernel-level code in mmvq.cu, beyond the single vec_dot change).

## Gate
Both trees (`build-spike` = flag off, rebuilt from the new source; `build-spike-pipe` = identical CMake cache + the flag):
`test-backend-ops -o MUL_MAT -p type_a=iq2_xs` and `-o MUL_MAT_ID -p type_a=iq2_xs`: **2/2 backends OK, 0 FAIL** each.

## Perf (us, median of 3 interleaved off/pipe rounds, spread < 1%; `logs/step2-pipe-vs-stock.log`)

| shape (m x k) | n=1 stock | n=1 pipe | change | n=2 | n=4 | n=8 |
|---|---|---|---|---|---|---|
| 17408x5120 (ffn gate/up) | 148.57 | 151.94 | **+2.3%** | +27.1% | +87.0% | +173.0% |
| 5120x17408 (ffn down) | 135.79 | 133.95 | **-1.4%** | +7.6% | +96.6% | +172.7% |
| 6144x5120 | 56.71 | 58.41 | +3.0% | +26.9% | +88.6% | +162.7% |
| 5120x6144 | 50.88 | 52.10 | +2.4% | +26.3% | +85.3% | +162.9% |
| 4096x14336 | 89.26 | 89.55 | +0.3% | +8.8% | +93.6% | +152.5% |

Flag-off stock numbers match step 1 (88.9 / 136.1 / 149.0 / 50.7 / 57.0) within 0.5%.

## Why n>=2 regresses (verified in ISA)
In stock code the compiler shares the weight decode across the per-column vec_dot calls: the n=2/4/8 kernels all have 10 `dwordx2`
loads (gathers + qs, 2 rows) and 8 `v_bcnt`. With the sched_barriers this sharing is gone: 20 / 40 / 80 `dwordx2` and 16 / 32 / 64
`v_bcnt`, i.e. weights loaded, gathered and sign-decoded once per column. A side effect of the barrier (it blocks merging across the
calls), not of the phase ordering itself. Not fixed: decode is n=1 and n=1 showed nothing to gain.

## What this says about step 1 finding 5
Removing one of three serialized round trips per iteration changed nothing at n=1. With 6-7 waves per SIMD the per-wave chain
latency is already hidden by the other waves, so "latency-bound on the dependent chain" in its simple form is wrong. What step 1
still shows: loads alone cost 52-55% of the runtime and reach only ~340-350 GB/s of weights, and the decode/dot body adds the rest
without any single removable component. The next hypotheses should target the memory side's efficiency (many small, partly
scattered loads per useful byte: ubyte scale, ushort d, 2-byte-aligned qs, 8-byte gathers through the vector memory path) and the
work per thread (VDR / weights per load), not issue order. Untested ideas, in order of cost: grid in LDS (removes the gathers from
the vector memory path; probe 3 suggests at most -3..-10%), and a VDR 4 layout (one thread handles a full 64-weight sub-block pair,
fewer, wider loads per useful byte).

# Step 3: iq2xs_grid staged in LDS, 2026-11-23

**Result: net negative for decode.** The gather saving is real, but at n=1 the per-block 4 KB copy costs more than it saves
on the k=5120 shapes, and those include the dominant ffn gate/up.

## Change
Spike commit `f7d1a4e37` (patch also in `probes/step3-lds-grid-mmvq.patch`): opt-in `-DGGML_HIP_IQ2_XS_LDS_GRID=1` (HIP only, off by
default), mmvq.cu only, `mul_mat_vec_q` IQ2_XS instances only: all threads of a block copy the 512x8 B `iq2xs_grid` into
`__shared__` once, `__syncthreads()`, then `vec_dot_iq2_xs_q8_1_lds` (stock math, grid pointer argument) is used for both the main
and the fused-gate vec_dot. `mul_mat_vec_q_moe` is untouched. Step 2 flag off.
- Flag-off: whole-file mmvq ISA identical to HEAD (only the per-file `__hip_cuid` hash differs), so `build-spike` stays the stock reference.
- Flag-on resources: 0 spills; n=1 38 VGPR, occupancy 6 (unchanged), LDS 4352 B/block (was 256); n=4 occupancy 4 (unchanged);
  n=2 occupancy 5 (was 4).
- Build `build-spike-lds`: CMake cache identical to build-spike except the flag.

## Gate
`build-spike-lds`: `test-backend-ops -o MUL_MAT -p type_a=iq2_xs` and `-o MUL_MAT_ID -p type_a=iq2_xs`: **2/2 backends OK, 0 FAIL**.
Gap: IQ2_XS is not in `base_types`, so no `MUL_MAT_VEC_FUSION` case covers the fused-gate LDS call (code identical to the main call).

## Perf (us, median of 3 interleaved stock/lds rounds, spread < 0.5%; `logs/step3-lds-vs-stock.log`)

| shape (m x k) | n=1 stock | n=1 LDS | change | n=4 change | n=8 change |
|---|---|---|---|---|---|
| 17408x5120 (ffn gate/up, x128/token) | 148.74 | 173.29 | **+16.5%** | -7.3% | -2.2% |
| 5120x17408 (ffn down, x56) | 135.91 | 127.18 | **-6.4%** | -11.1% | -3.9% |
| 6144x5120 (x48) | 56.91 | 65.77 | +15.6% | -7.3% | -4.9% |
| 5120x6144 (x48) | 50.73 | 57.27 | +12.9% | -6.7% | -3.5% |
| 4096x14336 (reference) | 89.25 | 86.15 | -3.5% | -10.4% | -4.5% |

Stock numbers match step 1 (88.9 / 136.1 / 149.0 / 50.7 / 57.0) within 0.4%.
Per token at n=1 (counts from step 1): +3.14 (gate/up) - 0.49 (down) + 0.43 + 0.31 = **about +3.4 ms/token slower** (~ -6% tg).

## Reading
- Staging cost scales with block count: at n=1 there is 1 row per block, so a block copies 4096 B of grid to use one row's
  weights: 20 x 74 B = 1480 B at k=5120, 68 x 74 B = 5032 B at k=17408. Only the long-k shapes amortize it. At n=4/8
  (2 rows per block, decode shared over 4-8 columns) every shape gains.
- Measured value of removing the global gathers, where the copy is amortized: -6.4% (5120x17408) and -3.5% (4096x14336) at n=1,
  -10..-11% at n=4. Probe 3 (gathers replaced by arithmetic, no copy) predicted -10% and -9% at n=1. So the gather is worth single
  digits, as probe 3 said. It is not the lever for the 2x gap to Q4_K/Q8_0 bandwidth.
- Not tried (would change more than this one item): fewer staging copies via more rows per block at n=1 (step 1 rows probe: rows 4
  alone gives +5%/-5%), or a persistent grid-stride kernel that stages once per CU.

# Step 4: VDR 4 (64 weights per thread), 2026-11-23

**Result: negative at every n, and the loads-only floor got worse, not better** (4096x14336 n=1: 48.9 -> 72.7 us, 347 -> 233 GB/s).

## Change
Spike commit `a5d139cb5` (patch also in `probes/step4-vdr4-mmvq.patch`): opt-in `-DGGML_HIP_IQ2_XS_VDR4=1` (HIP only, off by default),
mmvq.cu only. `get_vdr_mmvq(IQ2_XS)` returns 4 and `get_vec_dot_q_cuda(IQ2_XS)` returns `vec_dot_iq2_xs_q8_1_vdr4`. Since both are
looked up by both the dense and the moe mmvq kernels, both use it. Each call covers 64 weights (4 threads per 256-weight block
instead of 8): the 4 adjacent qs ints as one 16-byte load (offset 2 + 16m, only 2-byte aligned: relies on gfx9 unaligned global
access), both scale bytes as one ushort, two q8_1 blocks. Same math per 32-weight half. Step 2/3 flags off, no LDS.
- Flag-off: whole-file mmvq ISA identical to HEAD (except the `__hip_cuid` hash).
- n=1 loop ISA: 16 loads per 64 weights (5 dwordx4, 9 dwordx2 incl. 8 gathers, 2 ushort) vs stock 2 x 10 = 20, i.e. non-gather loads
  12 -> 8 as designed; 343 instructions / 16 dot4 per iteration (stock 185 / 8).

## Register pressure (`logs/step4-resusage-vdr4.txt`): 0 VGPR / 0 SGPR spills everywhere, occupancy down

| instance (iq2_xs) | stock VGPR / occ | VDR4 VGPR / occ |
|---|---|---|
| mul_mat_vec_q n=1 (plain) | 38 / 6 | 44 / 5 |
| mul_mat_vec_q n=1 fused gate | 49 / 4 | 72 / 3 |
| mul_mat_vec_q n=1 fused gate, small_k | 63 / 4 | 122 / 2 |
| mul_mat_vec_q n=1 small_k | 49 / 4 | 42 / 5 |
| mul_mat_vec_q n=2 / 3 / 4 | 59 / 62 / 58, occ 4 | 79 / 85 / 98, occ 3 / 2 / 2 |
| mul_mat_vec_q n=5..8 | 58-64, occ 4 | 100-127, occ 2 |
| mul_mat_vec_q_moe (fusion / plain) | 61 / 60, occ 4 | 120 / 43, occ 2 / 5 |

## Gate
`build-spike-vdr4` (CMake cache = build-spike + the flag): `test-backend-ops -o MUL_MAT -p type_a=iq2_xs` and
`-o MUL_MAT_ID -p type_a=iq2_xs`: **2/2 backends OK, 0 FAIL** (so the unaligned 16-byte loads return correct data).

## Perf (us, median of 3 interleaved stock/VDR4 rounds, spread < 0.5%; `logs/step4-vdr4-vs-stock.log`)

| shape (m x k) | n=1 stock | n=1 VDR4 | change | n=4 change | n=8 change |
|---|---|---|---|---|---|
| 17408x5120 (ffn gate/up) | 148.70 | 168.80 | **+13.5%** | +65.8% | +37.5% |
| 5120x17408 (ffn down) | 136.28 | 158.55 | **+16.3%** | +125.7% | +12.1% |
| 6144x5120 | 57.01 | 63.71 | +11.8% | +60.4% | +34.4% |
| 5120x6144 | 50.81 | 56.20 | +10.6% | +80.9% | +31.4% |
| 4096x14336 | 89.07 | 102.09 | +14.6% | +128.0% | +7.5% |

Stock matches step 1 within 0.3%.

## Did it move the loads-only floor? No, the floor got slower
Temporary probe (reverted, not committed; same idea as step 1 probe 1): the VDR4 vec_dot with all its loads (16-byte qs, ushort
scales, d, 2 x 32 B activations + 2 ds) but no gather, sign math or dot.

| n=1 | stock loads-only (step 1 probe 1) | VDR4 loads-only | change |
|---|---|---|---|
| 4096x14336 | 48.9 (347 GB/s) | 72.7 (233 GB/s) | +49% |
| 5120x17408 | 74.1 | 129.7 | +75% |
| 17408x5120 | 76.6 | 77.5 | +1% |

So fewer, wider loads per weight did not raise the floor; on two shapes it fell. Not diagnosed further (this was a single-change
step). Plausible contributors, all untested: (a) the 16-byte qs load is only 2-byte aligned, and unaligned dwordx4 accesses may be
split or take a slower path in the gfx9 memory pipeline; (b) half the threads per row: at k=17408 (68 blocks per row) a 2-wave
block covers 32 blocks per iteration, 68/96 = 71% lane utilisation vs 68/80 = 85% at VDR 2, which fits 5120x17408 being hit
hardest; at k=5120 both layouts use 20/32 of the lanes; (c) a different per-wave address footprint (one wave spans 16 weight
blocks and 2x the activation span instead of 8). The full kernel adds the occupancy loss on top (above table).

## Where this leaves the spike (steps 2-4)
None of the three levers aimed at the stock IQ2_XS mmvq beats it at n=1: issue order (step 2, neutral), the codebook gathers
(step 3, worth 3.5-6.4% where the copy is amortized, net negative), and the load layout (step 4, negative, floor worse). The stock
VDR 2 layout already has the best loads-only floor measured (~347 GB/s). Getting closer to Q4_K/Q8_0's 400-480 GB/s would need a
different weight layout in memory (e.g. a repacked, aligned IQ2_XS copy at load time) rather than kernel-side changes on the
74-byte, 2-byte-aligned blocks. That would be a bigger design step, not tried here.

# Step 5: weight-layout microbenchmark (standalone HIP, no llama.cpp changes), 2026-11-23

**Decision gate (full kernel >= 20% faster than stock n=1 on the FFN shapes): no layout passes.** Best: L3 at -6.2% (ffn gate/up)
and -4.5% (ffn down) vs llama.cpp stock, and L3 needs fp16 fused scales that add error. Nothing integrated.

## Harness
`probes/layout-bench/` (`iq2xs_layout_bench.cpp`, `iq2xs_grid.h` + `gen_grid.sh`, `build.sh`, `run.sh`). ROCm 6.2 clang, gfx906.
`run.sh` looks up four real IQ2_XS tensors with gguf-py (one per decode shape: `blk.3.ffn_gate`, `blk.8.ffn_down`,
`blk.0.attn_gate`, `blk.0.ssm_out`). The binary `pread`s only those bytes from the GGUF (~70 MB, under an 8 GB scope), builds the
repacked copies on the host and uploads them. Kernels copy stock mmvq's n=1 GCN shape: 128 threads (2 waves) per row,
8 threads per 256-weight block, 32 weights per thread per step, q8_1 activations (random N(0,1), stock 36 B layout), codebook as a
`__device__` table, and llama.cpp's HIP `__vsub4` / `__vcmpne4` / `unpack_ksigns` / sdot4 code verbatim. Timing: 20 warm-up
launches, then 7 x 100 launches, median per-launch time (spread mostly < 2.5%). Oracle: double-precision CPU dot of
`dequantize_row_iq2_xs` weights with the same q8_1 activations. Error is reported as max |gpu - oracle| / rms(oracle) over the rows.

Layouts (all built from the same tensor bytes):
- **S**: stock AoS `block_iq2_xs`, 74 B (d | qs[32] | scales[8], 2-byte aligned); stock `vec_dot_iq2_xs_q8_1` math.
- **L1 planar**: qs plane (64 B/block, each thread's 8 B naturally aligned), scales plane (8 B/block), d plane (fp16). 74 B/block.
- **L2 aligned AoS**: block padded to 80 B, 16-byte aligned (qs at 0, scales at 64, d at 72). +8% bytes.
- **L3 qs + fused scales**: qs plane + one `half2` per 32-weight sub-block = (d*(0.5+ls_lo)/4, d*(0.5+ls_hi)/4). Removes the d load
  and the integer scale fixup: 2 weight loads per thread step (uint2 + dword). 96 B/block, +30% bytes.
Each layout has (a) a loads-only kernel (same loads incl. activations, no gather / sign / dot) and (b) the full dp4a kernel.
Registers: all 8 kernels 24-34 VGPR, 0 spills, occupancy 7-8.

## Results (`logs/step5-layout-bench-run2.log`)
\* GB/s = 74-byte-equivalent weight bytes / time, i.e. useful IQ2_XS data rate, comparable across layouts and with steps 1-4.
"vs llama.cpp stock" = against `test-backend-ops` stock n=1 measured the same day (step 4 run: 148.70 / 136.28 / 57.01 / 50.81 us).
That run includes the q8_1 quantize of the activations (a few us), so it slightly favours the layouts here.

| tensor (m x k) | layout | B/block | loads-only us | floor GB/s* | full us | eff GB/s* | vs harness S | vs llama.cpp stock | max err / rms vs oracle |
|---|---|---|---|---|---|---|---|---|---|
| blk.3.ffn_gate (17408x5120) | S  stock AoS 74B | 74 | 68.69 | 375 | 153.33 | 168 | +0.0% | **+3.1%** (stock 148.7) | 1.74e-04 |
| blk.3.ffn_gate (17408x5120) | L1 planar qs / sc / d | 74 | 69.85 | 369 | 146.78 | 176 | -4.3% | **-1.3%** (stock 148.7) | 1.74e-04 |
| blk.3.ffn_gate (17408x5120) | L2 aligned AoS 80B | 80 | 69.68 | 370 | 144.73 | 178 | -5.6% | **-2.7%** (stock 148.7) | 1.74e-04 |
| blk.3.ffn_gate (17408x5120) | L3 qs + fused scale | 96 | 69.52 | 371 | 139.44 | 185 | -9.1% | **-6.2%** (stock 148.7) | 9.66e-04 |
| blk.8.ffn_down (5120x17408) | S  stock AoS 74B | 74 | 64.75 | 398 | 170.55 | 151 | +0.0% | **+25.1%** (stock 136.28) | 1.14e-04 |
| blk.8.ffn_down (5120x17408) | L1 planar qs / sc / d | 74 | 67.75 | 380 | 144.46 | 178 | -15.3% | **+6.0%** (stock 136.28) | 1.14e-04 |
| blk.8.ffn_down (5120x17408) | L2 aligned AoS 80B | 80 | 69.82 | 369 | 135.95 | 190 | -20.3% | **-0.2%** (stock 136.28) | 1.14e-04 |
| blk.8.ffn_down (5120x17408) | L3 qs + fused scale | 96 | 70.77 | 364 | 130.09 | 198 | -23.7% | **-4.5%** (stock 136.28) | 8.28e-04 |
| blk.0.attn_gate (6144x5120) | S  stock AoS 74B | 74 | 25.54 | 356 | 56.88 | 160 | +0.0% | **-0.2%** (stock 57.01) | 1.21e-04 |
| blk.0.attn_gate (6144x5120) | L1 planar qs / sc / d | 74 | 26.36 | 345 | 54.79 | 166 | -3.7% | **-3.9%** (stock 57.01) | 1.21e-04 |
| blk.0.attn_gate (6144x5120) | L2 aligned AoS 80B | 80 | 26.63 | 341 | 54.01 | 168 | -5.1% | **-5.3%** (stock 57.01) | 1.21e-04 |
| blk.0.attn_gate (6144x5120) | L3 qs + fused scale | 96 | 25.81 | 352 | 52.40 | 174 | -7.9% | **-8.1%** (stock 57.01) | 7.86e-04 |
| blk.0.ssm_out (5120x6144) | S  stock AoS 74B | 74 | 22.35 | 407 | 50.21 | 181 | +0.0% | **-1.2%** (stock 50.81) | 6.22e-04 |
| blk.0.ssm_out (5120x6144) | L1 planar qs / sc / d | 74 | 23.88 | 381 | 48.72 | 187 | -3.0% | **-4.1%** (stock 50.81) | 6.22e-04 |
| blk.0.ssm_out (5120x6144) | L2 aligned AoS 80B | 80 | 23.88 | 381 | 48.11 | 189 | -4.2% | **-5.3%** (stock 50.81) | 6.22e-04 |
| blk.0.ssm_out (5120x6144) | L3 qs + fused scale | 96 | 24.19 | 376 | 46.44 | 196 | -7.5% | **-8.6%** (stock 50.81) | 1.13e-02 |

## Findings
1. **Layout does not move the loads-only floor.** All four layouts land at 340-410 GB/s (74 B-equivalent) loads-only on every shape.
   That includes L3, which moves 30% more weight bytes, and L1, which makes every per-thread weight load naturally aligned. So the
   floor is not set by weight-load alignment or weight bytes. With activations (36 B per 9.25 B of weights per thread) and per-instruction
   costs in the mix, the memory side looks bound by the number of load instructions / the activation stream, not by weight layout.
   This matches step 4, where fewer, wider weight loads did not help either.
2. **Full-kernel gains are modest and come from less work per step, not from the floor.** L3 (fewest loads and no integer scale math)
   is fastest everywhere, -4.5..-8.6% vs llama.cpp stock. L2 / L1 give -5% on the small shapes and ~0..-3% on the FFN shapes.
3. **Correctness:** S, L1, L2 are bit-identical to S (same integer math) at 1.1e-4..6.2e-4 of rms vs the oracle (S itself has this
   error from the stock `(sumi*ls + sumi/2)/4` integer truncation). L3 is 7.9e-4..1.1e-2 (fp16 rounding of the fused scale; 1.1e-2 on
   `ssm_out`), about 5-20x stock's error. It would need a PPL check before any use, or fp32 scales (+32 B/block).
4. **Harness caveat:** the harness's own stock-equivalent S matches llama.cpp stock on three shapes (-1.2..+3.1%) but is 25% slower on
   ffn down (170.6 vs 136.3 us; 32-bit vs 64-bit index math made no difference, run1 vs run2). Against harness S, L2 / L3 would show
   -20% / -24% on ffn down. That is a weakness of the harness skeleton on long-k rows, not a real gain: llama.cpp's stock kernel already
   reaches L2's level there. The gate is judged against llama.cpp stock.

## Verdict
No layout earns integration (needed -20% on both FFN shapes; best -6.2% / -4.5%). Across steps 2-5 the IQ2_XS n=1 mmvq on gfx906 sits at
~170-200 GB/s useful weight rate, against a loads-only floor of ~350-400 GB/s that neither issue order, codebook placement, VDR nor weight
layout moved. The remaining gap to Q4_K/Q8_0 (400-480 GB/s) is the decode/dot body (gather + emulated byte-wise sign ops + scale fixup,
185 instructions per 8 dot4) on top of an activation-heavy floor. The candidates left are compute-side: a cheaper sign application than
`__vcmpne4`/`__vsub4` (which costs quarter-rate `v_mul_lo_u32`), or sign-folded codebook entries (a 64K-entry table is too big; the
512 x 128 sign combinations would need another decomposition). Neither is tested.

# Step 6: cheaper sign application (layout-bench only), 2026-11-23

**Result: bit-exact and a real but modest win on the k=5120 shapes (best L1c: -10.9% on ffn gate/up, -12% on the two small shapes vs
llama.cpp stock). It does not help ffn down, and nothing reaches the -20% gate.** The ksigns64-table variant has the fewest
instructions but is 21-37% slower. Another case (as with step 1 probe 5) where instruction count does not predict speed.

## Premise correction
The iq2xs grid bytes are **8, 25 and 43** (checked over all 512 x 8 bytes), not 1..7. The two-op trick still holds, because it only
needs nonzero magnitudes below 128: with a per-byte mask m (0xFF = negative) and b = m & 1, a negative byte becomes
(g ^ 0xFF) + 1 = 256 - g with no carry out of the byte, and positive bytes add 0. It does need a **byte mask**, not sign bits, and
building that mask is what costs in stock (`unpack_ksigns`' `s * 0x01010101` broadcast plus `__vcmpne4`'s `* 0xff`: 12 of the loop's 14
quarter-rate `v_mul_lo_u32`).

## Variants (`probes/layout-bench`, `signed_grid<SIGN>`; all keep the stock dot order and integer scale math)
- **SIGN 0** stock: `unpack_ksigns` + `vcmpne4` + xor + `vsub4` (llama.cpp's HIP code).
- **SIGN 1 "c" computed mask**: 8 sign bits (parity fixed via popc) -> per-byte 0/1 with a 24-bit multiply by 0x00204081 per nibble
  (bits do not overlap, so no carries) -> mask via `(0x80808080 - b) ^ 0x80808080` -> `(g ^ m) + b`.
- **SIGN 2 "t" table**: llama.cpp's `ksigns64` (128 x 8-byte byte masks, parity included) as a `__device__` table, `(g ^ m) + (m & 0x01010101)`.
- **SIGN 3 "f" stock folded** (your folded-multiply variant): stock math, with the broadcast done as one `v_perm_b32` and
  `vcmpne4`'s `*0xff` replaced by the same mask expansion.
- Applied to L1 (planar) and L2 (aligned 80 B), plus one extra row, **Sc** = stock 74 B layout + SIGN 1 (the integration-cheapest form,
  no repack).

Compiler pitfalls found in the ISA (both fixed, documented in the source): clang lowers `__umul24` to the quarter-rate `v_mul_lo_u32`
on gfx906 (forced `v_mul_u32_u24` via inline asm), and LLVM canonicalizes `(b << 8) - b` into `b * 255`, also `v_mul_lo_u32`.

## ISA, n=1 loop (`logs/step6-isa-loop.txt`)

| variant | loop instructions | v_mul_lo_u32 (quarter rate) | other | loads | VGPR / occ |
|---|---|---|---|---|---|
| S / L1 / L2 stock signs | 186 / 188 / 186 | 14 | - | 10 | 31-34 / 7-8 |
| L1c / L2c computed | 137 / 134 | 2 (scales only) | 8 v_mul_u32_u24 | 10 | 32-33 / 7-8 |
| L1t / L2t table | 109 / 106 | 2 | - | 14 (+4 sign gathers) | 24-26 / 8 |
| L1f / L2f folded | 188 / 186 | 2 | 4 v_perm | 10 | 29-34 / 7-8 |

0 spills everywhere.

## Correctness
- Exhaustive device self-test over all 65536 qs values (9-bit grid index x 7-bit sign index): signed bytes of SIGN 1 / 2 / 3 vs stock:
  **0 / 0 / 0 mismatches**.
- Full kernels on the four real tensors: every c / t / f variant (and Sc) is **bitwise identical to S on every row** (0 differing rows
  of 17408 / 5120 / 6144 / 5120), so the oracle error is exactly S's (1.1e-4..6.2e-4 of rms). L3 rows are shown for comparison only.

## Perf (`logs/step6-sign-variants.log`; median of 7 x 100 launches; "vs llama.cpp stock" = test-backend-ops stock n=1, same day)

| tensor (m x k) | variant | full us (spread) | eff GB/s* | vs harness S | vs llama.cpp stock | max err / rms vs oracle | rows bitwise != S |
|---|---|---|---|---|---|---|---|
| blk.3.ffn_gate (17408x5120) | S  stock AoS 74B | 153.36 (spread 1.8%) | 168 | +0.0% | **+3.1%** | 1.74e-04 | 0/17408 |
| blk.3.ffn_gate (17408x5120) | L1 planar qs/sc/d | 146.02 (spread 0.9%) | 176 | -4.8% | **-1.8%** | 1.74e-04 | 0/17408 |
| blk.3.ffn_gate (17408x5120) | L2 aligned AoS 80B | 146.27 (spread 1.0%) | 176 | -4.6% | **-1.6%** | 1.74e-04 | 0/17408 |
| blk.3.ffn_gate (17408x5120) | L3 qs + fused scale | 140.06 (spread 1.6%) | 184 | -8.7% | **-5.8%** | 9.66e-04 | 17403/17408 |
| blk.3.ffn_gate (17408x5120) | Sc stock AoS, computed sign mask | 140.78 (spread 0.9%) | 183 | -8.2% | **-5.3%** | 1.74e-04 | 0/17408 |
| blk.3.ffn_gate (17408x5120) | L1c planar, computed sign mask | 132.49 (spread 1.4%) | 194 | -13.6% | **-10.9%** | 1.74e-04 | 0/17408 |
| blk.3.ffn_gate (17408x5120) | L2c aligned, computed sign mask | 137.40 (spread 0.3%) | 188 | -10.4% | **-7.6%** | 1.74e-04 | 0/17408 |
| blk.3.ffn_gate (17408x5120) | L1t planar, ksigns64 table | 184.70 (spread 0.4%) | 139 | +20.4% | **+24.2%** | 1.74e-04 | 0/17408 |
| blk.3.ffn_gate (17408x5120) | L2t aligned, ksigns64 table | 186.19 (spread 0.5%) | 138 | +21.4% | **+25.2%** | 1.74e-04 | 0/17408 |
| blk.3.ffn_gate (17408x5120) | L1f planar, stock signs folded | 143.65 (spread 1.5%) | 179 | -6.3% | **-3.4%** | 1.74e-04 | 0/17408 |
| blk.3.ffn_gate (17408x5120) | L2f aligned, stock signs folded | 142.13 (spread 0.9%) | 181 | -7.3% | **-4.4%** | 1.74e-04 | 0/17408 |
| blk.8.ffn_down (5120x17408) | S  stock AoS 74B | 172.72 (spread 2.6%) | 149 | +0.0% | **+26.7%** | 1.14e-04 | 0/5120 |
| blk.8.ffn_down (5120x17408) | L1 planar qs/sc/d | 145.40 (spread 1.1%) | 177 | -15.8% | **+6.7%** | 1.14e-04 | 0/5120 |
| blk.8.ffn_down (5120x17408) | L2 aligned AoS 80B | 138.11 (spread 1.0%) | 187 | -20.0% | **+1.3%** | 1.14e-04 | 0/5120 |
| blk.8.ffn_down (5120x17408) | L3 qs + fused scale | 134.49 (spread 1.6%) | 192 | -22.1% | **-1.3%** | 8.28e-04 | 5119/5120 |
| blk.8.ffn_down (5120x17408) | Sc stock AoS, computed sign mask | 170.52 (spread 0.6%) | 151 | -1.3% | **+25.1%** | 1.14e-04 | 0/5120 |
| blk.8.ffn_down (5120x17408) | L1c planar, computed sign mask | 140.38 (spread 0.8%) | 184 | -18.7% | **+3.0%** | 1.14e-04 | 0/5120 |
| blk.8.ffn_down (5120x17408) | L2c aligned, computed sign mask | 151.80 (spread 0.8%) | 170 | -12.1% | **+11.4%** | 1.14e-04 | 0/5120 |
| blk.8.ffn_down (5120x17408) | L1t planar, ksigns64 table | 186.86 (spread 0.2%) | 138 | +8.2% | **+37.1%** | 1.14e-04 | 0/5120 |
| blk.8.ffn_down (5120x17408) | L2t aligned, ksigns64 table | 187.00 (spread 0.5%) | 138 | +8.3% | **+37.2%** | 1.14e-04 | 0/5120 |
| blk.8.ffn_down (5120x17408) | L1f planar, stock signs folded | 144.89 (spread 0.8%) | 178 | -16.1% | **+6.3%** | 1.14e-04 | 0/5120 |
| blk.8.ffn_down (5120x17408) | L2f aligned, stock signs folded | 138.12 (spread 1.0%) | 187 | -20.0% | **+1.4%** | 1.14e-04 | 0/5120 |
| blk.0.attn_gate (6144x5120) | S  stock AoS 74B | 56.32 (spread 8.2%) | 161 | +0.0% | **-1.2%** | 1.21e-04 | 0/6144 |
| blk.0.attn_gate (6144x5120) | L1 planar qs/sc/d | 54.80 (spread 2.6%) | 166 | -2.7% | **-3.9%** | 1.21e-04 | 0/6144 |
| blk.0.attn_gate (6144x5120) | L2 aligned AoS 80B | 54.47 (spread 2.2%) | 167 | -3.3% | **-4.5%** | 1.21e-04 | 0/6144 |
| blk.0.attn_gate (6144x5120) | L3 qs + fused scale | 52.00 (spread 2.2%) | 175 | -7.7% | **-8.8%** | 7.86e-04 | 6142/6144 |
| blk.0.attn_gate (6144x5120) | Sc stock AoS, computed sign mask | 53.19 (spread 4.8%) | 171 | -5.6% | **-6.7%** | 1.21e-04 | 0/6144 |
| blk.0.attn_gate (6144x5120) | L1c planar, computed sign mask | 50.17 (spread 2.5%) | 181 | -10.9% | **-12.0%** | 1.21e-04 | 0/6144 |
| blk.0.attn_gate (6144x5120) | L2c aligned, computed sign mask | 51.33 (spread 2.8%) | 177 | -8.9% | **-10.0%** | 1.21e-04 | 0/6144 |
| blk.0.attn_gate (6144x5120) | L1t planar, ksigns64 table | 69.19 (spread 1.1%) | 131 | +22.8% | **+21.4%** | 1.21e-04 | 0/6144 |
| blk.0.attn_gate (6144x5120) | L2t aligned, ksigns64 table | 69.53 (spread 0.8%) | 131 | +23.5% | **+22.0%** | 1.21e-04 | 0/6144 |
| blk.0.attn_gate (6144x5120) | L1f planar, stock signs folded | 53.81 (spread 2.5%) | 169 | -4.5% | **-5.6%** | 1.21e-04 | 0/6144 |
| blk.0.attn_gate (6144x5120) | L2f aligned, stock signs folded | 53.55 (spread 1.8%) | 170 | -4.9% | **-6.1%** | 1.21e-04 | 0/6144 |
| blk.0.ssm_out (5120x6144) | S  stock AoS 74B | 50.10 (spread 8.3%) | 181 | +0.0% | **-1.4%** | 6.22e-04 | 0/5120 |
| blk.0.ssm_out (5120x6144) | L1 planar qs/sc/d | 48.40 (spread 2.9%) | 188 | -3.4% | **-4.7%** | 6.22e-04 | 0/5120 |
| blk.0.ssm_out (5120x6144) | L2 aligned AoS 80B | 48.42 (spread 1.8%) | 188 | -3.4% | **-4.7%** | 6.22e-04 | 0/5120 |
| blk.0.ssm_out (5120x6144) | L3 qs + fused scale | 46.34 (spread 2.8%) | 196 | -7.5% | **-8.8%** | 1.13e-02 | 5120/5120 |
| blk.0.ssm_out (5120x6144) | Sc stock AoS, computed sign mask | 47.59 (spread 2.8%) | 191 | -5.0% | **-6.3%** | 6.22e-04 | 0/5120 |
| blk.0.ssm_out (5120x6144) | L1c planar, computed sign mask | 44.67 (spread 2.3%) | 204 | -10.8% | **-12.1%** | 6.22e-04 | 0/5120 |
| blk.0.ssm_out (5120x6144) | L2c aligned, computed sign mask | 44.70 (spread 2.9%) | 203 | -10.8% | **-12.0%** | 6.22e-04 | 0/5120 |
| blk.0.ssm_out (5120x6144) | L1t planar, ksigns64 table | 61.45 (spread 1.1%) | 148 | +22.6% | **+20.9%** | 6.22e-04 | 0/5120 |
| blk.0.ssm_out (5120x6144) | L2t aligned, ksigns64 table | 61.66 (spread 1.0%) | 147 | +23.1% | **+21.4%** | 6.22e-04 | 0/5120 |
| blk.0.ssm_out (5120x6144) | L1f planar, stock signs folded | 47.25 (spread 2.5%) | 192 | -5.7% | **-7.0%** | 6.22e-04 | 0/5120 |
| blk.0.ssm_out (5120x6144) | L2f aligned, stock signs folded | 47.38 (spread 1.5%) | 192 | -5.4% | **-6.8%** | 6.22e-04 | 0/5120 |

## Reading
1. **The computed mask is the only sign variant that wins clearly**: on the stock layout (Sc) -5..-7% vs llama.cpp stock; with L1
   -11..-12% on the three k=5120 shapes (gate/up, attn_gate, ssm_out). Within the harness, L1 vs L1c isolates the sign change:
   -9.3% (gate/up), -3.5% (down), -8.4% / -7.7% (small shapes).
2. **ffn down (k=17408) does not benefit**: Sc -1.3% vs harness S, L2c is even slower than L2 (151.8 vs 138.1 us, reproducible, spread
   < 1%). Against llama.cpp stock every down variant except L3 is >= 0%, but the harness's stock S is 26% slower than llama.cpp stock on
   this shape (step 5 caveat), so down numbers are pessimistic and not conclusive either way.
3. **The table variant is a clear loss** (+21..+37%) despite 43% fewer loop instructions: one extra dependent 8-byte gather per entry
   (4 per step) costs far more than the ALU it saves. This is consistent with step 3: gathers through the vector memory path are
   expensive on this part; trading ALU for gathers is the wrong direction.
4. **Folding only the multiplies (f)** gives -1.6..-2.8% over the same layout (L1 -> L1f, L2 -> L2f) on k=5120 shapes. Most of the
   computed-mask gain comes from replacing the whole vcmpne4/vsub4 emulation, not just from removing the quarter-rate multiplies.

## Gate verdict and what it means for integration
No variant meets -20% on both FFN shapes (best gate/up: L1c -10.9%; best down: L3 -1.3%, which is not bit-exact). Rough per-token
estimate for L1c (using the harness down number, i.e. pessimistic): -2.07 ms (gate/up x128) + 0.23 (down x56) - 0.33 (attn_gate x48)
- 0.29 (ssm_out x48) = **about -2.5 ms of a 50.8 ms token (~5% tg)**. Sc (no repack, drop-in for the stock vec_dot) would give roughly
-1.0..-1.5 ms (-2..-3% tg), assuming its harness down result transfers. These are the numbers to weigh at the consolidation step;
neither is integrated.

# Step 7: Sc integrated in llama.cpp (GGML_HIP_IQ2_XS_FAST_SIGN), real-model A/B, 2026-11-23

**Result: bit-exact on the real model (wikitext-2 PPL identical in all three modes, KLD identical to a stock-vs-stock control) and
tg128 +3.1% (19.53 -> 20.14 t/s, non-overlapping ranges), pp512 unchanged.** Profiling shows the gain is in the fused-gate mmvq
instance (-7.5% per call) while the plain mmvq instance gets slower (+17% per call), which explains why the plain-MUL_MAT op
benchmark shows a regression.

## Change
Spike commit `214d49d49` (patch also in `probes/step7-fast-sign.patch`): opt-in `-DGGML_HIP_IQ2_XS_FAST_SIGN=1` (HIP only, off by default)
inside the stock `vec_dot_iq2_xs_q8_1` (vecdotq.cuh). Its only consumer is mmvq, so the plain, fused-gate and moe IQ2_XS paths all use
it. The sign application is replaced by the step 6 SIGN 1 computed mask (24-bit multiply spread via inline-asm `v_mul_u32_u24`,
mask `(0x80808080 - b) ^ 0x80808080`, `(g ^ m) + b`), identical to the harness code that passed the exhaustive 65536-value self-test.
That self-test stays in `probes/layout-bench` as the unit check (it exercises the same expression; adding a device test to llama.cpp's
test suite was not cheap). No layout change, no repack. Step 2/3/4 flags off.
- Flag-off: whole-file mmvq ISA identical to stock (except `__hip_cuid`).
- Flag-on ISA: n=1 plain kernel 351 -> 299 instructions, `v_mul_lo_u32` 14 -> 2, VGPR 38 -> 32, occupancy 6 -> 8. Decode sharing
  across columns is preserved at n=2/4/8 (dwordx2 10 -> 11, `v_bcnt` 8 -> 8; contrast step 2). Fused-gate n=1: 49 -> 57 VGPR, occupancy 4
  (unchanged). Two instances drop to occupancy 2: fused + small_k (101 VGPR), never launched for IQ2_XS at n=1 because small_k is forced off
  for IQ2 types, and moe with fusion (99 VGPR), not used by this dense model. 0 spills everywhere (`logs/step7/step7-isa-res.txt`).
- Builds: `build-spike-flagon` = build-spike's CMake cache + the flag. `build-spike` rebuilt from the same source (flag off).

## Gates (`logs/step7/progress.log`)
`test-backend-ops -o MUL_MAT -p type_a=iq2_xs` and `-o MUL_MAT_ID -p type_a=iq2_xs`: **2/2 backends OK, 0 FAIL** on both builds.

## Bit-exactness on the real model (wikitext-2-raw test, `-ngl 99 -fa on -lm none`)

| mode | path exercised | stock PPL | flag-on PPL |
|---|---|---|---|
| default batch, `--chunks 40` | prefill (MMQ; code unchanged, control) | 6.9276 +/- 0.16737 | **6.9276 +/- 0.16737** |
| `-b 8 -ub 8 --chunks 8` | mmvq ncols=8 incl. fused gate | 7.7504 +/- 0.42721 | **7.7504 +/- 0.42721** |
| `-b 1 -ub 1 --chunks 2` | mmvq ncols=1, plain + fused gate (the decode kernels) | 6.9722 +/- 0.84288 | **6.9722 +/- 0.84288** |

Per-chunk values are identical too. KL divergence of flag-on against stock's saved logits (`-b 8`, 8 chunks): mean 0.000000, max 0.000049,
same top p 100%. A **stock-vs-stock control run against the same base file gives exactly the same numbers and an identical per-chunk table**
(including "PPL(base) 7.747440"), so the residual comes from llama.cpp's compressed storage of the base logits, not from any logit difference.
No red flag.

## Speed

| test | stock median (min-max of 9) | flag-on median (min-max of 9) | change |
|---|---|---|---|
| pp512 | 214.29 (212.17-216.04) | 214.25 (212.49-215.25) | -0.02% |
| tg128 | 19.53 (19.45-19.69) | 20.14 (20.09-20.21) | +3.09% |


| shape (m x k) | n=1 stock | n=1 flag-on | change | n=4 change | n=8 change |
|---|---|---|---|---|---|
| 17408x5120 | 148.63 | 160.86 | **+8.2%** | +2.9% | +5.6% |
| 5120x17408 | 136.03 | 144.11 | **+5.9%** | +8.1% | +0.4% |
| 6144x5120 | 56.77 | 62.19 | **+9.5%** | +1.7% | +7.4% |
| 5120x6144 | 50.79 | 55.62 | **+9.5%** | +2.2% | +5.6% |
| 4096x14336 | 89.13 | 95.39 | **+7.0%** | +8.1% | -6.6% |

llama-bench: `-ngl 99 -fa 1 -lm none -p 512 -n 128 -r 3`, three interleaved invocations per build = 9 samples each (`logs/step7/bench.jsonl`).
The op table is plain `MUL_MAT` (test-backend-ops perf, median of 3 interleaved rounds, `logs/step7/opperf.log`).

## Why the plain op benchmark regresses while decode gets faster (rocprofv3 kernel trace, `llama-bench -p 0 -n 64`, `logs/step7/summary.txt`)

| kernel group (64-token decode) | stock calls / total / per call | flag-on calls / total / per call | per call |
|---|---|---|---|
| iq2_xs mmvq n=1 **fused gate** (ffn gate+up) | 7800 / 1916.3 ms / 245.68 us | 7800 / 1772.5 ms / 227.24 us | **-7.5%** |
| iq2_xs mmvq n=1 **plain** (ffn down, attn_gate, ssm_out) | 6240 / 306.2 ms / 49.07 us | 6240 / 358.3 ms / 57.41 us | **+17.0%** |
| other mmvq (q4_K / q8_0 / q2_K) | 7865 / 554.0 ms | 7865 / 549.3 ms | -0.9% |
| all other kernels | 90999 / 592.5 ms | 90999 / 577.9 ms | -2.5% |
| **total GPU kernel time** | 3369.0 ms | 3258.0 ms | **-3.3%** |

So the end-to-end gain is the fused-gate instance (ffn gate+up, the largest IQ2_XS consumer), net of a real regression in the plain
instance. The test-backend-ops `MUL_MAT` perf cases only run the plain instance, hence +6..+9.5% there. This settles the ffn-down caveat
from steps 5/6: on the real model the plain path (which carries ffn down) is slower with the fast sign, not faster. Why the plain
instance regresses although its ISA is shorter and occupancy rises 6 -> 8 is not diagnosed. Untested candidate: higher occupancy adds
contention on the vector memory path the grid gathers use (steps 3 and 6 showed gathers are the expensive resource).

Obvious follow-up (not done, one template condition): enable the fast sign only in the fused instances (`has_fusion`). From the
profile that would keep the -143.8 ms fused gain and drop the +52.1 ms plain loss: about -4.3% GPU time per token instead of -3.3%.

# Final summary, steps 1-7

| step | lever | where | n=1 ffn gate/up | n=1 ffn down | end-to-end / other | correctness | verdict |
|---|---|---|---|---|---|---|---|
| 1 | diagnosis + ablations | llama.cpp probes | stock 148.7 us (173 GB/s) | stock 136.1 us (189 GB/s) | IQ2_XS = 76% of decode matmul; README's n=5..8 step not reproducible; tg128 19.68 | gate OK | loads-only floor ~350 GB/s; no single removable bottleneck |
| 2 | phased loads (3 -> 2 round trips) | llama.cpp, `GGML_HIP_IQ2_XS_PIPE_MMVQ` | +2.3% | -1.4% | n>=2 up to +173% (lost cross-column sharing) | gate OK | rejected; latency hypothesis refuted |
| 3 | codebook in LDS | llama.cpp, `GGML_HIP_IQ2_XS_LDS_GRID` | +16.5% | -6.4% | ~+3.4 ms/token net | gate OK | rejected; gathers worth single digits |
| 4 | VDR 4 (64 weights/thread) | llama.cpp, `GGML_HIP_IQ2_XS_VDR4` | +13.5% | +16.3% | loads-only floor worse (347 -> 233 GB/s) | gate OK | rejected |
| 5 | weight layouts L1/L2/L3 | layout-bench | best L3 -6.2% | best L3 -4.5% | loads-only floor unchanged by layout | L3 not bit-exact (fp16 scales) | gate (-20%) not met |
| 6 | cheaper sign application | layout-bench | best L1c -10.9% (Sc -5.3%) | no gain (harness caveat) | table variant +21..+37% | exhaustive self-test 0 mismatches, bitwise = stock | gate not met; Sc picked as drop-in |
| 7 | Sc in llama.cpp | llama.cpp, `GGML_HIP_IQ2_XS_FAST_SIGN` | fused gate+up -7.5% per call | plain path +17% per call | **tg128 +3.1% (19.53 -> 20.14), pp512 0%** | PPL identical in 3 modes, KLD = control | **only positive result; candidate (better: fused-only)** |

Bottom line for the integration decision: the IQ2_XS mmvq on gfx906 resisted every load-side lever. The only end-to-end gain found is the
bit-exact fast sign (+3.1% tg as implemented, plausibly ~+4% if restricted to the fused-gate instances). It stays off by default on the spike
branch; production (build16) is untouched.

# Step 8: fast sign only in the fused instances (GGML_HIP_IQ2_XS_FAST_SIGN=2), 2026-11-24

**Result: tg128 +4.3% vs stock (19.47 -> 20.31 t/s), about 1 point better than fast-sign-all (+3.4%); pp512 unchanged; PPL identical.**

## Change
Spike commit `820ac2476` (patch in `probes/step8-fast-sign-fused-only.patch`): `vec_dot_iq2_xs_q8_1` gets a `fast_sign` template parameter, present only
when the flag is defined. Its default is `GGML_HIP_IQ2_XS_FAST_SIGN != 2`, so FS=1 keeps the step 7 behaviour. With FS=2, both mmvq kernels
(`mul_mat_vec_q` and `mul_mat_vec_q_moe`) pick `vec_dot_iq2_xs_q8_1<has_fusion>` for IQ2_XS, so the fast sign runs in fused instances only.
ISA check (whole mmvq.cu): flag off is identical to stock, FS=1 is identical to the step 7 flag-on build, and with FS=2 **every IQ2_XS kernel is
byte-identical to one of the two references**: fused instances (dense n=1 incl. small_k, moe) match fast-all, all 10 plain instances
(n=1..8, small_k) match stock. Build `build-spike-fused` = build-spike's CMake cache + `-DGGML_HIP_IQ2_XS_FAST_SIGN=2`. The step 7 builds
were reused unchanged (their source is ISA-identical to this commit's flag-off / FS=1 code).

## Gates and bit-exactness
- `test-backend-ops -o MUL_MAT / MUL_MAT_ID -p type_a=iq2_xs` on build-spike-fused: **2/2 backends OK, 0 FAIL**.
- wikitext-2 PPL vs stock (step 7 stock logs, same binary): `-b 1 -ub 1 --chunks 2`: **6.9722 +/- 0.84288 = stock**, per-chunk identical;
  `-b 8 -ub 8 --chunks 8`: **7.7504 +/- 0.42721 = stock**, per-chunk identical.

## Three-way comparison (`logs/step8/`)
llama-bench `-ngl 99 -fa 1 -lm none -p 512 -n 128 -r 3`, 3 invocations per build interleaved stock / all / fused-only (9 samples each):

| build | pp512 median (min-max, 9 samples) | vs stock | tg128 median (min-max, 9 samples) | vs stock |
|---|---|---|---|---|
| stock | 214.40 (212.13-216.99) | +0.00% | 19.47 (19.14-19.64) | **+0.00%** |
| fast-sign-all (FS=1) | 213.93 (212.14-215.39) | -0.22% | 20.12 (20.09-20.25) | **+3.35%** |
| fast-sign-fused-only (FS=2) | 213.79 (212.14-214.82) | -0.28% | 20.31 (20.22-20.44) | **+4.34%** |

rocprofv3 kernel trace, `llama-bench -p 0 -n 64` (one run per build; stock and fast-all are the step 7 traces):

| kernel group (64-token decode, rocprofv3) | stock us/call | fast-all us/call | fused-only us/call | stock total ms | fast-all total ms | fused-only total ms |
|---|---|---|---|---|---|---|
| iq2_xs mmvq n=1 fusion=false | 49.07 | 57.41 | 49.34 | 306.2 | 358.3 | 307.9 |
| iq2_xs mmvq n=1 fusion=true | 245.68 | 227.24 | 230.64 | 1916.3 | 1772.5 | 1799.0 |
| other kernels | 6.51 | 6.35 | 6.56 | 592.5 | 577.9 | 597.4 |
| other mmvq | 70.44 | 69.84 | 70.94 | 554.0 | 549.3 | 557.9 |
| **total GPU kernel time** | | | | 3369.0 | 3258.0 | 3262.2 |

total vs stock: fast-all -3.3%, fused-only -3.2%

Reading:
- The plain instance is back to stock speed (57.41 -> 49.34 us/call vs stock 49.07), as expected from its identical code. That recovers the
  ~50 ms per 64 tokens that fast-all lost there.
- llama-bench (the primary measurement: 9 interleaved samples per build) puts fused-only ~1 point above fast-all (+4.34% vs +3.35% tg128;
  ranges 20.22-20.44 vs 20.09-20.25, i.e. just touching). That matches the step 7 estimate of about -4.3% GPU time.
- The single-run profiles cannot resolve that 1 point: total GPU time -3.2% (fused-only) vs -3.3% (fast-all). The fused instance has
  byte-identical code in both builds yet measures 230.6 vs 227.2 us/call, so run-to-run noise between profile runs is ~1.5% per kernel group.
- pp512 is unchanged in all three builds (prefill uses MMQ, which this does not touch).

## Final three-way summary

| build | flag | tg128 (t/s) | vs stock | pp512 (t/s) | PPL -b 1 / -b 8 | plain mmvq us/call | fused mmvq us/call |
|---|---|---|---|---|---|---|---|
| stock | - | 19.47 | - | 214.40 | 6.9722 / 7.7504 | 49.07 | 245.68 |
| fast-sign-all | FS=1 | 20.12 | +3.35% | 213.93 | identical | 57.41 (+17.0%) | 227.24 (-7.5%) |
| **fast-sign-fused-only** | **FS=2** | **20.31** | **+4.34%** | 213.79 | **identical** | 49.34 (= stock) | 230.64 (-6.1%) |

Fused-only is the variant to carry forward for the integration decision: bit-exact, no regression in any kernel instance, +4.3% decode.
It stays off by default on the spike branch. Production (build16) is untouched.

# Step 9: L4 pre-dequantized int8 weights ("spill RAM to reduce compute"), layout-bench only, 2026-11-24

**Decision gate: not met.** n=512: L4 is 24-29% *slower* than llama.cpp's MMQ, and within the harness pre-dequantizing saves only
0.1-1.2% over decoding the stock format into the tile. n=4/n=8 vs fast-sign-fused-only: n=4 beats it by 18.5% on ffn gate/up, but ffn down
at n=4 (+3.7%) and both FFN shapes at n=8 (+5.5%, +12.9%) are slower. Bit-exact everywhere. Nothing integrated; llama.cpp and build16
untouched.

## L4 format and harness
`probes/layout-bench/iq2xs_l4_bench.cpp` (+ `build_l4.sh`, `run_l4.sh`; same tensor lookup as step 5).
- **int8 plane**: the sign-applied iq2xs_grid values as plain signed bytes, 256 B per 256-weight block, in weight order.
- **scale plane**: one dword per 32-weight sub-block = fp16 d | scale byte << 16 (32 B per block). Deviation from the brief (a fused fp16
  scale per sub-block): a fused scale cannot reproduce stock's `(sumi0*ls0 + sumi1*ls1 + (sumi0+sumi1)/2)/4` (step 5 L3 was 5-20x
  stock's error). Keeping d and the 4-bit scales keeps the stock formula and makes bit-exactness possible.
- Total 288 B per block = 3.89x the 74 B stock block (3.46x for the int8 plane alone).
- Kernels, each templated on S (stock 74 B + stock sign math), Sf (stock 74 B + step 6 fast sign) and L4, with an identical compute core:
  - `gemv<NC>` (n = 1..8): mmvq-shaped (GCN config: nwarps 2 for n<=4 else 1, 1 row per block at n=1 else 2, 8 threads per 256-weight block).
    The weight decode is done once per step and shared by all columns.
  - `gemm` (n = 512): 64x64 output tile per 256-thread block, K step of 4 sub-blocks. Weights are decoded (S/Sf) or copied (L4) into an
    int8 LDS tile together with a (d, sc) tile; q8_1 activations go into LDS; each thread does 4x4 outputs with the stock per-sub-block
    integer formula.
- Registers (`logs/step9/resusage-l4-bench.txt`): 0 spills everywhere. gemv n=1 21-31 VGPR, occupancy 8; n=4 74-79, occ 3; n=8 123-128,
  occ 2; gemm 106-109 VGPR, occupancy 2, 20 KB LDS.

## Correctness
- Exhaustive dequant self-test over all 65536 qs values (device stock-sign and fast-sign decode vs the host L4 plane builder): **0 / 0 mismatches**.
- Every Sf and L4 kernel output is **bitwise identical to S** at n = 1, 2, 4, 8 and 512 on all four tensors (0 differing values of up to
  8.9 M per case). Oracle error (double CPU, all rows for n<=8, 64 sampled rows x 512 columns for the GEMM) 1.3e-4..7.5e-4 of rms,
  i.e. stock's integer-truncation error.

## Results vs llama.cpp (same-session references: `test-backend-ops perf` MUL_MAT, build-spike = stock, build-spike-fused = FS=2, median of
3 rounds; harness: median of 7 x 100 / 7 x 10 launches; `logs/step9/`)

| shape | n | llama.cpp stock us | llama.cpp FS=2 us | harness S us | harness L4 us | **L4 vs llama.cpp stock** | **L4 vs llama.cpp FS=2** | L4 vs harness S |
|---|---|---|---|---|---|---|---|---|
| ffn gate/up 17408x5120 | 1 | 149.65 | 149.87 | 149.67 | 163.22 | **+9.1%** | **+8.9%** | +9.1% |
| ffn gate/up 17408x5120 | 4 | 275.89 | 276.19 | 327.88 | 225.08 | **-18.4%** | **-18.5%** | -31.4% |
| ffn gate/up 17408x5120 | 8 | 342.30 | 342.50 | 425.96 | 361.35 | **+5.6%** | **+5.5%** | -15.2% |
| ffn gate/up 17408x5120 | 512 | 7453.67 | 7453.98 | 9759.40 | 9638.46 | **+29.3%** | **+29.3%** | -1.2% |
| ffn down 5120x17408 | 1 | 136.10 | 136.05 | 171.18 | 215.17 | **+58.1%** | **+58.2%** | +25.7% |
| ffn down 5120x17408 | 4 | 227.68 | 227.67 | 359.97 | 236.11 | **+3.7%** | **+3.7%** | -34.4% |
| ffn down 5120x17408 | 8 | 315.26 | 316.65 | 419.96 | 357.57 | **+13.4%** | **+12.9%** | -14.9% |
| ffn down 5120x17408 | 512 | 7902.66 | 7905.83 | 9925.36 | 9860.56 | **+24.8%** | **+24.7%** | -0.7% |
| attn_gate 6144x5120 | 1 | 56.93 | 57.18 | 52.11 | 61.42 | **+7.9%** | **+7.4%** | +17.9% |
| attn_gate 6144x5120 | 4 | 103.76 | 103.86 | 118.59 | 82.89 | **-20.1%** | **-20.2%** | -30.1% |
| attn_gate 6144x5120 | 8 | 129.15 | 128.94 | 151.58 | 128.60 | **-0.4%** | **-0.3%** | -15.2% |
| attn_gate 6144x5120 | 512 | 2739.37 | 2739.56 | 3472.41 | 3442.19 | **+25.7%** | **+25.6%** | -0.9% |
| ssm_out 5120x6144 | 1 | 50.75 | 50.85 | 46.57 | 62.70 | **+23.5%** | **+23.3%** | +34.6% |
| ssm_out 5120x6144 | 4 | 91.78 | 91.75 | 117.89 | 81.39 | **-11.3%** | **-11.3%** | -31.0% |
| ssm_out 5120x6144 | 8 | 115.25 | 115.20 | 145.32 | 128.25 | **+11.3%** | **+11.3%** | -11.7% |
| ssm_out 5120x6144 | 512 | 2821.58 | 2823.71 | 3515.02 | 3509.79 | **+24.4%** | **+24.3%** | -0.1% |

(llama.cpp FS=2 equals stock at n=4/8/512 by construction: fused mmvq instances exist only at n=1, and MMQ is untouched. At n=1 the
plain-MUL_MAT test also runs the plain instance, so FS=2's n=1 gain, which lives in the fused gate/up kernel, does not show here.)

## Byte-math prediction at n=1: confirmed in direction, much smaller than predicted
L4 reads 3.89x the bytes (int8 + scale planes) but only costs +8..+58% time, because it streams far closer to HBM peak:

| shape | stock GB/s | L4 GB/s (all 288 B) | time ratio L4 / stock |
|---|---|---|---|
| ffn gate/up 17408x5120 | 172 | 614 | 1.09 |
| ffn down 5120x17408 | 189 | 466 | 1.58 |
| attn_gate 6144x5120 | 160 | 576 | 1.08 |
| ssm_out 5120x6144 | 179 | 564 | 1.24 |

So the achievable-bandwidth ratio is ~2.5-3.6x, not ~2.5x. That is evidence, independent of steps 2-6, that the stock IQ2_XS kernel is
compute/latency-bound, not bandwidth-bound. Still, L4 loses at n=1 on every shape, so it could only ever be an additional copy for
batched/prefill paths, never a replacement for decode.

## Reading
1. **Prefill (n=512): pre-dequantization is irrelevant to an MMQ-style kernel.** Decoding a 64-row x 128-weight tile is amortized over
   64 columns. In the harness S -> L4 saves 0.1-1.2% and Sf 2.1-2.4%. The harness GEMM itself reaches only 9.2-9.6 TFLOPS against
   llama.cpp MMQ's ~12 TFLOPS on the same shapes, so L4 vs llama.cpp is +24..+29%. llama.cpp's MMQ also decodes once per tile,
   reused over up to 128 columns, so the loader share there should be similar or smaller. A >= 15% prefill gain from L4 is not plausible.
2. **Batched decode n=2..4 is where L4 shines.** Once weights are reused across 2-4 columns the kernel turns compute-bound, and L4 removes
   the gather + sign work: vs the harness S -31..-34% at n=4; vs llama.cpp -18.4% (gate/up), -20.1% (attn_gate), -11.3% (ssm_out) at n=4,
   but +3.7% on ffn down.
3. **n=8 falls back.** The mmvq GCN config drops to 1 warp per block for n>=5, and the n=8 kernels sit at 123-128 VGPR, occupancy 2. The
   harness L4 n=8 is only -11..-15% vs harness S, and llama.cpp's stock n=8 kernel is already faster than the harness S at n=8 by 19-25%,
   so L4 ends up +5.5% / +12.9% on the FFN shapes vs llama.cpp. A better n=5..8 configuration for L4 (2 warps, fewer rows) might change
   this. Not tried.
4. **Harness realism caveat (as in steps 5/6):** the harness's own S is 19-58% slower than llama.cpp at n>=2 and its GEMM is ~25% slower
   than MMQ. Against llama.cpp the L4 advantage at n=4 shrinks from ~-31% to -11..-20%.

## VRAM footprint
IQ2_XS tensors of Qwen3.8-27B: 5.23 GiB (of 9.85 GiB). L4 copies: **int8 plane 18.1 GiB, int8 + scale planes 20.35 GiB**.
- Kept *alongside* the stock weights (n=1 decode must stay on stock, see above): 9.85 + 20.35 = **30.2 GiB** of the MI50's 31.98 GiB,
  leaving ~1.8 GiB for KV cache, compute buffers and the ROCm context. Not workable for the Vane server's contexts.
- Replacing IQ2_XS: 24.97 GiB, but n=1 decode would be 8-58% slower per IQ2_XS matmul.
- A partial copy only for the shapes that win at n=4 (ffn gate/up: 128 tensors = ~3.4 GiB stock -> ~13.3 GiB as L4) would still need
  ~13 GiB extra for a gain that only shows at batch 2-4.

## Verdict
L4 does not earn integration. Prefill does not benefit (decode cost is already amortized in tiled kernels); batched decode benefits
only around n=2-4 on three of four shapes, at a VRAM cost (+13..+20 GiB) the 32 GB MI50 cannot spare next to a usable KV cache.
The fast-sign-fused-only variant (step 8) remains the integration candidate.
