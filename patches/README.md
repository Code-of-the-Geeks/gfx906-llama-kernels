# Patch series (against llama.cpp `ecdd29e8b`)

Apply in order, or cherry-pick. All experimental kernels are CMake options,
default OFF — applying the series changes nothing until you enable a flag.

| patch | flag | verdict |
|---|---|---|
| 0001 base: Q2_0/TQ2_0 HIP mmvq + planar MMQ engine kernels | `GGML_HIP_Q2_0_PLANAR_MMQ` (+ rows-per-block knobs) | **keeper** — the ternary engine (prefill about +100%, decode about +48% vs pre-patch) |
| 0002 test-backend-ops perf cases for Qwen3.8-27B decode shapes | none | measurement infra only |
| 0003 phased loads in iq2_xs vec_dot | `GGML_HIP_IQ2_XS_PIPE_MMVQ` | **rejected** (+150% at n>=2; sched_barrier kills cross-column sharing) |
| 0004 LDS-staged grid for iq2_xs mmvq | `GGML_HIP_IQ2_XS_LDS_GRID` | keeper **only for batched decode** (n=4/8 -2 to -11%; net negative at n=1) |
| 0005 VDR-4 (64 weights/thread, one wide qs load) | `GGML_HIP_IQ2_XS_VDR4` | **rejected** everywhere (unaligned loads lower the memory floor) |
| 0006 bit-exact fast sign, all mmvq instances | `GGML_HIP_IQ2_XS_FAST_SIGN=1` | partial (+3.1% decode; plain instances regress +17%) |
| 0007 fast sign restricted to fused instances | `GGML_HIP_IQ2_XS_FAST_SIGN=2` | **winner** (+4.3% tg128, bit-exact, PPL identical) |
| 0008 TQ2_0 get_rows + _nc converts + Maple-shaped MUL_MAT_ID tests | none (correctness path) | keeper — required for native-ternary MoE GGUFs |
| 0009 TQ2_0 MMQ tile kernel + per-expert J sizing | dispatch by batch size | **winner** (MoE prefill pp512 x2.9, batch>8 cliff removed) |
| 0010 branchless exact UE4M3 decode for NVFP4 scales | none (always on for HIP) | **winner** (Inkling nvfp4 tg +5.9%, bit-exact all 256 codes, greedy identical) |

`git am patches/*.patch` from a clean `ecdd29e8b` checkout. If you only want the
two winners without the rejected experiments: 0001, 0007 (needs 0006 as its
parent), 0008, 0009. Patch 0010 applies on `66a3b01b0` and is independent of the
others (it only touches `common.cuh` + test cases).
