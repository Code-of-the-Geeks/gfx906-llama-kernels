# gfx906 / GCN 5.1 instruction notes (measured)

Canonical reference: AMD **"GCN 5.1 Instruction Set Architecture"** whitepaper
(the gfx906/Vega document). What follows are our measured additions — things the
whitepaper does not tell you when writing llama.cpp-style dequant kernels.

## Instruction rates

| instruction | rate | notes |
|---|---|---|
| `v_dot4_i32_i8` | full rate | dp4a; the workhorse for int8 quantized dots on gfx906. No MFMA exists on this ISA generation for int8. |
| `v_mul_lo_u32` | **quarter rate** | any 32x32 multiply that LLVM emits as `v_mul_lo_u32` costs 4x. This is the trap below. |
| `v_mul_u32_u24` | full rate | 24-bit multiply — keep products under 2^24 and use inline asm to force it. |
| `v_bcnt_per_thread` (`v_bcntperm`) | moderate | popcount tricks beat LUT gathers at this wave size. |
| byte-wise ops (`__vcmpne4`, `__vsub4`) | emulated | no byte-wise ALU on GCN; these expand to masks/shuffles. Avoid. |
| buffer/image loads vs flat | buffer faster | `buffer_load_dwordx2/x4` are the aligned 128-bit path; unaligned scalar loads fall to `vmem` dword path and lose (VDR-4 experiment: loads-only floor 347 to 233 GB/s). |

## Compiler gotchas (all observed, all fixed with inline asm)

1. **`__umul24()` becomes `v_mul_lo_u32`.** The HIP intrinsic does not guarantee
   the 24-bit instruction; LLVM lowered it to the quarter-rate 32-bit multiply.
   Inline asm `v_mul_u32_u24` restored full rate.
2. **`(b << 8) - b` is strength-reduced to `b * 255`.** Cute algebra, but the
   multiply is quarter-rate. Write the shift-subtract explicitly or in asm.
3. **Sign-extension idioms expand.** `(x ^ mask) + carry` patterns survive well;
   table lookups for sign application do not (see gathers below).

## Gathers vs ALU

Vector-memory LUT gathers on gfx906 cost more than recomputing the value in
registers. Three independent confirmations:

- IQ2_XS ksigns64 table variant: **+21 to +37% slower** despite 43% fewer
  instructions.
- IQ2_XS grid-in-LDS variant: gathers worth only 3-6%; net negative at n=1
  (+16.5% on gate/up shapes where the LDS copy is not amortized), net positive
  only at batched decode (n=4/8: -2 to -11%).
- Ternary sign tables: same pattern as Q2_0 work — computed masks beat tables.

Rule of thumb: if the "table" is derivable from bits you already hold in one
register, compute it. The winning fast-sign trick:

```
b = per-byte 0/1 from 8 sign bits via v_mul_u32_u24 with constant 0x00204081
mask = (0x80808080 - b) ^ 0x80808080      // carry-free because magnitudes < 128
signed_byte = (g ^ mask) + b
```

Loop ISA went 185 to 137 instructions, `v_mul_lo_u32` 14 to 2, VGPR 38 to 32,
occupancy 6 to 8 waves/SIMD. End-to-end: tg128 +4.3% (fused instances only;
applying it to plain mmvq calls was +17% slower there — restrict by instance).

## Occupancy rules

- Wave size 64. Each SIMD is 16 lanes wide x4 queues; occupancy = waves/SIMD.
- VGPR budget dominates: 38 VGPR gives 8 waves (with other limits 6); above
  about 100 VGPR you fall to 2-3 waves and latency hiding dies.
- Full unrolling of dequant loops is a spill disaster: one fused variant hit
  49 to 72 VGPR (occ 4 to 3) just from widening loads; `#pragma unroll 1` fixed
  it. Manual partial unroll beats compiler full unroll here.
- `sched_barrier` in decode kernels destroys cross-column work sharing: the
  phased-load experiment (+0.3% at n=1, **+150% at n>=2**) because 6-7
  waves/SIMD already hide the load chain — barriers removed the overlap the
  other columns relied on.

## Bandwidth reality

Effective decode bandwidth for quantized matmuls on this card: about 170-200
GB/s (iq2_xs shapes measured 160-189 GB/s while Q4_K/Q8_0 reach 400-480 GB/s —
the gap is instruction cost, not the memory system; loads-only floor for the
same bytes: 347 GB/s). Implication: decode at n=1 is compute-bound for iq2_xs,
so pre-dequant "spill-RAM" layouts lose (int8-plane experiment: streams at
466-614 GB/s but +8 to +58% end-to-end at n=1). MMQ already amortizes tile
decode across columns, so pre-dequant does not help prefill either (+24 to +29%
at n=512).
