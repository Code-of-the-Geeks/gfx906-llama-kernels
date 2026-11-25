# gfx906-llama-kernels

Kernel notes, benchmarks and patch series for running **llama.cpp on AMD gfx906**
(Vega 20 / Radeon Pro Vega II / MI50 / Radeon VII) with ROCm/HIP.

Everything here was measured on one 32 GB Vega II card (ROCm 6.x, HIP build of
llama.cpp pinned to upstream commit `ecdd29e8b`). We publish the findings —
including the levers that failed — so anyone running an LLM on this GPU class
can pick up what works and skip what does not.

**Not a fork and not maintained.** The patches are a series against one pinned
commit; expect conflicts on newer trees. The documentation is the valuable part.

## Headline results

Measured head-to-head vs stock llama.cpp: **[benchmarks.html](benchmarks.html)** (root of this repo).

| change | effect | how |
|---|---|---|
| IQ2_XS fast sign application (fused mmvq only) | **tg128 +4.3%** (19.5 to 20.3 t/s on Qwen3.8-27B-IQ2_XS), bit-exact output | replace the grid sign lookup with a computed per-byte mask (one `v_mul_u32_u24`); loop ISA 185 to 137 instructions, VGPR 38 to 32 |
| TQ2_0 MMQ (native ternary MoE prefill) | **pp9 167 to 509 t/s, pp512 821 to 2394** on Maple 20B-A1B TQ2_0 | tile loader unpacking ternary weights to int8, reusing the q8_0 dp4a dot; plus per-expert J sizing for GCN |
| TQ2_0 get_rows + non-contiguous converts | makes native-ternary MoE GGUFs loadable/runnable on HIP at all | convert-table registration + MUL_MAT_ID regression cases at the batch 8/9 boundary |
| Q2_0 engine kernels (patch 0001) | full ternary engine: planar MMQ prefill about +100%, rows-per-block decode about +48% | bit-plane dp4a kernels; see `source/iq2-xs-mmq/README.md` history |

**Results so far:** these kernels make native-ternary MoE GGUFs (2.06 bpw) run
fully on-GPU at about 200 tok/s decode on a single Vega II. The model behind
that benchmark (a 20B-A1B ternary preview) was then evaluated head-to-head
against our purpose-trained production model: fast and well-formed, but it
failed execution-graded coding tasks (2/8 vs 8/8) and produced misattributed
citations and invented facts in grounded search synthesis. The kernels and the
ISA findings stand on their own; the speed ceiling they unlock is real, the
current ternary preview models are not yet usable for agentic or search duties.

## Why gfx906 is different

- **No MFMA.** Tensor-core math does not exist; int8 work rides `v_dot4_i32_i8`
  (dp4a), which runs at full rate on gfx906.
- **Gathers are expensive.** LUT gathers through the vector memory path cost more
  than ALU recompute — we confirmed this three independent times (sign-table
  variants measured +21 to +37% slower).
- **The compiler is a trap.** `__umul24` lowers to quarter-rate `v_mul_lo_u32`;
  LLVM strength-reduces `(b << 8) - b` into `b * 255` (quarter-rate again);
  byte-wise intrinsics (`__vcmpne4`, `__vsub4`) are emulated. Inline asm is the
  fix — see `docs/isa-notes.md`.
- **Occupancy dominates.** A kernel at 100+ VGPR drops to 2 waves/SIMD and loses
  everywhere; `#pragma unroll 1` fixed our first spill disaster.

## Layout

```
docs/isa-notes.md      measured instruction rates, compiler gotchas, occupancy rules
docs/methodology.md    how every number was produced (ISA dumps, loads-only floor, ablations)
patches/               9-patch series against ecdd29e8b (see patches/README.md for which win)
probes/                standalone microbench harnesses (layout bench, ISA stats)
source/iq2-xs-mmq/     full IQ2_XS kernel spike: README + RESULTS + logs
source/maple-ternary-moe/  native-ternary MoE spike: README + RESULTS + logs
```

## Applying the patches

```bash
git checkout ecdd29e8b
git am patches/0001-*.patch ... patches/0009-*.patch   # or cherry-pick individually
```

All optional kernels are flag-guarded CMake options, default OFF:
`GGML_HIP_Q2_0_PLANAR_MMQ`, `GGML_HIP_IQ2_XS_PIPE_MMVQ`, `GGML_HIP_IQ2_XS_LDS_GRID`,
`GGML_HIP_IQ2_XS_VDR4`, `GGML_HIP_IQ2_XS_FAST_SIGN` (1 = all mmvq instances,
2 = fused only — the winner), plus the TQ2_0 MMQ from patch 0009.

Build note for our environment: the ROCm runtime on the box provides
`libamdhip64.so.6`; llama.cpp HIP links against `.so.7`. We patched the ELF
dependency string post-build (`probes/build.sh` does it). On a stock ROCm install
this step is unnecessary.

## Verification standard used throughout

Every kernel change had to be **bit-exact or explain its error** (exhaustive
self-tests, PPL bit-identical checks, KLD against a stock-vs-stock control) and
show end-to-end llama-bench movement, not just microbench wins. Negative results
are recorded in full — six rejected levers with data are in the source dirs.

## Attribution

llama.cpp is MIT (ggml-org). Patches here are derivative work under the same
terms. GCN 5.1 instruction behaviour is from AMD's public "GCN 5.1 Instruction
Set Architecture" whitepaper plus our own `llvm-objdump` measurements.

Hardware reference: 32 GB Vega II (gfx906:sramecc+:xnack-, wave 64), power cap
160 W — about 170-200 GB/s effective decode bandwidth for quantized matmuls.
