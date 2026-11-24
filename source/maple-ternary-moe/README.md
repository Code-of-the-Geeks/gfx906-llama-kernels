# Maple ternary-MoE spike (step 1 of the ternary-MoE-offload ladder)

Goal: prove native-ternary MoE tensors run on our stack; measure the CPU-expert path
our end goal (big ternary MoE + RAM offload) depends on.

## Findings (2026-11-01)
- Official GGUFs: deepgrove/maple-preview-GGUF — TQ2_0 + Q4_K head = 5.5 GiB, arch maple
  (24 layers, 256 experts x8 active, SWA-512:GA 3:1, ctx 131072). Tensor histogram:
  168x TQ2_0 (type 35) + Q4_K head + F32 norms/router/embeddings.
- DeepGrove fork (deepgrove-ai/llama.cpp, commit 8ce8ca6c6 maplesupport): arch + graph +
  converter only — TQ2_0 kernels are CPU-ONLY (ops.cpp); no CUDA/HIP path in their tree.
  GPU kernels exist only in stamsam fork (type-35 vec_dot) — port target for step 1b.
- llama.cpp fork build (CPU-only, build-cpu): loads and answers correctly (Paris smoke).
- llama-bench CPU 12 threads (Ryzen 5 3600), pp512 307.4 +/- 2.8, tg128 66.5 +/- 1.4 t/s.
  => native ternary MoE decode at 66 t/s on our exact CPU class WITHOUT any GPU: the
  offload ceiling for step 2/3 looks comfortable (Flash-Next 6B active ~= 4x this traffic).
## Next steps
- 1b: port type-35 vec_dot + MUL_MAT_ID dispatch into build16 HIP (stamsam reference),
  run -ngl 99; compare tg128 vs CPU 66.5.
- sanity-vs-quality: wikitext PPL + fact set vs Bonsai before any deployment thought.

## Step 1b: GPU offload via build17 (TQ2_0 convert graft) — 2026-11-01 night

Ported stamsam dequantize_tq2_0 + convert-table registrations into mainline tree
(build17; build16 untouched). Fixed the GGML_ASSERT(convert_func != nullptr) crash.
Functional: chat + code tasks correct, co-resident with Vane (VRAM ~23 GB combined).

llama-bench (build17, -fa on, -r 3):
| config | pp512 | pp2048 | tg128 |
|---|---|---|---|
| full GPU (-ngl 99) | 830.9 ± 18.1 | 853.8 ± 4.1 | 200.0 ± 3.7 |
| -ncmoe 64 (experts partly in RAM) | 308.0 ± 7.2 | 323.4 ± 0.7 | 80.3 ± 0.6 |
| -ncmoe 999 (all MoE in RAM) | 309.3 ± 7.5 | 324.1 ± 0.9 | 74.8 ± 1.4 |
| CPU-only (maple-fork build-cpu, earlier) | 307.4 ± 2.8 | — | 66.5 ± 1.4 |

Findings:
- Full-GPU ternary MoE decode 200 t/s on MI50 — patched TQ2 mmvq handles MUL_MAT_ID.
- Expert offload via -ncmoe costs ~60% decode (200→75-80): PCIe-bound. Offload only
  worth it when model cannot fit VRAM; at Flash-Next scale the dense+KV still wins GPU.
- Maple caveat: burns entire completion budget in reasoning block on list/factual prompts
  (14k tokens, no answer at max_tokens 4096); enable_thinking:false not honored by template.

## Rung 1 (2026-11-24): full-GPU on spike tree
- Subtask 1, q2_0/tq2_0 MUL_MAT_ID gap analysis + patch plan: see [RESULTS.md](RESULTS.md).
- Subtask 2, P0 (TQ2_0 get_rows/_nc converts + Maple-shaped MMID tests), spike commit 27ca6cb51: all gates green, tg128 203.6. See RESULTS.md.
- Subtask 3, P1 (TQ2_0 MMQ + GCN per-expert J sizing), spike commit 197884098: pp9 167.7 → 537, pp16 842, pp512 838 → 2407, tg128 199.6, all suites OK. See RESULTS.md.
- Subtask 4, quality sanity: TQ2_0 vs Q4_K_M PPL parity (parent rerun: 84.32 vs 83.84, +0.57%), 5-prompt parity at temp 0; verdict = ternary MoE usable as fast short-task worker, not yet a general engine model (thinking-budget problem). See RESULTS.md.
- Outcome rung 1: CLOSED. TQ2_0 native ternary MoE = model-grade answers at half the bytes and full GPU speed on one card. Open decisions: graft P0+P1 into production build line; head-to-head Maple vs Gyro-M/Bonsai eval; stream-k polish for dense batch 9-16.
