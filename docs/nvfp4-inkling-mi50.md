# Running Inkling-2-Mini native NVFP4 on an MI50 (gfx906)

Inkling-2-Mini is an NVFP4-QAT model: its experts are trained in NVFP4, so the
native NVFP4 GGUF is bit-faithful to the checkpoint (perplexity 7.6669 vs 7.6679
for q8_0 on wikitext-2 c=512) and scores better on our coding battery (6/8 vs
5/8). It fits a 32 GB MI50 with room for KV cache.

## 1. Model

```
hf download ggml-org/Inkling-2-Mini-GGUF Inkling-2-Mini-NVFP4.gguf --repo-type model
```

~22.9 GB. Loads fully on GPU (~25 GB VRAM incl. KV at 16k ctx) — you need the
32 GB MI50, not a 16 GB card.

## 2. Build llama.cpp

Use this repo's `build.sh` (pins the tested upstream commit and applies the
patch series). Patch 0010 (`0010-HIP-branchless-exact-UE4M3-decode-for-NVFP4-scales-g.patch`)
is optional but recommended: branchless UE4M3 scale decode, bit-exact for all
256 UE4M3 codes, +3.6 % prefill / +5.9 % decode on NVFP4 tensors.

## 3. Serve

```
HSA_OVERRIDE_GFX_VERSION=9.0.6 llama-server \
  -m Inkling-2-Mini-NVFP4.gguf --alias inkling-2-mini \
  --jinja -ngl 99 -fa on -lm none -c 16384 --port 8083
```

- `HSA_OVERRIDE_GFX_VERSION=9.0.6` is required for gfx906 under ROCm.
- `-lm none` keeps logits on GPU (fits on 32 GB; drop it if VRAM is tight).

Measured on MI50 (stock build, no patch 0010): pp512 392.8 t/s, tg256 26.6 t/s;
served decode ~24–25 t/s at effort 0.7–0.9. Patch 0010 moves the hybrid
(nvfp4 experts + q8_0 dense layers) to pp ~476 / tg ~30 t/s.

## 4. Reasoning effort (important)

Inkling's chat template embeds a numeric effort knob; the default is **0.9**.
Effort map: `none 0.0, minimal 0.1, low 0.2, medium 0.7, high 0.9, xhigh/max 0.99`.

Pass it per request via `chat_template_kwargs`:

```json
{"model": "inkling-2-mini",
 "messages": [...],
 "chat_template_kwargs": {"reasoning_effort": "low"}}
```

Measured coding battery (8 tasks) vs effort: 0.9 → 6/8, 0.7 → 7/8, 0.5 → 8/8 —
lower effort is both **faster** (-45 % wall time) and **better**. For agentic /
patch-style coding use `minimal`–`low` (0.1–0.2); the model then emits tight,
correct diffs without reasoning bloat.

## 5. Verify the template

```
curl -s localhost:8083/apply-template -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}],"chat_template_kwargs":{"reasoning_effort":"low"}}'
```

The rendered prompt should contain a `Thinking effort level:` line with the
numeric value. If it does not, the embedded template is missing the effort map —
supply one with `--chat-template-file`.
