#!/bin/bash
# run.sh <binary> [gguf] : looks up the 4 IQ2_XS decode-shape tensors (one per shape) with gguf-py and runs the bench under a RAM cap
BIN=$1; GGUF=${2:-$HOME/llmstuff/models/Qwen3.8-27B-IQ2_XS.gguf}; GP=${GGUF_PY:-$HOME/llmstuff/llama.cpp-iq2xs-spike/gguf-py}
ARGS=$(python3 -c "
import sys; sys.path.insert(0, '$GP')
from gguf import GGUFReader
r = GGUFReader('$GGUF')
want = ['blk.3.ffn_gate.weight', 'blk.8.ffn_down.weight', 'blk.0.attn_gate.weight', 'blk.0.ssm_out.weight']
by = {t.name: t for t in r.tensors}
out = []
for n in want:
    t = by[n]; assert t.tensor_type.name == 'IQ2_XS', n
    k, m = int(t.shape[0]), int(t.shape[1])
    out += [n, str(m), str(k), str(int(t.data_offset)), str(int(t.n_bytes))]
print(' '.join(out))")
systemd-run --user --scope -q -p MemoryMax=10G -p MemorySwapMax=1G "$BIN" "$GGUF" $ARGS
