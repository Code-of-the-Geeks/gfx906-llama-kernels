#!/bin/bash
# regenerate iq2xs_grid.h and ksigns64.h from a llama.cpp tree: gen_grid.sh <llama.cpp root>
H=$(dirname "$0"); C="$1/ggml/src/ggml-common.h"
tab() { echo "// $1 ($3 x uint64) extracted from llama.cpp ggml/src/ggml-common.h (ecdd29e8b) by gen_grid.sh"; echo "static const uint64_t $2[$3] = {"
        awk -v pat="GGML_TABLE_BEGIN\\\\(uint64_t, $1, $3\\\\)" '$0 ~ pat {f=1;next} f&&/GGML_TABLE_END/{exit} f{print}' "$C"; echo "};"; }
tab iq2xs_grid h_iq2xs_grid 512 > "$H/iq2xs_grid.h"
tab ksigns64 h_ksigns64 128 > "$H/ksigns64.h"
