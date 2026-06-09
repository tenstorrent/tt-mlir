#!/bin/bash
# Regenerate all softmax flatbuffers (ttnn + d2m fused) from the MLIR sources in mlir/.
# Run from the tt-mlir repo root with env activated.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPT=build/bin/ttmlir-opt
TR=build/bin/ttmlir-translate
OUT="$HERE/../../../../_fb"   # repo-root/_fb
OUT="$(cd "$(dirname "$OUT")" 2>/dev/null && pwd)/_fb" 2>/dev/null || OUT="_fb"
mkdir -p "$OUT"

# d2m fused: full backend pipeline -> ttmetal flatbuffer
REST="d2m-insert-scratch-buffers,d2m-generic-apply-interchange,d2m-generate-outer-loops,d2m-mark-synchronized-buffers,d2m-allocate,d2m-lower-multicast-loads,d2m-generic-lower-to-explicit-form,canonicalize,d2m-be-pipeline{use-tile-matmul=0},d2m-to-ttkernel-pre-emitc-pipeline,d2m-to-ttmetal-pipeline,ttkernel-hoist-inits,d2m-emitc-pipeline"
for sz in 1x1 2x2 3x3 4x4; do
  src="$HERE/mlir/d2m_fused_${sz}.mlir"
  [ -f "$src" ] || continue
  $OPT --pass-pipeline="builtin.module($REST)" "$src" -o "$OUT/d2m_fused_${sz}_lowered.mlir" 2>/dev/null \
    && $TR --ttmetal-to-flatbuffer "$OUT/d2m_fused_${sz}_lowered.mlir" -o "$OUT/d2m_fused_${sz}.ttm" 2>/dev/null
  echo "d2m_fused_${sz}: $(ls -la "$OUT/d2m_fused_${sz}.ttm" 2>/dev/null | awk '{print $5}') bytes"
done

# ttnn: ttir.softmax -> ttnn backend pipeline -> flatbuffer.
# NOTE: mlir/ttnn_softmax_*.mlir already have numericStable=true. If regenerating from a
# fresh ttir.softmax, the default lowering is numericStable=false -> flip it before translate.
for sz in 1x1 2x2 3x3; do
  src="$HERE/mlir/ttnn_softmax_${sz}.mlir"
  [ -f "$src" ] || continue
  $TR --ttnn-to-flatbuffer "$src" -o "$OUT/ttnn_softmax_${sz}.ttnn" 2>/dev/null
  echo "ttnn_softmax_${sz}: $(ls -la "$OUT/ttnn_softmax_${sz}.ttnn" 2>/dev/null | awk '{print $5}') bytes ($(grep -oE 'numericStable = (true|false)' "$src" | head -1))"
done
echo "flatbuffers in: $OUT"
