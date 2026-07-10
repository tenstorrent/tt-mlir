// Disabled: hangs under --benchmark on the 2nd (program-cache-hit) run — device
// deadlocks at a reshape_view/binary_ng boundary; needs tt-metal investigation.
// Tracking: https://github.com/tenstorrent/tt-mlir/issues/9000
// UNSUPPORTED: true
// REQUIRES: opmodel, perf
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% optimization-level=2 experimental-weight-dtype=bfp_bf8 enable-permute-matmul-fusion=false" -o gemma_2_2b_decode_layer_ttnn.mlir %models/single_blocks_and_layers/gemma_2_2b_decode_layer.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn gemma_2_2b_decode_layer_ttnn.mlir
// RUN: ttrt run --benchmark %t.ttnn
