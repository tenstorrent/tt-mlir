# Milestone: Single-Chip Qwen 3 4B Inference Debugging

## Description

End-to-end Chisel debugging works for single-chip Qwen 3 4B inference in tt-mlir. Compile a Qwen 3 4B layer, run it through TTRT with Chisel callbacks enabled, and produce a per-op accuracy report (PCC, atol, rtol).

## Test Files

- `test/ttmlir/models/single_blocks_and_layers/qwen_3_4b_decode_layer.mlir`
- `test/ttmlir/models/single_blocks_and_layers/qwen_3_4b_decode_block.mlir`
- `test/ttmlir/models/single_blocks_and_layers/qwen_3_4b_prefill_layer.mlir`
