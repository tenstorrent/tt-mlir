// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --ttnn-decompose-layouts -o %t %s
// RUN: FileCheck %s --input-file=%t
//
// We expect AffineMaps to be uniqued in MLIR, so identical types (e.g., return operand and function result)
// should compare equal if their affine expressions are mathematically equivalent (among other properties).
//
// However, prior to this fix, we were constructing a new AffineMap programmatically for the return type,
// which produced an equivalent but structurally different expression tree from the one parsed in the function signature.
//
// This led to type mismatches during verification, even though the printed types appeared identical:
//
//   error: type of return operand 0 (...) doesn't match function result type (...) in function @forward
//
// The mismatch occurred because MLIR compares AffineMaps by structural identity, not by semantic equality.
//
// We resolved this by simplifying and canonicalizing the programmatically generated affine map,
// ensuring its internal structure matches the parsed one and enabling proper unification.

#dram = #ttnn.buffer_type<dram>
#ttnn_layout_dram_rm = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 8 + d1 * 8 + d2, d3), <1x1>, memref<8x2048xbf16, #ttnn.buffer_type<dram>>, <interleaved>>
#ttnn_layout_dram_tile = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x64x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>
module {
  func.func @forward(%arg0: tensor<1x1x8x2048xbf16, #ttnn_layout_dram_rm>) -> tensor<1x1x8x2048xbf16, #ttnn_layout_dram_tile> {
    %0 = "ttnn.to_layout"(%arg0) <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<1x1x8x2048xbf16, #ttnn_layout_dram_rm>) -> tensor<1x1x8x2048xbf16, #ttnn_layout_dram_tile>
    // CHECK: "ttnn.to_layout"
    // CHECK-SAME: layout = #ttnn.layout<tile>
    return %0 : tensor<1x1x8x2048xbf16, #ttnn_layout_dram_tile>
  }
}
