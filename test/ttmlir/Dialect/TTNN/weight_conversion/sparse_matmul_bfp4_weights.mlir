// RUN: ttmlir-opt --ttnn-weight-dtype-conversion="target-dtype=bfp_bf4" %s | FileCheck %s

// Test that the weight dtype conversion pass correctly:
// 1. Inserts a ttnn.typecast operation before the sparse_matmul
// 2. Converts the weight tensor (B operand) to bfp_bf4
// 3. Updates the sparse_matmul to use the typecast result
// 4. Keeps the output of sparse_matmul as bf16 (unchanged)

#dram = #ttnn.buffer_type<dram>

module attributes {} {
  func.func @test_sparse_matmul_bfp4_weights(%arg0: tensor<2x4x32x2880xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 4 + d1, d2, d3), <1x1>, memref<8x1x90x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>, %arg1: tensor<1x32x2880x5760xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1, d2, d3), <1x1>, memref<32x90x180x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg2: tensor<2x4x1x32xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 4 + d1, d2, d3), <1x1>, memref<8x1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>) -> tensor<2x4x1x32x32x5760xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3, d4, d5) -> (d0 * 4 + d1, d2 * 32 + d3, d4, d5), <1x1>, memref<8x32x1x180x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>> {

    // CHECK-LABEL: func.func @test_sparse_matmul_bfp4_weights

    // CHECK: %[[TYPECAST:.*]] = "ttnn.typecast"(%arg1)
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bfp_bf4>

    // CHECK: "ttnn.sparse_matmul"(%arg0, %[[TYPECAST]], %arg2)
    %0 = "ttnn.sparse_matmul"(%arg0, %arg1, %arg2) <{is_input_a_sparse = false, is_input_b_sparse = true, nnz = 0 : i64}> : (tensor<2x4x32x2880xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 4 + d1, d2, d3), <1x1>, memref<8x1x90x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>, tensor<1x32x2880x5760xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1, d2, d3), <1x1>, memref<32x90x180x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>, tensor<2x4x1x32xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 4 + d1, d2, d3), <1x1>, memref<8x1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>) -> tensor<2x4x1x32x32x5760xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3, d4, d5) -> (d0 * 4 + d1, d2 * 32 + d3, d4, d5), <1x1>, memref<8x32x1x180x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>

    return %0 : tensor<2x4x1x32x32x5760xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3, d4, d5) -> (d0 * 4 + d1, d2 * 32 + d3, d4, d5), <1x1>, memref<8x32x1x180x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>
  }
}
