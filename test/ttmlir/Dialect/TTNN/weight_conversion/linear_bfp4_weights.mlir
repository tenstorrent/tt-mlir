// RUN: ttmlir-opt --ttnn-weight-dtype-conversion="target-dtype=bfp_bf4" %s | FileCheck %s

// Test that the weight dtype conversion pass correctly:
// 1. Inserts a ttnn.typecast operation before the linear
// 2. Converts the weight tensor (B operand) to bfp_bf4
// 3. Updates the linear to use the typecast result
// 4. Keeps the output of linear as bf16 (unchanged)
// 5. Keeps the activation input (A operand) and bias of linear as bf16 (unchanged)

#dram = #ttnn.buffer_type<dram>

module attributes {} {
  func.func @test_linear_bfp4_weights(%arg0: tensor<1024xbf16, #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x32x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>, %arg1: tensor<1024x2048xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x64x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg2: tensor<2048x2048xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x64x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>, %arg3: tensor<2048x1024xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x32x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>) -> tensor<2048x1024xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x32x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>> {

    // CHECK-LABEL: func.func @test_linear_bfp4_weights

    // CHECK: %[[TYPECAST:.*]] = "ttnn.typecast"(%arg1)
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bfp_bf4>
    // CHECK-SAME: -> tensor<1024x2048x!ttcore.tile<32x32, bfp_bf4>,

    // CHECK: "ttnn.linear"(%arg2, %[[TYPECAST]], %arg3)
    // CHECK-SAME: -> tensor<2048x1024xbf16,
    %0 = "ttnn.linear"(%arg2, %arg1, %arg3) <{transpose_a = false, transpose_b = true}> : (tensor<2048x2048xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x64x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>, tensor<1024x2048xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x64x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>, tensor<2048x1024xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x32x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>) -> tensor<2048x1024xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x32x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>

    // CHECK: return {{.*}} : tensor<2048x1024xbf16,
    return %0 : tensor<2048x1024xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x32x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>
  }
}
